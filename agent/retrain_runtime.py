"""Configurable v11 training/inference for single-change experiments.

Runs in the competition Python 3.11 environment. With ``baseline_config()``
it reproduces ``train.training_matrix``/``train.fit``/``match_lib.predict``
exactly (checked by the worker's ``parity`` mode). Hard negatives are chosen
inside each inner training fold only; validation rows are predicted from their
own input plus artifacts fitted on the outer training rows.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile
from typing import Any, Iterator

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from agent.team_v11.source import train as trainer
from .retrain_config import KNOBS, validate_config

ml = trainer.ml
FEATURES = tuple(ml.FEATURE_NAMES)
_COL = {name: i for i, name in enumerate(FEATURES)}
GAP_COLUMNS = ("gap_gross", "gap_intl", "gap_best")
NUMBER_ASYMMETRY = ("num_q_only", "num_c_only")
NUMBER_ALL = ("num_equal", "num_conflict", "q_has_num", "c_has_num", "num_q_only", "num_c_only")
FLAG_TYPE = ("flag_prob", "flag_seen", "type_prob", "type_seen")
HARDNESS = ("s_compact", "s_skel_rom", "s_var_ratio", "s_learn_ratio", "s_alpha_ratio")
CATBOOST_PROFILE = dict(loss_function="YetiRank", iterations=600, learning_rate=0.08,
                        depth=6, random_seed=42, thread_count=8, verbose=False,
                        allow_writing_files=False, task_type="CPU")
MLP_PROFILE = {"hidden_sizes": (64, 32), "activation": "ReLU", "dropout": 0.1,
               "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
               "epochs": 20, "batch_queries": 128, "input_clip": 10.0,
               "seed": 42, "thread_count": 8, "device": "cpu",
               "loss": "query_listwise_softmax"}


def _mlp_network(input_dim: int):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(input_dim, MLP_PROFILE["hidden_sizes"][0]), nn.ReLU(),
                         nn.Dropout(MLP_PROFILE["dropout"]),
                         nn.Linear(MLP_PROFILE["hidden_sizes"][0], MLP_PROFILE["hidden_sizes"][1]),
                         nn.ReLU(), nn.Dropout(MLP_PROFILE["dropout"]),
                         nn.Linear(MLP_PROFILE["hidden_sizes"][1], 1))


def listwise_group_loss(scores, labels, groups: np.ndarray):
    """Average one softmax cross-entropy term for every query group."""
    import torch
    sizes = np.asarray(groups, dtype=np.int64)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels):
        raise ValueError("score와 label은 같은 길이의 1차원 배열이어야 합니다")
    if len(sizes) == 0 or np.any(sizes <= 0) or int(sizes.sum()) != len(scores):
        raise ValueError("group 크기는 양수이며 후보 행 수 합계와 같아야 합니다")
    losses, offset = [], 0
    for size in sizes:
        end = offset + int(size)
        group_labels = labels[offset:end]
        positives = torch.nonzero(group_labels > 0.5, as_tuple=False).flatten()
        if len(positives) != 1:
            raise ValueError("각 학습 쿼리는 정답 후보가 정확히 하나여야 합니다")
        group_scores = scores[offset:end]
        losses.append(torch.logsumexp(group_scores, dim=0) - group_scores[positives[0]])
        offset = end
    return torch.stack(losses).mean()


class MLPRanker:
    """Fixed-profile CPU listwise MLP whose candidate scores are row-independent."""

    def __init__(self) -> None:
        self.network = None
        self.input_mean_ = None
        self.input_scale_ = None
        self.input_dim_ = None
        self.training_stats_: dict[str, Any] = {}

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.input_dim_:
            raise ValueError("MLP 입력 특징 차원이 학습과 다릅니다")
        if not np.isfinite(values).all():
            raise ValueError("MLP 입력 특징에 비유한 값이 있습니다")
        normalized = (values - self.input_mean_) / self.input_scale_
        return np.clip(normalized, -MLP_PROFILE["input_clip"], MLP_PROFILE["input_clip"]).astype(
            np.float32, copy=False)

    def fit(self, features: np.ndarray, labels: np.ndarray, groups: np.ndarray):
        import torch
        import torch.nn as nn

        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32)
        sizes = np.asarray(groups, dtype=np.int64)
        if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
            raise ValueError("학습 특징은 유한한 2차원 배열이고 label 행 수와 같아야 합니다")
        if len(sizes) == 0 or np.any(sizes <= 0) or int(sizes.sum()) != len(x):
            raise ValueError("group 크기 합계가 학습 행 수와 다릅니다")
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("MLP label은 0/1이어야 합니다")
        offsets = np.concatenate(([0], np.cumsum(sizes)))
        if any(int(y[a:b].sum()) != 1 for a, b in zip(offsets[:-1], offsets[1:])):
            raise ValueError("각 학습 쿼리는 정답 후보가 정확히 하나여야 합니다")
        self.input_dim_ = int(x.shape[1])
        self.input_mean_ = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
        self.input_scale_ = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
        normalized = self._normalize(x)
        torch.set_num_threads(MLP_PROFILE["thread_count"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(MLP_PROFILE["seed"])
            self.network = _mlp_network(self.input_dim_)
            optimizer = torch.optim.AdamW(self.network.parameters(),
                                          lr=MLP_PROFILE["learning_rate"],
                                          weight_decay=MLP_PROFILE["weight_decay"])
            xt = torch.from_numpy(normalized)
            yt = torch.from_numpy(y)
            rng = np.random.default_rng(MLP_PROFILE["seed"])
            batches = 0
            self.network.train()
            for _ in range(MLP_PROFILE["epochs"]):
                query_order = rng.permutation(len(sizes))
                for start in range(0, len(query_order), MLP_PROFILE["batch_queries"]):
                    query_ids = query_order[start:start + MLP_PROFILE["batch_queries"]]
                    row_ids = np.concatenate([np.arange(offsets[q], offsets[q + 1]) for q in query_ids])
                    batch_sizes = sizes[query_ids]
                    optimizer.zero_grad(set_to_none=True)
                    scores = self.network(xt[row_ids]).reshape(-1)
                    loss = listwise_group_loss(scores, yt[row_ids], batch_sizes)
                    loss.backward()
                    optimizer.step()
                    batches += 1
            self.network.eval()
        self.training_stats_ = {"ranker_family": "mlp_listwise_v1_cpu", "queries": int(len(sizes)),
                                "candidate_rows": int(len(y)), "optimizer_batches": batches,
                                "epochs": MLP_PROFILE["epochs"], "normalizer_fit_rows": int(len(x)),
                                "zero_variance_features": int((scale <= 1e-6).sum())}
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        import torch
        if self.network is None:
            raise ValueError("MLP 랭커가 아직 학습되지 않았습니다")
        normalized = self._normalize(features)
        self.network.eval()
        with torch.no_grad():
            parts = [self.network(torch.from_numpy(normalized[start:start + 32768])).reshape(-1).numpy()
                     for start in range(0, len(normalized), 32768)]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)

    def save_model(self, path: str | Path) -> None:
        import torch
        if self.network is None:
            raise ValueError("학습되지 않은 MLP는 저장할 수 없습니다")
        torch.save({"profile": MLP_PROFILE, "input_dim": self.input_dim_,
                    "input_mean": torch.from_numpy(self.input_mean_.copy()),
                    "input_scale": torch.from_numpy(self.input_scale_.copy()),
                    "feature_names": list(FEATURES), "state_dict": self.network.state_dict(),
                    "training_stats": self.training_stats_}, path)

    @classmethod
    def load_model(cls, path: str | Path):
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["profile"] != MLP_PROFILE:
            raise ValueError("저장된 MLP 프로필이 현재 고정 프로필과 다릅니다")
        if payload["feature_names"] != list(FEATURES):
            raise ValueError("저장된 MLP 특징 순서가 현재 v11 특징과 다릅니다")
        model = cls()
        model.input_dim_ = int(payload["input_dim"])
        model.input_mean_ = payload["input_mean"].numpy().astype(np.float32, copy=False)
        model.input_scale_ = payload["input_scale"].numpy().astype(np.float32, copy=False)
        model.training_stats_ = payload["training_stats"]
        model.network = _mlp_network(model.input_dim_)
        model.network.load_state_dict(payload["state_dict"])
        model.network.eval()
        return model

    def feature_importance(self) -> dict[str, float]:
        if self.network is None:
            raise ValueError("MLP 랭커가 아직 학습되지 않았습니다")
        weights = self.network[0].weight.detach().cpu().numpy()
        values = np.mean(np.abs(weights), axis=0)
        total = float(values.sum()) or 1.0
        return {name: round(float(value / total), 6) for name, value in zip(FEATURES, values)}

    def model_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(sorted(MLP_PROFILE.items())).encode("utf-8"))
        digest.update("\n".join(FEATURES).encode("utf-8"))
        digest.update(array_digest(self.input_mean_, self.input_scale_).encode("ascii"))
        for name, tensor in sorted(self.network.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(array_digest(tensor.detach().cpu().contiguous().numpy()).encode("ascii"))
        return digest.hexdigest()


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str((values.dtype.str, values.shape)).encode("ascii"))
        digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


@contextmanager
def v11_globals(config: dict[str, str]) -> Iterator[None]:
    """v11 submission settings (--no-dict, vocab off) plus the monotone knob; restored after."""
    saved = (trainer.NO_DICT, ml.USE_VOCAB, ml.VOCAB_TOP, trainer.USE_MONOTONE)
    trainer.NO_DICT, ml.USE_VOCAB, ml.VOCAB_TOP = True, False, 3
    trainer.USE_MONOTONE = config["monotone_constraints"] == "on"
    try:
        yield
    finally:
        trainer.NO_DICT, ml.USE_VOCAB, ml.VOCAB_TOP, trainer.USE_MONOTONE = saved


def zeroed_columns(config: dict[str, str]) -> tuple[str, ...]:
    columns: tuple[str, ...] = ()
    if config["number_features"] == "no_asymmetry":
        columns += NUMBER_ASYMMETRY
    elif config["number_features"] == "none":
        columns += NUMBER_ALL
    if config["flag_type_features"] == "off":
        columns += FLAG_TYPE
    return columns


def transform_features(features: np.ndarray, config: dict[str, str]) -> tuple[np.ndarray, dict]:
    """Feature-level knobs, applied identically to training and validation matrices."""
    out = features
    stats = {"nan_gap_cells": 0, "zeroed_columns": list(zeroed_columns(config))}
    if config["tonnage_missing_encoding"] == "nan" or stats["zeroed_columns"]:
        out = features.copy()
    if config["tonnage_missing_encoding"] == "nan":
        for name in GAP_COLUMNS:
            column = out[:, _COL[name]]
            missing = column >= 2.0  # relative_gap() returns exactly 2.0 only for a missing side
            stats["nan_gap_cells"] += int(missing.sum())
            column[missing] = np.nan
    for name in stats["zeroed_columns"]:
        out[:, _COL[name]] = 0.0
    return out, stats


def _select_negatives(feats: np.ndarray, labels: np.ndarray, top_n: int) -> np.ndarray:
    """Row mask keeping every positive and the top_n hardest negatives (ties: candidate order)."""
    hardness = feats[:, [_COL[name] for name in HARDNESS]].max(axis=1)
    negatives = np.flatnonzero(labels == 0)
    order = negatives[np.argsort(-hardness[negatives], kind="mergesort")][:top_n]
    keep = labels == 1
    keep[order] = True
    return keep


def training_matrix(train: pd.DataFrame, catalog: pd.DataFrame, index, config: dict[str, str],
                    inner: int = 5):
    """Same inner GroupKFold construction as train.training_matrix, plus training knobs."""
    xs, ys, groups = [], [], []
    hits = 0
    stats = {"queries_total": 0, "queries_dropped_missing_truth": 0, "negatives_kept": 0,
             "negatives_before_selection": 0, "max_negatives_per_query": 0, "nan_gap_cells": 0,
             "zeroed_columns": list(zeroed_columns(config))}
    top_n = None if config["hard_negative_top_n"] == "all" else int(config["hard_negative_top_n"])
    splitter = GroupKFold(n_splits=inner)
    for fit_idx, prep_idx in splitter.split(train, groups=train["target_vessel_id"]):
        res = trainer.resources(train.iloc[fit_idx], catalog)
        rows = train.iloc[prep_idx].to_dict("records")
        queries = [ml.prepare_query(r, res) for r in rows]
        truth = [index.id_to_index[r["target_vessel_id"]] for r in rows]
        hit_flags: list[bool] = []
        cands, feats = ml.candidates_and_features(queries, index, res, force=truth, hits=hit_flags)
        hits += sum(hit_flags)
        feats, fstats = transform_features(feats, config)
        stats["nan_gap_cells"] += fstats["nan_gap_cells"]
        offset = 0
        for cand, t, hit in zip(cands, truth, hit_flags):
            block = feats[offset:offset + len(cand)]
            offset += len(cand)
            labels = (cand == t).astype(np.int8)
            stats["queries_total"] += 1
            if not hit and config["missing_truth_policy"] == "drop_query":
                stats["queries_dropped_missing_truth"] += 1
                continue
            stats["negatives_before_selection"] += int((labels == 0).sum())
            if top_n is not None:
                keep = _select_negatives(block, labels, top_n)
                block, labels = block[keep], labels[keep]
            kept = int((labels == 0).sum())
            stats["negatives_kept"] += kept
            stats["max_negatives_per_query"] = max(stats["max_negatives_per_query"], kept)
            xs.append(block)
            ys.append(labels)
            groups.append(len(labels))
    stats["queries_used"] = len(groups)
    stats["mean_negatives_per_query"] = stats["negatives_kept"] / max(len(groups), 1)
    return np.vstack(xs), np.concatenate(ys), np.asarray(groups), hits / len(train), stats


def ranker_params(config: dict[str, str]) -> dict[str, Any]:
    params = trainer.ranker_params()
    params["num_leaves"] = int(config["num_leaves"])
    params["n_estimators"] = int(config["n_estimators"])
    return params


def fit(train: pd.DataFrame, catalog: pd.DataFrame, index, config: dict[str, str]):
    validate_config(config)
    with v11_globals(config):
        x, y, groups, recall, stats = training_matrix(train, catalog, index, config)
        if config["ranker_family"] == "catboost_v1_cpu":
            from catboost import CatBoostRanker, Pool
            params = dict(CATBOOST_PROFILE)
            model = CatBoostRanker(**params)
            pool = Pool(x, label=y, group_id=np.repeat(np.arange(len(groups)), groups),
                        feature_names=list(FEATURES))
            model.fit(pool)
        elif config["ranker_family"] == "mlp_listwise_v1_cpu":
            model = MLPRanker().fit(x, y, groups)
            params = dict(MLP_PROFILE)
        else:
            params = ranker_params(config)
            model = lgb.LGBMRanker(**params)
            model.fit(x, y, group=groups, feature_name=list(FEATURES))
    stats["training_matrix_rows"] = int(len(y))
    stats["ranker_params"] = {k: v for k, v in params.items() if k != "monotone_constraints"}
    stats["monotone_constraints_applied"] = "monotone_constraints" in params
    stats["ranker_family"] = config["ranker_family"]
    stats["training_matrix_sha256"] = array_digest(x, y, groups)
    return model, recall, stats


def resources(train: pd.DataFrame, catalog: pd.DataFrame, config: dict[str, str]) -> dict:
    with v11_globals(config):
        return trainer.resources(train, catalog)


class ReusedBooster:
    """Saved baseline ranker for postprocess-only knobs (no retraining)."""

    def __init__(self, model_text: str) -> None:
        self.booster_ = lgb.Booster(model_str=model_text)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.booster_.predict(features)

    def get_params(self) -> dict[str, Any]:
        return {}


def model_sha256(model) -> str:
    if hasattr(model, "model_sha256"):
        return model.model_sha256()
    if hasattr(model, "save_model") and not hasattr(model, "booster_"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.cbm"
            model.save_model(str(path))
            return hashlib.sha256(path.read_bytes()).hexdigest()
    booster = model.booster_ if hasattr(model, "booster_") else model
    return hashlib.sha256(booster.model_to_string().encode("utf-8")).hexdigest()


def _missing(row: dict) -> dict[str, bool]:
    out = {}
    for key in ("vessel_name", "call_sign", "flag_name", "vessel_type"):
        value = row.get(key)
        out[key] = value is None or (isinstance(value, float) and np.isnan(value)) or value == ""
    for key in ("gross_tonnage", "international_gross_tonnage"):
        out[key] = ml.positive_float(row.get(key)) is None
    return out


def predict_details(model, frame: pd.DataFrame, index, res: dict, config: dict[str, str]
                    ) -> list[dict[str, Any]]:
    """Per-row prediction record; the frame must not contain the label column."""
    if "target_vessel_id" in frame.columns:
        raise AssertionError("추론 입력에 정답 열이 있습니다")
    validate_config(config)
    policy = config["override_policy"]
    with v11_globals(config):
        rows = frame.to_dict("records")
        queries = [ml.prepare_query(r, res) for r in rows]
        cand_lists, features = ml.candidates_and_features(queries, index, res)
        features, _ = transform_features(features, config)
        scores = model.predict(features)
        out, offset = [], 0
        for row, cand in zip(rows, cand_lists):
            block = features[offset:offset + len(cand)]
            local = np.asarray(scores[offset:offset + len(cand)])
            offset += len(cand)
            best = str(index.ids[cand[int(np.argmax(local))]])
            name_id = ml.lookup_by_name(row, res, index)
            callsign_id = res["callsign"].get(row.get("call_sign"))
            final, source = best, "none"
            if policy in ("callsign_and_name", "name_only") and name_id:
                final, source = name_id, "name_lookup"
            if policy in ("callsign_and_name", "callsign_only") and callsign_id is not None:
                final, source = str(callsign_id), "callsign"
            out.append({"sample_id": str(row["sample_id"]), "candidates": cand, "scores": local,
                        "features": block, "model_pred_id": best, "final_pred_id": str(final),
                        "override_source": source, "name_lookup_id": name_id,
                        "callsign_id": callsign_id, "missing": _missing(row)})
        if offset != len(features):
            raise AssertionError("후보 특징과 예측 행 수가 다릅니다")
    return out


def feature_importance(model) -> dict[str, float]:
    if isinstance(model, MLPRanker):
        return model.feature_importance()
    gains = (model.booster_.feature_importance(importance_type="gain")
             if hasattr(model, "booster_") else
             np.asarray(model.get_feature_importance(type="PredictionValuesChange")))
    total = float(gains.sum()) or 1.0
    return {name: round(float(g) / total, 6) for name, g in zip(FEATURES, gains)}


def behavior_check(knob: str | None, value: str | None, baseline_stats: dict, stats: dict,
                   override_counts: dict[str, int], model) -> dict[str, Any]:
    """Compare what a setting promised with what training/inference actually did."""
    if knob is None:
        return {"passed": True, "checks": ["baseline"]}
    checks: list[str] = []
    passed = True

    def expect(condition: bool, text: str) -> None:
        nonlocal passed
        checks.append(("ok: " if condition else "FAILED: ") + text)
        passed = passed and condition

    if knob == "ranker_family":
        if value == "catboost_v1_cpu":
            from catboost import CatBoostRanker
            expect(isinstance(model, CatBoostRanker), "CatBoostRanker actually fitted")
            expect(all(model.get_params().get(key) == expected for key, expected in CATBOOST_PROFILE.items()),
                   "fixed CPU CatBoost profile matches")
            expect(model.tree_count_ == CATBOOST_PROFILE["iterations"],
                   "all 600 trees fitted without early stopping")
            expect(model.feature_names_ == list(FEATURES), "same ordered v11 feature names")
            expect(not stats["monotone_constraints_applied"], "CatBoost profile is unconstrained")
        elif value == "mlp_ranknet_v1_cpu":
            expect(isinstance(model, MLPRanker), "MLPRanker actually fitted")
            expect(stats.get("ranker_params") == MLP_PROFILE, "fixed CPU MLP profile matches")
            expect(model.network is not None and model.network.training is False,
                   "MLP is fitted and in evaluation mode")
            expect(model.training_stats_.get("normalizer_fit_rows") == stats.get("training_matrix_rows"),
                   "normalizer was fit only on ranker training rows")
            expect(model.input_dim_ == len(FEATURES), "same ordered v11 feature width")
    elif knob == "hard_negative_top_n":
        n = int(value)
        expect(stats["max_negatives_per_query"] <= n, f"max negatives per query <= {n}")
        expect(stats["mean_negatives_per_query"] < baseline_stats["mean_negatives_per_query"],
               "mean negatives per query decreased")
    elif knob == "missing_truth_policy":
        expect(stats["queries_dropped_missing_truth"] > 0, "some training queries were dropped")
    elif knob == "tonnage_missing_encoding":
        expect(stats["nan_gap_cells"] > 0, "missing tonnage gaps encoded as NaN in training")
    elif knob in ("number_features", "flag_type_features"):
        expect(bool(stats["zeroed_columns"]), "feature columns zeroed")
        unused = all(model.booster_.feature_importance("split")[_COL[c]] == 0
                     for c in stats["zeroed_columns"])
        expect(unused, "zeroed columns unused by the trained ranker")
    elif knob == "monotone_constraints":
        expect(stats["monotone_constraints_applied"] == (value == "on"),
               "monotone constraint presence matches setting")
    elif knob in ("num_leaves", "n_estimators"):
        expect(model.get_params()[knob] == int(value), f"ranker {knob} == {value}")
    elif knob == "override_policy":
        allowed = {"callsign_only": {"none", "callsign"}, "name_only": {"none", "name_lookup"},
                   "none": {"none"}}[value]
        expect(all(count == 0 for source, count in override_counts.items() if source not in allowed),
               "disabled override sources never applied")
    else:
        expect(False, f"no behavior check defined for {knob}")
    if KNOBS[knob]["retrain"] is False:
        checks.append("postprocess-only knob: baseline ranker reused")
    return {"passed": passed, "checks": checks}
