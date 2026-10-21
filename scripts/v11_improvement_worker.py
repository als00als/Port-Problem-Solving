"""Diagnose and retrain v11 on fixed reused splits, one allow-listed change at a time.

Run with the competition Python 3.11 environment. Validation labels are removed
before prediction and joined only in the evaluation step. Files with sample or
vessel IDs go to ``<run>/local_only/`` (experiments/results is git-ignored);
the JSON written to ``--output`` holds aggregates only.

  audit    --split S --output F
  diagnose --split S --run-dir D --output F      baseline v11 + error diagnostics
  retrain  --split S --run-dir D --knob K --value V --output F
  parity   --split S --rows N --output F         baseline config == team train/predict
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

# Import torch before numpy/pandas: loading it afterwards fails on this Windows
# environment with "WinError 1114 ... c10.dll". Non-MLP settings run without torch.
try:  # pragma: no cover - environment-dependent import ordering
    import torch as _torch_preload  # noqa: F401
except Exception:
    _torch_preload = None

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import retrain_runtime as runtime  # noqa: E402
from agent.error_diagnostics import DIAGNOSTIC_PROTOCOL, diagnose_rows, summarize_split, paired_outcomes  # noqa: E402
from agent.mlp_oof_protocol import (CUDA_PROTOCOL_VERSION, PROTOCOL_VERSION as MLP_OOF_PROTOCOL,
                                    normalized_name_key)  # noqa: E402
from agent.retrain_config import (KNOBS, baseline_config, canonical_name, config_for,  # noqa: E402
                                  fingerprint, validate_config)
from src.metric import calculate_score, is_korean  # noqa: E402

trainer, ml = runtime.trainer, runtime.ml
SPLITS = {
    "group_seed42": ("group_split_seed42.json", "target_vessel_id_group_split", 42),
    "group_seed123": ("group_split_seed123.json", "target_vessel_id_group_split", 123),
    "group_seed2026": ("group_split_seed2026.json", "target_vessel_id_group_split", 2026),
    "row_seed42": ("row_split_seed42.json", "row_split", 42),
}
MLP_OOF_SPLIT_FILE = "agent_v11_mlp_listwise_group5_seed20261019.json"
MLP_CUDA_OOF_SPLIT_FILE = "agent_v11_mlp_listwise_cuda_group5_seed20261019.json"
for _fold in range(5):
    SPLITS[f"mlp_oof_fold{_fold}"] = (MLP_OOF_SPLIT_FILE, "target_vessel_id_group_oof", 20261019)
    SPLITS[f"mlp_cuda_oof_fold{_fold}"] = (MLP_CUDA_OOF_SPLIT_FILE, "target_vessel_id_group_oof", 20261019)
CODE_FILES = ("agent/team_v11/source/match_lib.py", "agent/team_v11/source/train.py",
              "agent/retrain_runtime.py", "agent/retrain_config.py", "agent/error_diagnostics.py",
              "agent/mlp_oof_protocol.py", "scripts/v11_improvement_worker.py")
INDEPENDENCE_ROWS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def load_split(split_name: str):
    file_name, method, seed = SPLITS[split_name]
    split_path = ROOT / "experiments" / "splits" / file_name
    train_path = ROOT / "data" / "open" / "train.csv"
    catalog_path = ROOT / "data" / "open" / "vessel_catalog.csv"
    metadata = json.loads((ROOT / "team_v11_artifacts" / "v11" / "training_metadata.json")
                          .read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    hashes = {"train_sha256": sha256_file(train_path), "catalog_sha256": sha256_file(catalog_path),
              "split_sha256": sha256_file(split_path)}
    hashes.update({Path(p).name.replace(".", "_") + "_sha256": sha256_file(ROOT / p) for p in CODE_FILES})
    for current, recorded in (("train_sha256", "train_sha256"), ("catalog_sha256", "catalog_sha256"),
                              ("match_lib_py_sha256", "source_match_lib_sha256")):
        if hashes[current] != metadata[recorded]:
            raise ValueError(f"현재 {current}가 v11 학습 기록과 다릅니다")
    if method == "target_vessel_id_group_oof":
        expected_protocol = CUDA_PROTOCOL_VERSION if split_name.startswith("mlp_cuda_") else MLP_OOF_PROTOCOL
        if split.get("protocol_version") != expected_protocol or split.get("split_method") != method:
            raise ValueError("허용된 MLP OOF protocol이 아닙니다")
        if split.get("train_sha256") != hashes["train_sha256"] or split.get("catalog_sha256") != hashes["catalog_sha256"]:
            raise ValueError("고정 MLP OOF protocol의 공식 데이터 해시가 현재 입력과 다릅니다")
    elif split.get("method") != method or split.get("seed") != seed:
        raise ValueError("허용된 고정 분할 파일이 아닙니다")
    train = pd.read_csv(train_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    all_ids = train["sample_id"].astype(str).tolist()
    if method == "target_vessel_id_group_oof":
        fold_id = int(split_name.removeprefix("mlp_cuda_oof_fold")
                      if split_name.startswith("mlp_cuda_") else split_name.removeprefix("mlp_oof_fold"))
        fold = split["folds"][fold_id]
        fit_ids, valid_ids = fold["train_sample_ids"], fold["valid_sample_ids"]
    else:
        fit_ids, valid_ids = split["train_sample_ids"], split["valid_sample_ids"]
    if len(all_ids) != len(set(all_ids)) or len(catalog) != catalog["vessel_id"].nunique():
        raise ValueError("공식 데이터의 sample_id 또는 vessel_id가 중복됩니다")
    if (len(fit_ids) != len(set(fit_ids)) or len(valid_ids) != len(set(valid_ids))
            or set(fit_ids) & set(valid_ids) or set(fit_ids) | set(valid_ids) != set(all_ids)):
        raise ValueError("고정 분할이 공식 train 행을 정확히 한 번씩 포함하지 않습니다")
    keyed = train.set_index("sample_id", drop=False)
    fit = keyed.loc[fit_ids].reset_index(drop=True)
    valid = keyed.loc[valid_ids].reset_index(drop=True)
    fit_targets, valid_targets = set(fit["target_vessel_id"]), set(valid["target_vessel_id"])
    if method in ("target_vessel_id_group_split", "target_vessel_id_group_oof") and fit_targets & valid_targets:
        raise ValueError("학습·검증 target_vessel_id 그룹이 겹칩니다")
    seen = valid["target_vessel_id"].isin(fit_targets)
    audit = {
        "split_name": split_name, "split_file": file_name, "method": method, "seed": seed,
        "official_inputs_only": True, "train_rows": len(fit), "valid_rows": len(valid),
        "train_target_groups": len(fit_targets), "valid_target_groups": len(valid_targets),
        "target_group_overlap": len(fit_targets & valid_targets),
        "seen_target_validation_rows": int(seen.sum()), "unseen_target_validation_rows": int((~seen).sum()),
        "split_exact_partition": True, "hashes": hashes,
        "reuse_and_exposure": ("new five-fold partition; all official target groups were exposed in earlier "
                               "validation or error analysis, so this OOF is exploratory"
                               if method == "target_vessel_id_group_oof" else
                               "existing split reused; its target groups were exposed in earlier "
                               "validation and error analysis, so it is not an independent holdout"),
    }
    return fit, valid, catalog, audit


def stats(mask, correct, hits, sizes) -> dict:
    n = int(mask.sum())
    return {"rows": n, "correct": int(correct[mask].sum()),
            "accuracy": float(correct[mask].mean()) if n else 0.0,
            "candidate_hits": int(hits[mask].sum()),
            "candidate_recall": float(hits[mask].mean()) if n else 0.0,
            "mean_candidates": float(sizes[mask].mean()) if n else 0.0}


def metrics(cases: list[dict], valid: pd.DataFrame) -> dict:
    final = [c["final_pred_id"] for c in cases]
    truth = valid["target_vessel_id"].astype(str).tolist()
    official = calculate_score(truth, final, valid["vessel_name"])
    correct = np.array([c["final_correct"] for c in cases])
    model_correct = np.array([c["model_correct"] for c in cases])
    hits = np.array([c["truth_in_candidates"] for c in cases])
    sizes = np.array([c["candidate_count"] for c in cases])
    korean = np.array([c["is_korean"] for c in cases])
    seen = np.array([c["seen_target"] for c in cases])
    every = np.ones(len(cases), dtype=bool)
    return {"official": official, "weighted_raw": 0.7 * official["korean"] + 0.3 * official["non_korean"],
            "all": stats(every, correct, hits, sizes), "korean": stats(korean, correct, hits, sizes),
            "non_korean": stats(~korean, correct, hits, sizes),
            "seen_target": stats(seen, correct, hits, sizes), "unseen_target": stats(~seen, correct, hits, sizes),
            "model_before_override_accuracy": float(model_correct.mean())}


def independence_check(model, inputs: pd.DataFrame, index, res, config, details) -> dict:
    """Same rows must predict identically alone, in a reversed subset, and in the full batch."""
    korean = np.array([is_korean(v) for v in inputs["vessel_name"]])
    picked = sorted({0, len(inputs) // 2, len(inputs) - 1,
                     int(np.flatnonzero(korean)[0]), int(np.flatnonzero(~korean)[0])})
    full = [details[i]["final_pred_id"] for i in picked]
    reverse = runtime.predict_details(model, inputs.iloc[picked[::-1]], index, res, config)
    singles = [runtime.predict_details(model, inputs.iloc[[i]], index, res, config)[0]["final_pred_id"]
               for i in picked]
    passed = [d["final_pred_id"] for d in reverse][::-1] == full and singles == full
    if not passed:
        raise ValueError("부분집합·단독 행 독립성 검사에 실패했습니다")
    return {"checked_rows": len(picked), "passed": True, "modes": ["reversed_subset", "single_row"]}


def _evaluate(model, fit, valid, catalog, index, config):
    res = runtime.resources(fit, catalog, config)
    lookup_ids = set(res["callsign"].values())
    lookup_ids.update(v for votes in res["name_lookup"].values() for v in votes)
    if not lookup_ids <= set(fit["target_vessel_id"]):
        raise ValueError("조회표에 학습 외 target이 포함됐습니다")
    inputs = valid.drop(columns=["target_vessel_id"])
    started = time.monotonic()
    details = runtime.predict_details(model, inputs, index, res, config)
    predict_seconds = time.monotonic() - started
    if not {d["final_pred_id"] for d in details} <= set(index.ids):
        raise AssertionError("예측 ID가 카탈로그 밖에 있습니다")
    independence = independence_check(model, inputs, index, res, config, details)
    # Evaluation step: labels are joined only after every prediction above is final.
    truth = valid["target_vessel_id"].astype(str).tolist()
    seen = valid["target_vessel_id"].isin(set(fit["target_vessel_id"])).to_numpy(bool)
    korean = np.array([is_korean(v) for v in valid["vessel_name"]])
    cases = diagnose_rows(details, truth, seen, korean, runtime.FEATURES, index.ids, index.compact,
                          index.id_to_index)
    train_name_keys = {normalized_name_key(value) for value in fit["vessel_name"]}
    for row, case in zip(inputs.to_dict("records"), cases):
        name_key = normalized_name_key(row.get("vessel_name"))
        case["query_name_seen_in_train"] = bool(name_key and name_key in train_name_keys)
    overrides = {s: int(sum(d["override_source"] == s for d in details))
                 for s in ("none", "name_lookup", "callsign")}
    return res, inputs, details, cases, predict_seconds, independence, overrides


def _local(run_dir: Path) -> Path:
    path = run_dir / "local_only"
    path.mkdir(parents=True, exist_ok=True)
    return path


def case_file_path(run_dir: Path, split_name: str, setting: str) -> Path:
    return _local(run_dir) / f"cases_{split_name}_{setting}.json"


def prediction_input_hashes(details: list[dict]) -> dict[str, str]:
    candidates, features = hashlib.sha256(), hashlib.sha256()
    for row in details:
        candidates.update(row["sample_id"].encode("utf-8"))
        candidates.update(runtime.array_digest(row["candidates"]).encode("ascii"))
        features.update(row["sample_id"].encode("utf-8"))
        features.update(runtime.array_digest(row["features"]).encode("ascii"))
    return {"candidates_sha256": candidates.hexdigest(), "features_sha256": features.hexdigest()}


def diagnose(split_name: str, run_dir: Path) -> dict:
    config = baseline_config()
    fit, valid, catalog, audit = load_split(split_name)
    index = ml.CatalogIndex(catalog)
    print(f"{split_name}/baseline: 학습 ({len(fit)}행)", flush=True)
    started = time.monotonic()
    model, recall, train_stats = runtime.fit(fit, catalog, index, config)
    train_seconds = time.monotonic() - started
    print(f"{split_name}/baseline: 검증 예측·진단 ({len(valid)}행)", flush=True)
    res, inputs, details, cases, predict_seconds, independence, overrides = _evaluate(
        model, fit, valid, catalog, index, config)
    with runtime.v11_globals(config):
        team = ml.predict(model, inputs, index, res, use_callsign=True, use_name_lookup=True)
    if team != [d["final_pred_id"] for d in details]:
        raise AssertionError("기준 설정 예측이 팀 v11 predict와 다릅니다")
    local = _local(run_dir)
    model_text = model.booster_.model_to_string()
    (local / f"model_{split_name}_baseline.txt").write_text(model_text, encoding="utf-8")
    baseline_case_path = case_file_path(run_dir, split_name, "baseline")
    write_json(baseline_case_path, {"protocol": DIAGNOSTIC_PROTOCOL, "cases": cases})
    return {
        "split_name": split_name, "setting": "v11_baseline", "config": config,
        "config_fingerprint": fingerprint(config), "audit": audit, "metrics": metrics(cases, valid),
        "diagnostic_summary": summarize_split(split_name, cases), "diagnostic_protocol": DIAGNOSTIC_PROTOCOL,
        "training_stats": train_stats, "training_candidate_recall_before_forced_truth": float(recall),
        "timing_seconds": {"training": train_seconds, "validation_prediction": predict_seconds},
        "model_sha256": runtime.model_sha256(model), "feature_names_sha256":
            hashlib.sha256("\n".join(runtime.FEATURES).encode()).hexdigest(),
        "feature_importance_gain_share": runtime.feature_importance(model),
        "prediction_input_hashes": prediction_input_hashes(details),
        "override_counts": overrides,
        "checks": {"validation_label_removed_before_prediction": True,
                   "lookup_ids_subset_of_training_targets": True,
                   "prediction_matches_team_v11_predict": True, "row_independence": independence},
        "local_case_file": baseline_case_path.relative_to(ROOT).as_posix(),
    }


def retrain(split_name: str, run_dir: Path, knob: str, value: str) -> dict:
    config = config_for(knob, value)
    if validate_config(config) != [knob]:
        raise ValueError("한 가지 설정 변경만 실행합니다")
    name = canonical_name(knob, value)
    baseline_out = json.loads((run_dir / f"diagnose_{split_name}.json").read_text(encoding="utf-8"))
    local = _local(run_dir)
    base_cases = json.loads((local / f"cases_{split_name}_baseline.json").read_text(encoding="utf-8"))["cases"]
    fit, valid, catalog, audit = load_split(split_name)
    if audit["hashes"]["split_sha256"] != baseline_out["audit"]["hashes"]["split_sha256"]:
        raise ValueError("기준 진단과 분할 파일이 다릅니다")
    index = ml.CatalogIndex(catalog)
    started = time.monotonic()
    if KNOBS[knob]["retrain"]:
        print(f"{split_name}/{name}: 재학습 ({len(fit)}행)", flush=True)
        model, recall, train_stats = runtime.fit(fit, catalog, index, config)
        retrained = True
    else:
        print(f"{split_name}/{name}: 후처리 설정, 기준 랭커 재사용", flush=True)
        text = (local / f"model_{split_name}_baseline.txt").read_text(encoding="utf-8")
        model = runtime.ReusedBooster(text)
        recall, train_stats = baseline_out["training_candidate_recall_before_forced_truth"], \
            baseline_out["training_stats"]
        retrained = False
    train_seconds = time.monotonic() - started
    res, inputs, details, cases, predict_seconds, independence, overrides = _evaluate(
        model, fit, valid, catalog, index, config)
    if [c["sample_id"] for c in cases] != [c["sample_id"] for c in base_cases]:
        raise AssertionError("기준과 검증 행 순서가 다릅니다")
    if not retrained and [c["model_pred_id"] for c in cases] != [c["model_pred_id"] for c in base_cases]:
        raise AssertionError("재사용 랭커가 기준 랭커 예측을 재현하지 못했습니다")
    behavior = runtime.behavior_check(knob, value, baseline_out["training_stats"], train_stats,
                                      overrides, model)
    input_hashes = prediction_input_hashes(details)
    base_hashes = baseline_out.get("prediction_input_hashes", {})
    if base_hashes and input_hashes["candidates_sha256"] != base_hashes["candidates_sha256"]:
        raise AssertionError("변경 모델의 검증 후보가 기준과 다릅니다")
    model_roundtrip_passed = None
    if knob == "ranker_family":
        if (input_hashes != base_hashes
                or train_stats["training_matrix_sha256"] != baseline_out["training_stats"].get("training_matrix_sha256")):
            raise AssertionError("랭커 교체 실험의 학습 행렬 또는 검증 입력 특징이 다릅니다")
        suffix = ".cbm" if value == "catboost_v1_cpu" else ".pt"
        model_path = local / f"model_{split_name}_{name}{suffix}"
        model.save_model(str(model_path))
        if value in ("mlp_listwise_v1_cpu", "mlp_listwise_v1_cuda"):
            restored = runtime.MLPRanker.load_model(model_path)
            candidate_features = np.vstack([row["features"] for row in details])
            if not np.array_equal(model.predict(candidate_features), restored.predict(candidate_features)):
                raise AssertionError("저장·재로딩한 MLP의 검증 점수가 다릅니다")
            model_roundtrip_passed = True
    newly_correct = [(b, c) for b, c in zip(base_cases, cases) if c["final_correct"] and not b["final_correct"]]
    newly_wrong = [(b, c) for b, c in zip(base_cases, cases) if b["final_correct"] and not c["final_correct"]]
    changed = sum(b["final_pred_id"] != c["final_pred_id"] for b, c in zip(base_cases, cases))
    write_json(local / f"changes_{split_name}_{name}.json", {
        "newly_correct_sample_ids": [c["sample_id"] for _, c in newly_correct],
        "newly_wrong_sample_ids": [c["sample_id"] for _, c in newly_wrong],
        "changed_prediction_sample_ids": [c["sample_id"] for b, c in zip(base_cases, cases)
                                          if b["final_pred_id"] != c["final_pred_id"]]})
    alternative_case_path = case_file_path(run_dir, split_name, name)
    write_json(alternative_case_path, {"protocol": DIAGNOSTIC_PROTOCOL, "cases": cases})
    base_importance = baseline_out["feature_importance_gain_share"]
    importance = runtime.feature_importance(model)
    same_importance_method = config["ranker_family"] == "lightgbm_v11"
    shift = (sorted(((f, importance[f] - base_importance[f]) for f in importance),
                    key=lambda item: (-abs(item[1]), item[0]))[:8] if same_importance_method else [])
    base_m, new_m = baseline_out["metrics"], metrics(cases, valid)

    def by(pairs, key, source):
        out: dict[str, int] = {}
        for b, c in pairs:
            label = (b if source == "baseline" else c)[key] or "none"
            out[label] = out.get(label, 0) + 1
        return dict(sorted(out.items()))
    return {
        "split_name": split_name, "setting": name, "knob": knob, "value": value, "config": config,
        "config_fingerprint": fingerprint(config), "baseline_config_fingerprint": baseline_out["config_fingerprint"],
        "retrained": retrained, "seed": trainer.SEED, "audit_hashes": audit["hashes"],
        "metrics": new_m, "baseline_metrics": base_m,
        "delta": {"weighted_raw": new_m["weighted_raw"] - base_m["weighted_raw"],
                  "korean_accuracy": new_m["korean"]["accuracy"] - base_m["korean"]["accuracy"],
                  "non_korean_accuracy": new_m["non_korean"]["accuracy"] - base_m["non_korean"]["accuracy"],
                  "candidate_recall": new_m["all"]["candidate_recall"] - base_m["all"]["candidate_recall"],
                  "seen_target_accuracy": new_m["seen_target"]["accuracy"] - base_m["seen_target"]["accuracy"],
                  "unseen_target_accuracy": new_m["unseen_target"]["accuracy"] - base_m["unseen_target"]["accuracy"],
                  "training_seconds": train_seconds - baseline_out["timing_seconds"]["training"],
                  "prediction_seconds": predict_seconds - baseline_out["timing_seconds"]["validation_prediction"],
                  "mean_candidates": new_m["all"]["mean_candidates"] - base_m["all"]["mean_candidates"]},
        "row_changes": {"newly_correct": len(newly_correct), "newly_wrong": len(newly_wrong),
                        "changed_predictions": int(changed),
                        "newly_correct_korean": sum(c["is_korean"] for _, c in newly_correct),
                        "newly_wrong_korean": sum(c["is_korean"] for _, c in newly_wrong),
                        "newly_correct_by_baseline_error_type": by(newly_correct, "primary_error_type", "baseline"),
                        "newly_wrong_by_new_error_type": by(newly_wrong, "primary_error_type", "new")},
        "feature_importance_shift_top8": [{"feature": f, "delta_gain_share": round(d, 6)} for f, d in shift],
        "feature_importance_method": ("LightGBM gain share" if same_importance_method else
                                      "CatBoost PredictionValuesChange share; not comparable with LightGBM gain"
                                      if value == "catboost_v1_cpu" else
                                      "MLP first-layer mean absolute weight share; not comparable with tree gain"),
        "feature_importance_share": importance,
        "complementarity": paired_outcomes(base_cases, cases),
        "prediction_input_hashes": input_hashes,
        "baseline_prediction_input_hashes": base_hashes,
        "diagnostic_summary": summarize_split(split_name, cases),
        "training_stats": train_stats, "training_candidate_recall_before_forced_truth": float(recall),
        "timing_seconds": {"training": train_seconds, "validation_prediction": predict_seconds},
        "model_sha256": runtime.model_sha256(model), "override_counts": overrides,
        "local_model_file": model_path.relative_to(ROOT).as_posix() if knob == "ranker_family" else None,
        "local_case_file": alternative_case_path.relative_to(ROOT).as_posix(),
        "behavior_check": behavior,
        "checks": {"validation_label_removed_before_prediction": True,
                   "lookup_ids_subset_of_training_targets": True,
                   "same_validation_rows_as_baseline": True, "same_seed": True,
                   "same_candidate_generation": True, "row_independence": independence,
                   "model_save_reload_prediction_equal": model_roundtrip_passed},
    }


def parity(split_name: str, rows: int) -> dict:
    """Baseline config must rebuild the team training matrix and predictions exactly."""
    config = baseline_config()
    fit, valid, catalog, _ = load_split(split_name)
    fit = fit.iloc[:rows].reset_index(drop=True)
    valid = valid.iloc[: max(rows // 5, 50)].reset_index(drop=True)
    index = ml.CatalogIndex(catalog)
    with runtime.v11_globals(config):
        x0, y0, g0, r0 = trainer.training_matrix(fit, catalog, index)
        x1, y1, g1, r1, _ = runtime.training_matrix(fit, catalog, index, config)
        params_equal = trainer.ranker_params() == runtime.ranker_params(config)
        team_model = trainer.fit(fit, catalog, index)[0]
        res = trainer.resources(fit, catalog)
        inputs = valid.drop(columns=["target_vessel_id"])
        team_pred = ml.predict(team_model, inputs, index, res, use_callsign=True, use_name_lookup=True)
    model, _, _ = runtime.fit(fit, catalog, index, config)
    ours = [d["final_pred_id"] for d in runtime.predict_details(model, inputs, index,
                                                               runtime.resources(fit, catalog, config), config)]
    result = {"split_name": split_name, "train_rows": len(fit), "valid_rows": len(valid),
              "training_matrix_equal": bool(np.array_equal(x0, x1, equal_nan=True)
                                            and np.array_equal(y0, y1) and np.array_equal(g0, g1) and r0 == r1),
              "ranker_params_equal": params_equal,
              "model_string_equal": team_model.booster_.model_to_string() == model.booster_.model_to_string(),
              "predictions_equal": team_pred == ours}
    if not all(v for k, v in result.items() if k.endswith("_equal")):
        raise AssertionError(f"기준 설정 재현 실패: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "diagnose", "retrain", "parity"))
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--knob", choices=KNOBS)
    parser.add_argument("--value")
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "audit":
        result = load_split(args.split)[3]
    elif args.mode == "diagnose":
        result = diagnose(args.split, args.run_dir)
    elif args.mode == "retrain":
        if args.knob is None or args.value is None:
            parser.error("retrain에는 --knob와 --value가 필요합니다")
        result = retrain(args.split, args.run_dir, args.knob, args.value)
    else:
        result = parity(args.split, args.rows)
    write_json(args.output, result)
    print(f"기록: {args.output}", flush=True)


if __name__ == "__main__":
    main()
