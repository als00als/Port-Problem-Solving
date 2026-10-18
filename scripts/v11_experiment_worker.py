"""Train and evaluate one v11 search setting on the fixed vessel-group split.

Run this worker with the Python 3.11 competition environment. The experiment
graph calls it once per setting; no validation label is passed to inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.team_v11.source import train as trainer  # noqa: E402
from src.metric import calculate_score, is_korean  # noqa: E402


SPLITS = {
    "group": ROOT / "experiments" / "splits" / "group_split_seed42.json",
    "row": ROOT / "experiments" / "splits" / "row_split_seed42.json",
}
SETTINGS = {
    "v11_baseline": {"use_vocab": False, "vocab_top": 3},
    "vocab_top1": {"use_vocab": True, "vocab_top": 1},
    "vocab_top3": {"use_vocab": True, "vocab_top": 3},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_audited_data(split_name: str):
    split_path = SPLITS[split_name]
    train_path = ROOT / "data" / "open" / "train.csv"
    catalog_path = ROOT / "data" / "open" / "vessel_catalog.csv"
    metadata_path = ROOT / "team_v11_artifacts" / "v11" / "training_metadata.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hashes = {
        "train_sha256": sha256(train_path),
        "catalog_sha256": sha256(catalog_path),
        "split_sha256": sha256(split_path),
        "match_lib_sha256": sha256(ROOT / "agent" / "team_v11" / "source" / "match_lib.py"),
        "train_code_sha256": sha256(ROOT / "agent" / "team_v11" / "source" / "train.py"),
    }
    for current_key, recorded_key in (("train_sha256", "train_sha256"),
                                      ("catalog_sha256", "catalog_sha256"),
                                      ("match_lib_sha256", "source_match_lib_sha256")):
        if hashes[current_key] != metadata[recorded_key]:
            raise ValueError(f"현재 {current_key}가 v11 학습 기록과 다릅니다")
    expected_method = "target_vessel_id_group_split" if split_name == "group" else "row_split"
    if split.get("method") != expected_method or split.get("seed") != 42:
        raise ValueError("고정된 seed42 분할만 허용합니다")

    train = pd.read_csv(train_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    all_ids = train["sample_id"].astype(str).tolist()
    fit_ids = split["train_sample_ids"]
    valid_ids = split["valid_sample_ids"]
    if len(all_ids) != len(set(all_ids)) or len(catalog) != catalog["vessel_id"].nunique():
        raise ValueError("공식 데이터의 sample_id 또는 vessel_id가 중복됩니다")
    if (len(fit_ids) != len(set(fit_ids)) or len(valid_ids) != len(set(valid_ids))
            or set(fit_ids) & set(valid_ids) or set(fit_ids) | set(valid_ids) != set(all_ids)):
        raise ValueError("고정 분할이 공식 train 행을 정확히 한 번씩 포함하지 않습니다")
    keyed = train.set_index("sample_id", drop=False)
    fit = keyed.loc[fit_ids].reset_index(drop=True)
    valid = keyed.loc[valid_ids].reset_index(drop=True)
    fit_targets = set(fit["target_vessel_id"])
    valid_targets = set(valid["target_vessel_id"])
    if split_name == "group" and fit_targets & valid_targets:
        raise ValueError("학습·검증 target_vessel_id 그룹이 겹칩니다")
    if not fit_targets | valid_targets <= set(catalog["vessel_id"]):
        raise ValueError("정답 선박이 공식 카탈로그에 없습니다")

    seen = valid["target_vessel_id"].isin(fit_targets)
    audit = {
        "split_name": split_name,
        "official_inputs_only": True,
        "fixed_split_method": split["method"],
        "split_seed": split["seed"],
        "train_rows": len(fit),
        "valid_rows": len(valid),
        "train_target_groups": len(fit_targets),
        "valid_target_groups": len(valid_targets),
        "target_group_overlap": len(fit_targets & valid_targets),
        "seen_target_validation_rows": int(seen.sum()),
        "unseen_target_validation_rows": int((~seen).sum()),
        "split_exact_partition": True,
        "hashes": hashes,
        "previous_group_exposure": True,
        "interpretation": "탐색용 재사용 검증 구간; 독립 일반화 추정치 아님",
    }
    return fit, valid, catalog, audit


def group_stats(mask: np.ndarray, correct: np.ndarray, hits: np.ndarray, sizes: np.ndarray) -> dict:
    count = int(mask.sum())
    return {
        "rows": count,
        "correct": int(correct[mask].sum()),
        "accuracy": float(correct[mask].mean()) if count else 0.0,
        "candidate_hits": int(hits[mask].sum()),
        "candidate_recall": float(hits[mask].mean()) if count else 0.0,
        "mean_candidates": float(sizes[mask].mean()) if count else 0.0,
        "max_candidates": int(sizes[mask].max()) if count else 0,
    }


def evaluate(setting_name: str, split_name: str) -> dict:
    setting = SETTINGS[setting_name]
    fit, valid, catalog, audit = load_audited_data(split_name)
    index = trainer.ml.CatalogIndex(catalog)
    original = (trainer.NO_DICT, trainer.ml.USE_VOCAB, trainer.ml.VOCAB_TOP)
    trainer.NO_DICT = True
    trainer.ml.USE_VOCAB = setting["use_vocab"]
    trainer.ml.VOCAB_TOP = setting["vocab_top"]
    try:
        print(f"{split_name}/{setting_name}: 학습 시작 ({len(fit)}행)", flush=True)
        started = time.monotonic()
        model, training_candidate_recall = trainer.fit(fit, catalog, index)
        train_seconds = time.monotonic() - started

        resources = trainer.resources(fit, catalog)
        # Any lookup override must be learned from the outer training rows only.
        lookup_ids = set(resources["callsign"].values())
        lookup_ids.update(vid for votes in resources["name_lookup"].values() for vid in votes)
        if not lookup_ids <= set(fit["target_vessel_id"]):
            raise ValueError("조회표에 검증 target이 포함됐습니다")

        inputs = valid.drop(columns=["target_vessel_id"])
        if "target_vessel_id" in inputs.columns:
            raise AssertionError("검증 정답이 추론 입력에 남았습니다")
        candidates: list[np.ndarray] = []
        print(f"{split_name}/{setting_name}: 검증 예측 시작 ({len(valid)}행)", flush=True)
        started = time.monotonic()
        predictions = trainer.ml.predict(model, inputs, index, resources,
                                         use_callsign=True, use_name_lookup=True,
                                         candidates_out=candidates)
        predict_seconds = time.monotonic() - started
        if len(predictions) != len(valid) or len(candidates) != len(valid):
            raise AssertionError("검증 예측 행 수가 다릅니다")
        if not set(predictions) <= set(index.ids):
            raise AssertionError("예측 ID가 카탈로그 밖에 있습니다")

        # The same rows must predict identically alone and in changed order.
        hangul = np.array([is_korean(name) for name in valid["vessel_name"]], dtype=bool)
        chosen = {0, len(valid) // 2, len(valid) - 1}
        chosen.add(int(np.flatnonzero(hangul)[0]))
        chosen.add(int(np.flatnonzero(~hangul)[0]))
        picked = sorted(chosen)
        subset = inputs.iloc[picked[::-1]]
        shuffled = trainer.ml.predict(model, subset, index, resources,
                                      use_callsign=True, use_name_lookup=True)
        singles = [trainer.ml.predict(model, inputs.iloc[[i]], index, resources,
                                      use_callsign=True, use_name_lookup=True)[0] for i in picked]
        if shuffled != [predictions[i] for i in picked[::-1]] or singles != [predictions[i] for i in picked]:
            raise ValueError("부분집합·단독 행 독립성 검사에 실패했습니다")

        true_ids = valid["target_vessel_id"].astype(str).to_numpy()
        pred_ids = np.asarray(predictions, dtype=str)
        correct = true_ids == pred_ids
        truth_indices = [index.id_to_index[value] for value in true_ids]
        hits = np.array([truth in cand for truth, cand in zip(truth_indices, candidates)], dtype=bool)
        sizes = np.array([len(cand) for cand in candidates], dtype=np.int64)
        seen = valid["target_vessel_id"].isin(set(fit["target_vessel_id"])).to_numpy(bool)
        metric = calculate_score(true_ids, predictions, valid["vessel_name"])
        weighted_raw = 0.7 * metric["korean"] + 0.3 * metric["non_korean"]
        result = {
            "split_name": split_name,
            "setting": setting_name,
            "configuration": {**setting, "no_dict": True, "use_callsign": True, "use_name_lookup": True,
                              "ranker": "v11_LGBMRanker_retrained", "inner_group_folds": 5},
            "metrics": {"official": metric, "weighted_raw": weighted_raw,
                        "all": group_stats(np.ones(len(valid), dtype=bool), correct, hits, sizes),
                        "korean": group_stats(hangul, correct, hits, sizes),
                        "non_korean": group_stats(~hangul, correct, hits, sizes),
                        "seen_target": group_stats(seen, correct, hits, sizes),
                        "unseen_target": group_stats(~seen, correct, hits, sizes)},
            "timing_seconds": {"training": train_seconds, "validation_prediction": predict_seconds},
            "training_candidate_recall_before_forced_truth": float(training_candidate_recall),
            "leakage_checks": {"target_group_overlap": audit["target_group_overlap"],
                               "group_disjoint_when_required": split_name != "group" or audit["target_group_overlap"] == 0,
                               "validation_label_removed_before_prediction": True,
                               "lookup_ids_subset_of_training_targets": True,
                               "validation_candidates_not_forced": True,
                               "row_independence_checked_rows": len(picked),
                               "row_independence_passed": True},
            "input_hashes": audit["hashes"],
        }
        return result
    finally:
        trainer.NO_DICT, trainer.ml.USE_VOCAB, trainer.ml.VOCAB_TOP = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "evaluate"))
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--setting", choices=SETTINGS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "audit":
        _, _, _, result = load_audited_data(args.split)
        result["available_settings"] = SETTINGS
    else:
        if args.setting is None:
            parser.error("evaluate에는 --setting이 필요합니다")
        result = evaluate(args.setting, args.split)
    write_json(args.output, result)
    print(f"기록: {args.output}", flush=True)


if __name__ == "__main__":
    main()
