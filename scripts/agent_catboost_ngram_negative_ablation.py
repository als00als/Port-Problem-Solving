"""Compare CatBoost training negatives under one fixed expanded candidate list."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submit" / "model"))

from adaptive_lib import AdaptiveCatalogIndex, is_korean, normalize_name, romanized_name  # noqa: E402
from scripts import agent_catboost_char_ngram_candidate as source  # noqa: E402
from scripts.agent_learned_reranker_nested_oof import FEATURE_NAMES  # noqa: E402
from scripts.agent_syllable_transliteration_probe import file_sha256, learn_syllable_mappings  # noqa: E402

PREREG_PATH = ROOT / "experiments/splits/agent_catboost_ngram_negative_ablation_seed31438.json"
SOURCE_PREREG_PATH = ROOT / "experiments/splits/agent_catboost_char_ngram_candidate_seed31437.json"
SOURCE_RESULT_PATH = ROOT / "experiments/results/agent_catboost_char_ngram_candidate_20261018/summary.json"
CAT_PREREG_PATH = ROOT / "experiments/splits/agent_catboost_ranker_nested_oof_seed31432.json"
SPLIT_PATH = ROOT / "experiments/splits/agent_syllable_transliteration_cv_seed31426.json"
RESULT_PATH = ROOT / "experiments/results/agent_catboost_ngram_negative_ablation_20261018/summary.json"
DATA_ROOT = ROOT / "data/open"
ARMS = ("original_negatives", "expanded_negatives")


def append_training_queries(frame: pd.DataFrame, index, mappings: dict, data: dict) -> dict:
    kept = 0
    skipped = 0
    added_negatives = 0
    for row in frame.to_dict("records"):
        if not is_korean(row.get("vessel_name")):
            continue
        query = source.make_treatment_query(index, row, mappings)
        control_count = len(query["candidate_ids"]) - query["ngram_added_candidates"]
        if control_count < 1:
            raise AssertionError("The original candidate union is empty")
        target = query["target"]
        original_labels = [int(candidate_id == target) for candidate_id in query["candidate_ids"][:control_count]]
        added_labels = [int(candidate_id == target) for candidate_id in query["candidate_ids"][control_count:]]
        if any(added_labels):
            raise AssertionError("N-gram-only positive candidate violates the negative-only comparison")
        if not any(original_labels):
            skipped += 1
            continue
        if sum(original_labels) != 1:
            raise AssertionError("Training query must have exactly one positive candidate")
        expanded_labels = original_labels + added_labels
        for arm, features, labels in (
            ("original_negatives", query["features"][:control_count], original_labels),
            ("expanded_negatives", query["features"], expanded_labels),
        ):
            data[arm]["features"].extend(features)
            data[arm]["labels"].extend(labels)
            data[arm]["groups"].append(len(labels))
            data[arm]["queries"] += 1
        added_negatives += len(added_labels)
        kept += 1
    return {"kept": kept, "skipped": skipped, "added_negatives": added_negatives}


def fit_ranker(data: dict, settings: dict, task_type: str):
    if not data["groups"] or sum(data["groups"]) != len(data["labels"]):
        raise AssertionError("Ranking group sizes do not match training labels")
    if len(data["features"]) != len(data["labels"]):
        raise AssertionError("Feature rows do not match training labels")
    if sum(data["labels"]) != data["queries"]:
        raise AssertionError("Expected one positive candidate for every training query")
    frame = pd.DataFrame(data["features"], columns=FEATURE_NAMES)
    group_ids = [query_id for query_id, size in enumerate(data["groups"]) for _ in range(size)]
    pool = Pool(frame, label=data["labels"], group_id=group_ids, feature_names=list(FEATURE_NAMES))
    ranker = source.create_ranker(settings)
    if task_type == "GPU":
        ranker.set_params(task_type="GPU")
    started = time.perf_counter()
    ranker.fit(pool)
    return ranker, time.perf_counter() - started


def summarize(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    target = frame["target"]
    original_correct = frame["original_prediction"].eq(target)
    expanded_correct = frame["expanded_prediction"].eq(target)
    return {
        "korean_rows": int(len(frame)),
        "candidate_recall": float(frame["has_target"].mean()),
        "mean_candidates": float(frame["candidate_count"].mean()),
        "max_candidates": int(frame["candidate_count"].max()),
        "original_negatives_top1": float(original_correct.mean()),
        "expanded_negatives_top1": float(expanded_correct.mean()),
        "top1_delta_expanded_minus_original": float(expanded_correct.mean() - original_correct.mean()),
        "expanded_correct_original_wrong": int((expanded_correct & ~original_correct).sum()),
        "original_correct_expanded_wrong": int((original_correct & ~expanded_correct).sum()),
    }


def evaluate() -> dict:
    if RESULT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite result: {RESULT_PATH}")
    if catboost.__version__ != "1.2.8":
        raise AssertionError(f"Expected CatBoost 1.2.8; found {catboost.__version__}")
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    source_prereg = json.loads(SOURCE_PREREG_PATH.read_text(encoding="utf-8"))
    source_result = json.loads(SOURCE_RESULT_PATH.read_text(encoding="utf-8"))
    cat_prereg = json.loads(CAT_PREREG_PATH.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if prereg["data_hashes"] != source_prereg["data_hashes"] or prereg["data_hashes"] != cat_prereg["data_hashes"]:
        raise AssertionError("Registered source data hashes differ")
    for filename, expected in prereg["data_hashes"].items():
        if file_sha256(DATA_ROOT / filename) != expected:
            raise AssertionError(f"Frozen input hash mismatch: {filename}")

    train = pd.read_csv(DATA_ROOT / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(DATA_ROOT / "vessel_catalog.csv", encoding="utf-8-sig")
    group_to_fold = {
        str(group_id): int(fold["fold"])
        for fold in split["folds"] for group_id in fold["target_group_ids"]
    }
    if set(group_to_fold) != set(train["target_vessel_id"].astype(str)):
        raise AssertionError("Frozen split does not cover target groups exactly")
    index = AdaptiveCatalogIndex(catalog.reset_index(drop=True))
    source._NGRAM_CACHE.clear()
    source._VECTORIZER = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 4), lowercase=True, norm="l2", dtype=np.float32
    )
    source._CATALOG_TFIDF = source._VECTORIZER.fit_transform(index.romanized_names)
    settings = cat_prereg["method"]["ranker"]
    use_gpu = source.probe_catboost_gpu(settings)
    task_type = "GPU" if use_gpu else "CPU"

    observations: list[dict] = []
    fold_reports: list[dict] = []
    feature_seconds = 0.0
    fit_seconds = {arm: 0.0 for arm in ARMS}

    for outer_fold in split["folds"]:
        outer_id = int(outer_fold["fold"])
        fold_ids = train["target_vessel_id"].astype(str).map(group_to_fold)
        outer_train = train[fold_ids.ne(outer_id)].copy()
        outer_valid = train[fold_ids.eq(outer_id)].copy()
        valid_groups = set(outer_valid["target_vessel_id"].astype(str))
        if valid_groups & set(outer_train["target_vessel_id"].astype(str)):
            raise AssertionError(f"Outer target leakage in fold {outer_id}")
        train_data = {
            arm: {"features": [], "labels": [], "groups": [], "queries": 0}
            for arm in ARMS
        }
        inner_reports: list[dict] = []
        for inner_fold in split["folds"]:
            inner_id = int(inner_fold["fold"])
            if inner_id == outer_id:
                continue
            inner_ids = outer_train["target_vessel_id"].astype(str).map(group_to_fold)
            inner_valid = outer_train[inner_ids.eq(inner_id)].copy()
            inner_train = outer_train[inner_ids.ne(inner_id)].copy()
            heldout_groups = set(inner_valid["target_vessel_id"].astype(str))
            if heldout_groups & valid_groups or heldout_groups & set(inner_train["target_vessel_id"].astype(str)):
                raise AssertionError(f"Inner target leakage in outer {outer_id}, inner {inner_id}")
            mappings, _, _ = learn_syllable_mappings(inner_train, catalog, normalize_name, romanized_name)
            started = time.perf_counter()
            counts = append_training_queries(inner_valid, index, mappings, train_data)
            feature_seconds += time.perf_counter() - started
            inner_reports.append({
                "inner_fold": inner_id,
                "mapping_fit_target_groups": len(set(inner_train["target_vessel_id"].astype(str))),
                "heldout_target_groups": len(heldout_groups),
                "kept_queries": counts["kept"],
                "skipped_without_original_target": counts["skipped"],
                "ngram_added_training_negatives": counts["added_negatives"],
            })

        if train_data["original_negatives"]["queries"] != train_data["expanded_negatives"]["queries"]:
            raise AssertionError("Training query sets differ")
        if sum(train_data["original_negatives"]["labels"]) != sum(train_data["expanded_negatives"]["labels"]):
            raise AssertionError("Training positive counts differ")
        models = {}
        fold_fit_seconds = {}
        for arm in ARMS:
            models[arm], fold_fit_seconds[arm] = fit_ranker(train_data[arm], settings, task_type)
            fit_seconds[arm] += fold_fit_seconds[arm]

        mappings, _, _ = learn_syllable_mappings(outer_train, catalog, normalize_name, romanized_name)
        fold_rows: list[dict] = []
        for row in outer_valid.to_dict("records"):
            if not is_korean(row.get("vessel_name")):
                continue
            started = time.perf_counter()
            query = source.make_treatment_query(index, row, mappings)
            feature_seconds += time.perf_counter() - started
            target_id = str(row["target_vessel_id"])
            item = {
                "target": target_id,
                "has_target": target_id in query["candidate_ids"],
                "candidate_count": query["candidate_count"],
                "original_prediction": source.rank_prediction(models["original_negatives"], query),
                "expanded_prediction": source.rank_prediction(models["expanded_negatives"], query),
            }
            observations.append(item)
            fold_rows.append(item)
        fold_reports.append({
            "outer_fold": outer_id,
            "valid_target_groups": len(valid_groups),
            "training_queries_per_arm": train_data["original_negatives"]["queries"],
            "positive_training_rows_per_arm": sum(train_data["original_negatives"]["labels"]),
            "training_candidate_rows": {arm: len(train_data[arm]["labels"]) for arm in ARMS},
            "added_negative_training_rows": len(train_data["expanded_negatives"]["labels"]) - len(train_data["original_negatives"]["labels"]),
            "catboost_fit_seconds": fold_fit_seconds,
            "metrics": summarize(fold_rows),
            "inner_folds": inner_reports,
        })
        print(f"Completed outer fold {outer_id}", flush=True)

    metrics = summarize(observations)
    expected = source_result["metrics"]["treatment"]
    exact_reference_reproduction = (
        abs(metrics["expanded_negatives_top1"] - expected["top1"]) <= 1e-12
        and abs(metrics["candidate_recall"] - expected["candidate_recall"]) <= 1e-12
        and abs(metrics["mean_candidates"] - expected["mean_candidate_count"]) <= 1e-12
        and all(
            abs(fold["metrics"]["expanded_negatives_top1"] - source_result["folds"][fold["outer_fold"]]["metrics"]["treatment"]["top1"]) <= 1e-12
            for fold in fold_reports
        )
    )
    if task_type == "CPU" and not exact_reference_reproduction:
        raise AssertionError("Expanded-negatives model differs from frozen seed31437 CPU result")
    gates = prereg["diagnostic_gates"]
    delta = metrics["top1_delta_expanded_minus_original"]
    folds_not_worse = sum(fold["metrics"]["top1_delta_expanded_minus_original"] >= 0.0 for fold in fold_reports)
    gate_results = {
        "top1_gain": delta >= gates["minimum_top1_gain_expanded_minus_control"],
        "weighted_contribution_gain": 0.7 * delta >= gates["minimum_weighted_contribution_gain_expanded_minus_control"],
        "folds_not_worse": folds_not_worse >= gates["minimum_folds_not_worse"],
        "same_validation_candidate_lists_and_features": True,
        "same_training_queries_and_positives": True,
        "all_added_training_candidates_negative": True,
        "no_target_group_leakage": True,
    }
    report = {
        "experiment": prereg["experiment"],
        "preregistration": str(PREREG_PATH.relative_to(ROOT)),
        "source_split": str(SPLIT_PATH.relative_to(ROOT)),
        "historical_validation_overlap": True,
        "historical_error_analysis_overlap": True,
        "independent_performance_estimate": False,
        "data_hashes": prereg["data_hashes"],
        "python_version": sys.version.split()[0],
        "catboost_version": catboost.__version__,
        "selected_device": task_type,
        "expanded_negatives_exact_cpu_reference_reproduction": exact_reference_reproduction,
        "inference_candidate_policy": prereg["method"]["inference"],
        "metrics": metrics,
        "weighted_contribution_delta": 0.7 * delta,
        "folds_expanded_not_worse": folds_not_worse,
        "training_queries_per_arm": sum(fold["training_queries_per_arm"] for fold in fold_reports),
        "training_candidate_rows": {
            arm: sum(fold["training_candidate_rows"][arm] for fold in fold_reports)
            for arm in ARMS
        },
        "ngram_added_negative_training_rows": sum(fold["added_negative_training_rows"] for fold in fold_reports),
        "folds": fold_reports,
        "timing": {"feature_generation_seconds_total": feature_seconds, "catboost_fit_seconds_total": fit_seconds},
        "diagnostic_gates": gates,
        "gate_results": gate_results,
        "exploratory_gates_passed": all(gate_results.values()),
        "submission_promotion_eligible": False,
        "submission_code_or_archive_modified": False,
        "leaderboard_submitted": False,
        "per_row_inputs_predictions_or_candidate_lists_saved": False,
        "limitations": prereg["limitations"],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps({
        "metrics": result["metrics"],
        "gate_results": result["gate_results"],
        "expanded_negatives_exact_cpu_reference_reproduction": result["expanded_negatives_exact_cpu_reference_reproduction"],
    }, ensure_ascii=False, indent=2))
