"""Group-bootstrap stability analysis for the fixed nested-OOF LambdaRank."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = ROOT / "experiments" / "splits" / "agent_lambdarank_group_bootstrap_seed31431.json"
MODEL_PREREG_PATH = ROOT / "experiments" / "splits" / "agent_learned_reranker_nested_oof_seed31430.json"
SPLIT_PATH = ROOT / "experiments" / "splits" / "agent_syllable_transliteration_cv_seed31426.json"
SOURCE_RESULT_PATH = (
    ROOT
    / "experiments"
    / "results"
    / "agent_learned_reranker_nested_oof_20261018"
    / f"summary_lgbm_{lgb.__version__.replace('.', '_')}.json"
)
RESULT_PATH = ROOT / "experiments" / "results" / "agent_lambdarank_group_bootstrap_20261018" / "summary.json"
DATA_ROOT = ROOT / "data" / "open"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submit" / "model"))

from adaptive_lib import (  # noqa: E402
    FUZZY_SCORE_BONUS,
    AdaptiveCatalogIndex,
    base_features,
    is_korean,
    normalize_name,
    romanized_name,
)
from scripts.agent_learned_reranker_nested_oof import (  # noqa: E402
    FEATURE_NAMES,
    collect_training_queries,
    make_query_examples,
    rank_with_model,
)
from scripts.agent_syllable_transliteration_probe import (  # noqa: E402
    file_sha256,
    learn_syllable_mappings,
)


def compute_group_counts() -> tuple[dict[str, dict[str, int]], dict]:
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    model_prereg = json.loads(MODEL_PREREG_PATH.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if lgb.__version__ != "4.6.0":
        raise AssertionError(f"Expected LightGBM 4.6.0, found {lgb.__version__}")
    for filename in ("train.csv", "vessel_catalog.csv"):
        if file_sha256(DATA_ROOT / filename) != prereg["data_hashes"][filename]:
            raise AssertionError(f"Frozen data hash mismatch: {filename}")
    if prereg["data_hashes"] != model_prereg["data_hashes"]:
        raise AssertionError("Bootstrap and source experiment use different data hashes")

    train = pd.read_csv(DATA_ROOT / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(DATA_ROOT / "vessel_catalog.csv", encoding="utf-8-sig")
    group_to_fold = {
        str(group_id): int(fold["fold"])
        for fold in split["folds"]
        for group_id in fold["target_group_ids"]
    }
    train_groups = set(train["target_vessel_id"].astype(str))
    if train_groups != set(group_to_fold):
        raise AssertionError("Frozen fold assignments do not cover training target groups exactly")

    index = AdaptiveCatalogIndex(catalog.reset_index(drop=True))
    group_counts: dict[str, dict[str, int]] = {}
    for outer_fold in split["folds"]:
        outer_id = int(outer_fold["fold"])
        outer_train = train[
            train["target_vessel_id"].astype(str).map(group_to_fold).ne(outer_id)
        ].copy()
        outer_valid = train[
            train["target_vessel_id"].astype(str).map(group_to_fold).eq(outer_id)
        ].copy()
        outer_train_groups = set(outer_train["target_vessel_id"].astype(str))
        outer_valid_groups = set(outer_valid["target_vessel_id"].astype(str))
        if outer_train_groups & outer_valid_groups:
            raise AssertionError(f"Outer target-group leakage in fold {outer_id}")

        train_features: list[list[float]] = []
        train_labels: list[int] = []
        train_group_sizes: list[int] = []
        for inner_fold in split["folds"]:
            inner_id = int(inner_fold["fold"])
            if inner_id == outer_id:
                continue
            inner_valid = outer_train[
                outer_train["target_vessel_id"].astype(str).map(group_to_fold).eq(inner_id)
            ].copy()
            inner_train = outer_train[
                outer_train["target_vessel_id"].astype(str).map(group_to_fold).ne(inner_id)
            ].copy()
            inner_train_groups = set(inner_train["target_vessel_id"].astype(str))
            inner_valid_groups = set(inner_valid["target_vessel_id"].astype(str))
            if inner_train_groups & inner_valid_groups or inner_valid_groups & outer_valid_groups:
                raise AssertionError(
                    f"Nested target-group leakage in outer {outer_id}, inner {inner_id}"
                )
            mappings, _, _ = learn_syllable_mappings(
                inner_train, catalog, normalize_name, romanized_name
            )
            features, labels, groups, _, _ = collect_training_queries(
                inner_valid, index, mappings
            )
            train_features.extend(features)
            train_labels.extend(labels)
            train_group_sizes.extend(groups)

        if sum(train_group_sizes) != len(train_labels) or not sum(train_labels):
            raise AssertionError(f"Invalid ranker training rows for outer fold {outer_id}")
        settings = model_prereg["ranker"]
        ranker = lgb.LGBMRanker(
            objective=settings["objective"],
            metric="ndcg",
            n_estimators=int(settings["n_estimators"]),
            learning_rate=float(settings["learning_rate"]),
            num_leaves=int(settings["num_leaves"]),
            max_depth=int(settings["max_depth"]),
            min_child_samples=int(settings["min_child_samples"]),
            reg_lambda=float(settings["reg_lambda"]),
            random_state=int(settings["random_state"]),
            n_jobs=int(settings["n_jobs"]),
            verbosity=-1,
            label_gain=settings["label_gain"],
            deterministic=True,
            force_col_wise=True,
        )
        ranker.fit(
            pd.DataFrame(train_features, columns=FEATURE_NAMES),
            train_labels,
            group=train_group_sizes,
        )
        mappings, _, _ = learn_syllable_mappings(
            outer_train, catalog, normalize_name, romanized_name
        )
        for row in outer_valid.to_dict("records"):
            if not is_korean(row.get("vessel_name")):
                continue
            query = make_query_examples(index, row, mappings)
            baseline_ranked = []
            for candidate_index in query["baseline_indices"]:
                features = base_features(row, index.catalog.iloc[candidate_index])
                score = features["score"] + FUZZY_SCORE_BONUS * query[
                    "baseline_fuzzy_scores"
                ].get(candidate_index, 0.0)
                baseline_ranked.append((score, index.ids[candidate_index]))
            baseline_ranked.sort(key=lambda item: (-item[0], item[1]))
            baseline_prediction = baseline_ranked[0][1] if baseline_ranked else index.ids[0]
            ranker_prediction = rank_with_model(ranker, query)
            group_id = str(row["target_vessel_id"])
            bucket = group_counts.setdefault(
                group_id,
                {"rows": 0, "baseline_correct": 0, "ranker_correct": 0},
            )
            target = str(row["target_vessel_id"])
            bucket["rows"] += 1
            bucket["baseline_correct"] += int(baseline_prediction == target)
            bucket["ranker_correct"] += int(ranker_prediction == target)

    if not group_counts or any(item["rows"] <= 0 for item in group_counts.values()):
        raise AssertionError("No Korean target-group observations were produced")
    return group_counts, prereg


def bootstrap(group_counts: dict[str, dict[str, int]], prereg: dict) -> dict:
    groups = list(group_counts.values())
    row_counts = np.asarray([item["rows"] for item in groups], dtype=np.int64)
    baseline_correct = np.asarray(
        [item["baseline_correct"] for item in groups], dtype=np.int64
    )
    ranker_correct = np.asarray(
        [item["ranker_correct"] for item in groups], dtype=np.int64
    )
    total_rows = int(row_counts.sum())
    baseline_accuracy = float(baseline_correct.sum() / total_rows)
    ranker_accuracy = float(ranker_correct.sum() / total_rows)
    point_delta = ranker_accuracy - baseline_accuracy

    settings = prereg["method"]
    rng = np.random.default_rng(int(prereg["seed"]))
    repetitions = int(settings["bootstrap_resamples"])
    draws_per_sample = len(groups)
    deltas = np.empty(repetitions, dtype=np.float64)
    batch_size = 200
    start = 0
    while start < repetitions:
        count = min(batch_size, repetitions - start)
        sampled = rng.integers(0, draws_per_sample, size=(count, draws_per_sample))
        denominators = row_counts[sampled].sum(axis=1)
        base_rates = baseline_correct[sampled].sum(axis=1) / denominators
        ranker_rates = ranker_correct[sampled].sum(axis=1) / denominators
        deltas[start : start + count] = ranker_rates - base_rates
        start += count

    low, high = np.quantile(deltas, [0.025, 0.975], method="linear")
    return {
        "korean_rows": total_rows,
        "target_groups_with_korean_rows": int(len(groups)),
        "baseline_top1": baseline_accuracy,
        "lambdarank_top1": ranker_accuracy,
        "point_delta": point_delta,
        "weighted_score_contribution_point_delta": 0.7 * point_delta,
        "bootstrap_resamples": repetitions,
        "confidence_level": 0.95,
        "cluster_bootstrap_unit": "target_vessel_id",
        "top1_delta_percentile_interval": [float(low), float(high)],
        "weighted_score_contribution_percentile_interval": [
            float(0.7 * low),
            float(0.7 * high),
        ],
        "probability_bootstrap_delta_at_or_below_zero": float(np.mean(deltas <= 0.0)),
        "interval_excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def run() -> dict:
    if not PREREG_PATH.exists():
        raise FileNotFoundError(PREREG_PATH)
    if RESULT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite result: {RESULT_PATH}")

    group_counts, prereg = compute_group_counts()
    metrics = bootstrap(group_counts, prereg)
    expected = json.loads(SOURCE_RESULT_PATH.read_text(encoding="utf-8"))["metrics"]
    tolerance = float(prereg["decision_rules"]["point_estimate_must_match_seed31430_with_absolute_tolerance"])
    if abs(metrics["baseline_top1"] - expected["baseline_top1"]) > tolerance:
        raise AssertionError("Recomputed baseline top-1 differs from the frozen 4.6.0 result")
    if abs(metrics["lambdarank_top1"] - expected["learned_ranker_top1"]) > tolerance:
        raise AssertionError("Recomputed LambdaRank top-1 differs from the frozen 4.6.0 result")
    if abs(
        metrics["weighted_score_contribution_point_delta"]
        - expected["weighted_score_absolute_gain"]
    ) > tolerance:
        raise AssertionError("Recomputed weighted contribution differs from the frozen result")

    report = {
        "experiment": prereg["experiment"],
        "preregistration": str(PREREG_PATH.relative_to(ROOT)),
        "source_model_result": str(SOURCE_RESULT_PATH.relative_to(ROOT)),
        "lightgbm_version": lgb.__version__,
        "historical_validation_overlap": True,
        "historical_error_analysis_overlap": True,
        "independent_performance_estimate": False,
        "parameters_or_model_changed": False,
        "group_outcomes_saved": False,
        "metrics": metrics,
        "decision": (
            "Positive fixed-OOF difference under these observed groups; this interval is descriptive and does not remove historical selection bias."
            if metrics["top1_delta_percentile_interval"][0] > 0.0
            else "The fixed-OOF difference is uncertain across target groups; this interval does not remove historical selection bias."
        ),
        "data_hashes": prereg["data_hashes"],
        "limitations": prereg["limitations"],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
