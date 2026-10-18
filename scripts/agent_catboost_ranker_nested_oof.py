"""Fixed nested grouped-OOF CatBoostRanker comparison for Korean candidates."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import catboost
import lightgbm as lgb
import pandas as pd
from catboost import CatBoostRanker, Pool


ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = ROOT / "experiments" / "splits" / "agent_catboost_ranker_nested_oof_seed31432.json"
MODEL_PREREG_PATH = ROOT / "experiments" / "splits" / "agent_learned_reranker_nested_oof_seed31430.json"
SPLIT_PATH = ROOT / "experiments" / "splits" / "agent_syllable_transliteration_cv_seed31426.json"
LIGHTGBM_RESULT_PATH = (
    ROOT
    / "experiments"
    / "results"
    / "agent_learned_reranker_nested_oof_20261018"
    / "summary_lgbm_4_6_0.json"
)
RESULT_PATH = (
    ROOT
    / "experiments"
    / "results"
    / "agent_catboost_ranker_nested_oof_20261018"
    / f"summary_catboost_{catboost.__version__.replace('.', '_')}.json"
)
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
    summarize,
)
from scripts.agent_syllable_transliteration_probe import (  # noqa: E402
    file_sha256,
    learn_syllable_mappings,
)


def create_ranker(settings: dict) -> CatBoostRanker:
    return CatBoostRanker(
        loss_function=settings["loss_function"],
        iterations=int(settings["iterations"]),
        learning_rate=float(settings["learning_rate"]),
        depth=int(settings["depth"]),
        l2_leaf_reg=float(settings["l2_leaf_reg"]),
        random_seed=int(settings["random_seed"]),
        thread_count=int(settings["thread_count"]),
        task_type=settings["task_type"],
        verbose=False,
        allow_writing_files=False,
    )


def evaluate() -> dict:
    if RESULT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite result: {RESULT_PATH}")
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    model_prereg = json.loads(MODEL_PREREG_PATH.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if catboost.__version__ != "1.2.8":
        raise AssertionError(f"Expected CatBoost 1.2.8, found {catboost.__version__}")
    if lgb.__version__ != "4.6.0":
        raise AssertionError(f"Expected LightGBM 4.6.0 reference, found {lgb.__version__}")
    if prereg["data_hashes"] != model_prereg["data_hashes"]:
        raise AssertionError("CatBoost preregistration differs from the frozen source data")
    for filename, expected_hash in prereg["data_hashes"].items():
        if file_sha256(DATA_ROOT / filename) != expected_hash:
            raise AssertionError(f"Frozen input hash mismatch: {filename}")

    train = pd.read_csv(DATA_ROOT / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(DATA_ROOT / "vessel_catalog.csv", encoding="utf-8-sig")
    group_to_fold = {
        str(group_id): int(fold["fold"])
        for fold in split["folds"]
        for group_id in fold["target_group_ids"]
    }
    if set(group_to_fold) != set(train["target_vessel_id"].astype(str)):
        raise AssertionError("Frozen folds do not cover all target groups exactly")

    index = AdaptiveCatalogIndex(catalog.reset_index(drop=True))
    observations: list[dict] = []
    fold_reports: list[dict] = []
    fit_seconds_total = 0.0
    feature_seconds_total = 0.0
    settings = prereg["method"]["ranker"]

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
        inner_reports: list[dict] = []
        inner_skipped_queries = 0

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

            started = time.perf_counter()
            mappings, _, _ = learn_syllable_mappings(
                inner_train, catalog, normalize_name, romanized_name
            )
            fit_seconds_total += time.perf_counter() - started
            started = time.perf_counter()
            features, labels, groups, kept, skipped = collect_training_queries(
                inner_valid, index, mappings
            )
            feature_seconds_total += time.perf_counter() - started
            train_features.extend(features)
            train_labels.extend(labels)
            train_group_sizes.extend(groups)
            inner_skipped_queries += skipped
            inner_reports.append(
                {
                    "inner_fold": inner_id,
                    "mapping_fit_target_groups": len(inner_train_groups),
                    "heldout_training_target_groups": len(inner_valid_groups),
                    "ranker_training_queries_kept": kept,
                    "ranker_training_queries_without_target_candidate": skipped,
                }
            )

        if not train_group_sizes or not any(train_labels):
            raise AssertionError(f"No CatBoost ranking examples for outer fold {outer_id}")
        if sum(train_group_sizes) != len(train_labels) or len(train_features) != len(train_labels):
            raise AssertionError(f"CatBoost query group sizes do not match features in fold {outer_id}")
        group_ids = [
            query_id
            for query_id, query_size in enumerate(train_group_sizes)
            for _ in range(query_size)
        ]
        pool = Pool(
            data=pd.DataFrame(train_features, columns=FEATURE_NAMES),
            label=train_labels,
            group_id=group_ids,
            feature_names=list(FEATURE_NAMES),
        )
        ranker = create_ranker(settings)
        started = time.perf_counter()
        ranker.fit(pool)
        ranker_fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        outer_mappings, _, _ = learn_syllable_mappings(
            outer_train, catalog, normalize_name, romanized_name
        )
        fit_seconds_total += time.perf_counter() - started
        started = time.perf_counter()
        fold_observations: list[dict] = []
        for row in outer_valid.to_dict("records"):
            if not is_korean(row.get("vessel_name")):
                continue
            query = make_query_examples(index, row, outer_mappings)
            baseline_ranked = []
            for candidate_index in query["baseline_indices"]:
                features = base_features(row, index.catalog.iloc[candidate_index])
                score = features["score"] + FUZZY_SCORE_BONUS * query[
                    "baseline_fuzzy_scores"
                ].get(candidate_index, 0.0)
                baseline_ranked.append((score, index.ids[candidate_index]))
            baseline_ranked.sort(key=lambda item: (-item[0], item[1]))
            baseline_prediction = baseline_ranked[0][1] if baseline_ranked else index.ids[0]
            candidate_frame = pd.DataFrame(query["features"], columns=FEATURE_NAMES)
            scores = ranker.predict(candidate_frame)
            order = sorted(
                range(len(query["candidate_ids"])),
                key=lambda i: (-float(scores[i]), query["candidate_ids"][i]),
            )
            ranker_prediction = query["candidate_ids"][order[0]] if order else index.ids[0]
            item = {
                "target": query["target"],
                "baseline_prediction": baseline_prediction,
                "ranker_prediction": ranker_prediction,
                "baseline_has_target": query["target"] in {
                    index.ids[i] for i in query["baseline_indices"]
                },
                "union_has_target": query["target"] in query["candidate_ids"],
                "candidate_count": query["candidate_count"],
            }
            observations.append(item)
            fold_observations.append(item)
        feature_seconds_total += time.perf_counter() - started
        fold_reports.append(
            {
                "outer_fold": outer_id,
                "train_target_groups": len(outer_train_groups),
                "valid_target_groups": len(outer_valid_groups),
                "inner_oof_training_queries_kept": sum(
                    item["ranker_training_queries_kept"] for item in inner_reports
                ),
                "inner_oof_training_queries_without_target_candidate": inner_skipped_queries,
                "ranker_training_candidate_rows": len(train_labels),
                "ranker_training_positive_rows": int(sum(train_labels)),
                "ranker_training_group_count": len(train_group_sizes),
                "ranker_fit_seconds": ranker_fit_seconds,
                "metrics": summarize(fold_observations),
                "inner_folds": inner_reports,
            }
        )

    catboost_metrics = summarize(observations)
    reference = json.loads(LIGHTGBM_RESULT_PATH.read_text(encoding="utf-8"))
    lgb_metrics = reference["metrics"]
    comparison = {
        "korean_top1_delta_catboost_minus_lightgbm": (
            catboost_metrics["learned_ranker_top1"] - lgb_metrics["learned_ranker_top1"]
        ),
        "weighted_score_contribution_delta_catboost_minus_lightgbm": 0.7
        * (catboost_metrics["learned_ranker_top1"] - lgb_metrics["learned_ranker_top1"]),
        "candidate_recall_delta_catboost_minus_lightgbm": (
            catboost_metrics["union_candidate_recall"] - lgb_metrics["union_candidate_recall"]
        ),
        "folds_improved_over_lightgbm": sum(
            fold["metrics"]["learned_ranker_top1"]
            > reference["folds"][fold["outer_fold"]]["metrics"]["learned_ranker_top1"]
            for fold in fold_reports
        ),
    }
    gates = prereg["acceptance_gates"]
    gate_results = {
        "korean_top1_gain": comparison["korean_top1_delta_catboost_minus_lightgbm"]
        >= gates["minimum_korean_top1_gain_over_lightgbm"],
        "weighted_contribution_gain": comparison[
            "weighted_score_contribution_delta_catboost_minus_lightgbm"
        ]
        >= gates["minimum_weighted_score_contribution_gain_over_lightgbm"],
        "minimum_fold_count": comparison["folds_improved_over_lightgbm"]
        >= gates["minimum_outer_folds_improved"],
        "target_group_leakage": False,
    }
    report = {
        "experiment": prereg["experiment"],
        "preregistration": str(PREREG_PATH.relative_to(ROOT)),
        "source_split": prereg["source_group_split"],
        "historical_validation_overlap": True,
        "historical_error_analysis_overlap": True,
        "independent_performance_estimate": False,
        "data_hashes": prereg["data_hashes"],
        "catboost_version": catboost.__version__,
        "lightgbm_reference_version": lgb.__version__,
        "features": list(FEATURE_NAMES),
        "ranker_settings": settings,
        "metrics": catboost_metrics,
        "lightgbm_reference_metrics": lgb_metrics,
        "comparison": comparison,
        "folds": fold_reports,
        "timing": {
            "mapping_fit_seconds_total": fit_seconds_total,
            "feature_generation_seconds_total": feature_seconds_total,
            "ranker_fitting_seconds_total": sum(
                fold["ranker_fit_seconds"] for fold in fold_reports
            ),
        },
        "acceptance_gates": gates,
        "gate_results": gate_results,
        "exploratory_gates_passed": all(gate_results.values()),
        "submission_promotion_eligible": False,
        "submission_package_modified": False,
        "leaderboard_submitted": False,
        "per_row_inputs_or_predictions_saved": False,
        "limitations": prereg["limitations"],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
