"""Nested grouped-OOF LightGBM LambdaRank experiment for Korean vessel candidates."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from rapidfuzz import fuzz, process


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submit" / "model"))

from adaptive_lib import (  # noqa: E402
    FUZZY_SCORE_BONUS,
    AdaptiveCatalogIndex,
    base_features,
    is_korean,
    normalize_name,
    positive_float,
    romanized_name,
)
from scripts.agent_syllable_transliteration_probe import (  # noqa: E402
    FUZZY_CUTOFF,
    FUZZY_LIMIT,
    MAX_EXPANDED_CANDIDATES,
    MAX_UNION_CANDIDATES,
    MAX_QUERY_VARIANTS,
    SPLIT_PATH as SOURCE_SPLIT_PATH,
    file_sha256,
    generate_variants,
    learn_syllable_mappings,
)
from scripts.exp014_validation import read_data  # noqa: E402


PREREG_PATH = ROOT / "experiments" / "splits" / "agent_learned_reranker_nested_oof_seed31430.json"
RESULT_PATH = (
    ROOT
    / "experiments"
    / "results"
    / "agent_learned_reranker_nested_oof_20261018"
    / f"summary_lgbm_{lgb.__version__.replace('.', '_')}.json"
)
DATA_ROOT = ROOT / "data" / "open"
FEATURE_NAMES = (
    "exact_match",
    "name_ratio",
    "prefix_ratio",
    "length_ratio",
    "tonnage_ratio",
    "base_score",
    "baseline_fuzzy_score",
    "max_syllable_fuzzy_ratio",
    "baseline_candidate",
    "syllable_search_candidate",
    "query_gt_present",
    "query_international_gt_present",
    "catalog_gt_present",
)


def make_query_examples(index, row: dict, mappings: dict) -> dict:
    baseline_indices, baseline_fuzzy_scores = index.candidates_with_scores(row)
    baseline_set = set(baseline_indices)
    variants = generate_variants(row["vessel_name"], mappings, romanized_name)
    query = romanized_name(row["vessel_name"])
    syllable_scores: dict[int, float] = {}
    for variant in [query, *variants]:
        if not variant:
            continue
        for _, score, candidate_index in process.extract(
            variant,
            index.romanized_names,
            scorer=fuzz.ratio,
            limit=FUZZY_LIMIT,
            score_cutoff=FUZZY_CUTOFF,
        ):
            candidate_index = int(candidate_index)
            syllable_scores[candidate_index] = max(
                syllable_scores.get(candidate_index, 0.0), float(score) / 100.0
            )

    expanded = sorted(
        syllable_scores,
        key=lambda candidate_index: (
            -syllable_scores[candidate_index],
            index.ids[candidate_index],
        ),
    )[:MAX_EXPANDED_CANDIDATES]
    candidate_indices = list(baseline_indices)
    seen = set(candidate_indices)
    for candidate_index in expanded:
        if candidate_index not in seen:
            candidate_indices.append(candidate_index)
            seen.add(candidate_index)
        if len(candidate_indices) >= MAX_UNION_CANDIDATES:
            break

    feature_rows: list[list[float]] = []
    candidate_ids: list[str] = []
    for candidate_index in candidate_indices:
        catalog_row = index.catalog.iloc[candidate_index]
        features = base_features(row, catalog_row)
        feature_rows.append(
            [
                float(features["exact_match"]),
                float(features["name_ratio"]),
                float(features["prefix_ratio"]),
                float(features["length_ratio"]),
                float(features["tonnage_ratio"]),
                float(features["score"]),
                float(baseline_fuzzy_scores.get(candidate_index, 0.0)),
                float(syllable_scores.get(candidate_index, 0.0)),
                float(candidate_index in baseline_set),
                float(candidate_index in syllable_scores),
                float(positive_float(row.get("gross_tonnage")) is not None),
                float(positive_float(row.get("international_gross_tonnage")) is not None),
                float(positive_float(catalog_row.get("gross_tonnage")) is not None),
            ]
        )
        candidate_ids.append(index.ids[candidate_index])

    return {
        "features": feature_rows,
        "candidate_ids": candidate_ids,
        "baseline_indices": baseline_indices,
        "baseline_fuzzy_scores": baseline_fuzzy_scores,
        "target": str(row["target_vessel_id"]),
        "candidate_count": len(candidate_indices),
    }


def collect_training_queries(frame: pd.DataFrame, index, mappings: dict):
    features: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    kept_queries = 0
    skipped_no_target = 0
    for row in frame.to_dict("records"):
        if not is_korean(row.get("vessel_name")):
            continue
        query = make_query_examples(index, row, mappings)
        query_labels = [int(candidate_id == query["target"]) for candidate_id in query["candidate_ids"]]
        if not any(query_labels):
            skipped_no_target += 1
            continue
        features.extend(query["features"])
        labels.extend(query_labels)
        groups.append(len(query_labels))
        kept_queries += 1
    return features, labels, groups, kept_queries, skipped_no_target


def rank_with_model(model, query: dict) -> str:
    scores = model.predict(pd.DataFrame(query["features"], columns=FEATURE_NAMES))
    order = sorted(
        range(len(query["candidate_ids"])),
        key=lambda i: (-float(scores[i]), query["candidate_ids"][i]),
    )
    return query["candidate_ids"][order[0]] if order else ""


def summarize(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    baseline_correct = frame["baseline_prediction"] == frame["target"]
    ranker_correct = frame["ranker_prediction"] == frame["target"]
    return {
        "korean_rows": int(len(frame)),
        "baseline_top1": float(baseline_correct.mean()),
        "learned_ranker_top1": float(ranker_correct.mean()),
        "korean_top1_absolute_gain": float(ranker_correct.mean() - baseline_correct.mean()),
        "baseline_candidate_recall": float(frame["baseline_has_target"].mean()),
        "union_candidate_recall": float(frame["union_has_target"].mean()),
        "mean_union_candidates": float(frame["candidate_count"].mean()),
        "max_union_candidates": int(frame["candidate_count"].max()),
        "baseline_wrong_to_correct": int((~baseline_correct & ranker_correct).sum()),
        "baseline_correct_to_wrong": int((baseline_correct & ~ranker_correct).sum()),
        "prediction_changed": int((frame["baseline_prediction"] != frame["ranker_prediction"]).sum()),
    }


def evaluate() -> dict:
    if RESULT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite result: {RESULT_PATH}")
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    split = json.loads(SOURCE_SPLIT_PATH.read_text(encoding="utf-8"))
    for filename, expected_hash in prereg["data_hashes"].items():
        if file_sha256(DATA_ROOT / filename) != expected_hash:
            raise AssertionError(f"Frozen input hash mismatch: {filename}")

    train, catalog = read_data()
    group_to_fold = {
        str(group_id): int(fold["fold"])
        for fold in split["folds"]
        for group_id in fold["target_group_ids"]
    }
    if set(group_to_fold) != set(train["target_vessel_id"].astype(str)):
        raise AssertionError("Frozen folds do not cover target groups exactly")

    index = AdaptiveCatalogIndex(catalog.reset_index(drop=True))
    observations: list[dict] = []
    fold_reports: list[dict] = []
    fit_seconds_total = 0.0
    feature_seconds_total = 0.0

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
        inner_kept_queries = 0
        inner_skipped_queries = 0
        inner_reports: list[dict] = []

        # The four non-outer folds form grouped inner OOF partitions.
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
                    f"Inner target-group leakage in outer {outer_id}, inner {inner_id}"
                )

            fit_started = time.perf_counter()
            mappings, _, _ = learn_syllable_mappings(
                inner_train, catalog, normalize_name, romanized_name
            )
            mapping_fit_seconds = time.perf_counter() - fit_started
            fit_seconds_total += mapping_fit_seconds

            feature_started = time.perf_counter()
            features, labels, groups, kept, skipped = collect_training_queries(
                inner_valid, index, mappings
            )
            feature_seconds = time.perf_counter() - feature_started
            feature_seconds_total += feature_seconds
            train_features.extend(features)
            train_labels.extend(labels)
            train_group_sizes.extend(groups)
            inner_kept_queries += kept
            inner_skipped_queries += skipped
            inner_reports.append(
                {
                    "inner_fold": inner_id,
                    "fit_target_groups": len(inner_train_groups),
                    "heldout_training_target_groups": len(inner_valid_groups),
                    "ranker_training_queries_kept": kept,
                    "ranker_training_queries_without_target_candidate": skipped,
                }
            )

        if not train_group_sizes or not any(train_labels):
            raise AssertionError(f"No positive ranking examples for outer fold {outer_id}")
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=5,
            min_child_samples=30,
            reg_lambda=1.0,
            random_state=int(prereg["ranker"]["random_state"]),
            n_jobs=1,
            verbosity=-1,
            label_gain=[0, 1],
            deterministic=True,
            force_col_wise=True,
        )
        ranker.fit(
            pd.DataFrame(train_features, columns=FEATURE_NAMES),
            train_labels,
            group=train_group_sizes,
        )

        fit_started = time.perf_counter()
        outer_mappings, _, _ = learn_syllable_mappings(
            outer_train, catalog, normalize_name, romanized_name
        )
        mapping_fit_seconds = time.perf_counter() - fit_started
        fit_seconds_total += mapping_fit_seconds
        feature_started = time.perf_counter()
        fold_observations: list[dict] = []
        for row in outer_valid.to_dict("records"):
            if not is_korean(row.get("vessel_name")):
                continue
            query = make_query_examples(index, row, outer_mappings)
            baseline_ranked = []
            for candidate_index in query["baseline_indices"]:
                base = base_features(row, index.catalog.iloc[candidate_index])
                score = base["score"] + FUZZY_SCORE_BONUS * query["baseline_fuzzy_scores"].get(candidate_index, 0.0)
                baseline_ranked.append((score, index.ids[candidate_index]))
            baseline_ranked.sort(key=lambda item: (-item[0], item[1]))
            baseline_prediction = baseline_ranked[0][1] if baseline_ranked else index.ids[0]
            ranker_prediction = rank_with_model(ranker, query)
            item = {
                "target": query["target"],
                "baseline_prediction": baseline_prediction,
                "ranker_prediction": ranker_prediction,
                "baseline_has_target": query["target"] in set(
                    index.ids[i] for i in query["baseline_indices"]
                ),
                "union_has_target": query["target"] in query["candidate_ids"],
                "candidate_count": query["candidate_count"],
            }
            observations.append(item)
            fold_observations.append(item)
        feature_seconds = time.perf_counter() - feature_started
        feature_seconds_total += feature_seconds
        fold_reports.append(
            {
                "outer_fold": outer_id,
                "train_target_groups": len(outer_train_groups),
                "valid_target_groups": len(outer_valid_groups),
                "inner_oof_training_queries_kept": inner_kept_queries,
                "inner_oof_training_queries_without_target_candidate": inner_skipped_queries,
                "ranker_training_candidate_rows": len(train_labels),
                "ranker_training_positive_rows": int(sum(train_labels)),
                "ranker_training_group_count": len(train_group_sizes),
                "metrics": summarize(fold_observations),
                "inner_folds": inner_reports,
            }
        )

    metrics = summarize(observations)
    metrics["weighted_score_absolute_gain"] = 0.7 * metrics["korean_top1_absolute_gain"]
    gates = prereg["acceptance_gates"]
    gate_results = {
        "korean_top1_gain": metrics["korean_top1_absolute_gain"]
        >= gates["korean_top1_absolute_gain_min"],
        "weighted_score_gain": metrics["weighted_score_absolute_gain"]
        >= gates["weighted_score_absolute_gain_min"],
        "candidate_recall_not_below_frozen_union": abs(
            metrics["union_candidate_recall"] - 0.945424013434089
        ) <= 1e-12,
        "no_target_group_leakage": True,
    }
    report = {
        "experiment": prereg["experiment"],
        "preregistration": str(PREREG_PATH.relative_to(ROOT)),
        "source_split": prereg["source_split"],
        "historical_validation_overlap": True,
        "historical_error_analysis_overlap": True,
        "independent_performance_estimate": False,
        "data_hashes": prereg["data_hashes"],
        "lightgbm_version": lgb.__version__,
        "features": list(FEATURE_NAMES),
        "ranker_settings": prereg["ranker"],
        "metrics": metrics,
        "folds": fold_reports,
        "timing": {
            "mapping_fit_seconds_total": fit_seconds_total,
            "feature_generation_seconds_total": feature_seconds_total,
        },
        "acceptance_gates": gates,
        "gate_results": gate_results,
        "exploratory_gates_passed": all(gate_results.values()),
        "submission_promotion_eligible": False,
        "submission_code_or_archive_modified": False,
        "per_row_inputs_or_predictions_saved": False,
        "limitations": prereg["limitations"],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
