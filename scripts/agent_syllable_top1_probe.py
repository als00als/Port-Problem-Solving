"""Exploratory OOF top-1 evaluation for the frozen syllable candidate union."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
    romanized_name,
)
from scripts.agent_syllable_transliteration_probe import (  # noqa: E402
    FUZZY_CUTOFF,
    FUZZY_LIMIT,
    MAX_EXPANDED_CANDIDATES,
    MAX_UNION_CANDIDATES,
    SPLIT_PATH as SOURCE_SPLIT_PATH,
    file_sha256,
    generate_variants,
    learn_syllable_mappings,
)
from scripts.exp014_validation import read_data  # noqa: E402


PREREG_PATH = ROOT / "experiments" / "splits" / "agent_syllable_top1_seed31427.json"
RESULT_PATH = ROOT / "experiments" / "results" / "agent_syllable_top1_20261018" / "summary.json"
DATA_ROOT = ROOT / "data" / "open"


def rank_prediction(index, row: dict, candidate_indices: list[int], fuzzy_scores: dict[int, float]) -> str:
    ranked: list[tuple[float, str]] = []
    for candidate_index in candidate_indices:
        catalog_row = index.catalog.iloc[candidate_index]
        features = base_features(row, catalog_row)
        score = features["score"] + FUZZY_SCORE_BONUS * fuzzy_scores.get(candidate_index, 0.0)
        ranked.append((score, index.ids[candidate_index]))
    if not ranked:
        return index.ids[0]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][1]


def add_syllable_candidates(index, row: dict, base_indices: list[int], mappings: dict) -> list[int]:
    variants = generate_variants(row["vessel_name"], mappings, romanized_name)
    query = romanized_name(row["vessel_name"])
    variant_scores: dict[int, float] = {}
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
            variant_scores[candidate_index] = max(
                variant_scores.get(candidate_index, 0.0), float(score)
            )

    expanded = sorted(
        variant_scores,
        key=lambda candidate_index: (-variant_scores[candidate_index], index.ids[candidate_index]),
    )[:MAX_EXPANDED_CANDIDATES]
    merged = list(base_indices)
    seen = set(merged)
    for candidate_index in expanded:
        if candidate_index not in seen:
            merged.append(candidate_index)
            seen.add(candidate_index)
        if len(merged) >= MAX_UNION_CANDIDATES:
            break
    return merged


def accuracy(frame: pd.DataFrame, prediction_column: str) -> float:
    return float((frame[prediction_column] == frame["target_vessel_id"].astype(str)).mean())


def summarize(frame: pd.DataFrame) -> dict:
    korean = frame[frame["is_korean"]]
    korean_accuracy = accuracy(korean, "union_prediction")
    baseline_korean_accuracy = accuracy(korean, "baseline_prediction")
    baseline_correct = korean["baseline_prediction"] == korean["target_vessel_id"].astype(str)
    union_correct = korean["union_prediction"] == korean["target_vessel_id"].astype(str)
    return {
        "rows": int(len(frame)),
        "korean_rows": int(len(korean)),
        "baseline_korean_top1": baseline_korean_accuracy,
        "union_korean_top1": korean_accuracy,
        "korean_top1_absolute_gain": korean_accuracy - baseline_korean_accuracy,
        "baseline_candidate_recall": float(korean["baseline_has_target"].mean()),
        "union_candidate_recall": float(korean["union_has_target"].mean()),
        "mean_baseline_candidates": float(korean["baseline_candidate_count"].mean()),
        "mean_union_candidates": float(korean["union_candidate_count"].mean()),
        "max_union_candidates": int(korean["union_candidate_count"].max()),
        "baseline_wrong_to_correct": int((~baseline_correct & union_correct).sum()),
        "baseline_correct_to_wrong": int((baseline_correct & ~union_correct).sum()),
        "prediction_changed": int(
            (korean["baseline_prediction"] != korean["union_prediction"]).sum()
        ),
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
        raise AssertionError("The frozen folds do not cover target groups exactly")

    index = AdaptiveCatalogIndex(catalog.reset_index(drop=True))
    observations: list[dict] = []
    fold_metrics: list[dict] = []
    total_fit_seconds = 0.0
    total_search_seconds = 0.0

    for fold in split["folds"]:
        fold_id = int(fold["fold"])
        valid = train[
            train["target_vessel_id"].astype(str).map(group_to_fold).eq(fold_id)
        ].copy()
        training = train[
            train["target_vessel_id"].astype(str).map(group_to_fold).ne(fold_id)
        ].copy()
        valid_groups = set(valid["target_vessel_id"].astype(str))
        train_groups = set(training["target_vessel_id"].astype(str))
        if valid_groups & train_groups:
            raise AssertionError(f"Target-group leakage in fold {fold_id}")

        fit_started = time.perf_counter()
        mappings, _, _ = learn_syllable_mappings(
            training,
            catalog,
            normalize_name,
            romanized_name,
        )
        fit_seconds = time.perf_counter() - fit_started
        total_fit_seconds += fit_seconds
        fold_observations: list[dict] = []
        search_seconds = 0.0

        for row in valid.to_dict("records"):
            if not is_korean(row.get("vessel_name")):
                continue
            base_indices, fuzzy_scores = index.candidates_with_scores(row)
            started = time.perf_counter()
            union_indices = add_syllable_candidates(index, row, base_indices, mappings)
            search_seconds += time.perf_counter() - started
            target = str(row["target_vessel_id"])
            observation = {
                "fold": fold_id,
                "target_vessel_id": target,
                "baseline_prediction": index.predict_row(row),
                "union_prediction": rank_prediction(index, row, union_indices, fuzzy_scores),
                "baseline_has_target": target in {index.ids[i] for i in base_indices},
                "union_has_target": target in {index.ids[i] for i in union_indices},
                "baseline_candidate_count": len(base_indices),
                "union_candidate_count": len(union_indices),
                "is_korean": True,
            }
            observations.append(observation)
            fold_observations.append(observation)

        total_search_seconds += search_seconds
        fold_frame = pd.DataFrame(fold_observations)
        fold_metrics.append(
            {
                "fold": fold_id,
                "train_target_groups": len(train_groups),
                "valid_target_groups": len(valid_groups),
                "korean_rows": len(fold_frame),
                "metrics": summarize(fold_frame),
                "fit_seconds": fit_seconds,
                "search_seconds": search_seconds,
            }
        )

    korean_frame = pd.DataFrame(observations)
    metrics = summarize(korean_frame)
    # Since only Korean predictions change, the exact competition-score delta
    # is 0.7 times the Korean top-1 accuracy delta; non-Korean predictions are
    # identical by construction and need not be recomputed.
    metrics["weighted_score_absolute_gain"] = 0.7 * metrics["korean_top1_absolute_gain"]

    gates = prereg["acceptance_gates"]
    gate_results = {
        "korean_top1_gain": metrics["korean_top1_absolute_gain"]
        >= gates["korean_top1_absolute_gain_min"],
        "weighted_gain": metrics["weighted_score_absolute_gain"]
        >= gates["weighted_score_absolute_gain_min"],
        "mean_union_candidates": metrics["mean_union_candidates"]
        <= gates["mean_union_candidate_count_max"],
        "max_union_candidates": metrics["max_union_candidates"]
        <= gates["max_union_candidate_count_max"],
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
        "metrics": metrics,
        "folds": fold_metrics,
        "timing": {
            "fit_seconds_total": total_fit_seconds,
            "search_seconds_total": total_search_seconds,
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
