"""Aggregate target-rank diagnostics for the frozen syllable retrieval OOF run."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd


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


PREREG_PATH = ROOT / "experiments" / "splits" / "agent_syllable_rank_diagnostic_seed31429.json"
RESULT_PATH = ROOT / "experiments" / "results" / "agent_syllable_rank_diagnostic_20261018" / "summary.json"
DATA_ROOT = ROOT / "data" / "open"
RANK_BINS = ("missing", "1", "2-5", "6-10", "11-20", "21-50", "51-120")


def build_union(index, row: dict, baseline_indices: list[int], mappings: dict) -> list[int]:
    variants = generate_variants(row["vessel_name"], mappings, romanized_name)
    query = romanized_name(row["vessel_name"])
    variant_scores: dict[int, float] = {}
    from rapidfuzz import fuzz, process

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
        key=lambda candidate_index: (
            -variant_scores[candidate_index],
            index.ids[candidate_index],
        ),
    )[:MAX_EXPANDED_CANDIDATES]
    union = list(baseline_indices)
    seen = set(union)
    for candidate_index in expanded:
        if candidate_index not in seen:
            union.append(candidate_index)
            seen.add(candidate_index)
        if len(union) >= MAX_UNION_CANDIDATES:
            break
    return union


def rank_candidates(index, row: dict, candidate_indices: list[int], baseline_fuzzy_scores: dict[int, float]):
    ranked = []
    for candidate_index in candidate_indices:
        features = base_features(row, index.catalog.iloc[candidate_index])
        score = features["score"] + FUZZY_SCORE_BONUS * baseline_fuzzy_scores.get(candidate_index, 0.0)
        ranked.append((score, index.ids[candidate_index]))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked


def rank_bin(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank == 1:
        return "1"
    if rank <= 5:
        return "2-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    return "51-120"


def target_rank(ranked: list[tuple[float, str]], target: str) -> tuple[int | None, float | None]:
    for position, (score, vessel_id) in enumerate(ranked, start=1):
        if vessel_id == target:
            return position, score
    return None, None


def summarize(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    baseline_hist = {bucket: 0 for bucket in RANK_BINS}
    union_hist = {bucket: 0 for bucket in RANK_BINS}
    recovered_union_hist = {bucket: 0 for bucket in RANK_BINS}
    for row in rows:
        baseline_hist[rank_bin(row["baseline_rank"])] += 1
        union_bucket = rank_bin(row["union_rank"])
        union_hist[union_bucket] += 1
        if row["baseline_rank"] is None and row["union_rank"] is not None:
            recovered_union_hist[union_bucket] += 1

    gaps = frame["target_minus_union_top1_score"].dropna()
    recovered = frame[frame["baseline_rank"].isna() & frame["union_rank"].notna()]
    recovered_ranks = recovered["union_rank"].astype(float)
    recovered_gaps = recovered["target_minus_union_top1_score"].dropna()
    improved = int(
        (
            frame["union_rank"].notna()
            & (frame["baseline_rank"].isna() | (frame["union_rank"] < frame["baseline_rank"]))
        ).sum()
    )
    return {
        "korean_rows": int(len(frame)),
        "baseline_target_recall": float(frame["baseline_rank"].notna().mean()),
        "union_target_recall": float(frame["union_rank"].notna().mean()),
        "newly_recovered_target_count": int(len(recovered)),
        "target_rank_improved_count": improved,
        "baseline_target_rank_histogram": baseline_hist,
        "union_target_rank_histogram": union_hist,
        "newly_recovered_target_rank_histogram": recovered_union_hist,
        "newly_recovered_targets_top1": int((recovered["union_rank"] == 1).sum()),
        "newly_recovered_target_rank_summary": {
            "count": int(recovered_ranks.count()),
            "mean": float(recovered_ranks.mean()) if len(recovered_ranks) else None,
            "median": float(recovered_ranks.median()) if len(recovered_ranks) else None,
            "min": int(recovered_ranks.min()) if len(recovered_ranks) else None,
            "max": int(recovered_ranks.max()) if len(recovered_ranks) else None,
        },
        "newly_recovered_target_minus_top1_score_gap": {
            "count": int(recovered_gaps.count()),
            "mean": float(recovered_gaps.mean()) if len(recovered_gaps) else None,
            "median": float(recovered_gaps.median()) if len(recovered_gaps) else None,
            "min": float(recovered_gaps.min()) if len(recovered_gaps) else None,
            "max": float(recovered_gaps.max()) if len(recovered_gaps) else None,
        },
        "target_minus_union_top1_score_gap": {
            "count": int(gaps.count()),
            "mean": float(gaps.mean()) if len(gaps) else None,
            "median": float(gaps.median()) if len(gaps) else None,
            "p10": float(gaps.quantile(0.10)) if len(gaps) else None,
            "p90": float(gaps.quantile(0.90)) if len(gaps) else None,
            "at_least_minus_0_05_count": int((gaps >= -0.05).sum()),
            "at_least_minus_0_10_count": int((gaps >= -0.10).sum()),
        },
        "mean_union_candidate_count": float(frame["union_candidate_count"].mean()),
        "max_union_candidate_count": int(frame["union_candidate_count"].max()),
    }


def evaluate() -> dict:
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    if RESULT_PATH.exists():
        previous = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if previous.get("preregistration") != str(PREREG_PATH.relative_to(ROOT)):
            raise FileExistsError(f"Refusing to overwrite unrelated result: {RESULT_PATH}")
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
    search_seconds_total = 0.0

    for fold in split["folds"]:
        fold_id = int(fold["fold"])
        valid = train[
            train["target_vessel_id"].astype(str).map(group_to_fold).eq(fold_id)
        ].copy()
        training = train[
            train["target_vessel_id"].astype(str).map(group_to_fold).ne(fold_id)
        ].copy()
        valid_groups = set(valid["target_vessel_id"].astype(str))
        training_groups = set(training["target_vessel_id"].astype(str))
        if valid_groups & training_groups:
            raise AssertionError(f"Target-group leakage in fold {fold_id}")

        fit_started = time.perf_counter()
        mappings, _, _ = learn_syllable_mappings(
            training, catalog, normalize_name, romanized_name
        )
        fit_seconds = time.perf_counter() - fit_started
        fit_seconds_total += fit_seconds
        fold_rows: list[dict] = []
        search_seconds = 0.0

        for row in valid.to_dict("records"):
            if not is_korean(row.get("vessel_name")):
                continue
            baseline_indices, baseline_fuzzy_scores = index.candidates_with_scores(row)
            started = time.perf_counter()
            union_indices = build_union(index, row, baseline_indices, mappings)
            search_seconds += time.perf_counter() - started
            target = str(row["target_vessel_id"])
            baseline_ranked = rank_candidates(
                index, row, baseline_indices, baseline_fuzzy_scores
            )
            union_ranked = rank_candidates(
                index, row, union_indices, baseline_fuzzy_scores
            )
            baseline_target_rank, _ = target_rank(baseline_ranked, target)
            union_target_rank, target_score = target_rank(union_ranked, target)
            top1_score = union_ranked[0][0] if union_ranked else None
            observation = {
                "fold": fold_id,
                "baseline_rank": baseline_target_rank,
                "union_rank": union_target_rank,
                "target_minus_union_top1_score": (
                    target_score - top1_score
                    if target_score is not None and top1_score is not None
                    else None
                ),
                "union_candidate_count": len(union_indices),
            }
            observations.append(observation)
            fold_rows.append(observation)

        search_seconds_total += search_seconds
        fold_reports.append(
            {
                "fold": fold_id,
                "train_target_groups": len(training_groups),
                "valid_target_groups": len(valid_groups),
                "metrics": summarize(fold_rows),
                "fit_seconds": fit_seconds,
                "search_seconds": search_seconds,
            }
        )

    report = {
        "experiment": prereg["experiment"],
        "preregistration": str(PREREG_PATH.relative_to(ROOT)),
        "source_split": prereg["source_split"],
        "historical_validation_overlap": True,
        "historical_error_analysis_overlap": True,
        "independent_performance_estimate": False,
        "data_hashes": prereg["data_hashes"],
        "metrics": summarize(observations),
        "folds": fold_reports,
        "timing": {
            "fit_seconds_total": fit_seconds_total,
            "search_seconds_total": search_seconds_total,
        },
        "interpretation_rule": prereg["interpretation_rule"],
        "parameter_tuning_performed": False,
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
