"""Exploratory grouped CV for syllable-conditioned Korean name variants.

This probe is not an independent estimate: every target group in official
train has prior validation or error-analysis exposure. It never changes the
submission predictor and records aggregate results only.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process


ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "experiments" / "splits" / "agent_syllable_transliteration_cv_seed31426.json"
RESULT_PATH = ROOT / "experiments" / "results" / "agent_syllable_transliteration_20261018" / "summary.json"
DATA_ROOT = ROOT / "data" / "open"

MIN_GROUP_SUPPORT = 2
MAX_CHUNK_LENGTH = 5
MAX_QUERY_VARIANTS = 16
FUZZY_LIMIT = 40
FUZZY_CUTOFF = 50
MAX_EXPANDED_CANDIDATES = 100
MAX_UNION_CANDIDATES = 120


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_hangul_syllable(char: str) -> bool:
    return len(char) == 1 and "가" <= char <= "힣"


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def align_syllables_to_latin(
    syllables: list[str], target: str, romanized_name
) -> list[tuple[str, str]] | None:
    """Align each Hangul syllable to a 0-5 character Latin substring."""
    n, m = len(syllables), len(target)
    if not n or not target or m > n * MAX_CHUNK_LENGTH:
        return None

    costs = [[math.inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [
        [None] * (m + 1) for _ in range(n + 1)
    ]
    costs[0][0] = 0.0
    romanized_syllables = [romanized_name(syllable) for syllable in syllables]

    for i, base in enumerate(romanized_syllables):
        for start in range(m + 1):
            if not math.isfinite(costs[i][start]):
                continue
            for length in range(MAX_CHUNK_LENGTH + 1):
                end = start + length
                if end > m:
                    break
                piece = target[start:end]
                cost = edit_distance(base, piece) + 0.10 * abs(len(base) - len(piece))
                if length == 0:
                    cost += 0.35
                candidate = costs[i][start] + cost
                if candidate < costs[i + 1][end]:
                    costs[i + 1][end] = candidate
                    back[i + 1][end] = (start, length)

    if not math.isfinite(costs[n][m]):
        return None
    aligned: list[tuple[str, str]] = [("", "")] * n
    end = m
    for i in range(n, 0, -1):
        step = back[i][end]
        if step is None:
            return None
        start, length = step
        aligned[i - 1] = (syllables[i - 1], target[start : start + length])
        end = start
    return aligned


def make_training_pairs(train: pd.DataFrame, catalog: pd.DataFrame, normalize_name):
    catalog_names = dict(
        zip(catalog["vessel_id"].astype(str), catalog["vessel_name"].map(normalize_name))
    )
    pairs: set[tuple[tuple[str, ...], str, str]] = set()
    for row in train.to_dict("records"):
        name = row.get("vessel_name")
        if not isinstance(name, str):
            continue
        normalized = normalize_name(name)
        if not normalized or any(not is_hangul_syllable(char) for char in normalized):
            continue
        vessel_id = str(row["target_vessel_id"])
        target = catalog_names.get(vessel_id, "")
        if target:
            pairs.add((tuple(normalized), target, vessel_id))
    return sorted(pairs)


def learn_syllable_mappings(
    training_rows: pd.DataFrame, catalog: pd.DataFrame, normalize_name, romanized_name
) -> tuple[dict[str, list[tuple[str, int]]], int, int]:
    pairs = make_training_pairs(training_rows, catalog, normalize_name)
    support: dict[tuple[str, str], set[str]] = defaultdict(set)
    aligned_pair_count = 0
    for syllables, target, vessel_id in pairs:
        alignment = align_syllables_to_latin(list(syllables), target, romanized_name)
        if alignment is None:
            continue
        aligned_pair_count += 1
        for syllable, latin_piece in alignment:
            support[(syllable, latin_piece)].add(vessel_id)

    mappings: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (syllable, latin_piece), vessel_ids in support.items():
        if len(vessel_ids) >= MIN_GROUP_SUPPORT:
            mappings[syllable].append((latin_piece, len(vessel_ids)))
    for syllable in mappings:
        mappings[syllable].sort(key=lambda item: (-item[1], item[0]))
    return dict(mappings), len(pairs), aligned_pair_count


def generate_variants(
    name: str, mappings: dict[str, list[tuple[str, int]]], romanized_name
) -> list[str]:
    syllables = [char for char in name if is_hangul_syllable(char)]
    if len(syllables) != len(name) or not syllables:
        return []

    # Keep common learned outputs and the per-syllable Unidecode spelling.
    beam: list[tuple[str, float]] = [("", 0.0)]
    for syllable in syllables:
        base = romanized_name(syllable)
        options: list[tuple[str, int]] = [(base, 0)]
        for latin_piece, count in mappings.get(syllable, []):
            if latin_piece != base and all(piece != latin_piece for piece, _ in options):
                options.append((latin_piece, count))
            if len(options) >= 3:
                break
        expanded: list[tuple[str, float]] = []
        for prefix, score in beam:
            for latin_piece, count in options:
                preference = 0.0 if count == 0 else math.log1p(count) - 0.5
                expanded.append((prefix + latin_piece, score + preference))
        expanded.sort(key=lambda item: (-item[1], item[0]))
        beam = expanded[: MAX_QUERY_VARIANTS * 2]

    base_query = romanized_name(name)
    variants: list[str] = []
    seen = {base_query}
    for candidate, _ in beam:
        if candidate and candidate not in seen:
            seen.add(candidate)
            variants.append(candidate)
            if len(variants) >= MAX_QUERY_VARIANTS:
                break
    return variants


def _aggregate_observations(observations: list[dict]) -> dict:
    if not observations:
        return {
            "rows": 0,
            "baseline_target_recall": None,
            "syllable_union_target_recall": None,
            "absolute_recall_gain": None,
            "baseline_miss_count": 0,
            "newly_recovered_targets": 0,
            "recovery_rate_on_baseline_misses": None,
            "mean_baseline_candidates": None,
            "mean_union_candidates": None,
            "max_union_candidates": 0,
            "mean_generated_variants": None,
        }
    frame = pd.DataFrame(observations)
    baseline = float(frame["target_in_baseline"].mean())
    union = float(frame["target_in_union"].mean())
    misses = ~frame["target_in_baseline"]
    return {
        "rows": int(len(frame)),
        "baseline_target_recall": baseline,
        "syllable_union_target_recall": union,
        "absolute_recall_gain": union - baseline,
        "baseline_miss_count": int(misses.sum()),
        "newly_recovered_targets": int(frame["newly_recovered"].sum()),
        "recovery_rate_on_baseline_misses": (
            float(frame.loc[misses, "target_in_union"].mean()) if misses.any() else None
        ),
        "mean_baseline_candidates": float(frame["baseline_candidate_count"].mean()),
        "mean_union_candidates": float(frame["union_candidate_count"].mean()),
        "max_union_candidates": int(frame["union_candidate_count"].max()),
        "mean_generated_variants": float(frame["variant_count"].mean()),
    }


def evaluate() -> dict:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Pre-registered split not found: {SPLIT_PATH}")
    if RESULT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite result: {RESULT_PATH}")

    import sys

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "submit" / "model"))
    from adaptive_lib import AdaptiveCatalogIndex, is_korean, normalize_name, romanized_name
    from scripts.exp014_validation import read_data

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if split["train_sha256"] != file_sha256(DATA_ROOT / "train.csv"):
        raise AssertionError("train.csv differs from the pre-registered input")
    if split["catalog_sha256"] != file_sha256(DATA_ROOT / "vessel_catalog.csv"):
        raise AssertionError("vessel_catalog.csv differs from the pre-registered input")

    train, catalog = read_data()
    group_to_fold = {
        str(group_id): int(fold["fold"])
        for fold in split["folds"]
        for group_id in fold["target_group_ids"]
    }
    if set(group_to_fold) != set(train["target_vessel_id"].astype(str)):
        raise AssertionError("Frozen folds do not cover the train target groups exactly")
    if len(set(group_to_fold.values())) != len(split["folds"]):
        raise AssertionError("Fold assignments are invalid")

    index = AdaptiveCatalogIndex(catalog.reset_index(drop=True))
    observations: list[dict] = []
    fold_results: list[dict] = []
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
        training_groups = set(training["target_vessel_id"].astype(str))
        if valid_groups & training_groups:
            raise AssertionError(f"Target-group leakage in fold {fold_id}")

        fit_started = time.perf_counter()
        mappings, pair_count, aligned_pair_count = learn_syllable_mappings(
            training, catalog, normalize_name, romanized_name
        )
        fit_seconds = time.perf_counter() - fit_started
        total_fit_seconds += fit_seconds
        fold_observations: list[dict] = []
        search_seconds = 0.0

        for row in valid.to_dict("records"):
            if not is_korean(row.get("vessel_name")):
                continue
            baseline_indices, _ = index.candidates_with_scores(row)
            baseline_ids = {index.ids[int(i)] for i in baseline_indices}

            variants = generate_variants(row["vessel_name"], mappings, romanized_name)
            query = romanized_name(row["vessel_name"])
            variant_scores: dict[int, float] = {}
            search_started = time.perf_counter()
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
            search_seconds += time.perf_counter() - search_started

            expanded = sorted(
                variant_scores,
                key=lambda candidate_index: (
                    -variant_scores[candidate_index],
                    index.ids[candidate_index],
                ),
            )[:MAX_EXPANDED_CANDIDATES]
            merged = list(baseline_ids)
            merged_set = set(merged)
            for candidate_index in expanded:
                candidate_id = index.ids[candidate_index]
                if candidate_id not in merged_set:
                    merged.append(candidate_id)
                    merged_set.add(candidate_id)
                if len(merged) >= MAX_UNION_CANDIDATES:
                    break

            target = str(row["target_vessel_id"])
            observation = {
                "target_in_baseline": target in baseline_ids,
                "target_in_union": target in merged_set,
                "newly_recovered": target not in baseline_ids and target in merged_set,
                "baseline_candidate_count": len(baseline_ids),
                "union_candidate_count": len(merged_set),
                "variant_count": len(variants),
            }
            observations.append(observation)
            fold_observations.append(observation)

        total_search_seconds += search_seconds
        fold_results.append(
            {
                "fold": fold_id,
                "train_rows": int(len(training)),
                "train_target_groups": int(len(training_groups)),
                "valid_rows": int(len(valid)),
                "valid_korean_rows": len(fold_observations),
                "training_korean_pairs": pair_count,
                "aligned_training_pairs": aligned_pair_count,
                "learned_syllable_count": len(mappings),
                "fit_seconds": fit_seconds,
                "search_seconds": search_seconds,
                "metrics": _aggregate_observations(fold_observations),
            }
        )

    aggregate = _aggregate_observations(observations)
    gates = split["acceptance_gates"]
    gate_results = {
        "recall_gain": aggregate["absolute_recall_gain"]
        is not None
        and aggregate["absolute_recall_gain"]
        >= gates["korean_target_candidate_recall_absolute_gain_min"],
        "mean_union_candidates": aggregate["mean_union_candidates"]
        is not None
        and aggregate["mean_union_candidates"] <= gates["mean_union_candidate_count_max"],
        "max_union_candidates": aggregate["max_union_candidates"]
        <= gates["max_union_candidate_count_max"],
        "no_target_group_leakage": True,
    }
    report = {
        "experiment": split["experiment"],
        "split": str(SPLIT_PATH.relative_to(ROOT)),
        "historical_validation_overlap": True,
        "historical_error_analysis_overlap": True,
        "independent_performance_estimate": False,
        "data_hashes": {
            "train.csv": file_sha256(DATA_ROOT / "train.csv"),
            "vessel_catalog.csv": file_sha256(DATA_ROOT / "vessel_catalog.csv"),
        },
        "folds": fold_results,
        "korean_oof_metrics": aggregate,
        "timing": {
            "fit_seconds_total": total_fit_seconds,
            "search_seconds_total": total_search_seconds,
        },
        "fixed_method": split["fixed_method"],
        "acceptance_gates": gates,
        "gate_results": gate_results,
        "exploratory_gates_passed": all(gate_results.values()),
        "submission_promotion_eligible": False,
        "submission_code_or_archive_modified": False,
        "per_row_inputs_or_predictions_saved": False,
        "limitations": [
            "All official target groups were previously used in validation or seed31424 error analysis.",
            "Grouped out-of-fold mapping prevents same-target training leakage but does not remove historical model-selection bias.",
            "This is candidate-recall-only; no ranking-score or top-1 accuracy changes were evaluated.",
        ],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
