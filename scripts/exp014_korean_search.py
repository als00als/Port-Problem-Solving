"""Notebook support for diagnosing and modestly widening Korean fuzzy search."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from scripts.exp014_validation import (
    SEEDS,
    load_runtime,
    load_validation_split,
    read_data,
    select_prediction,
)


BASELINE = {"limit": 40, "cutoff": 50, "max_candidates": 60}
VARIANTS = (
    {"limit": 50, "cutoff": 50, "max_candidates": 70},
    {"limit": 80, "cutoff": 50, "max_candidates": 100},
    {"limit": 100, "cutoff": 50, "max_candidates": 120},
)


def diagnose_korean_missing(seed: int) -> pd.DataFrame:
    """For EXP-014 Korean misses, report target fuzzy score/rank in full catalog."""
    runtime = load_runtime()
    _, _, _, fuzz, is_korean, _, process, romanized_name = runtime
    train, catalog = read_data()
    valid = load_validation_split(train, seed)
    # Diagnostics only need one candidate pass; evaluate_split intentionally
    # requires two passes so its returned result always has a determinism check.
    details = _evaluate_once(seed, BASELINE)["details"]
    missing_ids = set(
        details.loc[
            details["is_korean"] & ~details["candidate_contains_target"], "sample_id"
        ]
    )
    row_by_id = valid.set_index("sample_id")
    catalog_ids = catalog.reset_index(drop=True)["vessel_id"].astype(str).tolist()
    catalog_names = [romanized_name(name) for name in catalog["vessel_name"]]
    catalog_names_normalized = [name or "" for name in catalog_names]
    rows: list[dict[str, Any]] = []
    for sample_id in missing_ids:
        row = row_by_id.loc[sample_id]
        if not is_korean(row.get("vessel_name")):
            continue
        query = romanized_name(row.get("vessel_name"))
        target_id = str(row["target_vessel_id"])
        target_index = catalog_ids.index(target_id)
        target_key = catalog_names_normalized[target_index]
        target_score = float(fuzz.ratio(query, target_key)) if query and target_key else 0.0
        matches = process.extract(
            query, catalog_names_normalized, scorer=fuzz.ratio, limit=None, score_cutoff=0
        ) if query else []
        target_rank = next(
            (rank for rank, (_, _, index) in enumerate(matches, start=1) if int(index) == target_index),
            None,
        )
        current_matches = process.extract(
            query,
            catalog_names_normalized,
            scorer=fuzz.ratio,
            limit=BASELINE["limit"],
            score_cutoff=BASELINE["cutoff"],
        ) if query else []
        rows.append(
            {
                "seed": seed,
                "sample_id": sample_id,
                "input_name": row.get("vessel_name"),
                "target_vessel_id": target_id,
                "target_catalog_name": catalog.iloc[target_index]["vessel_name"],
                "target_fuzzy_score": target_score,
                "target_rank_all_catalog": target_rank,
                "target_passes_exp014_cutoff": target_score >= BASELINE["cutoff"],
                "exp014_fuzzy_candidate_count": len(current_matches),
                "exp014_candidate_indices": ",".join(str(int(i)) for _, _, i in current_matches),
            }
        )
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)


def _evaluate_once(seed: int, config: dict[str, int]) -> dict[str, Any]:
    runtime = load_runtime()
    index_class, fuzzy_bonus, base_features, fuzz, is_korean, _, process, romanized_name = runtime
    _, catalog = read_data()
    valid = load_validation_split(read_data()[0], seed)
    catalog = catalog.reset_index(drop=True)
    index = index_class(catalog)
    ids = catalog["vessel_id"].astype(str).tolist()
    details: list[dict[str, Any]] = []
    started = time.perf_counter()
    for row in valid.to_dict("records"):
        base_indices = index.generate_base_candidates(row, config["max_candidates"])
        fuzzy_indices: list[int] = []
        fuzzy_scores: dict[int, float] = {}
        if is_korean(row.get("vessel_name")):
            query = romanized_name(row.get("vessel_name"))
            if query:
                for _, score, candidate_index in process.extract(
                    query,
                    index.romanized_names,
                    scorer=fuzz.ratio,
                    limit=config["limit"],
                    score_cutoff=config["cutoff"],
                ):
                    candidate_index = int(candidate_index)
                    fuzzy_indices.append(candidate_index)
                    fuzzy_scores[candidate_index] = float(score) / 100.0
        merged: list[int] = []
        seen: set[int] = set()
        for candidate_index in fuzzy_indices + base_indices:
            candidate_index = int(candidate_index)
            if candidate_index not in seen:
                merged.append(candidate_index)
                seen.add(candidate_index)
            if len(merged) >= config["max_candidates"]:
                break
        ranked = []
        for candidate_index in merged:
            item = catalog.iloc[candidate_index]
            feature_score = base_features(row, item)["score"]
            fuzzy_score = fuzzy_scores.get(candidate_index, 0.0) if is_korean(row.get("vessel_name")) else 0.0
            ranked.append(
                {
                    "score": feature_score + fuzzy_bonus * fuzzy_score,
                    "vessel_id": ids[candidate_index],
                    "vessel_name": item.get("vessel_name"),
                }
            )
        pred = select_prediction(ranked, tie_policy="vessel_id")
        target = str(row["target_vessel_id"])
        details.append(
            {
                "sample_id": row["sample_id"],
                "target_vessel_id": target,
                "prediction": pred,
                "is_korean": is_korean(row.get("vessel_name")),
                "correct": pred == target,
                "candidate_contains_target": target in {ids[i] for i in merged},
                "candidate_count": len(merged),
            }
        )
    frame = pd.DataFrame(details)
    korean = frame[frame["is_korean"]]
    non_korean = frame[~frame["is_korean"]]
    metrics = {
        "weighted": 0.7 * float(korean["correct"].mean()) + 0.3 * float(non_korean["correct"].mean()),
        "korean_accuracy": float(korean["correct"].mean()),
        "non_korean_accuracy": float(non_korean["correct"].mean()),
        "korean_coverage": float(korean["candidate_contains_target"].mean()),
        "overall_coverage": float(frame["candidate_contains_target"].mean()),
        "mean_candidate_count": float(frame["candidate_count"].mean()),
        "elapsed_seconds": time.perf_counter() - started,
        "korean_candidate_missing_rows": int((~korean["candidate_contains_target"]).sum()),
    }
    return {"metrics": metrics, "details": frame}


def evaluate_split(seed: int, config: dict[str, int], repeats: int = 2) -> dict[str, Any]:
    """Evaluate a config repeatedly and assert row predictions are deterministic."""
    if repeats < 2:
        raise ValueError("Use at least two repetitions for determinism verification")
    runs = [_evaluate_once(seed, config) for _ in range(repeats)]
    first_predictions = runs[0]["details"]["prediction"].reset_index(drop=True)
    repeatable = all(
        first_predictions.equals(run["details"]["prediction"].reset_index(drop=True))
        for run in runs[1:]
    )
    return {
        "seed": seed,
        "config": dict(config),
        "metrics": runs[0]["metrics"],
        "details": runs[0]["details"],
        "repeatable": repeatable,
        "repetitions": repeats,
    }


def compare_all(repeats: int = 2) -> tuple[pd.DataFrame, dict[tuple[int, str], dict[str, Any]]]:
    """Return metric rows and row details for baseline plus modest variants."""
    configs = [("EXP-014", BASELINE), *[(f"limit{c['limit']}_cutoff{c['cutoff']}_max{c['max_candidates']}", c) for c in VARIANTS]]
    rows = []
    outputs = {}
    for seed in SEEDS:
        for name, config in configs:
            result = evaluate_split(seed, config, repeats=repeats)
            outputs[(seed, name)] = result
            rows.append(
                {
                    "seed": seed,
                    "variant": name,
                    **result["config"],
                    **result["metrics"],
                    "repeatable": result["repeatable"],
                }
            )
    table = pd.DataFrame(rows)
    baseline = table[table["variant"] == "EXP-014"].set_index("seed")
    for metric in ("weighted", "korean_accuracy", "non_korean_accuracy", "korean_coverage", "overall_coverage", "mean_candidate_count", "elapsed_seconds"):
        table[f"delta_{metric}"] = table.apply(
            lambda row: row[metric] - baseline.loc[row["seed"], metric], axis=1
        )
    return table, outputs
