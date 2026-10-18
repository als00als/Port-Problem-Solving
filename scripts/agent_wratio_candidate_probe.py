"""Evaluate WRatio retrieval as an isolated candidate-recall signal."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "experiments" / "splits"
RESULTS = ROOT / "experiments" / "results" / "agent_wratio_20261018"
SEED = 31421
FUZZY_LIMIT = 100
FUZZY_CUTOFF = 50
EXCLUDED_SPLITS = (
    "group_split_seed42.json",
    "group_split_seed123.json",
    "group_split_seed2026.json",
    "review_policy_holdout_seed31415.json",
    "review_policy_holdout_seed31416.json",
    "review_policy_holdout_seed31417.json",
    "agent_expansion_holdout_seed31418.json",
    "agent_ngram_holdout_seed31419.json",
    "agent_wratio_holdout_seed31420.json",
)


def reserve_holdout(train: pd.DataFrame) -> dict:
    excluded_sample_ids: set[str] = set()
    for filename in EXCLUDED_SPLITS:
        split = json.loads((SPLITS / filename).read_text(encoding="utf-8"))
        excluded_sample_ids.update(split["valid_sample_ids"])

    available = train[~train["sample_id"].astype(str).isin(excluded_sample_ids)].copy()
    available["target_vessel_id"] = available["target_vessel_id"].astype(str)
    groups = available["target_vessel_id"].drop_duplicates().to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    _, valid_group_indices = next(splitter.split(groups, groups=groups))
    valid_groups = set(groups[valid_group_indices])
    valid = available[available["target_vessel_id"].isin(valid_groups)]
    valid_ids = sorted(valid["sample_id"].astype(str))

    payload = {
        "method": "new_target_group_holdout_excluding_all_prior_validation_and_review_splits",
        "seed": SEED,
        "valid_group_fraction": 0.20,
        "excluded_splits": list(EXCLUDED_SPLITS),
        "available_target_group_count": int(len(groups)),
        "valid_target_group_count": int(len(valid_groups)),
        "train_sample_ids": sorted(set(train["sample_id"].astype(str)) - set(valid_ids)),
        "valid_sample_ids": valid_ids,
    }
    (SPLITS / f"agent_wratio_holdout_seed{SEED}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def evaluate() -> dict:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "submit" / "model"))
    from adaptive_lib import AdaptiveCatalogIndex, is_korean, romanized_name
    from scripts.exp014_validation import read_data

    train, catalog = read_data()
    split = reserve_holdout(train)
    valid = train[train["sample_id"].astype(str).isin(split["valid_sample_ids"])].copy()
    index = AdaptiveCatalogIndex(catalog.reset_index(drop=True))

    rows = valid[valid["vessel_name"].map(is_korean)].to_dict("records")
    observations: list[dict] = []
    elapsed_seconds = 0.0
    for row in rows:
        base_indices, _ = index.candidates_with_scores(row)
        query = romanized_name(row.get("vessel_name"))
        started = time.perf_counter()
        matches = process.extract(
            query,
            index.romanized_names,
            scorer=fuzz.WRatio,
            limit=FUZZY_LIMIT,
            score_cutoff=FUZZY_CUTOFF,
        ) if query else []
        elapsed_seconds += time.perf_counter() - started

        wratio_indices = [int(candidate_index) for _, _, candidate_index in matches]
        base_ids = {index.ids[int(candidate_index)] for candidate_index in base_indices}
        wratio_ids = {index.ids[candidate_index] for candidate_index in wratio_indices}
        union_ids = base_ids | wratio_ids
        target = str(row["target_vessel_id"])
        observations.append(
            {
                "target_in_baseline": target in base_ids,
                "target_in_wratio_top100": target in wratio_ids,
                "target_in_wratio_top5": target in set(index.ids[i] for i in wratio_indices[:5]),
                "target_in_union": target in union_ids,
                "newly_recovered": target not in base_ids and target in wratio_ids,
                "baseline_candidate_count": len(base_ids),
                "wratio_candidate_count": len(wratio_ids),
                "union_candidate_count": len(union_ids),
            }
        )

    frame = pd.DataFrame(observations)
    baseline_missing = ~frame["target_in_baseline"] if len(frame) else pd.Series(dtype=bool)
    summary = {
        "holdout_seed": SEED,
        "holdout_rows": int(len(valid)),
        "korean_rows": int(len(frame)),
        "search": {
            "scorer": "rapidfuzz.fuzz.WRatio",
            "limit": FUZZY_LIMIT,
            "score_cutoff": FUZZY_CUTOFF,
            "query_and_catalog_names": "romanized_name() output",
        },
        "korean_results": {
            "baseline_target_candidate_rate": float(frame["target_in_baseline"].mean()),
            "wratio_top100_target_rate": float(frame["target_in_wratio_top100"].mean()),
            "wratio_top5_target_rate": float(frame["target_in_wratio_top5"].mean()),
            "union_target_candidate_rate": float(frame["target_in_union"].mean()),
            "baseline_missing_rows": int(baseline_missing.sum()),
            "newly_recovered_targets": int(frame["newly_recovered"].sum()),
            "new_recovery_rate_on_baseline_misses": (
                float(frame.loc[baseline_missing, "target_in_wratio_top100"].mean())
                if baseline_missing.any() else None
            ),
            "mean_baseline_candidates": float(frame["baseline_candidate_count"].mean()),
            "mean_wratio_candidates": float(frame["wratio_candidate_count"].mean()),
            "mean_union_candidates": float(frame["union_candidate_count"].mean()),
            "wratio_search_mean_ms": elapsed_seconds / max(len(frame), 1) * 1000,
        },
        "predictions_and_ranking_unchanged": True,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"summary_seed{SEED}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
