"""One-shot evaluation of character n-gram candidate retrieval on a new holdout."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "experiments" / "splits"
RESULTS = ROOT / "experiments" / "results" / "agent_ngram_20261018"
SEED = 31419
TOP_NGRAM_CANDIDATES = 100
NGRAM_RANGE = (2, 5)
EXCLUDED_SPLITS = (
    "group_split_seed42.json",
    "group_split_seed123.json",
    "group_split_seed2026.json",
    "review_policy_holdout_seed31415.json",
    "review_policy_holdout_seed31416.json",
    "review_policy_holdout_seed31417.json",
    "agent_expansion_holdout_seed31418.json",
)


def reserve_holdout(train: pd.DataFrame) -> dict:
    excluded_sample_ids: set[str] = set()
    for filename in EXCLUDED_SPLITS:
        payload = json.loads((SPLITS / filename).read_text(encoding="utf-8"))
        excluded_sample_ids.update(payload["valid_sample_ids"])

    available = train[~train["sample_id"].astype(str).isin(excluded_sample_ids)].copy()
    available["target_vessel_id"] = available["target_vessel_id"].astype(str)
    groups = available["target_vessel_id"].drop_duplicates().to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    _, valid_group_indices = next(splitter.split(groups, groups=groups))
    valid_groups = set(groups[valid_group_indices])
    valid = available[available["target_vessel_id"].isin(valid_groups)]
    train_ids = sorted(set(train["sample_id"].astype(str)) - set(valid["sample_id"].astype(str)))
    valid_ids = sorted(valid["sample_id"].astype(str))

    if excluded_sample_ids.intersection(valid_ids):
        raise AssertionError("new holdout overlaps a previously evaluated split")
    if set(valid["target_vessel_id"]).intersection(
        train.loc[train["sample_id"].astype(str).isin(train_ids), "target_vessel_id"].astype(str)
    ):
        # train_ids here include excluded historical validation rows; remove those
        # groups before asserting the new target-group boundary.
        eligible_train = train[
            train["sample_id"].astype(str).isin(set(available["sample_id"].astype(str)) - set(valid_ids))
        ]
        if set(valid["target_vessel_id"]).intersection(eligible_train["target_vessel_id"].astype(str)):
            raise AssertionError("target groups overlap within the new split")

    payload = {
        "method": "new_target_group_holdout_excluding_all_prior_validation_and_review_splits",
        "seed": SEED,
        "valid_group_fraction": 0.20,
        "excluded_splits": list(EXCLUDED_SPLITS),
        "available_target_group_count": int(len(groups)),
        "valid_target_group_count": int(len(valid_groups)),
        "train_sample_ids": train_ids,
        "valid_sample_ids": valid_ids,
    }
    path = SPLITS / f"agent_ngram_holdout_seed{SEED}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def evaluate() -> dict:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "submit" / "model"))
    from adaptive_lib import AdaptiveCatalogIndex, is_korean, romanized_name
    from rapidfuzz import fuzz, process

    from scripts.exp014_validation import read_data

    train, catalog = read_data()
    split = reserve_holdout(train)
    valid = train[train["sample_id"].astype(str).isin(split["valid_sample_ids"])].copy()
    catalog = catalog.reset_index(drop=True)
    index = AdaptiveCatalogIndex(catalog)
    catalog_ids = catalog["vessel_id"].astype(str).to_numpy()

    # Fit only against the official catalog. Validation inputs are transformed,
    # never used to fit or update the vocabulary.
    catalog_names = [romanized_name(name) for name in catalog["vessel_name"]]
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=NGRAM_RANGE, lowercase=False, dtype=np.float32
    )
    catalog_matrix = vectorizer.fit_transform(catalog_names)

    korean_rows = valid[valid["vessel_name"].map(is_korean)]
    observations: list[dict] = []
    miss_diagnostics: list[dict] = []
    elapsed_seconds = 0.0
    for row in korean_rows.to_dict("records"):
        baseline_indices, _ = index.candidates_with_scores(row)
        query = romanized_name(row.get("vessel_name"))
        started = time.perf_counter()
        query_vector = vectorizer.transform([query]) if query else None
        if query_vector is None or query_vector.nnz == 0:
            ngram_indices: list[int] = []
            ngram_scores: dict[int, float] = {}
        else:
            similarities = (query_vector @ catalog_matrix.T).toarray().ravel()
            matching = np.flatnonzero(similarities > 0)
            order = np.lexsort((catalog_ids[matching], -similarities[matching]))
            chosen = matching[order[:TOP_NGRAM_CANDIDATES]]
            ngram_indices = [int(i) for i in chosen]
            ngram_scores = {int(i): float(similarities[i]) for i in chosen}
        elapsed_seconds += time.perf_counter() - started

        baseline_ids = {index.ids[int(i)] for i in baseline_indices}
        ngram_ids = {index.ids[i] for i in ngram_indices}
        union_ids = baseline_ids | ngram_ids
        target = str(row["target_vessel_id"])
        ordered_ngram_ids = [index.ids[i] for i in ngram_indices]
        target_index = index.ids.index(target)
        target_ngram_score = float(similarities[target_index]) if query_vector is not None and query_vector.nnz else 0.0
        if target not in baseline_ids:
            fuzzy_matches = process.extract(
                query,
                index.romanized_names,
                scorer=fuzz.ratio,
                limit=None,
                score_cutoff=0,
            ) if query else []
            fuzzy_target_rank = next(
                (rank for rank, (_, _, candidate_index) in enumerate(fuzzy_matches, start=1)
                 if int(candidate_index) == target_index),
                None,
            )
            target_romanized_name = index.romanized_names[target_index]
            miss_diagnostics.append(
                {
                    "target_fuzzy_score": float(fuzz.ratio(query, target_romanized_name)) if query else 0.0,
                    "target_fuzzy_rank": fuzzy_target_rank,
                    "target_ngram_score": target_ngram_score,
                    "target_ngram_rank": (
                        1 + int(np.sum(similarities > target_ngram_score))
                        + int(np.sum((similarities == target_ngram_score) & (catalog_ids < target)))
                        if target_ngram_score > 0 else None
                    ),
                    "query_equals_target_romanization": bool(query and query == target_romanized_name),
                }
            )
        observations.append(
            {
                "target_in_baseline": target in baseline_ids,
                "target_in_ngram_top100": target in ngram_ids,
                "target_in_ngram_top5": target in set(ordered_ngram_ids[:5]),
                "target_in_union": target in union_ids,
                "newly_recovered": target not in baseline_ids and target in ngram_ids,
                "baseline_candidate_count": len(baseline_ids),
                "ngram_candidate_count": len(ngram_ids),
                "union_candidate_count": len(union_ids),
                "target_ngram_score": ngram_scores.get(target_index),
            }
        )

    frame = pd.DataFrame(observations)
    baseline_missing = ~frame["target_in_baseline"] if len(frame) else pd.Series(dtype=bool)
    miss_frame = pd.DataFrame(miss_diagnostics)
    summary = {
        "holdout_seed": SEED,
        "holdout_rows": int(len(valid)),
        "korean_rows": int(len(frame)),
        "ngram": {
            "analyzer": "char",
            "ngram_range": list(NGRAM_RANGE),
            "top_candidates": TOP_NGRAM_CANDIDATES,
            "fit_corpus": "official vessel_catalog.csv romanized vessel_name only",
        },
        "korean_results": {
            "baseline_target_candidate_rate": float(frame["target_in_baseline"].mean()),
            "ngram_top100_target_rate": float(frame["target_in_ngram_top100"].mean()),
            "ngram_top5_target_rate": float(frame["target_in_ngram_top5"].mean()),
            "union_target_candidate_rate": float(frame["target_in_union"].mean()),
            "baseline_missing_rows": int(baseline_missing.sum()),
            "newly_recovered_targets": int(frame["newly_recovered"].sum()),
            "new_recovery_rate_on_baseline_misses": (
                float(frame.loc[baseline_missing, "target_in_ngram_top100"].mean())
                if baseline_missing.any()
                else None
            ),
            "mean_baseline_candidates": float(frame["baseline_candidate_count"].mean()),
            "mean_ngram_candidates": float(frame["ngram_candidate_count"].mean()),
            "mean_union_candidates": float(frame["union_candidate_count"].mean()),
            "ngram_search_mean_ms": elapsed_seconds / max(len(frame), 1) * 1000,
        },
        "baseline_miss_diagnostics": {
            "rows": int(len(miss_frame)),
            "romanized_query_equals_target_name": int(miss_frame["query_equals_target_romanization"].sum()) if len(miss_frame) else 0,
            "target_fuzzy_score_below_cutoff_50": int((miss_frame["target_fuzzy_score"] < 50).sum()) if len(miss_frame) else 0,
            "target_fuzzy_score_at_least_50_but_rank_over_40": int(((miss_frame["target_fuzzy_score"] >= 50) & (miss_frame["target_fuzzy_rank"] > 40)).sum()) if len(miss_frame) else 0,
            "target_fuzzy_score_median": float(miss_frame["target_fuzzy_score"].median()) if len(miss_frame) else None,
            "target_fuzzy_score_min": float(miss_frame["target_fuzzy_score"].min()) if len(miss_frame) else None,
            "target_fuzzy_rank_median": float(miss_frame["target_fuzzy_rank"].median()) if len(miss_frame) else None,
            "target_fuzzy_rank_min": int(miss_frame["target_fuzzy_rank"].min()) if len(miss_frame) else None,
            "target_fuzzy_rank_max": int(miss_frame["target_fuzzy_rank"].max()) if len(miss_frame) else None,
            "target_fuzzy_rank_at_most_80": int((miss_frame["target_fuzzy_rank"] <= 80).sum()) if len(miss_frame) else 0,
            "target_has_shared_char_ngram": int((miss_frame["target_ngram_score"] > 0).sum()) if len(miss_frame) else 0,
            "target_in_ngram_top100": int((miss_frame["target_ngram_rank"] <= TOP_NGRAM_CANDIDATES).sum()) if len(miss_frame) else 0,
            "target_ngram_rank_median_when_shared": (
                float(miss_frame.loc[miss_frame["target_ngram_score"] > 0, "target_ngram_rank"].median())
                if (miss_frame["target_ngram_score"] > 0).any() else None
            ),
        },
        "no_predictions_changed": True,
        "no_reranking_evaluated": True,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"summary_seed{SEED}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
