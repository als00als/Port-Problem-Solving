"""Probe train-learned romanization edit rules for Korean candidate retrieval."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "experiments" / "splits"
RESULTS = ROOT / "experiments" / "results" / "agent_learned_transliteration_20261018"
SEEDS = (31422, 31423)
MIN_GROUP_SUPPORT = 3
MAX_QUERY_VARIANTS = 20
VARIANT_FUZZY_LIMIT = 40
VARIANT_FUZZY_CUTOFF = 50
EXPANSION_LIMIT = 100
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
    "agent_wratio_holdout_seed31421.json",
)


def reserve_holdouts(train: pd.DataFrame) -> dict[int, dict]:
    excluded_sample_ids: set[str] = set()
    for filename in EXCLUDED_SPLITS:
        payload = json.loads((SPLITS / filename).read_text(encoding="utf-8"))
        excluded_sample_ids.update(payload["valid_sample_ids"])

    available = train[~train["sample_id"].astype(str).isin(excluded_sample_ids)].copy()
    available["target_vessel_id"] = available["target_vessel_id"].astype(str)
    reserved: dict[int, dict] = {}
    for seed in SEEDS:
        groups = available["target_vessel_id"].drop_duplicates().to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        _, valid_group_indices = next(splitter.split(groups, groups=groups))
        valid_groups = set(groups[valid_group_indices])
        valid = available[available["target_vessel_id"].isin(valid_groups)]
        valid_ids = sorted(valid["sample_id"].astype(str))
        payload = {
            "method": "new_target_group_holdout_excluding_all_prior_validation_and_review_splits",
            "seed": seed,
            "valid_group_fraction": 0.20,
            "excluded_splits": list(EXCLUDED_SPLITS),
            "excluded_new_holdout_seeds": [s for s in SEEDS if s < seed],
            "available_target_group_count": int(len(groups)),
            "valid_target_group_count": int(len(valid_groups)),
            "train_sample_ids": sorted(set(train["sample_id"].astype(str)) - set(valid_ids)),
            "valid_sample_ids": valid_ids,
        }
        (SPLITS / f"agent_learned_transliteration_holdout_seed{seed}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        reserved[seed] = payload
        excluded_sample_ids.update(valid_ids)
        available = available[~available["sample_id"].astype(str).isin(valid_ids)]
    return reserved


def learn_edit_rules(training_rows: pd.DataFrame, catalog: pd.DataFrame, romanized_name) -> list[dict]:
    catalog_names = dict(
        zip(catalog["vessel_id"].astype(str), catalog["vessel_name"].map(romanized_name))
    )
    support: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    pairs = set()
    for row in training_rows.to_dict("records"):
        name = row.get("vessel_name")
        if not isinstance(name, str) or not any("가" <= char <= "힣" for char in name):
            continue
        source = romanized_name(name)
        vessel_id = str(row["target_vessel_id"])
        target = catalog_names.get(vessel_id, "")
        if source and target:
            pairs.add((source, target, vessel_id))

    for source, target, vessel_id in pairs:
        matcher = SequenceMatcher(None, source, target, autojunk=False)
        for operation, i1, i2, j1, j2 in matcher.get_opcodes():
            if operation == "equal":
                continue
            source_piece = source[i1:i2]
            target_piece = target[j1:j2]
            if source_piece == target_piece or len(source_piece) > 2 or len(target_piece) > 2:
                continue
            left = source[i1 - 1] if i1 > 0 else ""
            right = source[i2] if i2 < len(source) else ""
            if not left and not right:
                continue
            support[(left, source_piece, right, target_piece)].add(vessel_id)

    rules = [
        {"left": left, "source": source, "right": right, "target": target, "support": len(groups)}
        for (left, source, right, target), groups in support.items()
        if len(groups) >= MIN_GROUP_SUPPORT
    ]
    return sorted(
        rules,
        key=lambda rule: (-rule["support"], rule["left"], rule["source"], rule["right"], rule["target"]),
    )


def generate_variants(query: str, rules: list[dict]) -> list[str]:
    variants: list[str] = []
    seen = {query}
    for rule in rules:
        source = rule["source"]
        for position in range(len(query) - len(source) + 1):
            if source and query[position : position + len(source)] != source:
                continue
            if rule["left"] and (position == 0 or query[position - 1] != rule["left"]):
                continue
            after = position + len(source)
            if rule["right"] and (after >= len(query) or query[after] != rule["right"]):
                continue
            candidate = query[:position] + rule["target"] + query[after:]
            if candidate and candidate not in seen:
                seen.add(candidate)
                variants.append(candidate)
                if len(variants) >= MAX_QUERY_VARIANTS:
                    return variants
    return variants


def evaluate() -> dict:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "submit" / "model"))
    from adaptive_lib import AdaptiveCatalogIndex, is_korean, romanized_name
    from scripts.exp014_validation import read_data

    train, catalog = read_data()
    holdouts = reserve_holdouts(train)
    excluded_ids: set[str] = set()
    for filename in EXCLUDED_SPLITS:
        excluded_ids.update(json.loads((SPLITS / filename).read_text(encoding="utf-8"))["valid_sample_ids"])
    for payload in holdouts.values():
        excluded_ids.update(payload["valid_sample_ids"])
    training_rows = train[~train["sample_id"].astype(str).isin(excluded_ids)].copy()
    training_groups = set(training_rows["target_vessel_id"].astype(str))

    split_results = {}
    for seed, payload in holdouts.items():
        valid = train[train["sample_id"].astype(str).isin(payload["valid_sample_ids"])].copy()
        valid_groups = set(valid["target_vessel_id"].astype(str))
        if training_groups.intersection(valid_groups):
            raise AssertionError(f"target-group leakage in holdout {seed}")

        index = AdaptiveCatalogIndex(catalog.reset_index(drop=True))
        rules_started = time.perf_counter()
        rules = learn_edit_rules(training_rows, catalog, romanized_name)
        rule_fit_seconds = time.perf_counter() - rules_started
        korean_rows = valid[valid["vessel_name"].map(is_korean)].to_dict("records")
        observations: list[dict] = []
        search_seconds = 0.0
        for row in korean_rows:
            baseline_indices, _ = index.candidates_with_scores(row)
            query = romanized_name(row.get("vessel_name"))
            variant_started = time.perf_counter()
            variants = generate_variants(query, rules) if query else []
            scored: dict[int, float] = {}
            for variant in [query, *variants]:
                for _, score, candidate_index in process.extract(
                    variant,
                    index.romanized_names,
                    scorer=fuzz.ratio,
                    limit=VARIANT_FUZZY_LIMIT,
                    score_cutoff=VARIANT_FUZZY_CUTOFF,
                ):
                    candidate_index = int(candidate_index)
                    scored[candidate_index] = max(scored.get(candidate_index, 0.0), float(score))
            ordered = sorted(scored, key=lambda candidate_index: (-scored[candidate_index], index.ids[candidate_index]))
            transformed_indices = ordered[:EXPANSION_LIMIT]
            search_seconds += time.perf_counter() - variant_started

            baseline_ids = {index.ids[int(i)] for i in baseline_indices}
            transformed_ids = {index.ids[i] for i in transformed_indices}
            union_ids = baseline_ids | transformed_ids
            target = str(row["target_vessel_id"])
            observations.append(
                {
                    "target_in_baseline": target in baseline_ids,
                    "target_in_transformed_top100": target in transformed_ids,
                    "target_in_union": target in union_ids,
                    "newly_recovered": target not in baseline_ids and target in transformed_ids,
                    "baseline_candidate_count": len(baseline_ids),
                    "transformed_candidate_count": len(transformed_ids),
                    "union_candidate_count": len(union_ids),
                    "variant_count": len(variants),
                }
            )

        frame = pd.DataFrame(observations)
        missing = ~frame["target_in_baseline"] if len(frame) else pd.Series(dtype=bool)
        split_results[str(seed)] = {
            "holdout_rows": int(len(valid)),
            "korean_rows": int(len(frame)),
            "training_rows": int(len(training_rows)),
            "training_target_groups": int(len(training_groups)),
            "learned_rule_count": int(len(rules)),
            "rule_fit_seconds": rule_fit_seconds,
            "korean_results": {
                "baseline_target_candidate_rate": float(frame["target_in_baseline"].mean()),
                "transformed_top100_target_rate": float(frame["target_in_transformed_top100"].mean()),
                "union_target_candidate_rate": float(frame["target_in_union"].mean()),
                "baseline_missing_rows": int(missing.sum()),
                "newly_recovered_targets": int(frame["newly_recovered"].sum()),
                "new_recovery_rate_on_baseline_misses": (
                    float(frame.loc[missing, "target_in_transformed_top100"].mean()) if missing.any() else None
                ),
                "mean_baseline_candidates": float(frame["baseline_candidate_count"].mean()),
                "mean_transformed_candidates": float(frame["transformed_candidate_count"].mean()),
                "mean_union_candidates": float(frame["union_candidate_count"].mean()),
                "mean_query_variants": float(frame["variant_count"].mean()),
                "search_mean_ms": search_seconds / max(len(frame), 1) * 1000,
            },
        }

    summary = {
        "holdout_seeds": list(SEEDS),
        "training_data": "official train/catalog only; all prior and new holdout target groups excluded",
        "rule_learning": {
            "alignment": "difflib.SequenceMatcher opcodes over romanized input and target catalog name",
            "max_edit_piece_length": 2,
            "minimum_distinct_target_group_support": MIN_GROUP_SUPPORT,
            "max_query_variants": MAX_QUERY_VARIANTS,
        },
        "candidate_search": {
            "scorer": "rapidfuzz.fuzz.ratio",
            "per_variant_limit": VARIANT_FUZZY_LIMIT,
            "score_cutoff": VARIANT_FUZZY_CUTOFF,
            "expanded_candidate_limit": EXPANSION_LIMIT,
            "baseline_and_final_prediction_changed": False,
        },
        "splits": split_results,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
