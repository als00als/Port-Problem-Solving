"""Allow-listed, single-change retraining settings for the v11 ranker.

Pure standard library so both the agent environment (planning, guard) and the
competition Python 3.11 environment (training worker) can import it. Nothing in
here reads data, labels, or files; a setting is a knob name plus one value from
its allow-list, and every other knob stays at the v11 baseline.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


PROTOCOL_VERSION = "v11-improvement-1"

# Each knob changes exactly one cause or one training setting. Values are
# written as strings so the LLM schema can use a single enum per knob.
KNOBS: dict[str, dict[str, Any]] = {
    "ranker_family": {
        "kind": "training", "baseline": "lightgbm_v11",
        "values": ["lightgbm_v11", "catboost_v1_cpu", "mlp_listwise_v1_cpu",
                   "mlp_listwise_v1_cuda"],
        "description": "Replace the v11 LightGBM profile with fixed CPU CatBoost or small "
                       "CPU/CUDA PyTorch MLP profiles. Keep the same candidate matrix, inner-fold resources, "
                       "features and lookup overrides; no model tuning.",
        "addresses": ["tonnage_favored_truth_lost", "name_similarity_favored_wrong",
                      "other_rank_failure", "same_name_collision"],
        "retrain": True,
    },
    "hard_negative_top_n": {
        "kind": "training",
        "baseline": "all",
        "values": ["all", "50", "20"],
        "description": "Inside each inner training fold keep the positive plus only the N hardest "
                       "negatives per query (hardness = max name-similarity feature, ties by catalog "
                       "index). Validation candidates are untouched.",
        "addresses": ["name_similarity_favored_wrong", "same_name_collision", "number_mismatch",
                      "other_rank_failure", "near_miss_runner_up_same_name"],
        "retrain": True,
    },
    "missing_truth_policy": {
        "kind": "training",
        "baseline": "force",
        "values": ["force", "drop_query"],
        "description": "Training queries whose truth is absent from the candidate pool are either "
                       "forced in (v11) or dropped from the ranker training set.",
        "addresses": ["candidate_missing", "other_rank_failure"],
        "retrain": True,
    },
    "tonnage_missing_encoding": {
        "kind": "feature",
        "baseline": "gap_2.0",
        "values": ["gap_2.0", "nan"],
        "description": "gap_gross/gap_intl/gap_best use 2.0 for a missing tonnage (v11) or NaN so "
                       "LightGBM learns a separate missing branch. Applied identically in training "
                       "and inference.",
        "addresses": ["query_tonnage_missing", "tonnage_favored_truth_lost",
                      "near_miss_query_tonnage_missing"],
        "retrain": True,
    },
    "number_features": {
        "kind": "feature",
        "baseline": "full",
        "values": ["full", "no_asymmetry", "none"],
        "description": "Number-token features: full (v11), drop num_q_only/num_c_only, or drop all "
                       "number features (num_equal, num_conflict, q_has_num, c_has_num, num_q_only, "
                       "num_c_only) by setting them to a constant 0 in training and inference.",
        "addresses": ["number_mismatch", "near_miss_runner_up_number_mismatch"],
        "retrain": True,
    },
    "flag_type_features": {
        "kind": "feature",
        "baseline": "full",
        "values": ["full", "off"],
        "description": "Flag/vessel-type co-occurrence features (flag_prob, flag_seen, type_prob, "
                       "type_seen) on (v11) or constant 0.",
        "addresses": ["same_name_collision", "near_miss_runner_up_same_name"],
        "retrain": True,
    },
    "monotone_constraints": {
        "kind": "training",
        "baseline": "on",
        "values": ["on", "off"],
        "description": "LightGBM monotone constraints on similarity/gap/number features.",
        "addresses": ["tonnage_favored_truth_lost", "name_similarity_favored_wrong", "number_mismatch"],
        "retrain": True,
    },
    "num_leaves": {
        "kind": "training",
        "baseline": "31",
        "values": ["15", "31", "63"],
        "description": "LGBMRanker num_leaves (all other ranker parameters unchanged).",
        "addresses": ["other_rank_failure", "name_similarity_favored_wrong"],
        "retrain": True,
    },
    "n_estimators": {
        "kind": "training",
        "baseline": "400",
        "values": ["200", "400", "800"],
        "description": "LGBMRanker n_estimators (learning rate unchanged).",
        "addresses": ["other_rank_failure", "name_similarity_favored_wrong"],
        "retrain": True,
    },
    "override_policy": {
        "kind": "postprocess",
        "baseline": "callsign_and_name",
        "values": ["callsign_and_name", "callsign_only", "name_only", "none"],
        "description": "Training-table overrides after ranking: name lookup then call-sign lookup "
                       "(v11), only one of them, or none. The trained ranker is unchanged, so the "
                       "per-split baseline model is reused (no retraining).",
        "addresses": ["override_introduced_error", "override_wrong_and_model_wrong"],
        "retrain": False,
    },
}
OOF_ONLY_RANKERS = frozenset(("mlp_listwise_v1_cpu", "mlp_listwise_v1_cuda"))

ERROR_TYPES = (
    "override_introduced_error", "override_wrong_and_model_wrong", "candidate_missing",
    "same_name_collision", "number_mismatch", "query_tonnage_missing",
    "tonnage_favored_truth_lost", "name_similarity_favored_wrong", "other_rank_failure",
)
NEAR_MISS_FLAGS = (
    "near_miss_runner_up_same_name", "near_miss_runner_up_number_mismatch",
    "near_miss_query_tonnage_missing", "near_miss_runner_up_closer_tonnage",
    "near_miss_override_applied",
)
EVIDENCE_TYPES = ERROR_TYPES + NEAR_MISS_FLAGS
METRICS = ("weighted_raw", "korean_accuracy", "non_korean_accuracy", "candidate_recall",
           "newly_correct_rows", "newly_wrong_rows", "unseen_target_accuracy",
           "seen_target_accuracy", "training_seconds", "prediction_seconds")
STOP_CONDITIONS = ("any_language_accuracy_drop", "mean_weighted_gain_below_threshold",
                   "behavior_check_failed", "newly_wrong_exceeds_newly_correct")

# Pre-registered before the first diagnostic/retrain run of this protocol.
# Changing any value requires a new PROTOCOL_VERSION; results must not tune it.
DECISION_RULE = {
    "protocol_version": PROTOCOL_VERSION,
    "min_mean_weighted_raw_gain": 0.0005,
    "min_splits_with_positive_gain": 2,
    "require_no_split_weighted_drop": True,
    "require_language_non_decrease_every_split": True,
    "regressed_if_mean_weighted_gain_below": 0.0,
    "labels": {
        "improved": "all language accuracies non-decreasing on every split, no split with a weighted "
                    "drop, mean gain >= threshold and gains on >= 2 splits",
        "no_effect": "identical predictions on every split",
        "regressed": "mean weighted_raw change < 0",
        "mixed": "anything else",
        "invalid_behavior_mismatch": "post-run behavior check showed the setting did not do what "
                                     "its name says; never counted as an improvement",
    },
    "interpretation": "exploratory on reused, previously exposed splits; never an independent "
                      "generalization estimate",
}


def baseline_config() -> dict[str, str]:
    return {name: spec["baseline"] for name, spec in KNOBS.items()}


def canonical_name(knob: str, value: str) -> str:
    return f"{knob}__{re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')}"


def config_for(knob: str, value: str) -> dict[str, str]:
    config = baseline_config()
    config[knob] = value
    return config


def fingerprint(config: dict[str, str]) -> str:
    # Adding a baseline-valued knob must not make the old ledger forget trials.
    identity = dict(config)
    if identity.get("ranker_family") == "lightgbm_v11":
        identity.pop("ranker_family")
    text = json.dumps({"protocol": PROTOCOL_VERSION, "config": dict(sorted(identity.items()))},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def changed_knobs(config: dict[str, str]) -> list[str]:
    base = baseline_config()
    if set(config) != set(base):
        raise ValueError("설정 키가 허용 목록과 다릅니다")
    return [name for name in base if config[name] != base[name]]


def validate_config(config: dict[str, str]) -> list[str]:
    """Return the changed knob list; raise unless every value is allowed and <=1 knob changed."""
    if not isinstance(config, dict):
        raise ValueError("설정은 dict여야 합니다")
    for name, value in config.items():
        if name not in KNOBS or not isinstance(value, str) or value not in KNOBS[name]["values"]:
            raise ValueError(f"허용되지 않은 설정값: {name}={value!r}")
    changed = changed_knobs(config)
    if len(changed) > 1:
        raise ValueError("한 실험은 한 가지 설정만 바꿀 수 있습니다")
    return changed


def allowed_changes_for_llm() -> list[dict[str, Any]]:
    return [{"knob": name, "kind": spec["kind"], "baseline": spec["baseline"],
             "allowed_values": [v for v in spec["values"]
                                if v != spec["baseline"] and v not in OOF_ONLY_RANKERS],
             "description": spec["description"], "addresses_evidence_types": spec["addresses"],
             "retrains_ranker": spec["retrain"]}
            for name, spec in KNOBS.items()]


def classify_outcome(per_split: list[dict[str, Any]], behavior_passed: bool) -> dict[str, Any]:
    """Apply DECISION_RULE to per-split deltas (keys: delta_weighted_raw, delta_korean_accuracy,
    delta_non_korean_accuracy, newly_correct, newly_wrong)."""
    rule = DECISION_RULE
    if not per_split:
        raise ValueError("분할 결과가 없습니다")
    mean_gain = sum(item["delta_weighted_raw"] for item in per_split) / len(per_split)
    positive = sum(item["delta_weighted_raw"] > 0 for item in per_split)
    language_ok = all(item["delta_korean_accuracy"] >= 0 and item["delta_non_korean_accuracy"] >= 0
                      for item in per_split)
    no_drop = all(item["delta_weighted_raw"] >= 0 for item in per_split)
    identical = all(item["newly_correct"] == 0 and item["newly_wrong"] == 0 for item in per_split)
    if not behavior_passed:
        label = "invalid_behavior_mismatch"
    elif identical:
        label = "no_effect"
    elif (language_ok and no_drop and mean_gain >= rule["min_mean_weighted_raw_gain"]
          and positive >= rule["min_splits_with_positive_gain"]):
        label = "improved"
    elif mean_gain < rule["regressed_if_mean_weighted_gain_below"]:
        label = "regressed"
    else:
        label = "mixed"
    return {"label": label, "mean_delta_weighted_raw": mean_gain,
            "splits_with_positive_gain": positive, "splits": len(per_split),
            "language_non_decrease_every_split": language_ok, "no_split_weighted_drop": no_drop,
            "protocol_version": rule["protocol_version"]}
