"""Post-prediction error diagnostics for v11 validation splits.

Labels enter only here, after predictions are final. Per-row cases (with
sample IDs and vessel IDs) are written to local, git-ignored result folders;
``summarize_split``/``cross_split`` reduce them to counts that
``assert_llm_safe`` checks before anything may be sent to an LLM.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .retrain_config import ERROR_TYPES, NEAR_MISS_FLAGS


# Fixed before the first run of this protocol; results must not tune these.
DIAGNOSTIC_PROTOCOL = {
    "version": "diag-1",
    "error_cases": "every validation row whose final prediction differs from the label",
    "near_miss_cases": "correct rows sorted by model top1-top2 score margin ascending, then sample_id "
                       "ascending; the first near_miss_max_cases per split",
    "near_miss_max_cases": 50,
    "small_margin_threshold": 1.0,
    "candidate_order": "model score descending, catalog index ascending (same as v11 argmax)",
    "primary_type_priority": list(ERROR_TYPES),
    "margin_scope": "raw model scores before call-sign/name overrides; do not compare absolute margins across model families",
    "labels_used": "only after predictions are complete, in this evaluation step",
}

SNAPSHOT = ("s_compact", "s_skel_rom", "s_skel_trans", "s_token_set", "s_var_ratio", "s_learn_ratio",
            "s_alpha_ratio", "exact", "num_equal", "num_conflict", "num_q_only", "num_c_only",
            "gap_best", "c_gt_missing", "flag_prob", "type_prob")
QUERY_LEVEL = ("is_hangul", "dict_cov", "whole_hit", "q_has_num", "q_gt_missing")
NAME_SIMS = ("s_compact", "s_skel_rom", "s_var_ratio", "s_learn_ratio", "s_alpha_ratio")
INPUT_FIELDS = ("vessel_name", "call_sign", "gross_tonnage", "international_gross_tonnage",
                "flag_name", "vessel_type")


def paired_outcomes(baseline: list[dict], alternative: list[dict]) -> dict:
    """Post-prediction correctness contingency; oracle uses labels, never inference."""
    if len(baseline) != len(alternative) or any(
            any(left[key] != right[key] for key in ("sample_id", "truth_id", "is_korean", "seen_target"))
            for left, right in zip(baseline, alternative)):
        raise ValueError("paired outcomes require identical validation rows and labels")
    output = {"interpretation": "oracle is label-dependent diagnostic headroom, not deployable prediction accuracy"}
    for scope, correct_key, prediction_key in (("final", "final_correct", "final_pred_id"),
                                               ("before_override", "model_correct", "model_pred_id")):
        tables = {}
        for group in ("all", "korean", "non_korean", "seen_target", "unseen_target"):
            pairs = [(left, right) for left, right in zip(baseline, alternative)
                     if group == "all" or (group == "korean" and left["is_korean"])
                     or (group == "non_korean" and not left["is_korean"])
                     or (group == "seen_target" and left["seen_target"])
                     or (group == "unseen_target" and not left["seen_target"])]
            table = {"rows": len(pairs)}
            for name, left_value, right_value in (("both_correct", True, True),
                                                  ("baseline_only_correct", True, False),
                                                  ("alternative_only_correct", False, True),
                                                  ("both_wrong", False, False)):
                table[name] = sum(left[correct_key] == left_value and right[correct_key] == right_value
                                  for left, right in pairs)
            table["prediction_disagreements"] = sum(left[prediction_key] != right[prediction_key]
                                                     for left, right in pairs)
            table["oracle_accuracy"] = ((len(pairs) - table["both_wrong"]) / len(pairs) if pairs else 0.0)
            tables[group] = table
        tables["oracle_weighted_raw"] = (0.7 * tables["korean"]["oracle_accuracy"]
                                         + 0.3 * tables["non_korean"]["oracle_accuracy"])
        output[scope] = tables
    return output


def _value(features: np.ndarray, names: dict[str, int], row: int, key: str) -> float:
    value = float(features[row, names[key]])
    if key.startswith("gap_") and not np.isfinite(value):
        return 2.0  # NaN-encoded missing tonnage means the same as v11's 2.0
    return value


def _snapshot(features, names, row) -> dict[str, float] | None:
    if row is None:
        return None
    return {key: round(_value(features, names, row, key), 6) for key in SNAPSHOT}


def _number_mismatch(snap: dict[str, float]) -> bool:
    return bool(snap["num_conflict"] or snap["num_q_only"] or snap["num_c_only"])


def ranking(candidates: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Candidate positions in model order (score desc, catalog index asc)."""
    return np.lexsort((candidates, -np.asarray(scores, dtype=np.float64)))


def diagnose_rows(details: list[dict[str, Any]], truth_ids: list[str], seen: np.ndarray,
                  korean: np.ndarray, feature_names: tuple[str, ...], catalog_ids: np.ndarray,
                  catalog_compact: list[str], id_to_index: dict[str, int]) -> list[dict[str, Any]]:
    """One local case record per validation row. ``details`` never contain labels."""
    names = {name: i for i, name in enumerate(feature_names)}
    cases = []
    for detail, truth_id, is_seen, is_korean in zip(details, truth_ids, seen, korean):
        candidates = detail["candidates"]
        scores = np.asarray(detail["scores"], dtype=np.float64)
        features = detail["features"]
        order = ranking(candidates, scores)
        position = {int(candidates[j]): rank for rank, j in enumerate(order)}
        truth_index = id_to_index[truth_id]
        top1, top2 = int(order[0]), (int(order[1]) if len(order) > 1 else None)
        truth_row = int(np.flatnonzero(candidates == truth_index)[0]) if truth_index in position else None
        model_pred = str(catalog_ids[candidates[top1]])
        final_pred = detail["final_pred_id"]
        snap_top1 = _snapshot(features, names, top1)
        snap_truth = _snapshot(features, names, truth_row)
        snap_top2 = _snapshot(features, names, top2)
        query = {key: round(_value(features, names, top1, key), 6) for key in QUERY_LEVEL}
        margin = float(scores[top1] - scores[top2]) if top2 is not None else None
        case = {
            "sample_id": detail["sample_id"],
            "truth_id": truth_id,
            "model_pred_id": model_pred,
            "final_pred_id": final_pred,
            "is_korean": bool(is_korean),
            "seen_target": bool(is_seen),
            "model_correct": model_pred == truth_id,
            "final_correct": final_pred == truth_id,
            "override_source": detail["override_source"],
            "override_changed": final_pred != model_pred,
            "name_lookup_available": detail["name_lookup_id"] is not None,
            "callsign_available": detail["callsign_id"] is not None,
            "candidate_count": int(len(candidates)),
            "truth_in_candidates": truth_row is not None,
            "truth_rank": position[truth_index] + 1 if truth_row is not None else None,
            "top1_score": float(scores[top1]),
            "top2_score": float(scores[top2]) if top2 is not None else None,
            "margin": margin,
            "truth_score": float(scores[truth_row]) if truth_row is not None else None,
            "score_gap_top1_minus_truth": (float(scores[top1] - scores[truth_row])
                                           if truth_row is not None else None),
            "query": query,
            "missing_inputs": [key for key in INPUT_FIELDS if detail["missing"].get(key)],
            "top1": snap_top1, "top2": snap_top2, "truth": snap_truth,
            "truth_same_name_as_model_top1": (truth_row is not None
                                              and catalog_compact[truth_index]
                                              == catalog_compact[int(candidates[top1])]
                                              and model_pred != truth_id),
        }
        case["error_flags"] = _error_flags(case) if not case["final_correct"] else []
        case["primary_error_type"] = case["error_flags"][0] if case["error_flags"] else None
        case["near_miss_flags"] = (_near_miss_flags(case, catalog_compact, candidates, top1, top2)
                                   if case["final_correct"] else [])
        cases.append(case)
    return cases


def _error_flags(case: dict[str, Any]) -> list[str]:
    """All matching error types, in DIAGNOSTIC_PROTOCOL priority order (first = primary)."""
    flags = []
    overridden = case["override_source"] != "none"
    if overridden and case["model_correct"]:
        flags.append("override_introduced_error")
    if overridden and not case["model_correct"]:
        flags.append("override_wrong_and_model_wrong")
    if not case["truth_in_candidates"]:
        flags.append("candidate_missing")
        return flags
    truth, top1 = case["truth"], case["top1"]
    if case["truth_same_name_as_model_top1"]:
        flags.append("same_name_collision")
    if _number_mismatch(truth) and not _number_mismatch(top1):
        flags.append("number_mismatch")
    if case["query"]["q_gt_missing"]:
        flags.append("query_tonnage_missing")
    if not case["model_correct"]:
        if truth["gap_best"] < top1["gap_best"]:
            flags.append("tonnage_favored_truth_lost")
        if max(truth[k] for k in NAME_SIMS) < max(top1[k] for k in NAME_SIMS):
            flags.append("name_similarity_favored_wrong")
    if not flags or flags == ["override_wrong_and_model_wrong"]:
        flags.append("other_rank_failure")
    return flags


def _near_miss_flags(case, compact, candidates, top1, top2) -> list[str]:
    flags = []
    if top2 is not None:
        if compact[int(candidates[top1])] == compact[int(candidates[top2])]:
            flags.append("near_miss_runner_up_same_name")
        if case["top1"]["num_equal"] != case["top2"]["num_equal"] or _number_mismatch(case["top2"]):
            flags.append("near_miss_runner_up_number_mismatch")
        if case["top2"]["gap_best"] < case["top1"]["gap_best"]:
            flags.append("near_miss_runner_up_closer_tonnage")
    if case["query"]["q_gt_missing"]:
        flags.append("near_miss_query_tonnage_missing")
    if case["override_source"] != "none":
        flags.append("near_miss_override_applied")
    return flags


def select_near_misses(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool = [c for c in cases if c["final_correct"] and c["margin"] is not None]
    pool.sort(key=lambda c: (c["margin"], c["sample_id"]))
    return pool[:DIAGNOSTIC_PROTOCOL["near_miss_max_cases"]]


def _counts(items, keys) -> dict[str, int]:
    return {key: int(sum(key in item for item in items)) for key in keys}


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "absent"
    return "1" if rank == 1 else "2" if rank == 2 else "3_5" if rank <= 5 else "6_plus"


def summarize_split(split: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts and rates only: no sample IDs, vessel IDs, names, or label values."""
    errors = [c for c in cases if not c["final_correct"]]
    near = select_near_misses(cases)
    rows = len(cases)
    korean = [c for c in cases if c["is_korean"]]
    margins = np.array([c["margin"] for c in cases if c["final_correct"] and c["margin"] is not None])
    summary = {
        "split": split,
        "validation_rows": rows,
        "korean_rows": len(korean),
        "non_korean_rows": rows - len(korean),
        "seen_target_rows": int(sum(c["seen_target"] for c in cases)),
        "final_errors": len(errors),
        "model_errors_before_override": int(sum(not c["model_correct"] for c in cases)),
        "candidate_recall": float(np.mean([c["truth_in_candidates"] for c in cases])) if rows else 0.0,
        "error_primary_counts": {k: int(sum(c["primary_error_type"] == k for c in errors))
                                 for k in ERROR_TYPES},
        "error_flag_counts": _counts([c["error_flags"] for c in errors], ERROR_TYPES),
        "errors_by_language": {
            "korean": int(sum(c["is_korean"] for c in errors)),
            "non_korean": int(sum(not c["is_korean"] for c in errors)),
        },
        "errors_by_target_exposure": {
            "seen": int(sum(c["seen_target"] for c in errors)),
            "unseen": int(sum(not c["seen_target"] for c in errors)),
        },
        "truth_rank_among_errors": {b: int(sum(_rank_bucket(c["truth_rank"]) == b for c in errors))
                                    for b in ("1", "2", "3_5", "6_plus", "absent")},
        "near_miss": {
            "selected_cases": len(near),
            "correct_rows_with_margin_below_threshold": int((margins < DIAGNOSTIC_PROTOCOL[
                "small_margin_threshold"]).sum()) if margins.size else 0,
            "flag_counts": _counts([c["near_miss_flags"] for c in near], NEAR_MISS_FLAGS),
            "korean_cases": int(sum(c["is_korean"] for c in near)),
            "margin_quantiles_p10_p50_p90_all_correct": (
                [round(float(np.quantile(margins, q)), 4) for q in (0.1, 0.5, 0.9)]
                if margins.size else []),
        },
        "override": {
            "applied_rows": int(sum(c["override_source"] != "none" for c in cases)),
            "changed_rows": int(sum(c["override_changed"] for c in cases)),
            "introduced_errors": int(sum(c["model_correct"] and not c["final_correct"] for c in cases)),
            "fixed_model_errors": int(sum(not c["model_correct"] and c["final_correct"] for c in cases)),
            "by_source": {s: int(sum(c["override_source"] == s for c in cases))
                          for s in ("none", "name_lookup", "callsign")},
        },
        "missing_input_rows": {key: int(sum(key in c["missing_inputs"] for c in cases))
                               for key in ("gross_tonnage", "international_gross_tonnage",
                                           "call_sign", "flag_name", "vessel_type")},
    }
    return summary


def cross_split(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate problems present in >=2 reused splits from split-specific ones."""
    def pattern(getter, keys):
        out = {}
        for key in keys:
            counts = [getter(s).get(key, 0) for s in summaries]
            present = sum(c > 0 for c in counts)
            out[key] = {"total": int(sum(counts)), "splits_present": int(present),
                        "per_split": counts,
                        "pattern": "absent" if present == 0 else
                                   "recurring" if present >= 2 else "split_specific"}
        return out
    return {
        "splits": [s["split"] for s in summaries],
        "error_flags": pattern(lambda s: s["error_flag_counts"], ERROR_TYPES),
        "near_miss_flags": pattern(lambda s: s["near_miss"]["flag_counts"], NEAR_MISS_FLAGS),
        "note": "splits are previously exposed and overlap in rows; recurring means repeated across "
                "reused splits, not independently replicated",
    }


_ID_PATTERN = re.compile(r"\b(TRAIN|TEST|VESSEL)_\d|\bCS_[0-9A-Za-z]")  # official ID formats, case-sensitive
_HANGUL = re.compile(r"[ᄀ-ᇿ㄰-㆏가-힣]")


def assert_llm_safe(value: Any, path: str = "$") -> None:
    """Reject identifiers, Hangul text, or non-JSON scalars anywhere in an LLM payload."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert_llm_safe(key, f"{path}.<key>")
            assert_llm_safe(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            assert_llm_safe(item, f"{path}[{i}]")
    elif isinstance(value, str):
        if _ID_PATTERN.search(value) or _HANGUL.search(value):
            raise ValueError(f"LLM 입력에 식별 가능한 값이 있습니다: {path}")
        if any(ord(ch) > 126 or (ord(ch) < 32 and ch not in "\n") for ch in value):
            raise ValueError(f"LLM 입력에 허용되지 않은 문자가 있습니다: {path}")
    elif isinstance(value, bool) or value is None or isinstance(value, int):
        return
    elif isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"LLM 입력에 유한하지 않은 숫자가 있습니다: {path}")
    else:
        raise ValueError(f"LLM 입력에 허용되지 않은 형식이 있습니다: {path}")
