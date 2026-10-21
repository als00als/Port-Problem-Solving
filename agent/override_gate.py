"""Fixed override policies for protocol v11-override-gate-1.

A policy decides, per row, whether a lookup result may replace the model top-1. The
call-sign lookup has priority over the name lookup, as in v11, and each source is gated
independently with the same rule. A blocked lookup leaves the model top-1, or the other
source's approved result.

Every function here is a pure function of one row's own candidates, scores, feature rows
and lookup ids. Nothing reads a label, another row, or a global RNG.

``gate_decision`` and ``apply_policy`` take the catalog index object (anything exposing
``id_to_index``) because the looked-up vessel's rank and feature row can only be found
through it; the protocol's own wording ("read every quantity from that candidate's
existing v11 feature row") requires that lookup.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


PROTOCOL_VERSION = "v11-override-gate-1"

# Fixed by the protocol (policies.P3_rank_and_attributes.fixed_constants); never tuned.
TOP_K = 3
MAX_RELATIVE_TONNAGE_GAP = 0.25
TONNAGE_MISSING_SENTINEL = 2.0

POLICIES = ("P0_always", "P1_off", "P2_candidate_member", "P3_rank_and_attributes")
# decision_rule.ranking tie-break: fewer conditions first.
TIE_ORDER = {"P1_off": 0, "P2_candidate_member": 1, "P3_rank_and_attributes": 2, "P0_always": 3}
SOURCES = ("callsign", "name_lookup")
GATE_FEATURES = ("gap_best", "flag_prob", "flag_seen", "type_prob", "type_seen")
APPLY_REASONS = ("applied_unconditionally", "candidate_member", "rank_and_attributes_ok")
BLOCK_REASONS = ("policy_off", "not_in_candidates", "rank_above_top_k",
                 "tonnage_contradiction", "flag_contradiction", "type_contradiction")


def feature_columns(feature_names: Sequence[str]) -> dict[str, int]:
    """Column index of each gate feature inside runtime.FEATURES."""
    missing = [name for name in GATE_FEATURES if name not in feature_names]
    if missing:
        raise ValueError(f"v11 특징에 게이트 컬럼이 없습니다: {missing}")
    return {name: list(feature_names).index(name) for name in GATE_FEATURES}


def check_constants(protocol: dict) -> dict[str, Any]:
    """Fail unless the module constants equal the frozen protocol values."""
    fixed = protocol["policies"]["P3_rank_and_attributes"]["fixed_constants"]
    if int(fixed["top_k"]) != TOP_K:
        raise ValueError(f"top_k 불일치: 프로토콜 {fixed['top_k']}, 코드 {TOP_K}")
    if float(fixed["max_relative_tonnage_gap"]) != MAX_RELATIVE_TONNAGE_GAP:
        raise ValueError("max_relative_tonnage_gap 불일치: "
                         f"프로토콜 {fixed['max_relative_tonnage_gap']}, 코드 {MAX_RELATIVE_TONNAGE_GAP}")
    declared = set(protocol["policies"]) - {"common", "model_score_margin"}
    if declared != set(POLICIES):
        raise ValueError(f"정책 목록 불일치: 프로토콜 {sorted(declared)}, 코드 {sorted(POLICIES)}")
    return {"top_k": TOP_K, "max_relative_tonnage_gap": MAX_RELATIVE_TONNAGE_GAP,
            "policies": list(POLICIES)}


def model_order(candidates: np.ndarray, scores: Sequence[float]) -> np.ndarray:
    """Candidate positions in v11 argmax order: score descending, catalog index ascending."""
    return np.lexsort((np.asarray(candidates), -np.asarray(scores, dtype=np.float64)))


def candidate_rank(candidates: np.ndarray, scores: Sequence[float], catalog_position: int
                   ) -> int | None:
    """1-based rank of a catalog position in the row's candidate list, or None if absent."""
    candidates = np.asarray(candidates)
    hits = np.flatnonzero(candidates == catalog_position)
    if not len(hits):
        return None
    order = model_order(candidates, scores)
    row = int(hits[0])
    return int(np.flatnonzero(order == row)[0]) + 1


def _contradictions(feature_row: np.ndarray, columns: dict[str, int]) -> list[str]:
    """Protocol P3 conditions (b), (c), (d) read from the candidate's own feature row."""
    reasons = []
    gap_best = float(feature_row[columns["gap_best"]])
    # gap_best == 2.0 means a missing side and counts as no evidence.
    if gap_best < TONNAGE_MISSING_SENTINEL and gap_best > MAX_RELATIVE_TONNAGE_GAP:
        reasons.append("tonnage_contradiction")
    if (float(feature_row[columns["flag_seen"]]) == 1.0
            and float(feature_row[columns["flag_prob"]]) == 0.0):
        reasons.append("flag_contradiction")
    if (float(feature_row[columns["type_seen"]]) == 1.0
            and float(feature_row[columns["type_prob"]]) == 0.0):
        reasons.append("type_contradiction")
    return reasons


def gate_decision(policy: str, candidates: np.ndarray, scores: Sequence[float],
                  features: np.ndarray, lookup_id: str | None, catalog_index: Any,
                  columns: dict[str, int] | None = None) -> tuple[bool, str]:
    """Whether this policy lets ``lookup_id`` replace the model top-1, and why."""
    if policy not in POLICIES:
        raise ValueError(f"알 수 없는 정책: {policy}")
    if lookup_id is None:
        return False, "no_lookup"
    if policy == "P1_off":
        return False, "policy_off"
    if policy == "P0_always":
        return True, "applied_unconditionally"
    position = catalog_index.id_to_index.get(str(lookup_id))
    candidates = np.asarray(candidates)
    if position is None:
        return False, "not_in_candidates"
    hits = np.flatnonzero(candidates == position)
    if not len(hits):
        return False, "not_in_candidates"
    if policy == "P2_candidate_member":
        return True, "candidate_member"
    rank = candidate_rank(candidates, scores, position)
    if rank is None or rank > TOP_K:
        return False, "rank_above_top_k"
    if columns is None:
        raise ValueError("P3는 게이트 특징 컬럼 인덱스가 필요합니다 (feature_columns)")
    reasons = _contradictions(np.asarray(features)[int(hits[0])], columns)
    if reasons:
        return False, reasons[0]
    return True, "rank_and_attributes_ok"


def apply_policy(details_row: dict[str, Any], policy: str, catalog_index: Any,
                 columns: dict[str, int]) -> dict[str, Any]:
    """Final prediction and source under one policy, gating each lookup source separately."""
    candidates = details_row["candidates"]
    scores = details_row["scores"]
    features = details_row["features"]
    lookups = {"name_lookup": details_row.get("name_lookup_id"),
               "callsign": (None if details_row.get("callsign_id") is None
                            else str(details_row["callsign_id"]))}
    decisions = {}
    for source in SOURCES:
        applied, reason = gate_decision(policy, candidates, scores, features,
                                        lookups[source], catalog_index, columns)
        decisions[source] = {"lookup_id": None if lookups[source] is None else str(lookups[source]),
                             "available": lookups[source] is not None,
                             "applied": applied, "reason": reason}
    final, source = str(details_row["model_pred_id"]), "none"
    # v11 order: the name lookup may be replaced by an approved call-sign lookup.
    if decisions["name_lookup"]["applied"]:
        final, source = str(lookups["name_lookup"]), "name_lookup"
    if decisions["callsign"]["applied"]:
        final, source = str(lookups["callsign"]), "callsign"
    return {"sample_id": details_row["sample_id"], "final_pred_id": final,
            "override_source": source, "model_pred_id": str(details_row["model_pred_id"]),
            "decisions": decisions}


def decide_rows(details: list[dict[str, Any]], policy: str, catalog_index: Any,
                columns: dict[str, int]) -> list[dict[str, Any]]:
    return [apply_policy(row, policy, catalog_index, columns) for row in details]


def rank_policies(nets: dict[str, float], eligible: dict[str, bool]) -> list[str]:
    """decision_rule.ranking: net descending, ties by fewer conditions first."""
    names = [name for name in POLICIES if eligible.get(name)]
    return sorted(names, key=lambda name: (-nets[name], TIE_ORDER[name]))


def recommended(ranked: Sequence[str]) -> list[str]:
    """The top policy, plus P1 when it is not already top."""
    if not ranked:
        return []
    out = [ranked[0]]
    if "P1_off" not in out and "P1_off" in ranked:
        out.append("P1_off")
    return out
