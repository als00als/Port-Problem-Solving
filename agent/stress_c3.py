"""C3 plausible call-sign collision for protocol v11-override-gate-1.

C3 needs the row's own clean model ranking, so it cannot run in the single pass that
``agent.stress_perturbations`` uses. It lives here as a second pass that reads the clean
prediction of that same row. ``agent.stress_perturbations`` is left untouched so the
C2/N2/N5/T1 results of run ``agent_v11_stress_oof_20261019_134700_8b55cd`` stay identical.

Row selection reuses the frozen v11-stress-oof-1 hash exactly: the protocol string stays
``v11-stress-oof-1`` and the selection salt stays ``select``.

Independence: the choice uses only the row's own clean model output and the fold's
training call-sign table. Training and validation targets are group-disjoint, so the
injected vessel is never that row's truth. No label and no other validation row is read.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .stress_perturbations import FORBIDDEN_COLUMNS, u


SCENARIO_ID = "C3_plausible_callsign_collision_10pct"
RATE = 0.10
SELECTION_SALT = "select"
RANK_LOW, RANK_HIGH = 2, 5


def callsigns_by_vessel(callsign_table: Mapping[str, Any]) -> dict[str, list[str]]:
    """Invert the fold's training call-sign table: vessel_id -> sorted call signs."""
    out: dict[str, list[str]] = {}
    for call_sign, vessel_id in callsign_table.items():
        out.setdefault(str(vessel_id), []).append(str(call_sign))
    return {vessel: sorted(signs) for vessel, signs in out.items()}


def selected(sample_id: str) -> bool:
    return u(str(sample_id), SCENARIO_ID, SELECTION_SALT) < RATE


def model_order(candidates: np.ndarray, scores: Sequence[float]) -> np.ndarray:
    """v11 argmax order: score descending, catalog index ascending."""
    return np.lexsort((np.asarray(candidates), -np.asarray(scores, dtype=np.float64)))


def plan_row(clean_detail: Mapping[str, Any], by_vessel: Mapping[str, list[str]],
             catalog_ids: Sequence[str]) -> dict[str, Any]:
    """Pick the call sign to inject for one row, from its clean ranking only."""
    sample_id = str(clean_detail["sample_id"])
    out: dict[str, Any] = {"sample_id": sample_id, "selected": selected(sample_id),
                           "eligible": False, "changes": {}, "injected_rank": None,
                           "injected_vessel_id": None}
    if not out["selected"]:
        return out
    candidates = np.asarray(clean_detail["candidates"])
    order = model_order(candidates, clean_detail["scores"])
    # Ranks are 1-based; only ranks 2-5 are eligible, best rank first (ties already
    # resolved by catalog index inside model_order).
    for rank in range(RANK_LOW, min(RANK_HIGH, len(order)) + 1):
        vessel_id = str(catalog_ids[int(candidates[int(order[rank - 1])])])
        signs = by_vessel.get(vessel_id)
        if not signs:
            continue
        out.update({"eligible": True, "injected_rank": rank, "injected_vessel_id": vessel_id})
        picked = min(signs)  # lexicographically smallest, whatever order the table came in
        if picked != clean_detail.get("call_sign"):
            out["changes"] = {"call_sign": picked}
        return out
    return out


def perturb_frame(frame, clean_details: Sequence[Mapping[str, Any]],
                  callsign_table: Mapping[str, Any], catalog_ids: Sequence[str]):
    """Apply C3 to a validation input frame -> (frame, audit, plans)."""
    for column in FORBIDDEN_COLUMNS:
        if column in frame.columns:
            raise AssertionError("섭동 입력에 정답 열이 있습니다")
    if len(frame) != len(clean_details):
        raise ValueError("깨끗한 예측 행 수가 입력 행 수와 다릅니다")
    ids = [str(value) for value in frame["sample_id"]]
    if ids != [str(detail["sample_id"]) for detail in clean_details]:
        raise ValueError("깨끗한 예측의 sample_id 순서가 입력과 다릅니다")
    by_vessel = callsigns_by_vessel(callsign_table)
    current = list(frame["call_sign"])
    plans = [plan_row({**detail, "call_sign": current[position]}, by_vessel, catalog_ids)
             for position, detail in enumerate(clean_details)]
    out = frame.copy()
    out["call_sign"] = out["call_sign"].astype(object)
    column = out.columns.get_loc("call_sign")
    ranks: dict[str, int] = {}
    changed = 0
    for position, plan in enumerate(plans):
        if plan["changes"]:
            out.iat[position, column] = plan["changes"]["call_sign"]
            changed += 1
        if plan["injected_rank"] is not None:
            key = f"rank_{plan['injected_rank']}"
            ranks[key] = ranks.get(key, 0) + 1
    audit = {
        "scenario_id": SCENARIO_ID, "protocol_string_for_hash": "v11-stress-oof-1",
        "selection_salt": SELECTION_SALT, "rate": RATE,
        "rows": len(plans),
        "selected_rows": int(sum(p["selected"] for p in plans)),
        "rule_eligible_rows": int(sum(p["eligible"] for p in plans)),
        "changed_rows": int(changed),
        "column_change_counts": {"call_sign": int(changed)} if changed else {},
        "injected_rank_counts": dict(sorted(ranks.items())),
        "not_applicable": bool(sum(p["eligible"] for p in plans) == 0),
    }
    return out, audit, plans
