"""Held-out evaluation scenarios added by protocol v11-korean-augment-1.

T2b, T4, N6 and N7 are not used in training, so they measure robustness the augmentation
did not see. They reuse the frozen ``v11-stress-oof-1`` hash and protocol string unchanged
with their own scenario ids; ``agent.stress_perturbations`` is not modified, so the
existing C2/N2/N5/T1 results stay identical.

Each perturbed value is a pure function of (stress protocol string, scenario id, salt,
sample_id) and that row's own input columns. No label and no other validation row is read.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .stress_perturbations import FORBIDDEN_COLUMNS, TONNAGE_COLUMNS, is_korean_name, u

SCENARIOS = ("T2b_tonnage_noise_30pct", "T4_gross_only_missing",
             "N6_syllable_duplicate", "N7_korean_transpose")
T2B_FACTORS = (1.3, 0.7)


def _is_hangul_char(char: str) -> bool:
    return "가" <= char <= "힣"


def _missing(value: object) -> bool:
    return value is None or value == "" or (isinstance(value, float) and math.isnan(value))


def _positive(value: object) -> float | None:
    if _missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _syllables(name: str) -> list[int]:
    return [i for i, c in enumerate(name) if _is_hangul_char(c)]


def _different_adjacent_pairs(name: str) -> list[int]:
    """Adjacent Hangul positions whose two syllables differ (identical pairs are not eligible)."""
    syllables = _syllables(name)
    return [i for i, j in zip(syllables, syllables[1:])
            if j == i + 1 and name[i] != name[j]]


def _check_row(row: dict[str, Any]) -> None:
    for column in FORBIDDEN_COLUMNS:
        if column in row:
            raise AssertionError("섭동 입력에 정답 열이 있습니다")


def rule_eligible(scenario_id: str, row: dict[str, Any]) -> bool:
    _check_row(row)
    name = row.get("vessel_name")
    name = name if isinstance(name, str) and name else None
    if scenario_id == "T2b_tonnage_noise_30pct":
        return any(_positive(row.get(c)) is not None for c in TONNAGE_COLUMNS)
    if scenario_id == "T4_gross_only_missing":
        return all(_positive(row.get(c)) is not None for c in TONNAGE_COLUMNS)
    if scenario_id == "N6_syllable_duplicate":
        return bool(name) and is_korean_name(name) and len(_syllables(name)) >= 2
    if scenario_id == "N7_korean_transpose":
        return bool(name) and is_korean_name(name) and bool(_different_adjacent_pairs(name))
    raise ValueError(f"알 수 없는 시나리오: {scenario_id}")


def perturb_row(scenario_id: str, row: dict[str, Any]) -> dict[str, Any]:
    if scenario_id not in SCENARIOS:
        raise ValueError(f"알 수 없는 시나리오: {scenario_id}")
    _check_row(row)
    sample_id = str(row["sample_id"])
    eligible = rule_eligible(scenario_id, row)
    out: dict[str, Any] = {"changes": {}, "operations": [], "rule_eligible": eligible}
    if not eligible:
        return out
    name = row.get("vessel_name")
    if scenario_id == "T2b_tonnage_noise_30pct":
        factor = T2B_FACTORS[0] if u(sample_id, scenario_id, "sign") < 0.5 else T2B_FACTORS[1]
        for column in TONNAGE_COLUMNS:
            value = _positive(row.get(column))
            if value is not None:
                out["changes"][column] = value * factor
        out["operations"].append(f"t2b_tonnage_scaled_{factor:g}")
    elif scenario_id == "T4_gross_only_missing":
        out["changes"]["gross_tonnage"] = np.nan
        out["operations"].append("t4_gross_tonnage_nan")
    elif scenario_id == "N6_syllable_duplicate":
        syllables = _syllables(name)
        at = syllables[min(len(syllables) - 1,
                           int(math.floor(u(sample_id, scenario_id, "pos") * len(syllables))))]
        out["changes"]["vessel_name"] = name[:at] + name[at] + name[at:]
        out["operations"].append("n6_duplicate_syllable")
    else:
        pairs = _different_adjacent_pairs(name)
        at = pairs[min(len(pairs) - 1,
                       int(math.floor(u(sample_id, scenario_id, "pos") * len(pairs))))]
        new = name[:at] + name[at + 1] + name[at] + name[at + 2:]
        if new == name:  # cannot happen: identical pairs are excluded
            raise AssertionError("동일 음절 쌍이 선택됐습니다")
        out["changes"]["vessel_name"] = new
        out["operations"].append("n7_transpose_syllables")
    return out


def perturb_frame(scenario_id: str, frame):
    """Apply one held-out scenario to a validation input frame -> (frame, audit, results)."""
    for column in FORBIDDEN_COLUMNS:
        if column in frame.columns:
            raise AssertionError("섭동 입력에 정답 열이 있습니다")
    records = frame.to_dict("records")
    results = [perturb_row(scenario_id, record) for record in records]
    out = frame.copy()
    touched = {column for result in results for column in result["changes"]}
    for column in touched:
        if any(isinstance(result["changes"].get(column), str) for result in results):
            out[column] = out[column].astype(object)
    column_changes: dict[str, int] = {}
    operations: dict[str, int] = {}
    changed = 0
    for position, result in enumerate(results):
        for column, value in result["changes"].items():
            out.iat[position, out.columns.get_loc(column)] = value
            column_changes[column] = column_changes.get(column, 0) + 1
        changed += bool(result["changes"])
        for label in result["operations"]:
            operations[label] = operations.get(label, 0) + 1
    audit = {
        "scenario_id": scenario_id, "protocol_string_for_hash": "v11-stress-oof-1",
        "rows": len(records),
        "rule_eligible_rows": int(sum(r["rule_eligible"] for r in results)),
        "changed_rows": int(changed),
        "column_change_counts": dict(sorted(column_changes.items())),
        "operation_counts": dict(sorted(operations.items())),
        "not_applicable": bool(changed == 0),
    }
    return out, audit, results
