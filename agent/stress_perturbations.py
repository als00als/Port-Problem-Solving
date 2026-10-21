"""Deterministic validation-input perturbations for protocol v11-stress-oof-1.

Every perturbed value is a pure function of ``(protocol_version, scenario_id, sample_id,
that row's own input columns)``. ``C2`` additionally receives the fold's training
call-sign table, which the protocol allows. Nothing here reads ``target_vessel_id``,
another validation row, or a global RNG, so a row perturbs identically alone and inside
a batch.

Two points the frozen protocol text leaves open are fixed here and published in
``SALTS`` / ``READINGS`` so a run stays reproducible:

* row-selection salt -- the protocol states ``u < rate`` but names salts only for the
  operation choices, so selection uses ``select`` (``select:<component>`` for ``M1``).
* ``N1`` Korean split position -- "insert one space after syllable index floor(len/2)"
  is applied as an insertion at character offset ``len // 2``, the reading under which
  the stated eligibility (``len >= 2``) always produces an interior split.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import numpy as np


PROTOCOL_VERSION = "v11-stress-oof-1"

INPUT_COLUMNS = ("sample_id", "vessel_name", "call_sign", "gross_tonnage",
                 "international_gross_tonnage", "flag_name", "vessel_type", "arrival_year")
FORBIDDEN_COLUMNS = ("target_vessel_id",)

SCENARIOS = ("S0_clean_replay", "C1_callsign_removed", "C2_callsign_collision_10pct",
             "T1_tonnage_both_missing", "T2_tonnage_noise_20pct", "T3_tonnage_swap",
             "F1_flag_missing", "F2_vessel_type_missing", "N1_spacing", "N2_char_edit_one",
             "N3_affix", "N4_number_format", "N5_token_order", "M1_mixed")

RATES = {"C2_callsign_collision_10pct": 0.10}
M1_COMPONENT_RATES = {"C1": 0.30, "T1": 0.20, "F1": 0.20, "name": 0.30}
M1_NAME_OPS = ("N1_spacing", "N2_char_edit_one", "N3_affix")
TONNAGE_COLUMNS = ("gross_tonnage", "international_gross_tonnage")
ALWAYS_ELIGIBLE = ("S0_clean_replay", "C1_callsign_removed", "T1_tonnage_both_missing",
                   "F1_flag_missing", "F2_vessel_type_missing", "N1_spacing", "N3_affix",
                   "M1_mixed")

SALTS = {"row_selection": "select", "m1_row_selection": "select:<component>",
         "c2_pick": "pick", "t2_sign": "sign", "n2_operation": "op", "n2_position": "pos",
         "m1_name_op": "name_op"}
READINGS = {
    "row_selection_salt": "u(sample_id, scenario_id, 'select') < rate; M1 uses 'select:<component>'",
    "n1_korean_split": "space inserted at character offset len // 2 (names of >= 2 characters)",
    "n2_position_salt": "edit position chosen with salt 'pos' among the eligible positions",
    "m1_name_op_internals": "the chosen N1/N2/N3 operation uses its own scenario_id for its salts",
    "n2_unavailable_operation": "if the drawn operation has no eligible position the row is left "
                                "unchanged and counted as unavailable; no substitute operation runs",
}


def _is_hangul_char(char: str) -> bool:
    return "가" <= char <= "힣"


def is_korean_name(value: object) -> bool:
    """Same rule as src.metric.is_korean, kept local so this module stays dependency-free."""
    return isinstance(value, str) and any(_is_hangul_char(c) for c in value)


def u(sample_id: str, scenario_id: str, salt: str) -> float:
    """Protocol hash: sha256('v11-stress-oof-1|<scenario>|<salt>|<sample_id>') -> [0, 1)."""
    text = f"{PROTOCOL_VERSION}|{scenario_id}|{salt}|{sample_id}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) / 2 ** 64


def _choice(sample_id: str, scenario_id: str, salt: str, count: int) -> int:
    if count <= 0:
        raise ValueError("선택할 항목이 없습니다")
    return min(count - 1, int(math.floor(u(sample_id, scenario_id, salt) * count)))


def _selected(sample_id: str, scenario_id: str, rate: float, salt: str = "select") -> bool:
    return u(sample_id, scenario_id, salt) < rate


def _check_row(row: dict[str, Any]) -> None:
    for column in FORBIDDEN_COLUMNS:
        if column in row:
            raise AssertionError("섭동 입력에 정답 열이 있습니다")


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


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value != "" else None


# --- name operations -------------------------------------------------------------------

def _n1(name: str, scenario_id: str, sample_id: str) -> str | None:
    if is_korean_name(name):
        if len(name) < 2:
            return None
        cut = len(name) // 2
        return name[:cut] + " " + name[cut:]
    stripped = "".join(name.split())
    return stripped if stripped != name else None


def _next_letter(char: str) -> str:
    base = "A" if char.isupper() else "a"
    return chr((ord(char) - ord(base) + 1) % 26 + ord(base))


def _n2(name: str, scenario_id: str, sample_id: str) -> tuple[str | None, str]:
    """One deterministic edit. Returns (new name or None, applied operation label)."""
    if is_korean_name(name):
        syllables = [i for i, c in enumerate(name) if _is_hangul_char(c)]
        if len(syllables) >= 4:
            cut = syllables[_choice(sample_id, scenario_id, "pos", len(syllables))]
            return name[:cut] + name[cut + 1:], "korean_delete_syllable"
        pairs = [i for i, j in zip(syllables, syllables[1:]) if j == i + 1]
        if not pairs:
            return None, "korean_transpose_unavailable"
        at = pairs[_choice(sample_id, scenario_id, "pos", len(pairs))]
        return name[:at] + name[at + 1] + name[at] + name[at + 2:], "korean_transpose_syllables"
    operation = ("delete", "substitute", "transpose")[_choice(sample_id, scenario_id, "op", 3)]
    letters = [i for i, c in enumerate(name) if c.isascii() and c.isalpha()]
    if operation == "transpose":
        pairs = [i for i, j in zip(letters, letters[1:]) if j == i + 1]
        if not pairs:
            return None, "transpose_unavailable"
        at = pairs[_choice(sample_id, scenario_id, "pos", len(pairs))]
        return name[:at] + name[at + 1] + name[at] + name[at + 2:], "transpose"
    if not letters:
        return None, f"{operation}_unavailable"
    at = letters[_choice(sample_id, scenario_id, "pos", len(letters))]
    if operation == "delete":
        return name[:at] + name[at + 1:], "delete"
    return name[:at] + _next_letter(name[at]) + name[at + 1:], "substitute"


def _n3(name: str, scenario_id: str, sample_id: str) -> str | None:
    if is_korean_name(name):
        if name.endswith("호"):
            return name[:-1] or None
        return name + "호"
    if name.upper().lstrip().startswith(("M/V", "MV")):
        return None
    return "M/V " + name


def _integer_token_positions(name: str) -> list[int]:
    return [i for i, token in enumerate(name.split()) if token.isdigit()]


def _n4(name: str, scenario_id: str, sample_id: str) -> str | None:
    if is_korean_name(name):
        return None
    tokens = name.split()
    positions = _integer_token_positions(name)
    if not positions:
        return None
    tokens[positions[-1]] = "NO." + tokens[positions[-1]]
    return " ".join(tokens)


def _n5(name: str, scenario_id: str, sample_id: str) -> str | None:
    if is_korean_name(name):
        return None
    tokens = name.split()
    if len(tokens) < 2:
        return None
    return " ".join(tokens[1:] + tokens[:1])


# --- row rule (the protocol's stated row set, before the operation runs) ----------------

def rule_eligible(scenario_id: str, row: dict[str, Any]) -> bool:
    _check_row(row)
    sample_id = str(row["sample_id"])
    name = _text(row.get("vessel_name"))
    if scenario_id in ALWAYS_ELIGIBLE:
        return True
    if scenario_id == "C2_callsign_collision_10pct":
        return _selected(sample_id, scenario_id, RATES[scenario_id])
    if scenario_id == "T2_tonnage_noise_20pct":
        return any(_positive(row.get(c)) is not None for c in TONNAGE_COLUMNS)
    if scenario_id == "T3_tonnage_swap":
        values = [_positive(row.get(c)) for c in TONNAGE_COLUMNS]
        return all(v is not None for v in values) and values[0] != values[1]
    if scenario_id == "N2_char_edit_one":
        return bool(name) and len("".join(name.split())) >= 4
    if scenario_id == "N4_number_format":
        return bool(name) and not is_korean_name(name) and bool(_integer_token_positions(name))
    if scenario_id == "N5_token_order":
        return bool(name) and not is_korean_name(name) and len(name.split()) >= 2
    raise ValueError(f"알 수 없는 시나리오: {scenario_id}")


# --- the perturbation ------------------------------------------------------------------

def perturb_row(scenario_id: str, row: dict[str, Any],
                train_callsigns: Sequence[str] | None = None) -> dict[str, Any]:
    """Return {'changes': {column: value}, 'operations': [...], 'rule_eligible': bool}."""
    if scenario_id not in SCENARIOS:
        raise ValueError(f"알 수 없는 시나리오: {scenario_id}")
    _check_row(row)
    sample_id = str(row["sample_id"])
    eligible = rule_eligible(scenario_id, row)
    out: dict[str, Any] = {"changes": {}, "operations": [], "rule_eligible": eligible}
    if not eligible or scenario_id == "S0_clean_replay":
        return out
    name = _text(row.get("vessel_name"))

    def drop(column: str) -> None:
        if not _missing(row.get(column)):
            out["changes"][column] = np.nan

    def set_name(new: str | None, label: str) -> None:
        if new is not None and new != row.get("vessel_name"):
            out["changes"]["vessel_name"] = new
            out["operations"].append(label)
        else:
            out["operations"].append(label + "_no_change")

    if scenario_id == "C1_callsign_removed":
        drop("call_sign")
        out["operations"].append("call_sign_nan")
    elif scenario_id == "C2_callsign_collision_10pct":
        if train_callsigns is None:
            raise ValueError("C2에는 fold의 학습 call_sign 표가 필요합니다")
        table = sorted(str(value) for value in train_callsigns)
        if not table:
            raise ValueError("학습 call_sign 표가 비어 있습니다")
        picked = table[_choice(sample_id, scenario_id, "pick", len(table))]
        if picked != row.get("call_sign"):
            out["changes"]["call_sign"] = picked
        out["operations"].append("call_sign_from_training_table")
    elif scenario_id == "T1_tonnage_both_missing":
        for column in TONNAGE_COLUMNS:
            drop(column)
        out["operations"].append("tonnage_both_nan")
    elif scenario_id == "T2_tonnage_noise_20pct":
        factor = 1.2 if u(sample_id, scenario_id, "sign") < 0.5 else 0.8
        for column in TONNAGE_COLUMNS:
            value = _positive(row.get(column))
            if value is not None:
                out["changes"][column] = value * factor
        out["operations"].append(f"tonnage_scaled_{factor:g}")
    elif scenario_id == "T3_tonnage_swap":
        first, second = (row.get(c) for c in TONNAGE_COLUMNS)
        out["changes"][TONNAGE_COLUMNS[0]] = float(second)
        out["changes"][TONNAGE_COLUMNS[1]] = float(first)
        out["operations"].append("tonnage_swapped")
    elif scenario_id == "F1_flag_missing":
        drop("flag_name")
        out["operations"].append("flag_name_nan")
    elif scenario_id == "F2_vessel_type_missing":
        drop("vessel_type")
        out["operations"].append("vessel_type_nan")
    elif scenario_id == "N1_spacing":
        set_name(_n1(name, scenario_id, sample_id) if name else None, "n1_spacing")
    elif scenario_id == "N2_char_edit_one":
        new, label = _n2(name, scenario_id, sample_id)
        set_name(new, "n2_" + label)
    elif scenario_id == "N3_affix":
        set_name(_n3(name, scenario_id, sample_id) if name else None, "n3_affix")
    elif scenario_id == "N4_number_format":
        set_name(_n4(name, scenario_id, sample_id), "n4_number_format")
    elif scenario_id == "N5_token_order":
        set_name(_n5(name, scenario_id, sample_id), "n5_token_order")
    elif scenario_id == "M1_mixed":
        if _selected(sample_id, scenario_id, M1_COMPONENT_RATES["C1"], "select:C1"):
            drop("call_sign")
            out["operations"].append("m1_c1")
        if _selected(sample_id, scenario_id, M1_COMPONENT_RATES["T1"], "select:T1"):
            for column in TONNAGE_COLUMNS:
                drop(column)
            out["operations"].append("m1_t1")
        if _selected(sample_id, scenario_id, M1_COMPONENT_RATES["F1"], "select:F1"):
            drop("flag_name")
            out["operations"].append("m1_f1")
        if _selected(sample_id, scenario_id, M1_COMPONENT_RATES["name"], "select:name"):
            chosen = M1_NAME_OPS[_choice(sample_id, scenario_id, "name_op", len(M1_NAME_OPS))]
            if name is None:
                out["operations"].append(f"m1_{chosen}_no_change")
            elif chosen == "N1_spacing":
                set_name(_n1(name, chosen, sample_id), f"m1_{chosen}")
            elif chosen == "N2_char_edit_one":
                if len("".join(name.split())) >= 4:
                    new, label = _n2(name, chosen, sample_id)
                    set_name(new, f"m1_{chosen}_{label}")
                else:
                    out["operations"].append(f"m1_{chosen}_rule_ineligible")
            else:
                set_name(_n3(name, chosen, sample_id), f"m1_{chosen}")
    return out


def perturb_frame(scenario_id: str, frame, train_callsigns: Sequence[str] | None = None):
    """Apply perturb_row to each row of a validation input frame -> (frame, audit, results)."""
    for column in FORBIDDEN_COLUMNS:
        if column in frame.columns:
            raise AssertionError("섭동 입력에 정답 열이 있습니다")
    records = frame.to_dict("records")
    results = [perturb_row(scenario_id, record, train_callsigns) for record in records]
    out = frame.copy()
    for column in {c for result in results for c in result["changes"]}:
        if any(isinstance(result["changes"].get(column), str) for result in results):
            out[column] = out[column].astype(object)
    column_changes: dict[str, int] = {}
    operations: dict[str, int] = {}
    changed_rows = 0
    for position, result in enumerate(results):
        for column, value in result["changes"].items():
            out.iat[position, out.columns.get_loc(column)] = value
            column_changes[column] = column_changes.get(column, 0) + 1
        changed_rows += bool(result["changes"])
        for label in result["operations"]:
            operations[label] = operations.get(label, 0) + 1
    audit = {
        "scenario_id": scenario_id, "protocol_version": PROTOCOL_VERSION,
        "rows": len(records),
        "rule_eligible_rows": int(sum(r["rule_eligible"] for r in results)),
        "changed_rows": int(changed_rows),
        "column_change_counts": dict(sorted(column_changes.items())),
        "operation_counts": dict(sorted(operations.items())),
        "not_applicable": bool(changed_rows == 0 and scenario_id != "S0_clean_replay"),
    }
    return out, audit, results
