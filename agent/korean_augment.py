"""Korean training-query augmentation for protocol v11-korean-augment-1.

One perturbed copy per Korean training row, added as an extra training query inside each
inner GroupKFold split. The copy keeps its ``target_vessel_id``; resources, lookup tables
and the original queries are untouched.

Every value is a pure function of ``(protocol_version, variant token, salt, sample_id)``.
The protocol string differs from ``v11-stress-oof-1``, so training draws never coincide
with the evaluation perturbations.

``A4`` picks one of A1/A2/A3 per row and then applies that rule with the *chosen*
variant's own token in the salts, so an A4 row that draws A3 gets exactly the same copy
the standalone A3 variant would produce for it.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Iterable, Sequence

import numpy as np


PROTOCOL_VERSION = "v11-korean-augment-1"

VARIANTS = ("A0_none", "A1_tonnage_missing", "A2_tonnage_noise", "A3_name_noise", "A4_mixed")
TOKENS = {"A1_tonnage_missing": "A1", "A2_tonnage_noise": "A2",
          "A3_name_noise": "A3", "A4_mixed": "A4"}
A2_FACTORS = (0.85, 0.9, 1.1, 1.15)
A4_CHOICES = ("A1", "A2", "A3")
TONNAGE_COLUMNS = ("gross_tonnage", "international_gross_tonnage")
LABEL_COLUMN = "target_vessel_id"
COPIES_PER_KOREAN_ROW = 1


def _is_hangul_char(char: str) -> bool:
    return "가" <= char <= "힣"


def is_korean_name(value: object) -> bool:
    """Same rule as src.metric.is_korean (any Hangul syllable)."""
    return isinstance(value, str) and any(_is_hangul_char(c) for c in value)


def a(sample_id: str, variant: str, salt: str) -> float:
    """Protocol hash: sha256('v11-korean-augment-1|<variant>|<salt>|<sample_id>') -> [0, 1)."""
    text = f"{PROTOCOL_VERSION}|{variant}|{salt}|{sample_id}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) / 2 ** 64


def _choice(sample_id: str, token: str, salt: str, count: int) -> int:
    if count <= 0:
        raise ValueError("선택할 항목이 없습니다")
    return min(count - 1, int(math.floor(a(sample_id, token, salt) * count)))


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


def _a1(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    changes = {column: np.nan for column in TONNAGE_COLUMNS if not _missing(row.get(column))}
    return changes, ["a1_tonnage_both_nan" if changes else "a1_already_missing"]


def _a2(sample_id: str, row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    factor = A2_FACTORS[_choice(sample_id, "A2", "factor", len(A2_FACTORS))]
    changes = {}
    for column in TONNAGE_COLUMNS:
        value = _positive(row.get(column))
        if value is not None:
            changes[column] = value * factor
    return changes, [f"a2_tonnage_scaled_{factor:g}" if changes else "a2_no_positive_tonnage"]


def _a3(sample_id: str, row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    name = row.get("vessel_name")
    if not isinstance(name, str) or not name:
        return {}, ["a3_no_name"]
    syllables = _syllables(name)
    if len(syllables) >= 3 and a(sample_id, "A3", "op") < 0.5:
        cut = syllables[_choice(sample_id, "A3", "pos", len(syllables))]
        new = name[:cut] + name[cut + 1:]
        label = "a3_delete_syllable"
    elif len(name) >= 2:
        offset = 1 + int(math.floor(a(sample_id, "A3", "pos") * (len(name) - 1)))
        offset = min(offset, len(name) - 1)
        new = name[:offset] + " " + name[offset:]
        label = "a3_insert_space"
    else:
        return {}, ["a3_name_too_short"]
    if new == name:
        return {}, [label + "_no_change"]
    return {"vessel_name": new}, [label]


def _a4(sample_id: str, row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    chosen = A4_CHOICES[_choice(sample_id, "A4", "mix", len(A4_CHOICES))]
    # The chosen rule runs with its own token in the salts.
    if chosen == "A1":
        changes, ops = _a1(row)
    elif chosen == "A2":
        changes, ops = _a2(sample_id, row)
    else:
        changes, ops = _a3(sample_id, row)
    return changes, [f"a4_{chosen}"] + ops


def copy_for_row(variant: str, row: dict[str, Any]) -> dict[str, Any]:
    """One augmented copy of a Korean training row. Returns changes and operation labels."""
    if variant not in VARIANTS:
        raise ValueError(f"알 수 없는 변형: {variant}")
    if variant == "A0_none":
        raise ValueError("A0_none은 증강하지 않습니다")
    if LABEL_COLUMN not in row:
        raise AssertionError("증강 입력은 정답 열을 가진 학습 행이어야 합니다")
    if not is_korean_name(row.get("vessel_name")):
        raise AssertionError("한글 행만 증강합니다")
    sample_id = str(row["sample_id"])
    if variant == "A1_tonnage_missing":
        changes, ops = _a1(row)
    elif variant == "A2_tonnage_noise":
        changes, ops = _a2(sample_id, row)
    elif variant == "A3_name_noise":
        changes, ops = _a3(sample_id, row)
    else:
        changes, ops = _a4(sample_id, row)
    copy = dict(row)
    copy.update(changes)
    return {"sample_id": sample_id, "row": copy, "changes": changes, "operations": ops,
             "changed": bool(changes)}


def augment_rows(variant: str, rows: Sequence[dict[str, Any]],
                 forbidden_sample_ids: Iterable[str] = ()) -> tuple[list[dict[str, Any]], dict]:
    """Exactly one copy per Korean row of ``rows``; non-Korean rows get none."""
    forbidden = {str(value) for value in forbidden_sample_ids}
    korean = [row for row in rows if is_korean_name(row.get("vessel_name"))]
    leaked = sorted({str(row["sample_id"]) for row in rows} & forbidden)
    if leaked:
        raise AssertionError(f"증강 대상에 외부 검증 행이 있습니다: {len(leaked)}건")
    copies, operations, changed = [], {}, 0
    for row in korean:
        made = copy_for_row(variant, row)
        copies.append(made["row"])
        changed += made["changed"]
        for label in made["operations"]:
            operations[label] = operations.get(label, 0) + 1
    audit = {
        "variant": variant, "protocol_version": PROTOCOL_VERSION,
        "prep_rows": len(rows), "korean_prep_rows": len(korean),
        "copies_added": len(copies), "copies_changed": int(changed),
        "copies_per_korean_row": COPIES_PER_KOREAN_ROW,
        "operation_counts": dict(sorted(operations.items())),
        "outer_validation_ids_present": 0,
        "labels_preserved": all(copy[LABEL_COLUMN] == row[LABEL_COLUMN]
                                for copy, row in zip(copies, korean)),
    }
    if len(copies) != len(korean) * COPIES_PER_KOREAN_ROW:
        raise AssertionError("한글 행당 복사본 수가 1이 아닙니다")
    return copies, audit


def make_augmenter(variant: str, forbidden_sample_ids: Iterable[str] = ()
                   ) -> Callable[[Sequence[dict[str, Any]]], tuple[list[dict[str, Any]], dict]]:
    """Callable for retrain_runtime.training_matrix(augment=...); None means A0."""
    if variant == "A0_none":
        return None
    forbidden = tuple(str(value) for value in forbidden_sample_ids)

    def augmenter(rows: Sequence[dict[str, Any]]):
        return augment_rows(variant, rows, forbidden)

    augmenter.variant = variant  # type: ignore[attr-defined]
    return augmenter


def merge_audits(audits: Sequence[dict]) -> dict:
    """Pool the per-inner-split audits of one fold."""
    if not audits:
        return {}
    operations: dict[str, int] = {}
    for audit in audits:
        for label, count in audit["operation_counts"].items():
            operations[label] = operations.get(label, 0) + count
    return {
        "variant": audits[0]["variant"], "protocol_version": PROTOCOL_VERSION,
        "inner_splits": len(audits),
        "prep_rows": sum(x["prep_rows"] for x in audits),
        "korean_prep_rows": sum(x["korean_prep_rows"] for x in audits),
        "copies_added": sum(x["copies_added"] for x in audits),
        "copies_changed": sum(x["copies_changed"] for x in audits),
        "operation_counts": dict(sorted(operations.items())),
        "outer_validation_ids_present": sum(x["outer_validation_ids_present"] for x in audits),
        "labels_preserved": all(x["labels_preserved"] for x in audits),
    }
