"""Input schema and validation for a single vessel matching request."""

from __future__ import annotations

import math
from dataclasses import dataclass
from uuid import uuid4

import pandas as pd


@dataclass
class VesselRecord:
    sample_id: str
    vessel_name: str | None
    call_sign: str | None
    gross_tonnage: float | None
    international_gross_tonnage: float | None
    flag_name: str | None
    vessel_type: str | None
    arrival_year: int | None


def validate_record(payload: dict) -> VesselRecord:
    """Validate one request dictionary and convert it to a VesselRecord."""
    if not isinstance(payload, dict):
        raise TypeError("payload는 딕셔너리여야 합니다")

    if "target_vessel_id" in payload:
        raise ValueError("target_vessel_id는 정답이므로 입력에 포함할 수 없습니다.")

    allowed_fields = {
        "sample_id",
        "vessel_name",
        "call_sign",
        "gross_tonnage",
        "international_gross_tonnage",
        "flag_name",
        "vessel_type",
        "arrival_year",
    }
    unknown_fields = set(payload) - allowed_fields
    if unknown_fields:
        raise ValueError(f"허용되지 않은 컬럼 존재: {unknown_fields}")

    data = {}
    for field in allowed_fields:
        value = payload.get(field)
        if (
            value is None
            or (isinstance(value, str) and value.strip() == "")
            or pd.isna(value)
        ):
            value = None
        data[field] = value

    if data["sample_id"] is None:
        data["sample_id"] = uuid4().hex

    text_fields = ("sample_id", "vessel_name", "call_sign", "flag_name", "vessel_type")
    for field in text_fields:
        value = data[field]
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{field}는 문자열이어야 합니다")

    numeric_fields = ("gross_tonnage", "international_gross_tonnage", "arrival_year")
    for field in numeric_fields:
        value = data[field]
        if value is None:
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            raise TypeError(f"{field}는 숫자여야 합니다") from None

        if not math.isfinite(number):
            raise ValueError(f"{field}는 유한한 숫자여야 합니다.")

        if field == "arrival_year":
            if not number.is_integer():
                raise ValueError("arrival_year는 정수여야 합니다")
            data[field] = int(number)
        else:
            data[field] = number

    return VesselRecord(**data)
