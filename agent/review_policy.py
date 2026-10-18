"""Deterministic review rules for vessel-match results."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any

from .legacy_matching.matching_service import MatchResult
from .legacy_matching.record_validation import VesselRecord

from .schemas import ReviewSignals


@dataclass(frozen=True)
class ReviewPolicy:
    """Versioned rule settings; score thresholds are not probabilities."""

    version: str = "review_v1"
    tie_tolerance: float = 1e-12


def _result_value(result: MatchResult | dict[str, Any], key: str) -> Any:
    if isinstance(result, dict):
        return result.get(key)
    return getattr(result, key, None)


def _candidate_value(candidate: Any, key: str) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(key)
    return getattr(candidate, key, None)


def _record_values(record: VesselRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, dict):
        return record
    return record.__dict__


def assess_review_need(
    result: MatchResult | dict[str, Any],
    policy: ReviewPolicy | None = None,
    *,
    record: VesselRecord | dict[str, Any] | None = None,
    tool_failed: bool = False,
) -> ReviewSignals:
    """Flag no-candidate, top-tie, all-name/tonnage-missing, or tool failures."""
    policy = policy or ReviewPolicy()
    candidates = _result_value(result, "top_rank") or _result_value(result, "candidates") or []
    candidates = list(candidates)
    reasons: list[str] = []
    missing_fields: list[str] = []

    if not candidates:
        reasons.append("no_candidates")

    scores = [
        float(score)
        for candidate in candidates
        if (score := _candidate_value(candidate, "final_score")) is not None
    ]
    top_score = max(scores) if scores else None
    second_score = sorted(scores, reverse=True)[1] if len(scores) > 1 else None
    score_margin = top_score - second_score if top_score is not None and second_score is not None else None
    tied_top_count = (
        sum(isclose(score, top_score, rel_tol=0.0, abs_tol=policy.tie_tolerance) for score in scores)
        if top_score is not None
        else 0
    )
    if tied_top_count > 1:
        reasons.append("top_tie")

    if record is not None:
        values = _record_values(record)
        missing_fields = [
            field
            for field in (
                "vessel_name",
                "gross_tonnage",
                "international_gross_tonnage",
            )
            if values.get(field) is None
        ]
        name_missing = values.get("vessel_name") is None
        tonnage_missing = (
            values.get("gross_tonnage") is None
            and values.get("international_gross_tonnage") is None
        )
        if name_missing and tonnage_missing:
            reasons.append("name_and_tonnage_missing")

    if tool_failed:
        reasons.append("tool_failure")

    return ReviewSignals(
        top_score=top_score,
        score_margin=score_margin,
        tied_top_count=tied_top_count,
        missing_fields=missing_fields,
        reasons=reasons,
        review_required=bool(reasons),
    )
