"""Bounded LangChain tools backed by the existing MatchingService."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from langchain_core.tools import tool

from submit.model.matching_service import MatchingService
from submit.model.record_validation import validate_record


def build_agent_tools(service: MatchingService) -> list[Any]:
    """Expose matching and review-request operations, with no filesystem tools."""

    @tool
    def match_vessel(payload: dict, top_k: int = 5) -> str:
        """Match one record against the vessel catalog and return ranked evidence."""
        record = validate_record(payload)
        return json.dumps(
            asdict(service.match_vessel(record, top_k=top_k)),
            ensure_ascii=False,
            allow_nan=False,
        )

    @tool
    def expand_search(payload: dict, profile: str = "wide_v1", top_k: int = 5) -> str:
        """Search a wider Korean-name candidate set without replacing baseline."""
        record = validate_record(payload)
        return json.dumps(
            asdict(service.expand_search(record, profile=profile, top_k=top_k)),
            ensure_ascii=False,
            allow_nan=False,
        )

    @tool
    def compare_candidates(payload: dict, candidate_ids: list[str]) -> str:
        """Compare up to five existing catalog IDs for one record."""
        record = validate_record(payload)
        evidence = service.compare_candidates(record, candidate_ids)
        return json.dumps(
            [asdict(candidate) for candidate in evidence],
            ensure_ascii=False,
            allow_nan=False,
        )

    @tool
    def request_review(run_id: str, reason_codes: list[str], candidate_ids: list[str]) -> str:
        """Create a human-review request; it does not approve a vessel."""
        if len(candidate_ids) > 5:
            raise ValueError("검토 후보는 최대 5개입니다")
        unknown_ids = set(candidate_ids) - set(service.index.ids)
        if unknown_ids:
            raise ValueError(f"카탈로그에 없는 vessel_id: {sorted(unknown_ids)}")
        request = {
            "review_id": uuid4().hex,
            "run_id": run_id,
            "reason_codes": list(dict.fromkeys(reason_codes)),
            "candidate_ids": candidate_ids,
            "status": "awaiting_review",
        }
        return json.dumps(request, ensure_ascii=False)

    return [match_vessel, expand_search, compare_candidates, request_review]
