"""Typed contracts shared by the offline review-agent graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

from .legacy_matching.record_validation import VesselRecord, validate_record


AgentStatus = Literal[
    "running",
    "completed",
    "needs_review",
    "awaiting_review",
    "reviewed",
    "invalid_input",
    "tool_error",
]


@dataclass(frozen=True)
class ReviewSignals:
    """Deterministic flags that explain why a result may need human review."""

    top_score: float | None
    score_margin: float | None
    tied_top_count: int
    missing_fields: list[str]
    reasons: list[str]
    review_required: bool


@dataclass(frozen=True)
class ReviewDecision:
    """A human decision attached to one review request."""

    review_id: str
    action: Literal["confirm_baseline", "select_candidate", "unresolved"]
    selected_vessel_id: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class ReviewRequest:
    """A review request; creating it does not count as human approval."""

    review_id: str
    run_id: str
    reason_codes: list[str]
    candidate_ids: list[str]


class AgentState(TypedDict, total=False):
    """LangGraph state for one input row; never includes its target label."""

    run_id: str
    session_id: str
    input_record: dict[str, Any]
    primary_result: dict[str, Any] | None
    baseline_result: dict[str, Any] | None
    legacy_result: dict[str, Any] | None
    expanded_result: dict[str, Any] | None
    review_signals: dict[str, Any] | None
    evidence: list[dict[str, Any]]
    tool_history: list[dict[str, Any]]
    tool_calls_used: int
    model_calls_used: int
    expansion_count: int
    status: AgentStatus
    review_decision: dict[str, Any] | None
    final_report: str | None
    errors: list[str]
    messages: Annotated[list[Any], add_messages]
    config: dict[str, Any]


def record_to_payload(record: VesselRecord | dict[str, Any]) -> dict[str, Any]:
    """Return a validated input payload safe to pass into the graph."""
    validated = validate_record(asdict(record) if isinstance(record, VesselRecord) else record)
    return asdict(validated)
