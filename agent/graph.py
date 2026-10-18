"""LangGraph orchestration for one offline vessel-review run."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from submit.model.matching_service import MatchingService, MatchResult
from submit.model.record_validation import validate_record

from .review_policy import ReviewPolicy, assess_review_need
from .schemas import AgentState
from .tools import build_agent_tools


DEFAULT_MAX_TOOL_CALLS = 6
DEFAULT_MAX_MODEL_CALLS = 4
DEFAULT_MAX_EXPANSIONS = 1


def _limit(config: dict[str, Any], name: str, default: int) -> int:
    value = config.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name}는 0 이상의 정수여야 합니다")
    return value


def _policy(config: dict[str, Any]) -> ReviewPolicy:
    raw = config.get("review_policy")
    if raw is None:
        return ReviewPolicy()
    if isinstance(raw, ReviewPolicy):
        return raw
    if isinstance(raw, dict):
        return ReviewPolicy(**raw)
    raise TypeError("review_policy는 ReviewPolicy 또는 dict여야 합니다")


def _format_report(state: AgentState) -> str:
    result = state.get("baseline_result") or {}
    signals = state.get("review_signals") or {}
    prediction = result.get("baseline_prediction")
    candidates = result.get("top_rank") or []
    if prediction is None:
        summary = "기본 매칭 결과를 만들지 못했습니다."
    else:
        summary = f"기본 추천: {prediction}"
    if candidates:
        first = candidates[0]
        summary += (
            f" ({first.get('catalog_name', '이름 미상')}, "
            f"점수 {first.get('final_score', 0.0):.6f})"
        )
    if signals.get("review_required"):
        reasons = ", ".join(signals.get("reasons", [])) or "unspecified"
        summary += f". 사람 검토 필요: {reasons}"
    elif state.get("status") == "awaiting_review":
        summary += ". 사람 검토 응답을 기다리는 중입니다."
    return summary


def build_agent_graph(
    service: MatchingService,
    model: Any = None,
    checkpointer: Any = None,
    config: dict[str, Any] | None = None,
):
    """Build the graph; without a model, run deterministic matching and review."""
    options = dict(config or {})
    max_tool_calls = _limit(options, "max_tool_calls", DEFAULT_MAX_TOOL_CALLS)
    max_model_calls = _limit(options, "max_model_calls", DEFAULT_MAX_MODEL_CALLS)
    max_expansions = _limit(options, "max_expansions", DEFAULT_MAX_EXPANSIONS)
    review_policy = _policy(options)
    tools = build_agent_tools(service)

    builder = StateGraph(AgentState)

    def validate_input(state: AgentState) -> dict[str, Any]:
        try:
            record = validate_record(state.get("input_record", {}))
        except (TypeError, ValueError) as exc:
            return {
                "status": "invalid_input",
                "errors": [str(exc)],
                "input_record": {},
            }
        return {"input_record": asdict(record), "status": "running", "errors": []}

    def route_after_validation(state: AgentState) -> str:
        return "finish" if state.get("status") == "invalid_input" else "match"

    def baseline_match(state: AgentState) -> dict[str, Any]:
        try:
            record = validate_record(state["input_record"])
            result = service.match_vessel(record)
            result_dict = asdict(result)
            signals = assess_review_need(
                result,
                review_policy,
                record=record,
            )
            status = "needs_review" if signals.review_required else "completed"
            return {
                "baseline_result": result_dict,
                "review_signals": asdict(signals),
                "evidence": result_dict.get("top_rank", []),
                "status": status,
            }
        except Exception as exc:
            return {
                "status": "tool_error",
                "errors": [f"기본 매칭 실패: {type(exc).__name__}: {exc}"],
                "review_signals": asdict(
                    assess_review_need(
                        MatchResult(baseline_prediction=None, top_rank=[]),
                        review_policy,
                        record=state.get("input_record", {}),
                        tool_failed=True,
                    )
                ),
            }

    def route_after_match(state: AgentState) -> str:
        if model is None or state.get("status") == "tool_error":
            return "finish"
        return "agent"

    def agent_step(state: AgentState) -> dict[str, Any]:
        model_calls = state.get("model_calls_used", 0)
        if model_calls >= max_model_calls:
            return {
                "errors": [*state.get("errors", []), "model_call_budget_exhausted"],
                "status": "needs_review" if state.get("review_signals", {}).get("review_required") else "completed",
            }

        try:
            bound_model = model.bind_tools(tools)
            messages = state.get("messages") or [
                SystemMessage(
                    content=(
                        "You are an offline vessel-match review planner. "
                        "Use only the supplied matching tools. Never infer or invent catalog facts. "
                        "The baseline prediction is fixed; additional tools cannot overwrite it. "
                        "Do not request external data or expose the input outside this run. "
                        "If the result is ambiguous, request human review."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Validated input: {state.get('input_record')}\n"
                        f"Baseline result: {state.get('baseline_result')}\n"
                        f"Review signals: {state.get('review_signals')}"
                    )
                ),
            ]
            response = bound_model.invoke(messages)
            if not isinstance(response, AIMessage):
                raise TypeError("model은 AIMessage를 반환해야 합니다")
            call_count = len(response.tool_calls)
            return {
                "messages": [response],
                "model_calls_used": model_calls + 1,
                "tool_calls_used": state.get("tool_calls_used", 0) + call_count,
            }
        except Exception as exc:
            return {
                "errors": [
                    *state.get("errors", []),
                    f"에이전트 판단 실패: {type(exc).__name__}: {exc}",
                ],
                "status": "needs_review" if state.get("review_signals", {}).get("review_required") else "completed",
            }

    def route_after_agent(state: AgentState) -> str:
        messages = state.get("messages", [])
        response = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )
        if response is None or not response.tool_calls:
            return "finish"
        if state.get("tool_calls_used", 0) > max_tool_calls:
            return "budget_error"
        expansion_calls = sum(
            tool_call.get("name") == "expand_search" for tool_call in response.tool_calls
        )
        if state.get("expansion_count", 0) + expansion_calls > max_expansions:
            return "budget_error"
        return "tools"

    def budget_error(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        response = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )
        expansion_calls = sum(
            tool_call.get("name") == "expand_search"
            for tool_call in (response.tool_calls if response else [])
        )
        expansion_over_budget = (
            state.get("expansion_count", 0) + expansion_calls > max_expansions
        )
        return {
            "errors": [
                *state.get("errors", []),
                "expansion_budget_exhausted"
                if expansion_over_budget
                else "tool_call_budget_exhausted",
            ],
            "status": "needs_review" if state.get("review_signals", {}).get("review_required") else "completed",
        }

    def collect_tool_results(state: AgentState) -> dict[str, Any]:
        seen = {entry.get("tool_call_id") for entry in state.get("tool_history", [])}
        history = list(state.get("tool_history", []))
        new_entries = []
        request_made = False
        expansion_count = state.get("expansion_count", 0)
        expanded_result = None
        for message in state.get("messages", []):
            if not isinstance(message, ToolMessage) or message.tool_call_id in seen:
                continue
            entry = {
                "tool_call_id": message.tool_call_id,
                "tool_name": message.name,
                "status": getattr(message, "status", "success"),
                "result": str(message.content),
            }
            new_entries.append(entry)
            if message.name == "expand_search":
                expansion_count += 1
                if getattr(message, "status", "success") != "error":
                    try:
                        expanded_result = json.loads(str(message.content))
                    except (TypeError, ValueError):
                        entry["status"] = "error"
            if message.name == "request_review":
                request_made = True
            if getattr(message, "status", "success") == "error":
                entry["status"] = "error"

        history.extend(new_entries)
        patch: dict[str, Any] = {
            "tool_history": history,
            "expansion_count": expansion_count,
        }
        if expanded_result is not None:
            patch["expanded_result"] = expanded_result
        if request_made:
            patch["status"] = "awaiting_review"
        if any(entry["status"] == "error" for entry in new_entries):
            patch["errors"] = [*state.get("errors", []), "tool_execution_failed"]
            patch["status"] = "needs_review"
        return patch

    def route_after_tools(state: AgentState) -> str:
        if state.get("status") == "awaiting_review":
            return "finish"
        if state.get("tool_calls_used", 0) >= max_tool_calls:
            return "finish"
        return "agent"

    def finish(state: AgentState) -> dict[str, Any]:
        status = state.get("status", "completed")
        if status == "running":
            status = "completed"
        if status == "completed" and state.get("review_signals", {}).get("review_required"):
            status = "needs_review"
        finalized = {**state, "status": status}
        return {"status": status, "final_report": _format_report(finalized)}

    builder.add_node("validate_input", validate_input)
    builder.add_node("baseline_match", baseline_match)
    builder.add_node("agent", agent_step)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    builder.add_node("collect_tools", collect_tool_results)
    builder.add_node("budget_error", budget_error)
    builder.add_node("finish", finish)
    builder.add_edge(START, "validate_input")
    builder.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {"match": "baseline_match", "finish": "finish"},
    )
    builder.add_conditional_edges(
        "baseline_match",
        route_after_match,
        {"agent": "agent", "finish": "finish"},
    )
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "finish": "finish", "budget_error": "budget_error"},
    )
    builder.add_edge("tools", "collect_tools")
    builder.add_conditional_edges(
        "collect_tools",
        route_after_tools,
        {"agent": "agent", "finish": "finish"},
    )
    builder.add_edge("budget_error", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)


def run_agent(
    record: dict[str, Any],
    *,
    service: MatchingService,
    session_id: str,
    config: dict[str, Any] | None = None,
    model: Any = None,
    checkpointer: Any = None,
) -> AgentState:
    """Run one validated row and return the final graph state."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id는 비어 있지 않은 문자열이어야 합니다")
    options = dict(config or {})
    graph = build_agent_graph(service, model=model, checkpointer=checkpointer, config=options)
    invoke_config = dict(options.get("invoke_config", {}))
    configurable = dict(invoke_config.get("configurable", {}))
    configurable["thread_id"] = session_id
    invoke_config["configurable"] = configurable

    payload = {
        "run_id": options.get("run_id", uuid4().hex),
        "session_id": session_id,
        "input_record": record,
        "baseline_result": None,
        "expanded_result": None,
        "review_signals": None,
        "evidence": [],
        "tool_history": [],
        "tool_calls_used": 0,
        "model_calls_used": 0,
        "expansion_count": 0,
        "status": "running",
        "review_decision": None,
        "final_report": None,
        "errors": [],
        "messages": [],
    }
    result = graph.invoke(payload, config=invoke_config)
    return result
