import unittest
from dataclasses import asdict

import pandas as pd
from langchain_core.messages import AIMessage

from agent.graph import run_agent
from agent.tools import build_agent_tools
from submit.model.matching_service import MatchingService
from submit.model.record_validation import VesselRecord


def make_service() -> MatchingService:
    catalog = pd.DataFrame(
        {
            "vessel_id": [f"VESSEL_{index:04d}" for index in range(90)],
            "vessel_name": ["HANGEUL"] * 90,
            "gross_tonnage": [None] * 90,
            "vessel_type": ["Tanker"] * 90,
            "flag_name": ["Liberia"] * 90,
        }
    )
    return MatchingService(catalog)


def make_record() -> VesselRecord:
    return VesselRecord(
        sample_id="SAMPLE_1",
        vessel_name="한글",
        call_sign=None,
        gross_tonnage=None,
        international_gross_tonnage=None,
        flag_name=None,
        vessel_type=None,
        arrival_year=None,
    )


class ScriptedModel:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def bind_tools(self, tools):
        self.tool_names = {tool.name for tool in tools}
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "expand_search",
                        "args": {"payload": self.payload, "profile": "wide_v1", "top_k": 5},
                        "id": "expand-call-1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="Expansion reviewed.")


class ExpandedSearchTest(unittest.TestCase):
    def test_wide_profile_adds_candidates_without_changing_baseline(self):
        service = make_service()
        record = make_record()
        baseline = service.match_vessel(record)

        expanded = service.expand_search(record, profile="wide_v1", top_k=90)

        self.assertEqual(baseline.baseline_prediction, service.match_vessel(record).baseline_prediction)
        self.assertEqual(baseline.baseline_prediction, "VESSEL_0000")
        self.assertGreater(expanded.candidate_count, len(baseline.top_rank))
        self.assertEqual(expanded.candidate_count, 80)
        self.assertEqual(expanded.suggested_prediction, "VESSEL_0000")
        self.assertIn("VESSEL_0070", {item.vessel_id for item in expanded.top_rank})

    def test_wide_profile_rejects_non_korean_records(self):
        service = make_service()
        record = make_record()
        record.vessel_name = "ENGLISH NAME"

        with self.assertRaises(ValueError):
            service.expand_search(record, profile="wide_v1")

    def test_expansion_tool_is_connected_to_run_agent_state(self):
        service = make_service()
        record = make_record()
        model = ScriptedModel(asdict(record))
        state = run_agent(
            asdict(record),
            service=service,
            session_id="expanded-search-test",
            model=model,
        )

        self.assertIn("expand_search", model.tool_names)
        self.assertIsNotNone(state.get("expanded_result"))
        self.assertEqual(state["expanded_result"]["candidate_count"], 80)
        self.assertEqual(
            state["baseline_result"]["baseline_prediction"],
            "VESSEL_0000",
        )
        self.assertEqual(state["expansion_count"], 1)

    def test_expansion_budget_prevents_tool_execution(self):
        service = make_service()
        record = make_record()
        model = ScriptedModel(asdict(record))

        state = run_agent(
            asdict(record),
            service=service,
            session_id="expanded-search-budget-test",
            config={"max_expansions": 0},
            model=model,
        )

        self.assertEqual(state["expansion_count"], 0)
        self.assertIsNone(state.get("expanded_result"))
        self.assertIn("expansion_budget_exhausted", state["errors"])
        self.assertEqual(
            state["baseline_result"]["baseline_prediction"],
            "VESSEL_0000",
        )


if __name__ == "__main__":
    unittest.main()
