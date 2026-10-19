"""Offline checks for the diagnosis-driven improvement loop (agent environment).

Run: .venv_agent_py311\\Scripts\\python.exe -m unittest tests.test_improvement_agent -v
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from agent import improvement_graph as graph_module
from agent.error_diagnostics import assert_llm_safe, cross_split, diagnose_rows, summarize_split
from agent.improvement_planner import build_payload, guard, propose_deterministic
from agent.retrain_config import (KNOBS, baseline_config, canonical_name, classify_outcome,
                                  config_for, fingerprint, validate_config)
from agent.search_policy import _query_value, behavior_probe, validate_proposals
from agent.team_v11.source import match_lib as ml


EMPTY_RES = {"token_dict": {}, "whole_dict": {}, "syl_map": {}, "vocab": None}
FEATURES = tuple(ml.FEATURE_NAMES)


def stage(**kw):
    base = {"when": "all", "query_form": "translated", "catalog_field": "compact", "scorer": "ratio",
            "top_k": 10, "min_similarity": 60.0, "max_gt_gap": None}
    return {**base, **kw}


class SearchPolicyFixTest(unittest.TestCase):
    def test_english_variant_has_no_non_korean_fallback(self):
        q = ml.prepare_query({"vessel_name": "OCEAN PIONEER NO.2"}, EMPTY_RES)
        self.assertIsNone(_query_value(q, "english_variant"))
        self.assertEqual(_query_value(q, "tokens"), "OCEAN PIONEER 2")
        k = ml.prepare_query({"vessel_name": "오션파이오니어"}, EMPTY_RES)
        self.assertEqual(_query_value(k, "english_variant"), k["variants"][1])

    def test_previous_variant_skeleton_policy_is_now_rejected(self):
        old = {"name": "variant_skeleton_fallback", "hypothesis": "x" * 20, "stages": [
            stage(when="non_korean", query_form="english_variant", catalog_field="alpha",
                  scorer="token_set_ratio", top_k=25, min_similarity=68.0, max_gt_gap=0.45)]}
        with self.assertRaises(ValueError):
            validate_proposals([old])

    def test_token_scorer_requires_spaced_fields(self):
        bad = {"name": "token_compact", "hypothesis": "x" * 20,
               "stages": [stage(query_form="translated", catalog_field="compact", scorer="token_set_ratio")]}
        with self.assertRaises(ValueError):
            validate_proposals([bad])
        good = {"name": "token_order", "hypothesis": "x" * 20,
                "stages": [stage(when="non_korean", query_form="tokens", catalog_field="tokens",
                                 scorer="token_set_ratio")]}
        self.assertEqual(len(validate_proposals([good])), 1)

    def test_behavior_probe_counts_applied_rows(self):
        queries = [ml.prepare_query({"vessel_name": n}, EMPTY_RES) for n in ("OCEAN STAR", "오션스타")]
        probe = behavior_probe(queries, {"stages": [stage(query_form="english_variant")]})
        self.assertEqual(probe[0]["applied_rows"], 1)


class RetrainConfigTest(unittest.TestCase):
    def test_single_change_only(self):
        config = config_for("num_leaves", "63")
        self.assertEqual(validate_config(config), ["num_leaves"])
        config["n_estimators"] = "800"
        with self.assertRaises(ValueError):
            validate_config(config)
        with self.assertRaises(ValueError):
            validate_config({**baseline_config(), "num_leaves": "64"})

    def test_fingerprint_and_name_are_stable(self):
        self.assertEqual(fingerprint(config_for("n_estimators", "800")),
                         fingerprint(dict(reversed(list(config_for("n_estimators", "800").items())))))
        self.assertEqual(canonical_name("tonnage_missing_encoding", "nan"), "tonnage_missing_encoding__nan")

    def test_decision_rule(self):
        def split(dw, dk=0.0, dn=0.0, nc=1, nw=0):
            return {"delta_weighted_raw": dw, "delta_korean_accuracy": dk, "delta_non_korean_accuracy": dn,
                    "newly_correct": nc, "newly_wrong": nw}
        self.assertEqual(classify_outcome([split(0.001)] * 3, True)["label"], "improved")
        self.assertEqual(classify_outcome([split(0.001), split(0.001), split(0.001, dk=-0.01)], True)["label"],
                         "mixed")
        self.assertEqual(classify_outcome([split(0, nc=0)] * 2, True)["label"], "no_effect")
        self.assertEqual(classify_outcome([split(-0.001)] * 2, True)["label"], "regressed")
        self.assertEqual(classify_outcome([split(0.01)] * 2, False)["label"], "invalid_behavior_mismatch")


def fake_cross(**counts):
    flags = {k: {"total": v, "splits_present": 2 if v else 0, "per_split": [v], "pattern":
                 "recurring" if v else "absent"} for k, v in counts.items()}
    from agent.retrain_config import ERROR_TYPES, NEAR_MISS_FLAGS
    return {"splits": ["a", "b"],
            "error_flags": {k: flags.get(k, {"total": 0, "splits_present": 0, "per_split": [0],
                                             "pattern": "absent"}) for k in ERROR_TYPES},
            "near_miss_flags": {k: flags.get(k, {"total": 0, "splits_present": 0, "per_split": [0],
                                                 "pattern": "absent"}) for k in NEAR_MISS_FLAGS},
            "note": "test"}


def proposal(**kw):
    base = {"hypothesis": "Missing tonnage is conflated with large gaps in the ranker.",
            "evidence_error_types": ["query_tonnage_missing"],
            "evidence_summary": "Several error cases have no query tonnage at all.",
            "knob": "tonnage_missing_encoding", "value": "nan", "change_kind": "feature",
            "expected_effect": "fewer tonnage related errors", "failure_risk": "may shift other rows",
            "duplicates_existing_feature": False, "requires_validation_labels_or_cross_row": False,
            "metrics_to_check": ["weighted_raw"], "stop_condition": "any_language_accuracy_drop"}
    return {**base, **kw}


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.cross = fake_cross(query_tonnage_missing=3, number_mismatch=2)

    def test_accepts_valid_single_change(self):
        accepted, rejected = guard([proposal()], self.cross, set(), 3)
        self.assertEqual((len(accepted), rejected), (1, []))
        self.assertEqual(accepted[0]["config"], config_for("tonnage_missing_encoding", "nan"))

    def test_rejections(self):
        tried = {fingerprint(config_for("number_features", "none"))}
        cases = {
            "no_change_duplicates_current_implementation": proposal(value="gap_2.0"),
            "value_not_allowed_for_knob": proposal(value="63"),
            "change_kind_does_not_match_knob": proposal(change_kind="training"),
            "requires_labels_cross_row_or_external_data": proposal(requires_validation_labels_or_cross_row=True),
            "declared_duplicate_of_existing_feature": proposal(duplicates_existing_feature=True),
            "evidence_not_observed_in_diagnostics": proposal(evidence_error_types=["same_name_collision"]),
            "knob_does_not_address_cited_evidence": proposal(evidence_error_types=["number_mismatch"]),
            "hypothesis_forbidden_content": proposal(hypothesis="Use the test set rows to calibrate gaps."),
            "duplicate_of_previous_experiment": proposal(knob="number_features", value="none",
                                                         evidence_error_types=["number_mismatch"]),
        }
        for reason, item in cases.items():
            accepted, rejected = guard([item], self.cross, tried, 3)
            self.assertEqual(accepted, [], reason)
            self.assertIn(reason, rejected[0]["reasons"], reason)

    def test_within_round_duplicate_and_limit(self):
        accepted, rejected = guard([proposal(), proposal(), proposal()], self.cross, set(), 2)
        self.assertEqual(len(accepted), 1)
        self.assertIn("duplicate_within_round", rejected[0]["reasons"])
        self.assertIn("over_proposal_limit", rejected[1]["reasons"])

    def test_schema_extra_field_rejected(self):
        _, rejected = guard([{**proposal(), "code": "print(1)"}], self.cross, set(), 3)
        self.assertEqual(rejected[0]["reasons"], ["schema_invalid"])

    def test_deterministic_planner_skips_tried(self):
        tried = {fingerprint(config_for("tonnage_missing_encoding", "nan"))}
        out = propose_deterministic(self.cross, tried, 3)
        names = [(p["knob"], p["value"]) for p in out["proposals"]]
        self.assertNotIn(("tonnage_missing_encoding", "nan"), names)
        accepted, _ = guard(out["proposals"], self.cross, tried, 3)
        self.assertEqual(len(accepted), len(out["proposals"]))


def synthetic_details():
    """Three rows over a 4-vessel catalog; features are mostly zero."""
    names = {n: i for i, n in enumerate(FEATURES)}

    def block(rows):
        f = np.zeros((len(rows), len(FEATURES)), dtype=np.float32)
        for r, values in enumerate(rows):
            for k, v in values.items():
                f[r, names[k]] = v
        return f
    base_missing = {k: False for k in ("vessel_name", "call_sign", "gross_tonnage",
                                       "international_gross_tonnage", "flag_name", "vessel_type")}
    return [
        # truth 0 absent from candidates -> candidate_missing
        {"sample_id": "S1", "candidates": np.array([1, 2]), "scores": np.array([2.0, 1.0]),
         "features": block([{"s_compact": 80}, {"s_compact": 70}]), "final_pred_id": "V1",
         "override_source": "none", "name_lookup_id": None, "callsign_id": None, "missing": base_missing},
        # model right (V1) but call-sign override -> V2: override_introduced_error
        {"sample_id": "S2", "candidates": np.array([1, 2]), "scores": np.array([3.0, 1.0]),
         "features": block([{}, {}]), "final_pred_id": "V2", "override_source": "callsign",
         "name_lookup_id": None, "callsign_id": "V2", "missing": base_missing},
        # same compact name V2/V3, wrong pick -> same_name_collision
        {"sample_id": "S3", "candidates": np.array([2, 3]), "scores": np.array([1.0, 1.5]),
         "features": block([{"gap_best": 0.1}, {"gap_best": 0.5}]), "final_pred_id": "V3",
         "override_source": "none", "name_lookup_id": None, "callsign_id": None, "missing": base_missing},
    ]


class DiagnosticsTest(unittest.TestCase):
    def test_classification_and_safe_summary(self):
        ids = np.array(["V0", "V1", "V2", "V3"])
        cases = diagnose_rows(synthetic_details(), ["V0", "V1", "V2"], np.array([False] * 3),
                              np.array([True, False, False]), FEATURES, ids,
                              ["A", "B", "SAME", "SAME"], {v: i for i, v in enumerate(ids)})
        self.assertEqual([c["primary_error_type"] for c in cases],
                         ["candidate_missing", "override_introduced_error", "same_name_collision"])
        self.assertIn("tonnage_favored_truth_lost", cases[2]["error_flags"])
        self.assertEqual(cases[2]["truth_rank"], 2)
        summary = summarize_split("toy", cases)
        assert_llm_safe(summary)
        self.assertNotIn("S1", json.dumps(summary))
        cross = cross_split([summary, summary])
        self.assertEqual(cross["error_flags"]["candidate_missing"]["pattern"], "recurring")
        payload = build_payload([summary], cross, [], {"llm_calls_remaining": 1})
        assert_llm_safe(payload)

    def test_safety_rejects_identifiers(self):
        for bad in ({"x": "TRAIN_00001"}, {"x": "VESSEL_12"}, {"x": "오션"}, {"x": float("nan")}):
            with self.assertRaises(ValueError):
                assert_llm_safe(bad)


class LoopBudgetTest(unittest.TestCase):
    """Run the LangGraph loop with a fake worker to check budgets and stop reasons."""

    flags = ["query_tonnage_missing", "number_mismatch"]

    def _fake_worker(self, args, output):
        mode, split = args[0], args[2]
        m = {"all": {"correct": 9, "rows": 10, "candidate_hits": 10}, "korean": {"correct": 5, "rows": 5},
             "non_korean": {"correct": 4, "rows": 5}, "weighted_raw": 0.94}
        cases = [{"final_correct": i != 0, "model_correct": i != 0, "is_korean": i < 5, "seen_target": False,
                  "margin": 0.1 * i, "sample_id": f"S{i}", "truth_in_candidates": True, "override_source": "none",
                  "override_changed": False, "primary_error_type": "query_tonnage_missing" if i == 0 else None,
                  "error_flags": list(self.flags) if i == 0 else [],
                  "near_miss_flags": ["near_miss_query_tonnage_missing"], "truth_rank": 2 if i == 0 else 1,
                  "missing_inputs": ["gross_tonnage"]} for i in range(10)]
        if mode == "audit":
            return {"split_name": split}
        if mode == "diagnose":
            return {"metrics": m, "diagnostic_summary": summarize_split(split, cases), "model_sha256": "0",
                    "config_fingerprint": fingerprint(baseline_config()), "checks": {},
                    "timing_seconds": {"training": 1.0, "validation_prediction": 1.0}, "local_case_file": "x"}
        knob, value = args[args.index("--knob") + 1], args[args.index("--value") + 1]
        config = config_for(knob, value)
        return {"split_name": split, "config": config, "config_fingerprint": fingerprint(config),
                "setting": canonical_name(knob, value), "retrained": KNOBS[knob]["retrain"],
                "behavior_check": {"passed": True, "checks": ["ok"]},
                "delta": {"weighted_raw": 0.0, "korean_accuracy": 0.0, "non_korean_accuracy": 0.0},
                "row_changes": {"newly_correct": 0, "newly_wrong": 0}, "feature_importance_shift_top8": [],
                "timing_seconds": {"training": 1.0, "validation_prediction": 1.0}, "model_sha256": "1",
                "training_stats": {}, "checks": {}, "metrics": {**m, "official": {}, "seen_target": {},
                                                                "unseen_target": {}}}

    def test_deterministic_loop_stops_on_budget_and_dedups(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.object(graph_module, "_worker", self._fake_worker), \
                    mock.patch.object(graph_module, "LEDGER", tmp / "ledger.json"), \
                    mock.patch.object(graph_module, "LOG", tmp / "LOG.md"), \
                    mock.patch.object(graph_module, "ROOT", tmp):
                budget = graph_module.Budget(max_llm_calls=3, max_training_experiments=2,
                                             max_proposals_per_round=1, patience_rounds_without_improvement=5)
                run_dir = tmp / "run1"
                run_dir.mkdir()
                state = graph_module.build_improvement_graph("deterministic").invoke(
                    {"run_dir": str(run_dir), "splits": ["s1", "s2"], "planner": "deterministic",
                     "budget": vars(budget) if hasattr(budget, "__dict__") else budget.__dict__},
                    {"recursion_limit": 40})
                report = json.loads(Path(state["report_path"]).read_text(encoding="utf-8"))
                self.assertEqual(report["stop_reason"], "training_experiment_budget_exhausted")
                self.assertEqual(len(report["experiments"]), 2)
                self.assertEqual(report["llm_calls_made"], 0)
                names = [e["name"] for e in report["experiments"]]
                self.assertEqual(len(set(names)), 2)
                ledger = json.loads((tmp / "ledger.json").read_text(encoding="utf-8"))
                executed = [e for e in ledger["entries"] if e["status"] == "executed"]
                self.assertEqual(len(executed), 2)
                self.assertNotIn("S0", (tmp / "LOG.md").read_text(encoding="utf-8"))

    def test_patience_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.object(graph_module, "_worker", self._fake_worker), \
                    mock.patch.object(graph_module, "LEDGER", tmp / "ledger.json"), \
                    mock.patch.object(graph_module, "LOG", tmp / "LOG.md"), \
                    mock.patch.object(graph_module, "ROOT", tmp):
                run_dir = tmp / "run2"
                run_dir.mkdir()
                budget = {"max_llm_calls": 3, "max_training_experiments": 5, "max_proposals_per_round": 1,
                          "patience_rounds_without_improvement": 1}
                state = graph_module.build_improvement_graph("deterministic").invoke(
                    {"run_dir": str(run_dir), "splits": ["s1"], "planner": "deterministic", "budget": budget},
                    {"recursion_limit": 40})
                self.assertEqual(state["stop_reason"], "no_improvement_within_patience")
                self.assertEqual(state["experiments_run"], 1)

    def test_stops_when_no_allowed_change_remains(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.object(self, "flags", ["query_tonnage_missing"]),                     mock.patch.object(graph_module, "_worker", self._fake_worker),                     mock.patch.object(graph_module, "LEDGER", tmp / "ledger.json"),                     mock.patch.object(graph_module, "LOG", tmp / "LOG.md"),                     mock.patch.object(graph_module, "ROOT", tmp):
                run_dir = tmp / "run3"
                run_dir.mkdir()
                budget = {"max_llm_calls": 3, "max_training_experiments": 5, "max_proposals_per_round": 1,
                          "patience_rounds_without_improvement": 5}
                state = graph_module.build_improvement_graph("deterministic").invoke(
                    {"run_dir": str(run_dir), "splits": ["s1"], "planner": "deterministic", "budget": budget},
                    {"recursion_limit": 40})
                self.assertEqual(state["stop_reason"], "all_proposals_rejected_or_none_proposed")
                self.assertEqual(state["experiments_run"], 1)

    def test_budget_bounds(self):
        with self.assertRaises(ValueError):
            graph_module.Budget(max_llm_calls=0)
        with self.assertRaises(ValueError):
            graph_module.Budget(max_training_experiments=99)

    def test_user_selected_setting_uses_same_guard_and_exact_change(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with mock.patch.object(graph_module, "_worker", self._fake_worker), \
                    mock.patch.object(graph_module, "LEDGER", folder / "ledger.json"), \
                    mock.patch.object(graph_module, "LOG", folder / "LOG.md"), \
                    mock.patch.object(graph_module, "ROOT", folder):
                run = folder / "fixed"
                run.mkdir()
                state = graph_module.build_improvement_graph(
                    "deterministic", ("number_features", "no_asymmetry")).invoke(
                        {"run_dir": str(run), "splits": ["toy"], "planner": "deterministic",
                         "budget": {"max_llm_calls": 1, "max_training_experiments": 1,
                                    "max_proposals_per_round": 1,
                                    "patience_rounds_without_improvement": 1}})
                self.assertEqual([e["name"] for e in state["experiments"]],
                                 ["number_features__no_asymmetry"])
                self.assertEqual(state["llm_calls"], 0)
                self.assertEqual(state["stop_reason"], "training_experiment_budget_exhausted")
        with self.assertRaises(ValueError):
            graph_module.build_improvement_graph("llm", ("ranker_family", "catboost_v1_cpu"))


if __name__ == "__main__":
    unittest.main()
