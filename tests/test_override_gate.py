"""Tests for the v11-override-gate-1 policies, the C3 collision and the run gates."""

import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from agent import override_gate as gate
from agent import stress_c3 as c3
from agent import stress_perturbations as perturb

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments" / "protocols" / "v11_override_gate_v1.json"
PROTOCOL_SHA256 = "73d4c464f63d050e8f68dbdadab64e9ac013b3ff2a24f6ef8b968755764735b1"

# A tiny catalog: vessel_id -> catalog position.
CATALOG = ["VESSEL_00", "VESSEL_01", "VESSEL_02", "VESSEL_03", "VESSEL_04", "VESSEL_05"]
COLUMNS = {name: i for i, name in enumerate(gate.GATE_FEATURES)}


class FakeIndex:
    def __init__(self, ids):
        self.ids = np.array(ids, dtype=object)
        self.id_to_index = {str(v): i for i, v in enumerate(ids)}


INDEX = FakeIndex(CATALOG)


def features_for(count, **overrides):
    """One clean feature row per candidate; no contradiction unless overridden."""
    rows = np.zeros((count, len(gate.GATE_FEATURES)), dtype=np.float32)
    rows[:, COLUMNS["gap_best"]] = 0.0
    for key, value in overrides.items():
        position, name = key.split("__")
        rows[int(position), COLUMNS[name]] = value
    return rows


def row(candidates, scores, features, callsign_id=None, name_lookup_id=None,
        model_pred_id=None, sample_id="TRAIN_00001"):
    order = gate.model_order(np.array(candidates), scores)
    top = str(CATALOG[candidates[int(order[0])]])
    return {"sample_id": sample_id, "candidates": np.array(candidates), "scores": np.array(scores),
            "features": features, "model_pred_id": model_pred_id or top,
            "callsign_id": callsign_id, "name_lookup_id": name_lookup_id}


class ConstantsTest(unittest.TestCase):
    def test_protocol_file_hash_and_module_constants_agree(self):
        self.assertEqual(hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(), PROTOCOL_SHA256)
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(protocol["protocol_version"], gate.PROTOCOL_VERSION)
        checked = gate.check_constants(protocol)
        self.assertEqual(checked["top_k"], 3)
        self.assertEqual(checked["max_relative_tonnage_gap"], 0.25)
        self.assertEqual(sorted(checked["policies"]), sorted(gate.POLICIES))

    def test_a_changed_constant_is_refused(self):
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        protocol["policies"]["P3_rank_and_attributes"]["fixed_constants"]["top_k"] = 5
        with self.assertRaisesRegex(ValueError, "top_k"):
            gate.check_constants(protocol)
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        protocol["policies"].pop("P2_candidate_member")
        with self.assertRaisesRegex(ValueError, "정책 목록"):
            gate.check_constants(protocol)

    def test_feature_columns_resolve_against_the_real_v11_feature_order(self):
        from agent.retrain_runtime import FEATURES
        columns = gate.feature_columns(FEATURES)
        self.assertEqual(columns, {"gap_best": 20, "flag_prob": 24, "flag_seen": 25,
                                   "type_prob": 26, "type_seen": 27})
        with self.assertRaises(ValueError):
            gate.feature_columns(("a", "b"))


class RankTest(unittest.TestCase):
    def test_rank_follows_v11_argmax_with_catalog_index_tie_break(self):
        candidates = [4, 1, 2]
        scores = [0.5, 0.5, 0.9]
        # VESSEL_02 wins on score; the 0.5 tie goes to the lower catalog index (1 before 4).
        self.assertEqual(gate.candidate_rank(np.array(candidates), scores, 2), 1)
        self.assertEqual(gate.candidate_rank(np.array(candidates), scores, 1), 2)
        self.assertEqual(gate.candidate_rank(np.array(candidates), scores, 4), 3)
        self.assertIsNone(gate.candidate_rank(np.array(candidates), scores, 5))


class P2Test(unittest.TestCase):
    def test_candidate_membership_is_the_only_condition(self):
        detail = row([0, 1, 2], [0.9, 0.5, 0.1], features_for(3), callsign_id="VESSEL_02")
        applied, reason = gate.gate_decision("P2_candidate_member", detail["candidates"],
                                             detail["scores"], detail["features"],
                                             "VESSEL_02", INDEX, COLUMNS)
        self.assertTrue(applied)
        self.assertEqual(reason, "candidate_member")

    def test_a_vessel_outside_the_candidate_list_is_blocked(self):
        detail = row([0, 1, 2], [0.9, 0.5, 0.1], features_for(3))
        for policy in ("P2_candidate_member", "P3_rank_and_attributes"):
            applied, reason = gate.gate_decision(policy, detail["candidates"], detail["scores"],
                                                 detail["features"], "VESSEL_05", INDEX, COLUMNS)
            self.assertFalse(applied, policy)
            self.assertEqual(reason, "not_in_candidates", policy)

    def test_a_vessel_absent_from_the_catalog_is_blocked(self):
        detail = row([0, 1, 2], [0.9, 0.5, 0.1], features_for(3))
        applied, reason = gate.gate_decision("P2_candidate_member", detail["candidates"],
                                             detail["scores"], detail["features"],
                                             "VESSEL_99", INDEX, COLUMNS)
        self.assertFalse(applied)
        self.assertEqual(reason, "not_in_candidates")


class P3Test(unittest.TestCase):
    def _decide(self, features, lookup="VESSEL_03", candidates=(0, 1, 2, 3),
                scores=(0.9, 0.8, 0.7, 0.6)):
        return gate.gate_decision("P3_rank_and_attributes", np.array(candidates), list(scores),
                                  features, lookup, INDEX, COLUMNS)

    def test_rank_four_is_blocked_and_rank_three_passes(self):
        applied, reason = self._decide(features_for(4))
        self.assertFalse(applied)
        self.assertEqual(reason, "rank_above_top_k")
        applied, reason = self._decide(features_for(3), lookup="VESSEL_02",
                                       candidates=(0, 1, 2), scores=(0.9, 0.8, 0.7))
        self.assertTrue(applied)
        self.assertEqual(reason, "rank_and_attributes_ok")

    def test_tonnage_gap_above_the_fixed_threshold_is_a_contradiction(self):
        applied, reason = self._decide(features_for(3, **{"2__gap_best": 0.3}),
                                       lookup="VESSEL_02", candidates=(0, 1, 2),
                                       scores=(0.9, 0.8, 0.7))
        self.assertFalse(applied)
        self.assertEqual(reason, "tonnage_contradiction")

    def test_a_gap_at_the_threshold_and_the_stress_noise_gaps_pass(self):
        for value in (0.25, 0.2, 0.167, 0.0):
            applied, reason = self._decide(features_for(3, **{"2__gap_best": value}),
                                           lookup="VESSEL_02", candidates=(0, 1, 2),
                                           scores=(0.9, 0.8, 0.7))
            self.assertTrue(applied, value)
            self.assertEqual(reason, "rank_and_attributes_ok", value)

    def test_gap_best_two_means_a_missing_side_and_counts_as_no_evidence(self):
        applied, reason = self._decide(features_for(3, **{"2__gap_best": 2.0}),
                                       lookup="VESSEL_02", candidates=(0, 1, 2),
                                       scores=(0.9, 0.8, 0.7))
        self.assertTrue(applied)
        self.assertEqual(reason, "rank_and_attributes_ok")

    def test_flag_contradiction_only_when_seen_is_one_and_prob_is_zero(self):
        blocked = self._decide(features_for(3, **{"2__flag_seen": 1.0, "2__flag_prob": 0.0}),
                               lookup="VESSEL_02", candidates=(0, 1, 2), scores=(0.9, 0.8, 0.7))
        self.assertEqual(blocked, (False, "flag_contradiction"))
        unseen = self._decide(features_for(3, **{"2__flag_seen": 0.0, "2__flag_prob": 0.0}),
                              lookup="VESSEL_02", candidates=(0, 1, 2), scores=(0.9, 0.8, 0.7))
        self.assertEqual(unseen, (True, "rank_and_attributes_ok"))
        positive = self._decide(features_for(3, **{"2__flag_seen": 1.0, "2__flag_prob": 0.4}),
                                lookup="VESSEL_02", candidates=(0, 1, 2), scores=(0.9, 0.8, 0.7))
        self.assertEqual(positive, (True, "rank_and_attributes_ok"))

    def test_type_contradiction_only_when_seen_is_one_and_prob_is_zero(self):
        blocked = self._decide(features_for(3, **{"2__type_seen": 1.0, "2__type_prob": 0.0}),
                               lookup="VESSEL_02", candidates=(0, 1, 2), scores=(0.9, 0.8, 0.7))
        self.assertEqual(blocked, (False, "type_contradiction"))
        unseen = self._decide(features_for(3, **{"2__type_seen": 0.0, "2__type_prob": 0.0}),
                              lookup="VESSEL_02", candidates=(0, 1, 2), scores=(0.9, 0.8, 0.7))
        self.assertEqual(unseen, (True, "rank_and_attributes_ok"))
        positive = self._decide(features_for(3, **{"2__type_seen": 1.0, "2__type_prob": 0.7}),
                                lookup="VESSEL_02", candidates=(0, 1, 2), scores=(0.9, 0.8, 0.7))
        self.assertEqual(positive, (True, "rank_and_attributes_ok"))

    def test_tonnage_is_reported_before_flag_and_type(self):
        applied, reason = self._decide(
            features_for(3, **{"2__gap_best": 0.9, "2__flag_seen": 1.0, "2__flag_prob": 0.0,
                               "2__type_seen": 1.0, "2__type_prob": 0.0}),
            lookup="VESSEL_02", candidates=(0, 1, 2), scores=(0.9, 0.8, 0.7))
        self.assertFalse(applied)
        self.assertEqual(reason, "tonnage_contradiction")

    def test_p3_without_feature_columns_is_refused(self):
        with self.assertRaisesRegex(ValueError, "컬럼"):
            gate.gate_decision("P3_rank_and_attributes", np.array([0, 1, 2]), [0.9, 0.8, 0.7],
                               features_for(3), "VESSEL_02", INDEX, None)


class PolicyApplicationTest(unittest.TestCase):
    def test_callsign_has_priority_over_name_lookup(self):
        detail = row([0, 1, 2], [0.9, 0.8, 0.7], features_for(3),
                     callsign_id="VESSEL_01", name_lookup_id="VESSEL_02")
        for policy in ("P0_always", "P2_candidate_member", "P3_rank_and_attributes"):
            result = gate.apply_policy(detail, policy, INDEX, COLUMNS)
            self.assertEqual(result["final_pred_id"], "VESSEL_01", policy)
            self.assertEqual(result["override_source"], "callsign", policy)

    def test_each_source_is_gated_independently(self):
        # The call sign points outside the candidates; the name lookup is rank 2 and clean.
        detail = row([0, 1, 2], [0.9, 0.8, 0.7], features_for(3),
                     callsign_id="VESSEL_05", name_lookup_id="VESSEL_01")
        result = gate.apply_policy(detail, "P2_candidate_member", INDEX, COLUMNS)
        self.assertEqual(result["final_pred_id"], "VESSEL_01")
        self.assertEqual(result["override_source"], "name_lookup")
        self.assertFalse(result["decisions"]["callsign"]["applied"])
        self.assertEqual(result["decisions"]["callsign"]["reason"], "not_in_candidates")
        self.assertTrue(result["decisions"]["name_lookup"]["applied"])
        # P0 keeps the v11 behaviour: the out-of-candidate call sign still wins.
        p0 = gate.apply_policy(detail, "P0_always", INDEX, COLUMNS)
        self.assertEqual(p0["final_pred_id"], "VESSEL_05")

    def test_p1_never_applies_and_keeps_the_model_top1(self):
        detail = row([0, 1, 2], [0.9, 0.8, 0.7], features_for(3),
                     callsign_id="VESSEL_01", name_lookup_id="VESSEL_02")
        result = gate.apply_policy(detail, "P1_off", INDEX, COLUMNS)
        self.assertEqual(result["final_pred_id"], "VESSEL_00")
        self.assertEqual(result["override_source"], "none")
        for source in gate.SOURCES:
            self.assertEqual(result["decisions"][source]["reason"], "policy_off")

    def test_a_row_without_lookups_is_untouched_by_every_policy(self):
        detail = row([0, 1, 2], [0.9, 0.8, 0.7], features_for(3))
        for policy in gate.POLICIES:
            result = gate.apply_policy(detail, policy, INDEX, COLUMNS)
            self.assertEqual(result["final_pred_id"], "VESSEL_00", policy)
            self.assertEqual(result["override_source"], "none", policy)
            for source in gate.SOURCES:
                self.assertEqual(result["decisions"][source]["reason"], "no_lookup")

    def test_an_unknown_policy_is_refused(self):
        detail = row([0, 1, 2], [0.9, 0.8, 0.7], features_for(3), callsign_id="VESSEL_01")
        with self.assertRaises(ValueError):
            gate.gate_decision("P9_unknown", detail["candidates"], detail["scores"],
                               detail["features"], "VESSEL_01", INDEX, COLUMNS)


class RankingTest(unittest.TestCase):
    def test_net_descending_then_fewer_conditions_first(self):
        nets = {p: 0 for p in gate.POLICIES}
        eligible = {p: True for p in gate.POLICIES}
        self.assertEqual(gate.rank_policies(nets, eligible),
                         ["P1_off", "P2_candidate_member", "P3_rank_and_attributes", "P0_always"])
        nets["P3_rank_and_attributes"] = 5
        self.assertEqual(gate.rank_policies(nets, eligible)[0], "P3_rank_and_attributes")
        eligible["P3_rank_and_attributes"] = False
        self.assertNotIn("P3_rank_and_attributes", gate.rank_policies(nets, eligible))

    def test_recommended_adds_p1_when_it_is_not_top(self):
        self.assertEqual(gate.recommended(["P3_rank_and_attributes", "P1_off"]),
                         ["P3_rank_and_attributes", "P1_off"])
        self.assertEqual(gate.recommended(["P1_off", "P2_candidate_member"]), ["P1_off"])
        self.assertEqual(gate.recommended([]), [])


class C3Test(unittest.TestCase):
    def _details(self, sample_id, candidates, scores):
        return {"sample_id": sample_id, "candidates": np.array(candidates),
                "scores": np.array(scores), "call_sign": "CS_valid"}

    def test_selection_uses_the_frozen_stress_hash_and_select_salt(self):
        self.assertEqual(c3.SELECTION_SALT, "select")
        self.assertEqual(c3.RATE, 0.10)
        for index in range(50):
            sample_id = f"TRAIN_{index:05d}"
            expected = perturb.u(sample_id, c3.SCENARIO_ID, "select") < 0.10
            self.assertEqual(c3.selected(sample_id), expected)
        ids = [f"TRAIN_{i:05d}" for i in range(4000)]
        rate = sum(c3.selected(i) for i in ids) / len(ids)
        self.assertLess(abs(rate - 0.10), 0.02)

    def test_only_a_training_vessel_at_rank_two_to_five_is_chosen(self):
        by_vessel = {"VESSEL_03": ["CS_b", "CS_a"], "VESSEL_00": ["CS_top"]}
        selected_id = next(f"TRAIN_{i:05d}" for i in range(500) if c3.selected(f"TRAIN_{i:05d}"))
        detail = self._details(selected_id, [0, 1, 2, 3], [0.9, 0.8, 0.7, 0.6])
        plan = c3.plan_row(detail, by_vessel, CATALOG)
        # VESSEL_00 is rank 1 and must be skipped even though it is in the table.
        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["injected_vessel_id"], "VESSEL_03")
        self.assertEqual(plan["injected_rank"], 4)
        self.assertEqual(plan["changes"], {"call_sign": "CS_a"})  # lexicographically smallest

    def test_rank_six_is_not_eligible(self):
        by_vessel = {"VESSEL_05": ["CS_x"]}
        selected_id = next(f"TRAIN_{i:05d}" for i in range(500) if c3.selected(f"TRAIN_{i:05d}"))
        detail = self._details(selected_id, [0, 1, 2, 3, 4, 5],
                               [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        plan = c3.plan_row(detail, by_vessel, CATALOG)
        self.assertFalse(plan["eligible"])
        self.assertEqual(plan["changes"], {})

    def test_the_best_ranked_eligible_candidate_wins(self):
        by_vessel = {"VESSEL_01": ["CS_one"], "VESSEL_03": ["CS_three"]}
        selected_id = next(f"TRAIN_{i:05d}" for i in range(500) if c3.selected(f"TRAIN_{i:05d}"))
        detail = self._details(selected_id, [0, 1, 2, 3], [0.9, 0.8, 0.7, 0.6])
        plan = c3.plan_row(detail, by_vessel, CATALOG)
        self.assertEqual(plan["injected_vessel_id"], "VESSEL_01")
        self.assertEqual(plan["injected_rank"], 2)

    def test_unselected_rows_are_untouched_and_deterministic(self):
        by_vessel = {"VESSEL_03": ["CS_a"]}
        unselected = next(f"TRAIN_{i:05d}" for i in range(500) if not c3.selected(f"TRAIN_{i:05d}"))
        detail = self._details(unselected, [0, 1, 2, 3], [0.9, 0.8, 0.7, 0.6])
        first = c3.plan_row(detail, by_vessel, CATALOG)
        second = c3.plan_row(detail, by_vessel, CATALOG)
        self.assertEqual(first, second)
        self.assertFalse(first["selected"])
        self.assertEqual(first["changes"], {})

    def test_not_applicable_when_no_candidate_is_a_training_vessel(self):
        frame = pd.DataFrame([{"sample_id": f"TRAIN_{i:05d}", "vessel_name": "X",
                               "call_sign": "CS_valid"} for i in range(200)])
        details = [self._details(f"TRAIN_{i:05d}", [0, 1, 2, 3], [0.9, 0.8, 0.7, 0.6])
                   for i in range(200)]
        out, audit, plans = c3.perturb_frame(frame, details, {}, CATALOG)
        self.assertTrue(audit["not_applicable"])
        self.assertEqual(audit["changed_rows"], 0)
        self.assertEqual(audit["rule_eligible_rows"], 0)
        self.assertGreater(audit["selected_rows"], 0)
        self.assertEqual(list(out["call_sign"]), list(frame["call_sign"]))

    def test_frame_level_c3_is_deterministic_and_refuses_labels(self):
        frame = pd.DataFrame([{"sample_id": f"TRAIN_{i:05d}", "vessel_name": "X",
                               "call_sign": "CS_valid"} for i in range(200)])
        details = [self._details(f"TRAIN_{i:05d}", [0, 1, 2, 3], [0.9, 0.8, 0.7, 0.6])
                   for i in range(200)]
        table = {"CS_t1": "VESSEL_03", "CS_t0": "VESSEL_03"}
        first = c3.perturb_frame(frame, details, table, CATALOG)
        second = c3.perturb_frame(frame, details, table, CATALOG)
        self.assertEqual(list(first[0]["call_sign"]), list(second[0]["call_sign"]))
        self.assertEqual(first[1], second[1])
        self.assertGreater(first[1]["changed_rows"], 0)
        self.assertEqual(first[1]["injected_rank_counts"], {"rank_4": first[1]["changed_rows"]})
        self.assertEqual(first[1]["protocol_string_for_hash"], "v11-stress-oof-1")
        with self.assertRaises(AssertionError):
            c3.perturb_frame(frame.assign(target_vessel_id="VESSEL_1"), details, table, CATALOG)

    def test_mismatched_clean_predictions_are_refused(self):
        frame = pd.DataFrame([{"sample_id": "TRAIN_00000", "vessel_name": "X", "call_sign": "CS"}])
        with self.assertRaises(ValueError):
            c3.perturb_frame(frame, [], {}, CATALOG)
        wrong = [self._details("TRAIN_00001", [0, 1], [0.9, 0.8])]
        with self.assertRaises(ValueError):
            c3.perturb_frame(frame, wrong, {}, CATALOG)

    def test_callsigns_by_vessel_sorts_and_groups(self):
        table = {"CS_b": "VESSEL_01", "CS_a": "VESSEL_01", "CS_c": "VESSEL_02"}
        self.assertEqual(c3.callsigns_by_vessel(table),
                         {"VESSEL_01": ["CS_a", "CS_b"], "VESSEL_02": ["CS_c"]})


class RunGateTest(unittest.TestCase):
    def test_a_g0_mismatch_is_detected(self):
        from scripts.run_v11_override_gate import g0_check
        details = [{"sample_id": "TRAIN_00000", "final_pred_id": "VESSEL_1",
                    "candidates": np.array([0, 1]), "features": np.zeros((2, 2), dtype=np.float32)},
                   {"sample_id": "TRAIN_00001", "final_pred_id": "VESSEL_2",
                    "candidates": np.array([0, 1]), "features": np.zeros((2, 2), dtype=np.float32)}]
        stored = [{"sample_id": "TRAIN_00000", "final_pred_id": "VESSEL_1"},
                  {"sample_id": "TRAIN_00001", "final_pred_id": "VESSEL_2"}]
        from scripts.v11_improvement_worker import prediction_input_hashes
        hashes = prediction_input_hashes(details)
        ok = g0_check("x", details, stored, hashes["candidates_sha256"], hashes["features_sha256"])
        self.assertTrue(ok["all_rows_match"])
        self.assertTrue(ok["candidates_sha256_matches"])
        moved = [dict(details[0]), dict(details[1], final_pred_id="VESSEL_9")]
        bad = g0_check("x", moved, stored, hashes["candidates_sha256"], hashes["features_sha256"])
        self.assertFalse(bad["all_rows_match"])
        self.assertEqual(bad["final_pred_mismatched"], 1)
        hashed = g0_check("x", details, stored, "0" * 64, hashes["features_sha256"])
        self.assertFalse(hashed["candidates_sha256_matches"])

    def test_non_identical_e1_predictions_fail_g1(self):
        from scripts.run_v11_override_gate import e1_policies_identical
        same = [{policy: {"final_pred_sha256": "a" * 64} for policy in gate.POLICIES}]
        self.assertTrue(e1_policies_identical(same))
        drifted = [{policy: {"final_pred_sha256": ("b" * 64 if policy == "P2_candidate_member"
                                                   else "a" * 64)}
                    for policy in gate.POLICIES}]
        self.assertFalse(e1_policies_identical(drifted))
        self.assertTrue(e1_policies_identical(drifted, "P3_rank_and_attributes"))
        self.assertFalse(e1_policies_identical(drifted, "P2_candidate_member"))
        two_folds = same + drifted
        self.assertFalse(e1_policies_identical(two_folds))

    def test_existing_stress_scenarios_are_untouched_by_this_protocol(self):
        # C2/N2/N5/T1 must keep behaving exactly as in v11-stress-oof-1.
        self.assertEqual(perturb.PROTOCOL_VERSION, "v11-stress-oof-1")
        self.assertIn("C2_callsign_collision_10pct", perturb.SCENARIOS)
        self.assertNotIn(c3.SCENARIO_ID, perturb.SCENARIOS)
        base = {"sample_id": "TRAIN_00001", "vessel_name": "ALPHA BRAVO 7", "call_sign": "CS_a",
                "gross_tonnage": 100.0, "international_gross_tonnage": 200.0,
                "flag_name": "라이베리아", "vessel_type": "원유운반선", "arrival_year": 2023}
        for scenario in ("N2_char_edit_one", "N5_token_order", "T1_tonnage_both_missing"):
            first = perturb.perturb_row(scenario, dict(base))
            second = perturb.perturb_row(scenario, dict(base))
            self.assertEqual(first, second, scenario)


if __name__ == "__main__":
    unittest.main()
