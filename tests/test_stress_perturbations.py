"""Tests for the v11-stress-oof-1 perturbations and the G0 clean-replay gate."""

import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from agent import stress_perturbations as perturb

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments" / "protocols" / "v11_stress_oof_v1.json"
PROTOCOL_SHA256 = "4fc3f8ba3fead033185470caf3b86452b32eb13e4db3df2515df22b3f11fc86a"


def row(sample_id="TRAIN_00001", name="MORNING COURIER 7", call_sign="CS_aaa",
        gross=1000.0, intl=1200.0, flag="라이베리아", vessel_type="원유운반선", year=2023):
    return {"sample_id": sample_id, "vessel_name": name, "call_sign": call_sign,
            "gross_tonnage": gross, "international_gross_tonnage": intl,
            "flag_name": flag, "vessel_type": vessel_type, "arrival_year": year}


def frame(rows):
    return pd.DataFrame(rows)


class ProtocolHashTest(unittest.TestCase):
    def test_frozen_protocol_file_is_unchanged_and_lists_the_implemented_scenarios(self):
        digest = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, PROTOCOL_SHA256)
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(protocol["protocol_version"], perturb.PROTOCOL_VERSION)
        self.assertEqual(tuple(s["id"] for s in protocol["scenarios"]), perturb.SCENARIOS)
        rates = {s["id"]: s["rate"] for s in protocol["scenarios"]}
        self.assertEqual(rates["C2_callsign_collision_10pct"],
                         perturb.RATES["C2_callsign_collision_10pct"])

    def test_u_matches_the_protocol_hash_formula_exactly(self):
        text = "v11-stress-oof-1|C2_callsign_collision_10pct|pick|TRAIN_00001"
        expected = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) / 2 ** 64
        self.assertEqual(perturb.u("TRAIN_00001", "C2_callsign_collision_10pct", "pick"), expected)
        self.assertTrue(0.0 <= expected < 1.0)


class DeterminismTest(unittest.TestCase):
    def test_same_input_gives_the_same_output_for_every_scenario(self):
        base = row()
        table = [f"CS_{i:03d}" for i in range(50)]
        for scenario in perturb.SCENARIOS:
            extra = table if scenario == "C2_callsign_collision_10pct" else None
            first = perturb.perturb_row(scenario, dict(base), extra)
            second = perturb.perturb_row(scenario, dict(base), extra)
            self.assertEqual(first, second, scenario)

    def test_a_row_perturbs_identically_alone_and_inside_a_batch(self):
        rows = [row(f"TRAIN_{i:05d}", name=f"VESSEL NAME {i}", call_sign=f"CS_x{i}",
                    gross=1000.0 + i, intl=1200.0 + i) for i in range(40)]
        rows.append(row("TRAIN_00099", name="모닝쿠리어", call_sign="CS_k1"))
        table = [f"CS_{i:03d}" for i in range(50)]
        for scenario in perturb.SCENARIOS:
            extra = table if scenario == "C2_callsign_collision_10pct" else None
            batch, _, batch_results = perturb.perturb_frame(scenario, frame(rows), extra)
            for position, source in enumerate(rows):
                alone, _, single = perturb.perturb_frame(scenario, frame([source]), extra)
                self.assertEqual(single[0], batch_results[position], f"{scenario}/{position}")
                for column in ("vessel_name", "call_sign", "flag_name", "vessel_type"):
                    one, many = alone.iloc[0][column], batch.iloc[position][column]
                    same = (one == many) or (isinstance(one, float) and isinstance(many, float)
                                             and math.isnan(one) and math.isnan(many))
                    self.assertTrue(same, f"{scenario}/{column}/{position}")
            # Reversing the frame must not move any row's own perturbation.
            reversed_frame, _, reversed_results = perturb.perturb_frame(
                scenario, frame(rows[::-1]), extra)
            self.assertEqual(reversed_results[::-1], batch_results, scenario)

    def test_a_label_column_in_the_input_is_refused(self):
        with self.assertRaises(AssertionError):
            perturb.perturb_row("C1_callsign_removed", dict(row(), target_vessel_id="VESSEL_1"))
        with self.assertRaises(AssertionError):
            perturb.perturb_frame("C1_callsign_removed",
                                  frame([dict(row(), target_vessel_id="VESSEL_1")]))


class MissingEncodingTest(unittest.TestCase):
    def test_removed_values_are_written_as_nan(self):
        cases = {"C1_callsign_removed": ["call_sign"], "F1_flag_missing": ["flag_name"],
                 "F2_vessel_type_missing": ["vessel_type"],
                 "T1_tonnage_both_missing": list(perturb.TONNAGE_COLUMNS)}
        for scenario, columns in cases.items():
            out, audit, _ = perturb.perturb_frame(scenario, frame([row()]))
            for column in columns:
                value = out.iloc[0][column]
                self.assertTrue(isinstance(value, float) and math.isnan(value),
                                f"{scenario}/{column} -> {value!r}")
                self.assertTrue(pd.isna(value))
            self.assertEqual(audit["changed_rows"], 1)

    def test_an_already_missing_value_is_not_counted_as_a_change(self):
        out, audit, _ = perturb.perturb_frame(
            "F1_flag_missing", frame([row(flag=np.nan), row(flag=None), row(flag="")]))
        self.assertEqual(audit["changed_rows"], 0)
        self.assertTrue(audit["not_applicable"])
        self.assertEqual(audit["rule_eligible_rows"], 3)


class EligibilityTest(unittest.TestCase):
    def test_n2_needs_four_non_space_characters(self):
        self.assertFalse(perturb.rule_eligible("N2_char_edit_one", row(name="A B")))
        self.assertTrue(perturb.rule_eligible("N2_char_edit_one", row(name="AB CD")))
        self.assertFalse(perturb.rule_eligible("N2_char_edit_one", row(name="ABC")))
        self.assertTrue(perturb.rule_eligible("N2_char_edit_one", row(name="ABCD")))
        self.assertFalse(perturb.rule_eligible("N2_char_edit_one", row(name=None)))

    def test_n2_makes_exactly_one_edit_and_never_touches_digits(self):
        for index in range(60):
            name = f"ALPHA BRAVO {index}"
            result = perturb.perturb_row("N2_char_edit_one", row(f"TRAIN_{index:05d}", name=name))
            new = result["changes"].get("vessel_name")
            if new is None:
                continue
            self.assertIn(len(new), (len(name) - 1, len(name)))
            self.assertEqual("".join(c for c in new if c.isdigit()),
                             "".join(c for c in name if c.isdigit()))

    def test_n2_korean_deletes_a_syllable_or_transposes_two(self):
        long_name = perturb.perturb_row("N2_char_edit_one", row(name="모닝쿠리어"))
        self.assertEqual(len(long_name["changes"]["vessel_name"]), 4)
        short = perturb.perturb_row("N2_char_edit_one", row(name="모닝호"))
        self.assertFalse(perturb.rule_eligible("N2_char_edit_one", row(name="모닝호")))
        self.assertEqual(short["changes"], {})

    def test_n4_needs_a_standalone_integer_token_and_rewrites_the_last_one(self):
        self.assertFalse(perturb.rule_eligible("N4_number_format", row(name="MORNING COURIER")))
        self.assertFalse(perturb.rule_eligible("N4_number_format", row(name="HYUNDAI7")))
        self.assertTrue(perturb.rule_eligible("N4_number_format", row(name="STAR 3 7")))
        self.assertFalse(perturb.rule_eligible("N4_number_format", row(name="모닝 7")))
        result = perturb.perturb_row("N4_number_format", row(name="STAR 3 7"))
        self.assertEqual(result["changes"]["vessel_name"], "STAR 3 NO.7")

    def test_n5_needs_two_tokens_and_moves_the_first_to_the_end(self):
        self.assertFalse(perturb.rule_eligible("N5_token_order", row(name="DILAM")))
        self.assertTrue(perturb.rule_eligible("N5_token_order", row(name="MORNING COURIER")))
        self.assertFalse(perturb.rule_eligible("N5_token_order", row(name="모닝 쿠리어")))
        result = perturb.perturb_row("N5_token_order", row(name="A B C"))
        self.assertEqual(result["changes"]["vessel_name"], "B C A")

    def test_n1_removes_non_korean_spacing_and_splits_a_korean_name_once(self):
        self.assertEqual(perturb.perturb_row("N1_spacing", row(name="MORNING COURIER"))
                         ["changes"]["vessel_name"], "MORNINGCOURIER")
        self.assertEqual(perturb.perturb_row("N1_spacing", row(name="DILAM"))["changes"], {})
        self.assertEqual(perturb.perturb_row("N1_spacing", row(name="모닝쿠리어"))
                         ["changes"]["vessel_name"], "모닝 쿠리어")
        self.assertEqual(perturb.perturb_row("N1_spacing", row(name="모"))["changes"], {})

    def test_n3_affix_rules_per_language(self):
        self.assertEqual(perturb.perturb_row("N3_affix", row(name="DILAM"))
                         ["changes"]["vessel_name"], "M/V DILAM")
        self.assertEqual(perturb.perturb_row("N3_affix", row(name="M/V DILAM"))["changes"], {})
        self.assertEqual(perturb.perturb_row("N3_affix", row(name="MV DILAM"))["changes"], {})
        self.assertEqual(perturb.perturb_row("N3_affix", row(name="모닝쿠리어"))
                         ["changes"]["vessel_name"], "모닝쿠리어호")
        self.assertEqual(perturb.perturb_row("N3_affix", row(name="모닝쿠리어호"))
                         ["changes"]["vessel_name"], "모닝쿠리어")

    def test_tonnage_rules(self):
        self.assertFalse(perturb.rule_eligible("T2_tonnage_noise_20pct",
                                               row(gross=np.nan, intl=np.nan)))
        self.assertTrue(perturb.rule_eligible("T2_tonnage_noise_20pct", row(gross=5.0, intl=np.nan)))
        noisy = perturb.perturb_row("T2_tonnage_noise_20pct", row(gross=100.0, intl=np.nan))
        self.assertEqual(set(noisy["changes"]), {"gross_tonnage"})
        self.assertIn(round(noisy["changes"]["gross_tonnage"], 6), (120.0, 80.0))
        both = perturb.perturb_row("T2_tonnage_noise_20pct", row(gross=100.0, intl=200.0))
        factor = both["changes"]["gross_tonnage"] / 100.0
        self.assertAlmostEqual(both["changes"]["international_gross_tonnage"] / 200.0, factor)
        self.assertFalse(perturb.rule_eligible("T3_tonnage_swap", row(gross=100.0, intl=100.0)))
        self.assertFalse(perturb.rule_eligible("T3_tonnage_swap", row(gross=100.0, intl=np.nan)))
        swapped = perturb.perturb_row("T3_tonnage_swap", row(gross=100.0, intl=200.0))
        self.assertEqual(swapped["changes"], {"gross_tonnage": 200.0,
                                              "international_gross_tonnage": 100.0})

    def test_s0_changes_nothing_and_is_not_reported_as_not_applicable(self):
        out, audit, _ = perturb.perturb_frame("S0_clean_replay", frame([row(), row("TRAIN_2")]))
        self.assertEqual(audit["changed_rows"], 0)
        self.assertFalse(audit["not_applicable"])
        pd.testing.assert_frame_equal(out, frame([row(), row("TRAIN_2")]))

    def test_an_unknown_scenario_is_refused(self):
        with self.assertRaises(ValueError):
            perturb.perturb_row("X9_unknown", row())


class CollisionTest(unittest.TestCase):
    def test_c2_selects_about_the_protocol_rate_of_rows(self):
        rows = [row(f"TRAIN_{i:05d}") for i in range(4000)]
        selected = sum(perturb.rule_eligible("C2_callsign_collision_10pct", r) for r in rows)
        self.assertLess(abs(selected / len(rows) - 0.10), 0.02)

    def test_c2_picks_from_the_training_table_and_never_the_rows_own_call_sign(self):
        table = [f"CS_train{i:03d}" for i in range(120)]
        picked_values = set()
        changed = 0
        for index in range(600):
            sample_id = f"TRAIN_{index:05d}"
            # The validation row's own call sign is group-disjoint from the training table.
            candidate = row(sample_id, call_sign=f"CS_valid{index:03d}")
            result = perturb.perturb_row("C2_callsign_collision_10pct", candidate, table)
            if "call_sign" not in result["changes"]:
                continue
            changed += 1
            value = result["changes"]["call_sign"]
            self.assertIn(value, table)
            self.assertNotEqual(value, candidate["call_sign"])
            picked_values.add(value)
        self.assertGreater(changed, 30)
        self.assertGreater(len(picked_values), 10)

    def test_c2_uses_the_sorted_table_and_the_pick_salt(self):
        table = ["CS_b", "CS_a", "CS_c"]
        expected = sorted(table)[int(perturb.u("TRAIN_X", "C2_callsign_collision_10pct", "pick") * 3)]
        result = perturb.perturb_row("C2_callsign_collision_10pct", row("TRAIN_X"), table)
        if result["rule_eligible"]:
            self.assertEqual(result["changes"]["call_sign"], expected)
        self.assertEqual(sorted(table), ["CS_a", "CS_b", "CS_c"])

    def test_c2_without_a_training_table_is_refused(self):
        eligible = next(r for r in (row(f"TRAIN_{i:05d}") for i in range(200))
                        if perturb.rule_eligible("C2_callsign_collision_10pct", r))
        with self.assertRaises(ValueError):
            perturb.perturb_row("C2_callsign_collision_10pct", eligible, None)


class MixedScenarioTest(unittest.TestCase):
    def test_m1_applies_independent_components_at_their_own_rates(self):
        rows = [row(f"TRAIN_{i:05d}", name=f"ALPHA BRAVO {i}") for i in range(4000)]
        _, audit, results = perturb.perturb_frame("M1_mixed", frame(rows))
        counts = {key: sum(op == f"m1_{key}" for r in results for op in r["operations"])
                  for key in ("c1", "t1", "f1")}
        self.assertLess(abs(counts["c1"] / len(rows) - 0.30), 0.03)
        self.assertLess(abs(counts["t1"] / len(rows) - 0.20), 0.03)
        self.assertLess(abs(counts["f1"] / len(rows) - 0.20), 0.03)
        name_rows = sum(any(op.startswith("m1_N") for op in r["operations"]) for r in results)
        self.assertLess(abs(name_rows / len(rows) - 0.30), 0.03)
        self.assertFalse(audit["not_applicable"])
        self.assertNotIn("m1_c2", audit["operation_counts"])

    def test_m1_name_component_reuses_the_standalone_scenario_result(self):
        for index in range(300):
            sample_id = f"TRAIN_{index:05d}"
            candidate = row(sample_id, name=f"ALPHA BRAVO {index}")
            result = perturb.perturb_row("M1_mixed", candidate)
            label = next((op for op in result["operations"] if op.startswith("m1_N")), None)
            if label is None or label.endswith("_no_change"):
                continue
            scenario = {"N1": "N1_spacing", "N2": "N2_char_edit_one",
                        "N3": "N3_affix"}[label[3:5]]
            standalone = perturb.perturb_row(scenario, candidate)
            if "vessel_name" in standalone["changes"] and "vessel_name" in result["changes"]:
                self.assertEqual(result["changes"]["vessel_name"],
                                 standalone["changes"]["vessel_name"], label)


class G0GateTest(unittest.TestCase):
    def _context_and_payload(self, stored_pred, replay_pred):
        from scripts.run_v11_stress_oof import g0_fold_check
        frozen = {"fold": 0, "split_name": "mlp_oof_fold0",
                  "clean_candidates_sha256": "c" * 64, "clean_features_sha256": "f" * 64,
                  "model_file_sha256": "m" * 64, "clean_correct": "2/2"}
        context = {"fold": 0, "frozen": frozen, "reloaded_model_sha256": "m" * 64,
                   "clean_cases": [{"sample_id": f"TRAIN_{i:05d}", "final_pred_id": value}
                                   for i, value in enumerate(stored_pred)]}
        payload = {"prediction_input_hashes": {"candidates_sha256": "c" * 64,
                                               "features_sha256": "f" * 64},
                   "metrics": {"all": {"correct": 2, "rows": 2}},
                   "cases": [{"sample_id": f"TRAIN_{i:05d}", "final_pred_id": value}
                             for i, value in enumerate(replay_pred)]}
        return g0_fold_check(context, payload)

    def test_identical_replay_passes_the_gate(self):
        check = self._context_and_payload(["VESSEL_1", "VESSEL_2"], ["VESSEL_1", "VESSEL_2"])
        self.assertTrue(check["all_rows_match"])
        self.assertTrue(check["candidates_sha256_matches"])
        self.assertEqual(check["final_pred_mismatched"], 0)

    def test_one_mismatched_row_fails_the_gate(self):
        check = self._context_and_payload(["VESSEL_1", "VESSEL_2"], ["VESSEL_1", "VESSEL_9"])
        self.assertFalse(check["all_rows_match"])
        self.assertEqual(check["final_pred_mismatched"], 1)
        self.assertEqual(check["final_pred_matched"], 1)

    def test_a_changed_candidate_hash_fails_the_gate(self):
        from scripts.run_v11_stress_oof import g0_fold_check
        frozen = {"fold": 0, "clean_candidates_sha256": "0" * 64, "clean_features_sha256": "f" * 64,
                  "model_file_sha256": "m" * 64, "clean_correct": "1/1"}
        context = {"fold": 0, "frozen": frozen, "reloaded_model_sha256": "m" * 64,
                   "clean_cases": [{"sample_id": "TRAIN_00000", "final_pred_id": "VESSEL_1"}]}
        payload = {"prediction_input_hashes": {"candidates_sha256": "c" * 64,
                                               "features_sha256": "f" * 64},
                   "metrics": {"all": {"correct": 1, "rows": 1}},
                   "cases": [{"sample_id": "TRAIN_00000", "final_pred_id": "VESSEL_1"}]}
        check = g0_fold_check(context, payload)
        self.assertFalse(check["candidates_sha256_matches"])
        self.assertTrue(check["all_rows_match"])


if __name__ == "__main__":
    unittest.main()
