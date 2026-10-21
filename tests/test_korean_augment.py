"""Tests for the v11-korean-augment-1 augmentation, held-out scenarios and run gates."""

import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from agent import korean_augment as aug
from agent import stress_heldout as heldout
from agent import stress_perturbations as perturb

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments" / "protocols" / "v11_korean_augment_v1.json"
PROTOCOL_SHA256 = "b3bf281ea0acb37f4c1a30d8ec7586476294ce7d5677e74e51fcb944f458f85c"


def train_row(sample_id="TRAIN_00001", name="모닝쿠리어", gross=1000.0, intl=1200.0,
              target="VESSEL_00001", call_sign="CS_a"):
    return {"sample_id": sample_id, "vessel_name": name, "call_sign": call_sign,
            "gross_tonnage": gross, "international_gross_tonnage": intl,
            "flag_name": "라이베리아", "vessel_type": "원유운반선", "arrival_year": 2023,
            "target_vessel_id": target}


def test_row(sample_id="TRAIN_00001", name="모닝쿠리어", gross=1000.0, intl=1200.0):
    row = train_row(sample_id, name, gross, intl)
    row.pop("target_vessel_id")
    return row


class ProtocolTest(unittest.TestCase):
    def test_protocol_hash_and_constants(self):
        self.assertEqual(hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(), PROTOCOL_SHA256)
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(protocol["protocol_version"], aug.PROTOCOL_VERSION)
        self.assertEqual(set(protocol["augmentation"]["variants"]), set(aug.VARIANTS))
        self.assertEqual(protocol["augmentation"]["copies_per_korean_row"],
                         aug.COPIES_PER_KOREAN_ROW)
        self.assertEqual(aug.A2_FACTORS, (0.85, 0.9, 1.1, 1.15))

    def test_hash_formula_matches_the_protocol_and_differs_from_the_stress_hash(self):
        text = "v11-korean-augment-1|A2|factor|TRAIN_00001"
        expected = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) / 2 ** 64
        self.assertEqual(aug.a("TRAIN_00001", "A2", "factor"), expected)
        self.assertNotEqual(aug.a("TRAIN_00001", "A2", "factor"),
                            perturb.u("TRAIN_00001", "A2", "factor"))

    def test_held_out_scenarios_reuse_the_stress_protocol_string(self):
        out, audit, _ = heldout.perturb_frame("T4_gross_only_missing",
                                              pd.DataFrame([test_row()]))
        self.assertEqual(audit["protocol_string_for_hash"], "v11-stress-oof-1")
        self.assertEqual(perturb.PROTOCOL_VERSION, "v11-stress-oof-1")
        for scenario in heldout.SCENARIOS:
            self.assertNotIn(scenario, perturb.SCENARIOS)


class AugmentRuleTest(unittest.TestCase):
    def test_determinism_for_every_variant(self):
        row = train_row()
        for variant in aug.VARIANTS[1:]:
            first = aug.copy_for_row(variant, dict(row))
            second = aug.copy_for_row(variant, dict(row))
            self.assertEqual(first, second, variant)

    def test_exactly_one_copy_per_korean_row_and_none_for_other_rows(self):
        rows = [train_row(f"TRAIN_{i:05d}", name="모닝쿠리어" if i % 3 == 0 else "MORNING",
                          target=f"VESSEL_{i:05d}") for i in range(30)]
        korean = sum(1 for r in rows if aug.is_korean_name(r["vessel_name"]))
        for variant in aug.VARIANTS[1:]:
            copies, audit = aug.augment_rows(variant, rows)
            self.assertEqual(len(copies), korean, variant)
            self.assertEqual(audit["korean_prep_rows"], korean)
            self.assertEqual(audit["copies_added"], korean)
            self.assertTrue(all(aug.is_korean_name(c["vessel_name"]) for c in copies))
            self.assertTrue(audit["labels_preserved"])

    def test_labels_are_preserved_on_every_copy(self):
        rows = [train_row(f"TRAIN_{i:05d}", target=f"VESSEL_{i:05d}") for i in range(10)]
        for variant in aug.VARIANTS[1:]:
            copies, _ = aug.augment_rows(variant, rows)
            self.assertEqual([c["target_vessel_id"] for c in copies],
                             [r["target_vessel_id"] for r in rows], variant)

    def test_outer_validation_ids_are_refused(self):
        rows = [train_row("TRAIN_00001"), train_row("TRAIN_00002")]
        with self.assertRaisesRegex(AssertionError, "외부 검증"):
            aug.augment_rows("A1_tonnage_missing", rows, forbidden_sample_ids=["TRAIN_00002"])
        copies, audit = aug.augment_rows("A1_tonnage_missing", rows,
                                         forbidden_sample_ids=["TRAIN_99999"])
        self.assertEqual(len(copies), 2)
        self.assertEqual(audit["outer_validation_ids_present"], 0)

    def test_a_row_without_a_label_or_a_non_korean_row_is_refused(self):
        with self.assertRaises(AssertionError):
            aug.copy_for_row("A1_tonnage_missing", test_row())
        with self.assertRaises(AssertionError):
            aug.copy_for_row("A1_tonnage_missing", train_row(name="MORNING COURIER"))
        with self.assertRaises(ValueError):
            aug.copy_for_row("A0_none", train_row())
        with self.assertRaises(ValueError):
            aug.copy_for_row("A9_unknown", train_row())

    def test_a1_sets_both_tonnage_values_to_nan(self):
        made = aug.copy_for_row("A1_tonnage_missing", train_row())
        for column in aug.TONNAGE_COLUMNS:
            self.assertTrue(math.isnan(made["row"][column]))
        self.assertTrue(made["changed"])
        already = aug.copy_for_row("A1_tonnage_missing",
                                   train_row(gross=np.nan, intl=np.nan))
        self.assertFalse(already["changed"])
        self.assertIn("a1_already_missing", already["operations"])

    def test_a2_uses_only_the_four_protocol_factors_on_present_values(self):
        seen = set()
        for i in range(200):
            made = aug.copy_for_row("A2_tonnage_noise", train_row(f"TRAIN_{i:05d}"))
            factor = made["row"]["gross_tonnage"] / 1000.0
            seen.add(round(factor, 6))
            self.assertAlmostEqual(made["row"]["international_gross_tonnage"] / 1200.0, factor)
        self.assertEqual(seen, {0.85, 0.9, 1.1, 1.15})
        one_side = aug.copy_for_row("A2_tonnage_noise", train_row(intl=np.nan))
        self.assertEqual(set(one_side["changes"]), {"gross_tonnage"})
        none = aug.copy_for_row("A2_tonnage_noise", train_row(gross=np.nan, intl=np.nan))
        self.assertFalse(none["changed"])

    def test_a3_deletes_a_syllable_or_inserts_one_space(self):
        deletes = inserts = 0
        for i in range(200):
            sample_id = f"TRAIN_{i:05d}"
            made = aug.copy_for_row("A3_name_noise", train_row(sample_id, name="모닝쿠리어"))
            new = made["row"]["vessel_name"]
            if aug.a(sample_id, "A3", "op") < 0.5:
                deletes += 1
                self.assertEqual(len(new), 4, new)
                self.assertNotIn(" ", new)
            else:
                inserts += 1
                self.assertEqual(len(new), 6, new)
                self.assertEqual(new.count(" "), 1)
                self.assertNotEqual(new[0], " ")
                self.assertNotEqual(new[-1], " ")
        self.assertGreater(deletes, 50)
        self.assertGreater(inserts, 50)

    def test_a3_two_syllable_name_always_takes_the_insertion_branch(self):
        for i in range(60):
            made = aug.copy_for_row("A3_name_noise", train_row(f"TRAIN_{i:05d}", name="모닝"))
            self.assertEqual(made["row"]["vessel_name"], "모 닝")
            self.assertIn("a3_insert_space", made["operations"])

    def test_a3_three_syllable_name_covers_both_branches_with_a_clamped_offset(self):
        lengths = set()
        for i in range(120):
            sample_id = f"TRAIN_{i:05d}"
            made = aug.copy_for_row("A3_name_noise", train_row(sample_id, name="모닝호"))
            new = made["row"]["vessel_name"]
            lengths.add(len(new))
            if " " in new:
                self.assertIn(new.index(" "), (1, 2))
            else:
                self.assertEqual(len(new), 2)
        self.assertEqual(lengths, {2, 4})

    def test_a3_single_syllable_name_is_copied_unchanged(self):
        made = aug.copy_for_row("A3_name_noise", train_row(name="호"))
        self.assertFalse(made["changed"])
        self.assertIn("a3_name_too_short", made["operations"])

    def test_a4_delegates_to_a1_a2_or_a3_with_that_variant_own_salts(self):
        counts = {"A1": 0, "A2": 0, "A3": 0}
        for i in range(300):
            sample_id = f"TRAIN_{i:05d}"
            row = train_row(sample_id)
            made = aug.copy_for_row("A4_mixed", row)
            chosen = next(op for op in made["operations"] if op.startswith("a4_"))[3:]
            counts[chosen] += 1
            standalone = aug.copy_for_row({"A1": "A1_tonnage_missing", "A2": "A2_tonnage_noise",
                                           "A3": "A3_name_noise"}[chosen], dict(row))
            self.assertEqual(made["changes"], standalone["changes"], f"{sample_id}/{chosen}")
        for value in counts.values():
            self.assertGreater(value, 50)


class HeldOutScenarioTest(unittest.TestCase):
    def test_t2b_uses_only_the_two_protocol_factors(self):
        seen = set()
        for i in range(120):
            out = heldout.perturb_row("T2b_tonnage_noise_30pct", test_row(f"TRAIN_{i:05d}"))
            seen.add(round(out["changes"]["gross_tonnage"] / 1000.0, 6))
        self.assertEqual(seen, {1.3, 0.7})
        self.assertFalse(heldout.rule_eligible("T2b_tonnage_noise_30pct",
                                               test_row(gross=np.nan, intl=np.nan)))

    def test_t4_clears_only_gross_tonnage_and_needs_both_present(self):
        out = heldout.perturb_row("T4_gross_only_missing", test_row())
        self.assertEqual(set(out["changes"]), {"gross_tonnage"})
        self.assertTrue(math.isnan(out["changes"]["gross_tonnage"]))
        self.assertFalse(heldout.rule_eligible("T4_gross_only_missing", test_row(intl=np.nan)))
        self.assertFalse(heldout.rule_eligible("T4_gross_only_missing", test_row(gross=np.nan)))

    def test_n6_duplicates_one_syllable_in_place(self):
        for i in range(60):
            out = heldout.perturb_row("N6_syllable_duplicate", test_row(f"TRAIN_{i:05d}",
                                                                        name="모닝쿠리어"))
            new = out["changes"]["vessel_name"]
            self.assertEqual(len(new), 6)
            self.assertTrue(any(new[j] == new[j + 1] for j in range(len(new) - 1)))
        self.assertFalse(heldout.rule_eligible("N6_syllable_duplicate", test_row(name="호")))
        self.assertFalse(heldout.rule_eligible("N6_syllable_duplicate", test_row(name="ABCD")))

    def test_n7_swaps_only_pairs_of_different_syllables(self):
        for i in range(60):
            out = heldout.perturb_row("N7_korean_transpose", test_row(f"TRAIN_{i:05d}",
                                                                      name="모닝쿠리어"))
            new = out["changes"]["vessel_name"]
            self.assertEqual(sorted(new), sorted("모닝쿠리어"))
            self.assertNotEqual(new, "모닝쿠리어")
        # Every adjacent pair identical -> not eligible, and no change is produced.
        self.assertFalse(heldout.rule_eligible("N7_korean_transpose", test_row(name="가가")))
        self.assertFalse(heldout.rule_eligible("N7_korean_transpose", test_row(name="가가가")))
        out = heldout.perturb_row("N7_korean_transpose", test_row(name="가가"))
        self.assertEqual(out["changes"], {})
        # A mixed name keeps only the differing pair as a candidate.
        self.assertTrue(heldout.rule_eligible("N7_korean_transpose", test_row(name="가가나")))

    def test_held_out_scenarios_are_deterministic_and_batch_independent(self):
        rows = [test_row(f"TRAIN_{i:05d}", name="모닝쿠리어" if i % 2 else f"ALPHA {i}")
                for i in range(40)]
        frame = pd.DataFrame(rows)
        for scenario in heldout.SCENARIOS:
            batch, audit, results = heldout.perturb_frame(scenario, frame)
            again, audit2, results2 = heldout.perturb_frame(scenario, frame)
            self.assertEqual(results, results2, scenario)
            self.assertEqual(audit, audit2, scenario)
            for position, row in enumerate(rows):
                alone, _, single = heldout.perturb_frame(scenario, pd.DataFrame([row]))
                self.assertEqual(single[0], results[position], f"{scenario}/{position}")
            reversed_out, _, reversed_results = heldout.perturb_frame(
                scenario, pd.DataFrame(rows[::-1]))
            self.assertEqual(reversed_results[::-1], results, scenario)

    def test_a_label_column_in_the_input_is_refused(self):
        with self.assertRaises(AssertionError):
            heldout.perturb_row("T4_gross_only_missing", train_row())
        with self.assertRaises(AssertionError):
            heldout.perturb_frame("T4_gross_only_missing", pd.DataFrame([train_row()]))

    def test_an_unknown_scenario_is_refused(self):
        with self.assertRaises(ValueError):
            heldout.perturb_row("X9_unknown", test_row())


class TrainingParityTest(unittest.TestCase):
    def test_augment_none_leaves_the_matrix_identical_on_synthetic_data(self):
        import importlib.util
        if not all(importlib.util.find_spec(n) for n in ("lightgbm", "sklearn", "rapidfuzz")):
            self.skipTest("경쟁 환경에서 실행하세요")
        from agent import retrain_runtime as runtime
        from agent.retrain_config import baseline_config
        rows, catalog = [], []
        for group in range(10):
            vessel = f"VESSEL_{group:05d}"
            catalog.append({"vessel_id": vessel, "vessel_type": "Tanker",
                            "vessel_name": f"SHIP {group}", "size": 1000 + group, "unit": "DWT",
                            "deadweight_tonnage": 1000 + group, "gross_tonnage": 900 + group,
                            "flag_name": "Liberia"})
            for copy in range(2):
                rows.append(train_row(f"TRAIN_{group:03d}{copy}", name=f"SHIP {group}",
                                      gross=900 + group, intl=900 + group, target=vessel))
        train = pd.DataFrame(rows)
        catalog_frame = pd.DataFrame(catalog)
        index = runtime.ml.CatalogIndex(catalog_frame)
        config = baseline_config()
        with runtime.v11_globals(config):
            first = runtime.training_matrix(train, catalog_frame, index, config, inner=2)
            second = runtime.training_matrix(train, catalog_frame, index, config, inner=2,
                                             augment=None)
        self.assertEqual(runtime.array_digest(*first[:3]), runtime.array_digest(*second[:3]))
        self.assertNotIn("augmentation", first[4])
        self.assertNotIn("augmentation", second[4])

    def test_an_augmenter_adds_queries_and_records_an_audit(self):
        import importlib.util
        if not all(importlib.util.find_spec(n) for n in ("lightgbm", "sklearn", "rapidfuzz")):
            self.skipTest("경쟁 환경에서 실행하세요")
        from agent import retrain_runtime as runtime
        from agent.retrain_config import baseline_config
        rows, catalog = [], []
        for group in range(10):
            vessel = f"VESSEL_{group:05d}"
            catalog.append({"vessel_id": vessel, "vessel_type": "Tanker",
                            "vessel_name": f"SHIP {group}", "size": 1000 + group, "unit": "DWT",
                            "deadweight_tonnage": 1000 + group, "gross_tonnage": 900 + group,
                            "flag_name": "Liberia"})
            for copy in range(2):
                rows.append(train_row(f"TRAIN_{group:03d}{copy}", name="모닝쿠리어",
                                      gross=900 + group, intl=900 + group, target=vessel))
        train = pd.DataFrame(rows)
        catalog_frame = pd.DataFrame(catalog)
        index = runtime.ml.CatalogIndex(catalog_frame)
        config = baseline_config()
        with runtime.v11_globals(config):
            plain = runtime.training_matrix(train, catalog_frame, index, config, inner=2)
            grown = runtime.training_matrix(train, catalog_frame, index, config, inner=2,
                                            augment=aug.make_augmenter("A1_tonnage_missing"))
        self.assertGreater(len(grown[2]), len(plain[2]))
        self.assertIn("augmentation", grown[4])
        merged = aug.merge_audits(grown[4]["augmentation"])
        self.assertEqual(merged["copies_added"], len(train))
        self.assertTrue(merged["labels_preserved"])
        self.assertEqual(aug.make_augmenter("A0_none"), None)


class RunGateTest(unittest.TestCase):
    def test_g0_mismatch_is_detected(self):
        from scripts.run_v11_korean_augment import g0_check
        frozen = {"clean_candidates_sha256": "c" * 64, "clean_features_sha256": "f" * 64}
        details = [{"sample_id": "TRAIN_00000", "final_pred_id": "VESSEL_1",
                    "candidates": np.array([0, 1]), "features": np.zeros((2, 2), dtype=np.float32)}]
        from scripts.v11_improvement_worker import prediction_input_hashes
        hashes = prediction_input_hashes(details)
        context = {"fold": 0, "frozen": dict(frozen, **{
            "clean_candidates_sha256": hashes["candidates_sha256"],
            "clean_features_sha256": hashes["features_sha256"]}),
            "clean_cases": [{"sample_id": "TRAIN_00000", "final_pred_id": "VESSEL_1"}]}
        self.assertTrue(g0_check(context, details)["all_rows_match"])
        moved = [dict(details[0], final_pred_id="VESSEL_9")]
        bad = g0_check(context, moved)
        self.assertFalse(bad["all_rows_match"])
        self.assertEqual(bad["final_pred_mismatched"], 1)
        hashed = g0_check({**context, "frozen": frozen}, details)
        self.assertFalse(hashed["candidates_sha256_matches"])

    def test_g2_and_robustness_gain_thresholds_match_the_protocol(self):
        from scripts.run_v11_korean_augment import GAIN_THRESHOLD, NON_KOREAN_TOLERANCE, HELD_OUT
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(GAIN_THRESHOLD, 24)
        self.assertEqual(NON_KOREAN_TOLERANCE, -3)
        self.assertIn("+24", protocol["decision_rule"]["robustness_gain"])
        self.assertIn("-3", protocol["gates"]["G2_clean_non_regression"])
        self.assertEqual(len(HELD_OUT), 6)

    def test_pairing_counts_newly_wrong_and_correct_by_language(self):
        from scripts.run_v11_korean_augment import pair_vs_a0
        a0 = {"cases": {"a": (True, True, None), "b": (False, False, "x"), "c": (True, False, None)},
              "korean": {"correct": 1}, "non_korean": {"correct": 1},
              "all": {"candidate_hits": 3}}
        variant = {"cases": {"a": (False, True, "tonnage_favored_truth_lost"),
                             "b": (True, False, None), "c": (True, False, None)},
                   "korean": {"correct": 0}, "non_korean": {"correct": 2},
                   "all": {"candidate_hits": 3}}
        paired = pair_vs_a0(variant, a0)
        self.assertEqual(paired["newly_wrong"], {"total": 1, "korean": 1, "non_korean": 0})
        self.assertEqual(paired["newly_correct"], {"total": 1, "korean": 0, "non_korean": 1})
        self.assertEqual(paired["korean_net_correct"], -1)
        self.assertEqual(paired["non_korean_net_correct"], 1)
        self.assertTrue(paired["candidate_recall_equal"])
        self.assertEqual(paired["newly_wrong_primary_error_types"],
                         {"tonnage_favored_truth_lost": 1})

    def test_n2_korean_delete_label_matches_the_stress_implementation(self):
        from scripts.run_v11_korean_augment import KOREAN_DELETE_LABEL
        result = perturb.perturb_row("N2_char_edit_one", test_row(name="모닝쿠리어"))
        self.assertIn(KOREAN_DELETE_LABEL, result["operations"])
        self.assertEqual(KOREAN_DELETE_LABEL, "n2_korean_delete_syllable")


if __name__ == "__main__":
    unittest.main()
