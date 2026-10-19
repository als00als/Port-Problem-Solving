"""Tests for the frozen target-group OOF protocol and paired decision rule."""

import unittest
import importlib.util
from pathlib import Path

import pandas as pd

from agent.mlp_oof_protocol import (build_group_oof_split, exact_mcnemar_pvalue,
                                    holm_adjust, normalized_name_key, summarize_paired_oof,
                                    validate_group_oof_split)
from agent.retrain_config import config_for, validate_config

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("sklearn") and importlib.util.find_spec("scipy"),
                     "run OOF statistics tests in competition Python 3.11 environment")
class GroupOofSplitTest(unittest.TestCase):
    def test_runner_writes_baseline_diagnosis_at_worker_expected_path(self):
        from scripts.run_v11_mlp_oof_experiment import diagnosis_output_path
        self.assertEqual(diagnosis_output_path(Path("run"), "mlp_oof_fold0"),
                         Path("run") / "diagnose_mlp_oof_fold0.json")

    def test_worker_case_artifact_path_matches_retrain_setting_name(self):
        from scripts.v11_improvement_worker import case_file_path
        self.assertEqual(case_file_path(Path("run"), "mlp_oof_fold0",
                                        "ranker_family__mlp_listwise_v1_cpu"),
                         Path("run") / "local_only" /
                         "cases_mlp_oof_fold0_ranker_family__mlp_listwise_v1_cpu.json")

    def test_mlp_family_is_a_single_allowed_ranker_change(self):
        config = config_for("ranker_family", "mlp_listwise_v1_cpu")
        self.assertEqual(validate_config(config), ["ranker_family"])

    def test_split_covers_each_row_once_without_target_overlap(self):
        rows = []
        for group in range(25):
            rows.extend([
                {"sample_id": f"S{group}_E", "target_vessel_id": f"V{group}", "is_korean": 0},
                {"sample_id": f"S{group}_K", "target_vessel_id": f"V{group}", "is_korean": 1},
            ])
        frame = pd.DataFrame(rows)
        split_a = build_group_oof_split(frame, seed=20261019)
        split_b = build_group_oof_split(frame, seed=20261019)
        self.assertEqual(split_a, split_b)
        validate_group_oof_split(frame, split_a)
        self.assertEqual(len(split_a["folds"]), 5)
        self.assertEqual(sum(len(fold["valid_sample_ids"]) for fold in split_a["folds"]), len(frame))

    @unittest.skipUnless((ROOT / "data/open/train.csv").is_file() and
                         (ROOT / "experiments/splits/agent_v11_mlp_listwise_group5_seed20261019.json").is_file(),
                         "requires official data and frozen MLP OOF split")
    def test_worker_loads_frozen_fold_without_target_group_leakage(self):
        from scripts.v11_improvement_worker import load_split
        fit, valid, catalog, audit = load_split("mlp_oof_fold0")
        self.assertEqual(len(fit) + len(valid), 12222)
        self.assertEqual(audit["target_group_overlap"], 0)
        self.assertEqual(audit["unseen_target_validation_rows"], len(valid))
        self.assertEqual(len(catalog), 78595)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "run exact tests in competition Python 3.11 environment")
class PairedOofStatisticsTest(unittest.TestCase):
    def test_name_normalization_is_deterministic_and_removes_formatting(self):
        self.assertEqual(normalized_name_key("  오션-스타 2호 "), "오션스타2호")
        self.assertEqual(normalized_name_key(None), "")

    def test_exact_mcnemar_handles_zero_and_one_sided_discordance(self):
        self.assertEqual(exact_mcnemar_pvalue(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_pvalue(4, 0), 0.125)
        self.assertEqual(exact_mcnemar_pvalue(3, 3), 1.0)

    def test_holm_adjustment_is_monotone_and_capped(self):
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_language_paired_counts_and_korean_name_strata(self):
        baseline, mlp = [], []
        outcomes = [
            (True, False, True, True),   # Korean baseline-only win, name seen
            (False, True, True, False),  # Korean MLP-only win, name unseen
            (False, True, False, False), # non-Korean MLP-only win
            (True, False, False, True),  # non-Korean baseline-only win
        ]
        for i, (base_ok, mlp_ok, korean, name_seen) in enumerate(outcomes):
            common = {"sample_id": str(i), "truth_id": "T", "is_korean": korean,
                      "seen_target": False, "query_name_seen_in_train": name_seen}
            baseline.append({**common, "final_correct": base_ok})
            mlp.append({**common, "final_correct": mlp_ok})
        result = summarize_paired_oof(baseline, mlp)
        self.assertEqual(result["korean"]["mlp_only_correct"], 1)
        self.assertEqual(result["korean"]["baseline_only_correct"], 1)
        self.assertEqual(result["non_korean"]["mlp_only_correct"], 1)
        self.assertEqual(result["korean_name_strata"]["train_name_unseen"]["rows"], 1)
        self.assertEqual(result["korean_name_strata"]["train_name_seen"]["rows"], 1)
        self.assertEqual(result["decision"]["language_net_positive"], False)
        with self.assertRaises(ValueError):
            summarize_paired_oof(baseline, mlp[::-1])


if __name__ == "__main__":
    unittest.main()
