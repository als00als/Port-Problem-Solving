"""Tests for the frozen target-group OOF protocol and paired decision rule."""

import unittest
import importlib.util
import json
from pathlib import Path
from unittest import mock

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

    def test_gpu_runner_rejects_missing_cuda_before_creating_results(self):
        from scripts import run_v11_mlp_oof_experiment as runner
        unavailable = mock.Mock(returncode=2, stdout='{"cuda_available": false}', stderr="")
        with mock.patch.object(runner.subprocess, "run", return_value=unavailable) as process:
            with self.assertRaisesRegex(RuntimeError, "학습 작업자 Python"):
                runner.run(gpu=True)
        command = process.call_args.args[0]
        self.assertEqual(command[:2], [str(runner.PYTHON), "-c"])
        self.assertIn("torch.cuda.is_available()", command[2])

    def test_worker_case_artifact_path_matches_retrain_setting_name(self):
        from scripts.v11_improvement_worker import case_file_path
        self.assertEqual(case_file_path(Path("run"), "mlp_oof_fold0",
                                        "ranker_family__mlp_listwise_v1_cpu"),
                         Path("run") / "local_only" /
                         "cases_mlp_oof_fold0_ranker_family__mlp_listwise_v1_cpu.json")

    def test_mlp_family_is_a_single_allowed_ranker_change(self):
        config = config_for("ranker_family", "mlp_listwise_v1_cpu")
        self.assertEqual(validate_config(config), ["ranker_family"])
        gpu_config = config_for("ranker_family", "mlp_listwise_v1_cuda")
        self.assertEqual(validate_config(gpu_config), ["ranker_family"])

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

    @unittest.skipUnless((ROOT / "data/open/train.csv").is_file() and
                         (ROOT / "experiments/splits/agent_v11_mlp_listwise_cuda_group5_seed20261019.json").is_file(),
                         "requires official data and frozen CUDA MLP OOF split")
    def test_worker_loads_frozen_cuda_fold_without_target_group_leakage(self):
        from agent.mlp_oof_protocol import CUDA_PROTOCOL_VERSION
        from agent.retrain_runtime import MLP_CUDA_PROFILE
        from scripts.v11_improvement_worker import load_split
        frozen = json.loads((ROOT / "experiments/splits/agent_v11_mlp_listwise_cuda_group5_seed20261019.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(frozen["protocol_version"], CUDA_PROTOCOL_VERSION)
        self.assertEqual(frozen["ranker_profile"], json.loads(json.dumps(MLP_CUDA_PROFILE)))
        self.assertEqual(frozen["analysis_role"], "secondary_device_check")
        self.assertTrue(frozen["prior_cpu_partial_results_reported_before_cuda_freeze"])
        fit, valid, catalog, audit = load_split("mlp_cuda_oof_fold0")
        self.assertEqual(len(fit) + len(valid), 12222)
        self.assertEqual(audit["target_group_overlap"], 0)
        self.assertEqual(audit["unseen_target_validation_rows"], len(valid))
        self.assertEqual(len(catalog), 78595)

    def test_refrozen_cpu_protocol_is_marked_as_exposed_and_still_accepted_by_the_runner(self):
        import tempfile
        from scripts import freeze_v11_mlp_oof_split as freezer
        from scripts import run_v11_mlp_oof_experiment as runner
        frame = pd.DataFrame([{"sample_id": f"S{g}_{t}", "vessel_name": "X", "target_vessel_id": f"V{g}",
                               "is_korean": t} for g in range(25) for t in (0, 1)])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "refrozen.json"
            with mock.patch.object(freezer.pd, "read_csv", return_value=frame),                  mock.patch.object(freezer, "sha256_file", return_value="0" * 64):
                frozen = freezer.freeze(out, refrozen_after_exposure=True)
            plain_role = "primary_cpu_oof"
            self.assertEqual(frozen["analysis_role"], "primary_cpu_oof_refrozen_after_exposure")
            self.assertTrue(frozen["refrozen_after_partial_results_seen"])
            self.assertTrue(frozen["prior_partial_results_reported_before_freeze"])
            self.assertIn("exploratory", frozen["selection_policy"])
            self.assertEqual(set(frozen["environment"]),
                             {"python", "numpy", "pandas", "scikit_learn"})
            self.assertTrue(out.is_file())
            with self.assertRaises(FileExistsError):
                freezer.freeze(out, refrozen_after_exposure=True)
            # Re-freezing without the flag keeps the un-marked role, so the marking is explicit.
            second = Path(tmp) / "plain.json"
            with mock.patch.object(freezer.pd, "read_csv", return_value=frame),                  mock.patch.object(freezer, "sha256_file", return_value="0" * 64):
                plain = freezer.freeze(second)
            self.assertEqual(plain["analysis_role"], plain_role)
            self.assertFalse(plain["refrozen_after_partial_results_seen"])
            self.assertEqual(freezer.fold_digest(plain), freezer.fold_digest(frozen))
        # The CPU runner accepts both primary roles and refuses the secondary device protocol.
        for role in (plain_role, "primary_cpu_oof_refrozen_after_exposure"):
            self.assertEqual(runner.check_analysis_role(role, gpu=False), role)
        with self.assertRaisesRegex(ValueError, "CPU OOF"):
            runner.check_analysis_role("secondary_device_check", gpu=False)
        with self.assertRaisesRegex(ValueError, "CUDA OOF"):
            runner.check_analysis_role(plain_role, gpu=True)
        self.assertEqual(runner.check_analysis_role("secondary_device_check", gpu=True),
                         "secondary_device_check")

    def test_cuda_protocol_copies_the_cpu_folds_verbatim_instead_of_recomputing(self):
        import tempfile
        from agent.mlp_oof_protocol import CUDA_PROTOCOL_VERSION, PROTOCOL_VERSION
        from agent.retrain_runtime import MLP_CUDA_PROFILE
        from scripts import freeze_v11_mlp_oof_split as freezer
        frame = pd.DataFrame([{"sample_id": f"S{g}_{t}", "vessel_name": "X", "target_vessel_id": f"V{g}",
                               "is_korean": t} for g in range(25) for t in (0, 1)])
        with tempfile.TemporaryDirectory() as tmp:
            cpu_path = Path(tmp) / "cpu.json"
            with mock.patch.object(freezer.pd, "read_csv", return_value=frame),                  mock.patch.object(freezer, "sha256_file", return_value="0" * 64):
                cpu = freezer.freeze(cpu_path)
            # A CPU protocol whose folds differ from whatever this environment would rebuild.
            cpu["folds"][0]["valid_sample_ids"] = sorted(cpu["folds"][0]["valid_sample_ids"],
                                                         reverse=True)
            cpu_path.write_text(json.dumps(cpu, ensure_ascii=False), encoding="utf-8")
            cpu_digest = freezer.fold_digest(cpu)
            self.assertNotEqual(cpu_digest, freezer.fold_digest(
                freezer.build_group_oof_split(frame, seed=20261019)))
            out = Path(tmp) / "cuda.json"
            with mock.patch.object(freezer, "sha256_file", return_value="0" * 64):
                cuda = freezer.freeze_cuda_from_cpu(cpu_path, out)
            self.assertEqual(freezer.fold_digest(cuda), cpu_digest)
            self.assertEqual(cuda["folds"], cpu["folds"])
            self.assertEqual(cuda["protocol_version"], CUDA_PROTOCOL_VERSION)
            # The runner reads the written file, so check the profile as it was serialized.
            written = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(written["ranker_profile"], json.loads(json.dumps(MLP_CUDA_PROFILE)))
            self.assertEqual(written["folds"], cpu["folds"])
            self.assertEqual(cuda["analysis_role"], "secondary_device_check")
            self.assertTrue(cuda["folds_copied_verbatim_from_cpu_protocol"])
            self.assertEqual(cuda["derived_from"]["fold_digest"], cpu_digest)
            self.assertEqual(cuda["acceptance_gates"], cpu["acceptance_gates"])
            with self.assertRaises(FileExistsError):
                with mock.patch.object(freezer, "sha256_file", return_value="0" * 64):
                    freezer.freeze_cuda_from_cpu(cpu_path, out)
            # A CUDA protocol is not an acceptable source, and mismatched data hashes are refused.
            with self.assertRaisesRegex(ValueError, PROTOCOL_VERSION):
                with mock.patch.object(freezer, "sha256_file", return_value="0" * 64):
                    freezer.freeze_cuda_from_cpu(out, Path(tmp) / "again.json")
            with self.assertRaisesRegex(ValueError, "공식 데이터 해시"):
                with mock.patch.object(freezer, "sha256_file", return_value="1" * 64):
                    freezer.freeze_cuda_from_cpu(cpu_path, Path(tmp) / "mismatch.json")

    def test_fold_comparison_detects_metadata_only_edits_and_fold_changes(self):
        import tempfile
        from scripts import freeze_v11_mlp_oof_split as freezer
        frame = pd.DataFrame([{"sample_id": f"S{g}_{t}", "vessel_name": "X", "target_vessel_id": f"V{g}",
                               "is_korean": t} for g in range(25) for t in (0, 1)])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "frozen.json"
            with mock.patch.object(freezer.pd, "read_csv", return_value=frame),                  mock.patch.object(freezer, "sha256_file", return_value="0" * 64):
                frozen = freezer.freeze(out)
            edited = dict(frozen, experiment="metadata only edit")
            out.write_text(json.dumps(edited, ensure_ascii=False, indent=2), encoding="utf-8")
            with mock.patch.object(freezer.pd, "read_csv", return_value=frame):
                same = freezer.compare_to(out)
            self.assertTrue(same["folds_identical"])
            self.assertEqual(same["existing_fold_digest"], same["rebuilt_fold_digest"])
            moved = json.loads(json.dumps(frozen))
            moved["folds"][0]["valid_sample_ids"] = moved["folds"][0]["valid_sample_ids"][1:]
            out.write_text(json.dumps(moved, ensure_ascii=False, indent=2), encoding="utf-8")
            with mock.patch.object(freezer.pd, "read_csv", return_value=frame):
                changed = freezer.compare_to(out)
            self.assertFalse(changed["folds_identical"])


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
