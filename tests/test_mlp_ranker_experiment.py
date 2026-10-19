"""Native PyTorch checks for the fixed v11 MLP reranker."""

import tempfile
import unittest
from pathlib import Path
import importlib.util

import numpy as np
from unittest import mock

from agent.retrain_config import config_for

_HAS_NATIVE_RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "sklearn", "lightgbm"))
if _HAS_NATIVE_RUNTIME:
    from agent import retrain_runtime as runtime
    from agent.retrain_runtime import FEATURES, MLPRanker, MLP_PROFILE, listwise_group_loss
else:
    FEATURES, MLPRanker, MLP_PROFILE, listwise_group_loss, runtime = (), object, {}, None, None


@unittest.skipUnless(_HAS_NATIVE_RUNTIME, "run MLP tests in competition Python 3.11 environment")
class MlpRankerTest(unittest.TestCase):
    def test_runtime_dispatches_the_fixed_mlp_profile_and_behavior_check_passes(self):
        x = np.zeros((12, len(FEATURES)), dtype=np.float32)
        x[:, 0] = np.arange(len(x), dtype=np.float32)
        y, groups = np.tile([1, 0], 6).astype(np.int8), np.full(6, 2)
        config = config_for("ranker_family", "mlp_listwise_v1_cpu")
        with mock.patch.object(runtime, "training_matrix", return_value=(x, y, groups, 1.0, {})):
            model, recall, stats = runtime.fit(None, None, None, config)
        self.assertIsInstance(model, MLPRanker)
        self.assertEqual(recall, 1.0)
        self.assertEqual(stats["ranker_params"], MLP_PROFILE)
        result = runtime.behavior_check("ranker_family", "mlp_listwise_v1_cpu", {}, stats, {}, model)
        self.assertTrue(result["passed"], result["checks"])

    def test_listwise_loss_contributes_one_term_per_variable_length_query(self):
        import torch
        scores = torch.tensor([2.0, 0.0, 0.0, 1.0, 2.0])
        labels = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])
        groups = np.array([2, 3])
        loss = float(listwise_group_loss(scores, labels, groups))
        expected_first = np.log(np.exp(2.0) + np.exp(0.0)) - 2.0
        expected_second = np.log(np.exp(0.0) + np.exp(1.0) + np.exp(2.0)) - 1.0
        self.assertAlmostEqual(loss, (expected_first + expected_second) / 2.0, places=6)

    def test_fixed_profile_is_cpu_and_train_only_scaler_handles_constant_features(self):
        self.assertEqual(MLP_PROFILE["hidden_sizes"], (64, 32))
        self.assertEqual(MLP_PROFILE["epochs"], 20)
        x = np.zeros((8, len(FEATURES)), dtype=np.float32)
        x[:, 0] = np.arange(8, dtype=np.float32)
        labels, groups = np.array([1, 0, 1, 0, 0, 1, 0, 0]), np.array([2, 3, 3])
        model = MLPRanker().fit(x, labels, groups)
        self.assertEqual(model.input_mean_.shape, (len(FEATURES),))
        self.assertEqual(model.input_scale_[1], 1.0)
        self.assertTrue(np.isfinite(model.predict(x)).all())
        self.assertEqual(set(model.feature_importance()), set(FEATURES))

    def test_fit_is_deterministic_and_serialized_predictions_match(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(18, len(FEATURES))).astype(np.float32)
        labels = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
        groups = np.full(6, 3)
        first = MLPRanker().fit(x, labels, groups)
        second = MLPRanker().fit(x, labels, groups)
        np.testing.assert_array_equal(first.predict(x), second.predict(x))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranker.pt"
            first.save_model(path)
            restored = MLPRanker.load_model(path)
            np.testing.assert_array_equal(first.predict(x), restored.predict(x))
            self.assertEqual(first.model_sha256(), restored.model_sha256())

    def test_invalid_groups_and_nonfinite_features_are_rejected(self):
        x = np.zeros((4, len(FEATURES)), dtype=np.float32)
        labels = np.array([1, 0, 1, 0])
        with self.assertRaises(ValueError):
            MLPRanker().fit(x, labels, np.array([3]))
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            MLPRanker().fit(x, labels, np.array([2, 2]))


if __name__ == "__main__":
    unittest.main()
