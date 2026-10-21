"""Checks for fixed-profile ranker comparison, not competition-score tests."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from agent.error_diagnostics import paired_outcomes
from agent.improvement_planner import propose_fixed_change, guard
from agent.retrain_config import baseline_config, config_for, fingerprint


class RankerComparisonTest(unittest.TestCase):
    def test_lightgbm_fingerprints_keep_existing_ledger_identity(self):
        config = config_for('number_features', 'no_asymmetry')
        legacy = {k: v for k, v in config.items() if k != 'ranker_family'}
        expected = hashlib.sha256(json.dumps(
            {'protocol': 'v11-improvement-1', 'config': dict(sorted(legacy.items()))},
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(fingerprint(config), expected)
        self.assertNotEqual(fingerprint(config_for('ranker_family', 'catboost_v1_cpu')),
                            fingerprint(baseline_config()))

    def test_fixed_proposal_passes_guard_and_addresses_observed_errors(self):
        cross = {'error_flags': {'other_rank_failure': {'total': 2}}, 'near_miss_flags': {}}
        plan = propose_fixed_change(cross, 'ranker_family', 'catboost_v1_cpu')
        accepted, rejected = guard(plan['proposals'], cross, set(), 1)
        self.assertEqual(rejected, [])
        self.assertEqual([(p['knob'], p['value']) for p in accepted],
                         [('ranker_family', 'catboost_v1_cpu')])

    @unittest.skipUnless(importlib.util.find_spec('sklearn') and importlib.util.find_spec('catboost'),
                         'run native ranker check in competition Python environment')
    def test_native_catboost_profile_fit_hash_importance_and_roundtrip(self):
        from agent import retrain_runtime as runtime
        from catboost import CatBoostRanker
        rng = np.random.default_rng(42)
        x = rng.normal(size=(30, len(runtime.FEATURES))).astype(np.float32)
        y = np.tile([1, 0, 0, 0, 0], 6).astype(np.int8)
        groups = np.full(6, 5)
        config = config_for('ranker_family', 'catboost_v1_cpu')
        with mock.patch.object(runtime, 'training_matrix',
                               return_value=(x, y, groups, 1.0, {})):
            model, _, stats = runtime.fit(None, None, None, config)
        self.assertIsInstance(model, CatBoostRanker)
        self.assertEqual(model.tree_count_, 600)
        self.assertEqual(model.get_params()['task_type'], 'CPU')
        self.assertEqual(model.feature_names_, list(runtime.FEATURES))
        self.assertTrue(runtime.behavior_check('ranker_family', 'catboost_v1_cpu', {}, stats, {}, model)['passed'])
        self.assertTrue(np.isfinite(model.predict(x)).all())
        self.assertEqual(len(runtime.model_sha256(model)), 64)
        self.assertEqual(runtime.model_sha256(model), runtime.model_sha256(model))
        self.assertEqual(set(runtime.feature_importance(model)), set(runtime.FEATURES))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.cbm'
            model.save_model(str(path))
            self.assertEqual(runtime.model_sha256(model), hashlib.sha256(path.read_bytes()).hexdigest())
            restored = CatBoostRanker().load_model(str(path))
            np.testing.assert_array_equal(model.predict(x), restored.predict(x))

    def test_complementarity_counts_do_not_claim_oracle_as_real_prediction(self):
        base, alt = [], []
        for i, (left, right) in enumerate([(True, True), (True, False), (False, True), (False, False)]):
            common = {'sample_id': str(i), 'truth_id': 'truth', 'is_korean': i < 2, 'seen_target': False}
            base.append({**common, 'final_correct': left, 'model_correct': left,
                         'final_pred_id': 'truth' if left else 'wrong_a',
                         'model_pred_id': 'truth' if left else 'wrong_a'})
            alt.append({**common, 'final_correct': right, 'model_correct': right,
                        'final_pred_id': 'truth' if right else 'wrong_b',
                        'model_pred_id': 'truth' if right else 'wrong_b'})
        result = paired_outcomes(base, alt)
        counts = result['final']['all']
        self.assertEqual([counts[k] for k in ('both_correct', 'baseline_only_correct',
                                              'alternative_only_correct', 'both_wrong')], [1, 1, 1, 1])
        self.assertEqual(counts['oracle_accuracy'], 0.75)
        self.assertEqual(counts['prediction_disagreements'], 3)
        self.assertIn('not deployable', result['interpretation'])
        with self.assertRaises(ValueError):
            paired_outcomes(base, alt[::-1])


if __name__ == '__main__':
    unittest.main()
