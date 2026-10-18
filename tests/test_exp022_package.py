"""Contract checks for the frozen EXP-022 submission candidate."""

import types
import unittest

import pandas as pd

from scripts.exp022_non_korean_fuzzy_gate import load_baseline_module
from scripts.package_exp022 import build_variant_library


class Exp022PackageTests(unittest.TestCase):
    def test_typo_retrieval_preserves_korean_path_and_candidate_cap(self):
        baseline = load_baseline_module()
        variant = types.ModuleType("exp022_test_variant")
        exec(compile(build_variant_library().decode("utf-8"), "variant.py", "exec"), variant.__dict__)
        catalog = pd.DataFrame(
            [
                {"vessel_id": "VESSEL_TARGET", "vessel_name": "MORNING STAR", "gross_tonnage": 8000},
                *[
                    {"vessel_id": f"VESSEL_{i:02d}", "vessel_name": f"UNRELATED {i}", "gross_tonnage": 1000}
                    for i in range(20)
                ],
            ]
        )
        base_index = baseline.AdaptiveCatalogIndex(catalog)
        new_index = variant.AdaptiveCatalogIndex(catalog)
        typo = {"vessel_name": "MORNNG STAR", "gross_tonnage": 1000}
        base_candidates, _ = base_index.candidates_with_scores(typo)
        new_candidates, scores = new_index.candidates_with_scores(typo)
        self.assertNotIn(0, base_candidates)
        self.assertIn(0, new_candidates)
        self.assertEqual(scores, {})  # no non-Korean fuzzy ranking bonus
        self.assertLessEqual(len(new_candidates), variant.MAX_CANDIDATES)

        korean = {"vessel_name": "모닝스타", "gross_tonnage": 1000}
        self.assertEqual(
            base_index.candidates_with_scores(korean),
            new_index.candidates_with_scores(korean),
        )
        self.assertEqual(base_index.predict_row(korean), new_index.predict_row(korean))


if __name__ == "__main__":
    unittest.main()
