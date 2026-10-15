import unittest

from scripts.exp014_validation import (
    count_prediction_changes,
    generate_fixed_candidates,
    load_runtime,
    select_prediction,
)


class SelectPredictionTest(unittest.TestCase):
    def test_baseline_uses_vessel_id_for_exact_score_ties(self):
        candidates = [
            {"score": 0.5, "vessel_id": "VESSEL_2", "vessel_name": "ALPHA"},
            {"score": 0.5, "vessel_id": "VESSEL_1", "vessel_name": "ZULU"},
        ]

        self.assertEqual(select_prediction(candidates, tie_policy="vessel_id"), "VESSEL_1")

    def test_name_policy_uses_normalized_name_only_for_score_ties(self):
        candidates = [
            {"score": 0.5, "vessel_id": "VESSEL_1", "vessel_name": "Lotus Sambu"},
            {"score": 0.5 + 5e-13, "vessel_id": "VESSEL_2", "vessel_name": "Giant-Sambu"},
        ]

        self.assertEqual(select_prediction(candidates, tie_policy="catalog_name"), "VESSEL_2")

    def test_prediction_change_count_treats_two_missing_values_as_equal(self):
        self.assertEqual(
            count_prediction_changes([None, "VESSEL_1"], [None, "VESSEL_2"]),
            1,
        )


class NonKoreanFuzzyCandidateTest(unittest.TestCase):
    def test_adds_non_korean_fuzzy_candidates_without_changing_base_candidates(self):
        class FakeIndex:
            romanized_names = ["OCEANSTAR", "OCEANSTARII", "OTHERSHIP"]

            def generate_base_candidates(self, row, max_candidates):
                self.asserted_max_candidates = max_candidates
                return [2]

        runtime = load_runtime()
        fuzz, process, romanized_name = runtime[3], runtime[6], runtime[7]
        row = {"vessel_name": "OCEAN STAR", "gross_tonnage": None}
        index = FakeIndex()

        base_only, _ = generate_fixed_candidates(
            row, index, fuzz, process, romanized_name
        )
        expanded, scores = generate_fixed_candidates(
            row,
            index,
            fuzz,
            process,
            romanized_name,
            include_non_korean_fuzzy=True,
        )

        self.assertEqual(base_only, [2])
        self.assertEqual(expanded, [0, 1, 2])
        self.assertEqual(set(scores), {0, 1})
        self.assertEqual(index.asserted_max_candidates, 60)

    def test_name_policy_does_not_override_a_non_tied_higher_score(self):
        candidates = [
            {"score": 0.5, "vessel_id": "VESSEL_1", "vessel_name": "ALPHA"},
            {"score": 0.5000001, "vessel_id": "VESSEL_2", "vessel_name": "ZULU"},
        ]

        self.assertEqual(select_prediction(candidates, tie_policy="catalog_name"), "VESSEL_2")


if __name__ == "__main__":
    unittest.main()
