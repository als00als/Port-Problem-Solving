import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "submit" / "model"))

import adaptive_lib  # noqa: E402


class Exp014CandidateConfigTest(unittest.TestCase):
    def test_exp014_uses_wide_fixed_fuzzy_search_configuration(self):
        self.assertEqual(adaptive_lib.FUZZY_LIMIT, 40)
        self.assertEqual(adaptive_lib.FUZZY_CUTOFF, 50)
        self.assertEqual(adaptive_lib.MAX_CANDIDATES, 60)
