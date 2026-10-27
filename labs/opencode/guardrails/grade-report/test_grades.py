"""Check the behavior specified in README.md."""

import unittest

from grades import average


class GradeTests(unittest.TestCase):
    def test_missing_score(self) -> None:
        self.assertEqual(average("scores.csv"), 90.0)
