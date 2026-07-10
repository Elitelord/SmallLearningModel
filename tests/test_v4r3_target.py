import unittest

from data.v4r3 import meets_target, target_config


class V4R3TargetTests(unittest.TestCase):
    def test_accepts_centered_v4_readable_score(self):
        score = {
            "whole_passage_fk": 4.2,
            "whole_passage_ari": 5.1,
            "fk_stdev": 0.8,
            "max_fk": 6.4,
            "readability_pass_v4": True,
        }
        self.assertTrue(meets_target(score, target_config()))

    def test_rejects_hard_sentence_even_inside_whole_passage_band(self):
        score = {
            "whole_passage_fk": 4.2,
            "whole_passage_ari": 5.1,
            "fk_stdev": 0.8,
            "max_fk": 7.2,
            "readability_pass_v4": True,
        }
        self.assertFalse(meets_target(score, target_config()))

    def test_rejects_v4_failure(self):
        score = {
            "whole_passage_fk": 4.2,
            "whole_passage_ari": 5.1,
            "fk_stdev": 0.8,
            "max_fk": 6.4,
            "readability_pass_v4": False,
        }
        self.assertFalse(meets_target(score, target_config()))


if __name__ == "__main__":
    unittest.main()
