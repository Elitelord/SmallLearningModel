import unittest

from litmus.concepts import CONCEPTS
from litmus.run_frontier_v2 import DEFAULT_MODELS, model_family, model_key


class FrontierV2Tests(unittest.TestCase):
    def test_default_matrix_has_three_models_and_36_outputs(self):
        identities = {
            (model, concept) for model in DEFAULT_MODELS for concept in CONCEPTS
        }
        self.assertEqual(len(DEFAULT_MODELS), 3)
        self.assertEqual(len(identities), 36)

    def test_default_models_have_distinct_keys_and_families(self):
        self.assertEqual(len({model_key(model) for model in DEFAULT_MODELS}), 3)
        self.assertEqual(
            {model_family(model) for model in DEFAULT_MODELS},
            {"openai", "anthropic", "google"},
        )


if __name__ == "__main__":
    unittest.main()
