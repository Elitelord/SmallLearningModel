import unittest

from litmus.concepts import CONCEPTS
from litmus.run_frontier_v2 import (
    DEFAULT_MODELS,
    completion_options,
    load_concepts,
    model_family,
    model_key,
)


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

    def test_opus_48_omits_deprecated_temperature(self):
        self.assertEqual(
            completion_options("claude-group/claude-opus-4-8", 0.0),
            {"model": "claude-group/claude-opus-4-8"},
        )
        self.assertEqual(
            completion_options("claude-group/claude-opus-4-7", 0.0)["temperature"],
            0.0,
        )

    def test_blind_holdout_loads_in_frozen_order(self):
        blind = load_concepts("blind_v4r5")
        self.assertEqual(len(blind), 24)
        self.assertEqual(blind[0], "How does a suction cup stick to a wall?")
        self.assertEqual(blind[-1], "How does rainwater slowly make a cave?")


if __name__ == "__main__":
    unittest.main()
