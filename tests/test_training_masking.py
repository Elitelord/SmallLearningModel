import unittest

from train.qlora_train import build_label_masked_example, training_prompt
from data.sft_format import build_messages


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    ):
        del tokenize, enable_thinking
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(ch) for ch in text]}


class TrainingMaskingTests(unittest.TestCase):
    def test_uses_phrasing_and_masks_prompt_tokens(self):
        tok = FakeTokenizer()
        rec = {
            "concept": "Canonical concept",
            "phrasing": "Child phrasing?",
            "explanation": "This is the assistant answer.",
        }

        example = build_label_masked_example(tok, rec, max_seq_len=512)
        prompt_text = tok.apply_chat_template(
            build_messages("Child phrasing?"),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_len = len(tok(prompt_text, add_special_tokens=False)["input_ids"])

        self.assertEqual(training_prompt(rec), "Child phrasing?")
        self.assertTrue(all(label == -100 for label in example["labels"][:prompt_len]))
        self.assertTrue(any(label != -100 for label in example["labels"][prompt_len:]))

    def test_falls_back_to_concept_when_phrasing_missing(self):
        self.assertEqual(training_prompt({"concept": "Only concept"}), "Only concept")


if __name__ == "__main__":
    unittest.main()
