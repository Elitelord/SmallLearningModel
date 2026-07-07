"""The minimal SFT prompt — the single source of truth for train AND inference.

Part B of the Day-2 brief: the behavior must live in the WEIGHTS, not in a
clever prompt. So the instruction is deliberately bare — just `Explain: {concept}`
— with no readability rules, no "for a 7-year-old", no few-shot. A base model
given this prompt will happily answer at an adult reading level; a *tuned* model
answers at grade 3 because that is what its weights learned. That gap is exactly
what the Day-5 demo shows.

CRITICAL: train and inference prompts must be byte-identical. Every consumer
(data/generate.py's SFT records, train/qlora_train.py, eval/base_vs_tuned.py)
imports build_messages() from here so they can never drift.

Note this is a DIFFERENT prompt from litmus.concepts.PROMPT_TEMPLATE. That verbose
prompt was for the litmus test ("can a well-prompted model already do this?").
Here we want the opposite: the weakest possible prompt, so any grade-3 behavior
is attributable to the fine-tune.
"""

# The bare user instruction. {concept} is a bare concept string, e.g.
# "Why is the sky blue?" -> "Explain: Why is the sky blue?"
MINIMAL_PROMPT = "Explain: {concept}"


def build_messages(concept: str) -> list[dict]:
    """The chat-format messages for one concept (user turn only).

    Pass to tokenizer.apply_chat_template(..., add_generation_prompt=True) at
    inference, or with the assistant turn appended for a training example.
    """
    return [{"role": "user", "content": MINIMAL_PROMPT.format(concept=concept)}]


def build_training_messages(concept: str, explanation: str) -> list[dict]:
    """Full user+assistant turn for one SFT example."""
    return [
        {"role": "user", "content": MINIMAL_PROMPT.format(concept=concept)},
        {"role": "assistant", "content": explanation},
    ]
