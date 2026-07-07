"""The fixed concept set and the eval prompt.

These are frozen for the whole week so every model (and the Day-3 base-vs-tuned
eval) is judged with the same ruler. Do NOT edit the wording — reuse verbatim.
"""

# The 12 fixed concepts (litmus_test_instructions.md).
CONCEPTS = [
    "Why is the sky blue?",
    "How do plants make their own food?",
    "Why do we have day and night?",
    "What makes ice melt?",
    "How do magnets work?",
    "Why do things fall to the ground?",
    "Where does a puddle go when it dries up?",
    "Why do we have seasons?",
    "How do our lungs help us breathe?",
    "What makes a rainbow?",
    "Why does the moon look like it changes shape?",
    "How do fish breathe underwater?",
]

# The eval prompt. Paste verbatim into each browser model; use as the user
# message for API/Qwen. {CONCEPT} is substituted per concept.
PROMPT_TEMPLATE = """You are explaining science to a 7-year-old in the 3rd grade.

Rules:
- Every single sentence must be simple enough for an 8-year-old to read on their own. Use short sentences and common, everyday words.
- Avoid technical terms. If you must use one, explain it in plain words right away.
- Stay scientifically accurate. Explain HOW or WHY it really works — the actual reason — not just what it is.
- Do NOT oversimplify so much that it becomes wrong.
- Write 4 to 8 sentences.

Explain: {CONCEPT}"""


def build_prompt(concept: str) -> str:
    return PROMPT_TEMPLATE.format(CONCEPT=concept)
