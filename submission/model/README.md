---
base_model: Qwen/Qwen3-4B
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
- en
tags:
- qwen3
- lora
- qlora
- education
- readability
- science
datasets:
- __DATASET_REPO__
---

# Qwen3-4B Grade-Level Science Explainer v4r8

A QLoRA adapter for Qwen3-4B that produces concise elementary-science explanations
at a third-grade reading level from the bare prompt `Explain: {concept}`.

## Results

| Model | Prompt | Readability | Accuracy-v2 | Overall-v2 |
|---|---|---:|---:|---:|
| Qwen3-4B base | full grade-3 prompt | 2/12 | 9/12 | **2/12** |
| **v4r8 adapter** | bare `Explain:` | **9/12** | 9/12 | **8/12** |

The fixed 12-prompt development litmus uses deterministic readability metrics and a
multi-judge accuracy rubric. GPT-5.4 and Claude Opus 4.7 are primary judges; Gemini
3.1 Pro resolves axis disagreements. The adapter gains six joint passes over the
well-prompted base while receiving a weaker prompt.

## Training

- Base: `Qwen/Qwen3-4B`
- Dataset: `__DATASET_REPO__` (485 unique clean records)
- Method: 4-bit QLoRA with assistant-only loss
- LoRA rank/alpha: 32/64
- Epochs: 2
- Learning rate: `2e-4`
- Batch size / gradient accumulation: 8 / 2
- Max sequence length: 512
- Decode setting: temperature 0, selected on a separate 24-prompt calibration set

## Usage

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3-4B"
adapter_id = "__MODEL_REPO__"
tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_id)

messages = [{"role": "user", "content": "Explain: Why is the sky blue?"}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(
    **inputs,
    max_new_tokens=220,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

Interactive demo: `__SPACE_REPO__`

Source, training notebook, eval harness, and raw judgments:
https://github.com/Elitelord/SmallLearningModel

## Intended Use

Educational prototyping, constrained-style research, and generation of short science
explanations that will still receive human review.

## Limitations

This adapter does not guarantee factual correctness. It failed four of twelve joint
litmus cases, including mechanism errors for seasons and lungs. Readability formulas
are proxies rather than direct measures of child comprehension. The evaluation set is
small and English-only, and the 12 development prompts were used to compare training
iterations. Do not use the model as an unsupervised curriculum authority.
