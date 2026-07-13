import os
import re
from functools import lru_cache

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scoring import score_text


MODEL_ID = os.getenv("MODEL_ID", "__MODEL_REPO__")
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen3-4B")


@lru_cache(maxsize=1)
def load_model():
    if MODEL_ID == "__MODEL_REPO__":
        raise RuntimeError("Set MODEL_ID to the published v4r8 adapter repository.")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    load_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float16 if not torch.cuda.is_available() else torch.float16,
    }
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)
    model = PeftModel.from_pretrained(base, MODEL_ID)
    model.eval()
    return model, tokenizer


def explain(concept: str):
    concept = concept.strip()
    if not concept:
        return "Please enter a science question.", ""
    model, tokenizer = load_model()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Explain: {concept}"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    metrics = score_text(text)
    if "error" in metrics:
        return text, metrics["error"]
    status = "PASS" if metrics["pass"] else "MISS"
    summary = (
        f"### Readability gate: {status}\n"
        f"FK **{metrics['whole_fk']}** · ARI **{metrics['whole_ari']}** · "
        f"sentence FK stdev **{metrics['fk_stdev']}** · "
        f"max sentence FK **{metrics['max_fk']}** · "
        f"{metrics['sentences']} sentences"
    )
    return text, summary


demo = gr.Interface(
    fn=explain,
    inputs=gr.Textbox(
        label="Elementary-science question",
        value="Why is the sky blue?",
    ),
    outputs=[
        gr.Textbox(label="v4r8 explanation", lines=9),
        gr.Markdown(),
    ],
    title="Grade-Level Science Explainer",
    description=(
        "Qwen3-4B + v4r8, using only the bare prompt `Explain: {question}`. "
        "The metrics are the same deterministic v4 readability gate used in evaluation."
    ),
    examples=[
        ["Why is the sky blue?"],
        ["How do magnets work?"],
        ["Where does a puddle go when it dries up?"],
        ["What makes a rainbow?"],
    ],
)


if __name__ == "__main__":
    demo.launch(share=os.getenv("GRADIO_SHARE") == "1")
