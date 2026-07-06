"""Day-1 checkpoint: prove the local environment can load and run a base model.

Usage:
    .venv\\Scripts\\python scripts\\smoke_test.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    messages = [{"role": "user", "content": "In one sentence, what is a QLoRA fine-tune?"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    output = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"Model: {MODEL_ID}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
