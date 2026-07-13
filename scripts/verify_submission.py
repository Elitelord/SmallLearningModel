"""Verify the final public Hugging Face model and dataset repositories."""

from __future__ import annotations

import argparse


MODEL_FILES = {
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
}
DATASET_FILES = {"README.md", "train.jsonl", "dataset_stats.json"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--dataset-repo", required=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    model = api.model_info(args.model_repo, files_metadata=True)
    dataset = api.dataset_info(args.dataset_repo, files_metadata=True)
    model_files = {sibling.rfilename for sibling in model.siblings}
    dataset_files = {sibling.rfilename for sibling in dataset.siblings}
    missing_model = sorted(MODEL_FILES - model_files)
    missing_dataset = sorted(DATASET_FILES - dataset_files)
    if missing_model or missing_dataset:
        raise RuntimeError(
            f"missing model files={missing_model}; missing dataset files={missing_dataset}"
        )
    adapter = next(
        sibling for sibling in model.siblings
        if sibling.rfilename == "adapter_model.safetensors"
    )
    print(f"model public:   https://huggingface.co/{args.model_repo}")
    print(f"adapter bytes:  {adapter.size}")
    print(f"dataset public: https://huggingface.co/datasets/{args.dataset_repo}")
    print(f"dataset files:  {', '.join(sorted(dataset_files))}")
    print("submission Hub verification passed")


if __name__ == "__main__":
    main()
