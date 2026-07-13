"""Publish the final v4r8 adapter, v4r7 dataset, and Gradio Space."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL_CARD = ROOT / "submission" / "model" / "README.md"
DATASET_CARD = ROOT / "data" / "v4" / "DATASET_CARD.md"
DATASET = ROOT / "data" / "v4" / "gold_v4_r7.jsonl"
DATASET_STATS = ROOT / "data" / "v4" / "gold_v4_r7.stats.json"
DEMO = ROOT / "demo"


def render(text: str, model_repo: str, dataset_repo: str, space_repo: str) -> str:
    return (
        text.replace("__MODEL_REPO__", model_repo)
        .replace("__DATASET_REPO__", dataset_repo)
        .replace("__SPACE_REPO__", space_repo)
    )


def require_submission_files(adapter: Path) -> None:
    required = [
        adapter / "adapter_config.json",
        adapter / "adapter_model.safetensors",
        MODEL_CARD,
        DATASET_CARD,
        DATASET,
        DATASET_STATS,
        DEMO / "app.py",
        DEMO / "requirements.txt",
        DEMO / "README.md",
        DEMO / "scoring.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing submission files:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=ROOT / "train" / "adapters" / "v4r8")
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--space-repo", required=True)
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--skip-space", action="store_true")
    parser.add_argument("--require-space", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    adapter = args.adapter.resolve()
    require_submission_files(adapter)
    print(f"adapter: {adapter}")
    print(f"model:   https://huggingface.co/{args.model_repo}")
    print(f"dataset: https://huggingface.co/datasets/{args.dataset_repo}")
    print(f"space:   https://huggingface.co/spaces/{args.space_repo}")
    if args.dry_run:
        print("dry run passed; no Hub writes performed")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_repo(args.model_repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.model_repo,
        repo_type="model",
        folder_path=adapter,
        commit_message="Upload final v4r8 QLoRA adapter",
    )

    with tempfile.TemporaryDirectory() as directory:
        staging = Path(directory)
        model_card = staging / "model_README.md"
        model_card.write_text(
            render(
                MODEL_CARD.read_text(encoding="utf-8"),
                args.model_repo,
                args.dataset_repo,
                args.space_repo,
            ),
            encoding="utf-8",
        )
        api.upload_file(
            repo_id=args.model_repo,
            repo_type="model",
            path_or_fileobj=model_card,
            path_in_repo="README.md",
            commit_message="Add final model card",
        )

        api.create_repo(args.dataset_repo, repo_type="dataset", private=args.private, exist_ok=True)
        dataset_card = staging / "dataset_README.md"
        dataset_card.write_text(DATASET_CARD.read_text(encoding="utf-8"), encoding="utf-8")
        for source, destination in (
            (DATASET, "train.jsonl"),
            (DATASET_STATS, "dataset_stats.json"),
            (dataset_card, "README.md"),
        ):
            api.upload_file(
                repo_id=args.dataset_repo,
                repo_type="dataset",
                path_or_fileobj=source,
                path_in_repo=destination,
                commit_message="Publish final v4r7 dataset",
            )

        if not args.skip_space:
            try:
                api.create_repo(
                    args.space_repo,
                    repo_type="space",
                    space_sdk="gradio",
                    private=args.private,
                    exist_ok=True,
                )
                staged_demo = staging / "demo"
                shutil.copytree(DEMO, staged_demo)
                for path in (staged_demo / "app.py", staged_demo / "README.md"):
                    path.write_text(
                        render(
                            path.read_text(encoding="utf-8"),
                            args.model_repo,
                            args.dataset_repo,
                            args.space_repo,
                        ),
                        encoding="utf-8",
                    )
                api.upload_folder(
                    repo_id=args.space_repo,
                    repo_type="space",
                    folder_path=staged_demo,
                    commit_message="Publish v4r8 Gradio demo",
                )
            except Exception as error:
                if args.require_space:
                    raise
                print(f"warning: Space publication skipped: {error}")
    print("submission artifacts published")


if __name__ == "__main__":
    main()
