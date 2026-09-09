"""Explicit Hugging Face publishing helpers with no import-time side effects."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_REPOSITORY = "sainivipin/fraud-model-final"


def upload_model(
    model_dir: str | Path = "models/fraud_model_final",
    repository: str | None = None,
    token: str | None = None,
) -> str:
    from huggingface_hub import HfApi

    access_token = token or os.getenv("HF_TOKEN")
    if not access_token:
        raise RuntimeError("HF_TOKEN is required to upload a model")

    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    repo_id = repository or os.getenv("HF_MODEL_REPO", DEFAULT_REPOSITORY)
    api = HfApi(token=access_token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Publish trained SafeGuard AI model",
    )
    return repo_id


def main() -> None:
    """Publish the model directory configured for steps 02 and 03."""
    from dotenv import load_dotenv

    from .pipeline_helpers import load_config

    load_dotenv()
    config = load_config()
    model_dir = config["train"].get("model_output_dir", "models/fraud_model_final")
    repository = upload_model(model_dir)
    print(f"Stage 04 complete: model uploaded to '{repository}'")


if __name__ == "__main__":
    main()
