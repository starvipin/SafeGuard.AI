# STEP 04: publish the local trained model to a Hugging Face MODEL repository; website deployment is separate.
"""Explicit Hugging Face publishing helpers with no import-time side effects."""

from __future__ import annotations

import os
from pathlib import Path


# Use this destination when neither a repository argument nor HF_MODEL_REPO is provided.
DEFAULT_REPOSITORY = "sainivipin/fraud-model-final"


# Upload only when this function is explicitly called; importing the module does not publish anything.
def upload_model(
    model_dir: str | Path = "models/fraud_model_final",
    repository: str | None = None,
    token: str | None = None,
) -> str:
    from huggingface_hub import HfApi

    # Prefer the supplied token, then HF_TOKEN from the environment; stop if neither exists.
    access_token = token or os.getenv("HF_TOKEN")
    if not access_token:
        raise RuntimeError("HF_TOKEN is required to upload a model")

    # Require a local model directory and report an invalid path clearly.
    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    # Destination priority: explicit repository argument, environment setting, then the default.
    repo_id = repository or os.getenv("HF_MODEL_REPO", DEFAULT_REPOSITORY)
    api = HfApi(token=access_token)
    # Create the model repository if it does not exist; allow an existing repository.
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    # Upload the configured folder; existing remote model files may be updated.
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Publish trained SafeGuard AI model",
    )
    return repo_id


# Run: uv run python -m src.model_training.step_04_upload_to_hf
def main() -> None:
    """Publish the model directory configured for steps 02 and 03."""
    from dotenv import load_dotenv

    from .pipeline_helpers import load_config

    # Load local .env settings and read the same model_output_dir used by Steps 02 and 03.
    load_dotenv()
    config = load_config()
    model_dir = config["train"].get("model_output_dir", "models/fraud_model_final")
    repository = upload_model(model_dir)
    print(f"Stage 04 complete: model uploaded to '{repository}'")


if __name__ == "__main__":
    main()
