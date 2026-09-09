# STEP 04: local trained model ko Hugging Face MODEL repository par publish karo; website Space deploy alag workflow hai.
"""Explicit Hugging Face publishing helpers with no import-time side effects."""

from __future__ import annotations

import os
from pathlib import Path


# Repository argument aur HF_MODEL_REPO dono missing hon to yeh destination use hota hai.
DEFAULT_REPOSITORY = "sainivipin/fraud-model-final"


# Explicit function call par upload hota hai; import karne se network par model publish nahi hota.
def upload_model(
    model_dir: str | Path = "models/fraud_model_final",
    repository: str | None = None,
    token: str | None = None,
) -> str:
    from huggingface_hub import HfApi

    # Function ka token pehle, warna environment ka HF_TOKEN; missing token par upload rok do.
    access_token = token or os.getenv("HF_TOKEN")
    if not access_token:
        raise RuntimeError("HF_TOKEN is required to upload a model")

    # Local model directory exist karni chahiye; galat path par clear error do.
    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    # Destination priority: repository argument, phir environment, phir default repository.
    repo_id = repository or os.getenv("HF_MODEL_REPO", DEFAULT_REPOSITORY)
    api = HfApi(token=access_token)
    # Model repository missing ho to banao; existing repo allowed hai.
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    # Configured folder ke files HF par bhejo; existing remote model files update ho sakti hain.
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

    # Local .env token/settings padho; steps 02/03 wala model_output_dir params.yaml se lo.
    load_dotenv()
    config = load_config()
    model_dir = config["train"].get("model_output_dir", "models/fraud_model_final")
    repository = upload_model(model_dir)
    print(f"Stage 04 complete: model uploaded to '{repository}'")


if __name__ == "__main__":
    main()
