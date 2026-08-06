"""Environment-driven application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AppConfig:
    """Default runtime settings, overridable through environment variables."""

    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(64 * 1024)))

    HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "sainivipin/fraud-model-final")
    MODEL_DIR = Path(
        os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models" / "fraud_model_final"))
    )
    MODEL_FILENAME = os.getenv("MODEL_FILENAME", "model.safetensors")
    HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "10"))
