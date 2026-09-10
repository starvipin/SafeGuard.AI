# Runtime settings come from the environment or .env, with defaults for missing values.
"""Environment-driven application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv


# Load the local .env file without replacing environment values that are already set.
load_dotenv()

# Two parent levels above settings.py is the project root, independent of the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Flask copies these uppercase attributes into app.config.
class AppConfig:
    """Default runtime settings, overridable through environment variables."""

    # Convert environment strings into appropriate types: boolean debug flags and integer ports and sizes.
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
    # Limit the request body size in bytes; the default is 64 KB.
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(64 * 1024)))

    # Download from this Hugging Face model repository when the local model is missing.
    HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "sainivipin/fraud-model-final")
    # Use MODEL_DIR when provided; otherwise use models/fraud_model_final under the project root.
    MODEL_DIR = Path(
        os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models" / "fraud_model_final"))
    )
    # The detector checks for this weight file to decide whether a model download is needed.
    MODEL_FILENAME = os.getenv("MODEL_FILENAME", "model.safetensors")
    # Maximum number of recent scan results retained in this process's memory.
    HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "10"))
