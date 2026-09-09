# Website ki settings: .env/environment se values aati hain, missing value par default lagta hai.
"""Environment-driven application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv


# Local .env ko environment mein load karo; pehle se set environment values ko default taur par nahi badalta.
load_dotenv()

# settings.py se do parent levels upar project root hai; model path current terminal folder par depend nahi karta.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Flask in uppercase attributes ko app.config mein copy karta hai.
class AppConfig:
    """Default runtime settings, overridable through environment variables."""

    # Environment strings ko sahi types mein badlo: debug boolean, port aur size integer.
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
    # Request body ki maximum size bytes mein; default 64 KB hai.
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(64 * 1024)))

    # Local model missing hone par is Hugging Face model repository se download hota hai.
    HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "sainivipin/fraud-model-final")
    # MODEL_DIR diya ho to use karo; warna project ke models/fraud_model_final mein model rakho.
    MODEL_DIR = Path(
        os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models" / "fraud_model_final"))
    )
    # Is weight file ki maujoodgi check karke detector download karne ka decision leta hai.
    MODEL_FILENAME = os.getenv("MODEL_FILENAME", "model.safetensors")
    # Ek process mein maximum kitne recent scan results memory mein rakhe jayenge.
    HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "10"))
