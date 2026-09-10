# Live prediction engine: lazily load the trained model and combine its verdict with keyword signals.
"""Fraud detection service with lazy model loading and keyword fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Mapping

from .prediction_result import Prediction


LOGGER = logging.getLogger(__name__)

# These phrases indicate possible risk; a match is a warning signal, not proof of fraud.
FRAUD_KEYWORDS = (
    "immediately", "24 hours", "tonight", "blocked", "suspended", "asap",
    "click here", "update kyc", "verify now", "call this number", "log in",
    "link below", "lottery", "cashback", "winner", "work from home",
    "easy money", "prize", "bonus", "gift", "claim now", "account",
    "police", "arrest", "fir", "court case", "legal notice", "warrant",
    "seized", "customs", "disconnected", "terminated", "jail", "income tax",
    "rbi", "govt", "support team", "admin", "manager", "ceo", "officer",
    "cbi", "narcotics", "unusual login", "device detected", "password changed",
    "security alert", "unauthorized access", "validation code", "otp",
    "reset pin", "account frozen", "limited time", "expires today",
    "last chance", "offer ends", "urgent attention", "wire transfer", "refund",
    "credit", "debit", "pending transaction", "invoice", "bill payment",
    "processing fee", "bitcoin", "wallet address", "wallet",
)


# This class performs inference only; training is implemented in src/model_training.
class FraudDetector:
    """Own model lifecycle and combine ML results with safety heuristics."""

    # Store paths and initial state without loading the heavy tokenizer or model yet.
    def __init__(self, model_dir: Path, model_repo: str, model_filename: str) -> None:
        self.model_dir = model_dir
        self.model_repo = model_repo
        self.model_filename = model_filename
        self.tokenizer = None
        self.model = None
        # After a failed load, this detector instance does not retry on every request.
        self.device = None
        # Protect downloading and loading when several requests arrive at the same time.
        self._load_attempted = False
        # Convert Flask configuration values into constructor arguments.
        self._load_lock = Lock()

    @classmethod
    # AI prediction is available only when both the model and tokenizer are present.
    def from_config(cls, config: Mapping) -> "FraudDetector":
        return cls(
            model_dir=Path(config["MODEL_DIR"]),
            model_repo=config["HF_MODEL_REPO"],
            model_filename=config["MODEL_FILENAME"],
        )

    @property
    # Reuse a loaded model; after a failed attempt, continue with keyword fallback.
    def model_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    # Check state again after acquiring the lock because another request may have loaded the model.
    def load_model(self) -> bool:
        if self.model_available:
            return True
        if self._load_attempted:
            return False

        # Import heavy libraries here to keep application startup and health checks lightweight.
        with self._load_lock:
            if self.model_available:
                return True
            if self._load_attempted:
                return False
            self._load_attempted = True

            try:
                # If the weight file is missing, download the repository snapshot into the local model directory.
                import torch
                from huggingface_hub import snapshot_download
                from transformers import (
                    DistilBertForSequenceClassification,
                    DistilBertTokenizerFast,
                )

                # Use CUDA when available, otherwise CPU; model and input tensors must share a device.
                model_path = self.model_dir / self.model_filename
                if not model_path.exists():
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                    LOGGER.info("Downloading model %s", self.model_repo)
                    snapshot_download(repo_id=self.model_repo, local_dir=self.model_dir)

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_dir)
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    self.model_dir
                )
                self.model.to(self.device)
                # Evaluation mode disables training-only behavior such as dropout.
                self.model.eval()
                LOGGER.info("Fraud model loaded on %s", self.device)
                return True
            # Log loading errors and allow the keyword detector to keep returning results.
            except Exception:
                LOGGER.exception("Model unavailable; keyword fallback enabled")
                self.tokenizer = None
                self.model = None
                return False

    # Trim the message, attempt model loading, then combine keyword and model signals.
    def predict(self, text: str) -> Prediction:
        normalized = text.strip()
        if not normalized:
            raise ValueError("Message cannot be empty")

        self.load_model()
        keywords = self._find_keywords(normalized)
        # If AI is unavailable, return a keyword-only WARNING or LEGIT result.
        if not self.model_available:
            return self._keyword_prediction(keywords)

        import torch

        # Convert text into token IDs and an attention mask; truncation enforces the model's length limit.
        inputs = self.tokenizer(
            normalized, return_tensors="pt", truncation=True, padding=True
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        # Disable gradient tracking to reduce memory use and computation during prediction.
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        # Logits are raw scores; softmax produces class scores, and argmax selects the winning label.
        probabilities = torch.softmax(logits, dim=1)
        label = int(torch.argmax(probabilities, dim=1).item())
        # Confidence is the model's score, not a guarantee of real-world correctness.
        confidence = float(probabilities[0, label].item())

        # Label 1 produces FRAUD; label 0 with risky phrases produces a hybrid WARNING.
        if label == 1:
            return Prediction("FRAUD", "Detected by AI model", "danger", "model", confidence)
        if keywords:
            return Prediction(
                "WARNING",
                f"AI result was safe, but risky phrases were found: {', '.join(keywords)}",
                "warning",
                "hybrid",
                confidence,
            )
        return Prediction("LEGIT", "No suspicious pattern detected", "success", "model", confidence)

    @staticmethod
    # Ignore letter case with casefold and search for phrases as substrings.
    def _find_keywords(text: str) -> list[str]:
        lowered = text.casefold()
        return [keyword for keyword in FRAUD_KEYWORDS if keyword in lowered]

    @staticmethod
    # Explain matching phrases in a warning; otherwise return a keyword-scan LEGIT result.
    def _keyword_prediction(keywords: list[str]) -> Prediction:
        if keywords:
            return Prediction(
                "WARNING",
                f"Suspicious phrases found: {', '.join(keywords)}",
                "warning",
                "keywords",
            )
        return Prediction(
            "LEGIT",
            "No suspicious phrase detected (keyword scan)",
            "success",
            "keywords",
        )
