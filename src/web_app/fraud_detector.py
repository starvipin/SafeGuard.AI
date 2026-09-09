# Live prediction ka engine: zaroorat par trained model load karo, phir AI aur keywords se verdict do.
"""Fraud detection service with lazy model loading and keyword fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Mapping

from .prediction_result import Prediction


LOGGER = logging.getLogger(__name__)

# Risky words/phrases ki list; match milna fraud ka pakka saboot nahi, warning ka signal hai.
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


# Yeh class sirf inference karti hai; model ki training src/model_training mein hoti hai.
class FraudDetector:
    """Own model lifecycle and combine ML results with safety heuristics."""

    # Model paths aur state save karo; heavy tokenizer/model abhi load nahi karte.
    def __init__(self, model_dir: Path, model_repo: str, model_filename: str) -> None:
        self.model_dir = model_dir
        self.model_repo = model_repo
        self.model_filename = model_filename
        self.tokenizer = None
        self.model = None
        # CUDA available ho to GPU, warna CPU; model aur input tensors same device par hone chahiye.
        self.device = None
        # Ek detector instance mein failed load ko har request par dobara try nahi karte.
        self._load_attempted = False
        # Multiple requests ek saath aayen to download/loading ko lock se protect karo.
        self._load_lock = Lock()

    @classmethod
    # Flask config ko constructor ke arguments mein badalne ka helper.
    def from_config(cls, config: Mapping) -> "FraudDetector":
        return cls(
            model_dir=Path(config["MODEL_DIR"]),
            model_repo=config["HF_MODEL_REPO"],
            model_filename=config["MODEL_FILENAME"],
        )

    @property
    # Model aur tokenizer dono present hon tabhi AI prediction available hai.
    def model_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    # Pehle loaded model reuse karo; pehle attempt fail hua ho to keyword fallback par raho.
    def load_model(self) -> bool:
        if self.model_available:
            return True
        if self._load_attempted:
            return False

        # Lock milne ke baad state dobara check karo; doosri request model load kar chuki ho sakti hai.
        with self._load_lock:
            if self.model_available:
                return True
            if self._load_attempted:
                return False
            self._load_attempted = True

            try:
                # Heavy libraries yahin import hoti hain, isliye startup aur health endpoint halka rehta hai.
                import torch
                from huggingface_hub import snapshot_download
                from transformers import (
                    DistilBertForSequenceClassification,
                    DistilBertTokenizerFast,
                )

                # Weight file missing ho to poora repository snapshot local model directory mein download karo.
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
                # Evaluation mode dropout jaise training-only behavior ko band karta hai.
                self.model.eval()
                LOGGER.info("Fraud model loaded on %s", self.device)
                return True
            # Loading fail ho to log mein error rakho aur keyword detector se app ko result dene do.
            except Exception:
                LOGGER.exception("Model unavailable; keyword fallback enabled")
                self.tokenizer = None
                self.model = None
                return False

    # Message saaf karo, model loading try karo, phir keyword aur AI signals combine karo.
    def predict(self, text: str) -> Prediction:
        normalized = text.strip()
        if not normalized:
            raise ValueError("Message cannot be empty")

        self.load_model()
        keywords = self._find_keywords(normalized)
        # AI unavailable ho to sirf phrases se WARNING ya LEGIT return hota hai.
        if not self.model_available:
            return self._keyword_prediction(keywords)

        import torch

        # Text ko token IDs/attention mask mein badlo; truncation model ki length limit follow karta hai.
        inputs = self.tokenizer(
            normalized, return_tensors="pt", truncation=True, padding=True
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        # Gradients store nahi hote: prediction ke liye memory aur calculation bachti hai.
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        # Logits raw scores hain; softmax unhe class scores mein, argmax winning label mein badalta hai.
        probabilities = torch.softmax(logits, dim=1)
        label = int(torch.argmax(probabilities, dim=1).item())
        # Confidence model ka score hai; ise real-world correctness ki guarantee mat samjho.
        confidence = float(probabilities[0, label].item())

        # Label 1 par FRAUD. AI label 0 ho lekin risky phrases milen to hybrid WARNING.
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
    # casefold se capital/small letters ka fark hatao; phrase matching substring ke roop mein hoti hai.
    def _find_keywords(text: str) -> list[str]:
        lowered = text.casefold()
        return [keyword for keyword in FRAUD_KEYWORDS if keyword in lowered]

    @staticmethod
    # Phrases milen to warning reason mein unhe dikhao; nahi milen to keyword-scan LEGIT do.
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
