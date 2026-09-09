# Website aur fraud detector ke behavior checks; mock predictions se real HF downloads avoid hote hain.
from unittest.mock import patch

import pytest

from src.web_app import create_app
from src.web_app.prediction_result import Prediction
from src.web_app.fraud_detector import FraudDetector


@pytest.fixture
# Har test ke liye TESTING mode mein fresh app; history limit 2 rakhkar boundary aasani se check hoti hai.
def application():
    return create_app({"TESTING": True, "HISTORY_LIMIT": 2})


@pytest.fixture
# Flask test client bina real port/server start kiye HTTP requests simulate karta hai.
def client(application):
    return application.test_client()


# Keyword fallback aur empty message handling ka group.
class TestFraudDetector:
    # Model load ko False stub karke verify karo ki risky phrases par WARNING milta hai.
    def test_keyword_fallback_warns_for_risky_message(self, tmp_path):
        detector = FraudDetector(tmp_path, "owner/model", "model.safetensors")
        with patch.object(detector, "load_model", return_value=False):
            result = detector.predict("Click here to update your account immediately")

        assert result.status == "WARNING"
        assert result.source == "keywords"
        assert "click here" in result.reason

    # Safe message par expected keyword-only Prediction object milna chahiye.
    def test_keyword_fallback_accepts_safe_message(self, tmp_path):
        detector = FraudDetector(tmp_path, "owner/model", "model.safetensors")
        with patch.object(detector, "load_model", return_value=False):
            result = detector.predict("Hello, how are you?")

        assert result == Prediction(
            status="LEGIT",
            reason="No suspicious phrase detected (keyword scan)",
            alert_class="success",
            source="keywords",
        )

    # Sirf spaces wali input ko detector reject karta hai.
    def test_empty_message_is_rejected(self, tmp_path):
        detector = FraudDetector(tmp_path, "owner/model", "model.safetensors")
        with pytest.raises(ValueError, match="cannot be empty"):
            detector.predict("   ")


# HTML, JSON API, history aur health endpoints ka group.
class TestWebApplication:
    # GET / se 200 aur page title milna template location sahi hone ka check hai.
    def test_index_renders(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"SafeGuard AI" in response.data

    # Fake FRAUD verdict ke saath HTML form submit karo; message page mein dikhna chahiye.
    def test_form_analysis_adds_history(self, client, application):
        detector = application.extensions["fraud_detector"]
        with patch.object(
            detector,
            "predict",
            return_value=Prediction("FRAUD", "Detected by AI", "danger", "model", 0.98),
        ):
            response = client.post("/", data={"message": "Test fraud message"})

        assert response.status_code == 200
        assert b"Test fraud message" in response.data

    # Blank JSON message ko 400, valid message ko mocked verdict ke saath 200 milna chahiye.
    def test_json_api_validates_and_analyzes(self, client, application):
        empty_response = client.post("/api/v1/analyze", json={"message": " "})
        assert empty_response.status_code == 400

        detector = application.extensions["fraud_detector"]
        with patch.object(
            detector,
            "predict",
            return_value=Prediction("LEGIT", "Safe", "success", "model", 0.91),
        ):
            response = client.post("/api/v1/analyze", json={"message": "Team lunch at noon"})

        assert response.status_code == 200
        assert response.get_json()["status"] == "LEGIT"

    # 3 scans aur limit 2: latest do results bachne chahiye; clear API unhe khaali kare.
    def test_history_is_bounded_and_can_be_cleared(self, client, application):
        detector = application.extensions["fraud_detector"]
        with patch.object(
            detector,
            "predict",
            return_value=Prediction("LEGIT", "Safe", "success", "keywords"),
        ):
            for message in ("First", "Second", "Third"):
                client.post("/api/v1/analyze", json={"message": message})

        history = application.extensions["analysis_history"].snapshot()
        assert [item["text"] for item in history] == ["Third", "Second"]

        response = client.post("/clear_history", headers={"Accept": "application/json"})
        assert response.status_code == 200
        assert application.extensions["analysis_history"].snapshot() == []

    # Health request model load kiye bina status de; startup lightweight rehna chahiye.
    def test_health_does_not_load_model(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.get_json()["model_loaded"] is False
