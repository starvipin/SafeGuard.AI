# Check website and detector behavior using mocked predictions to avoid real HF downloads.
from unittest.mock import patch

import pytest

from src.web_app import create_app
from src.web_app.prediction_result import Prediction
from src.web_app.fraud_detector import FraudDetector


@pytest.fixture
# Create a fresh test application with a history limit of two to exercise its boundary.
def application():
    return create_app({"TESTING": True, "HISTORY_LIMIT": 2})


@pytest.fixture
# Flask's test client simulates HTTP requests without starting a server or opening a port.
def client(application):
    return application.test_client()


# Group checks for keyword fallback and empty-message handling.
class TestFraudDetector:
    # Stub model loading to fail and verify that risky phrases produce WARNING.
    def test_keyword_fallback_warns_for_risky_message(self, tmp_path):
        detector = FraudDetector(tmp_path, "owner/model", "model.safetensors")
        with patch.object(detector, "load_model", return_value=False):
            result = detector.predict("Click here to update your account immediately")

        assert result.status == "WARNING"
        assert result.source == "keywords"
        assert "click here" in result.reason

    # Verify that a safe message produces the expected keyword-only Prediction.
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

    # Reject input that contains only whitespace.
    def test_empty_message_is_rejected(self, tmp_path):
        detector = FraudDetector(tmp_path, "owner/model", "model.safetensors")
        with pytest.raises(ValueError, match="cannot be empty"):
            detector.predict("   ")


# Group checks for HTML, JSON, history, and health endpoints.
class TestWebApplication:
    # A successful home-page response and title verify that the template can be located.
    def test_index_renders(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"SafeGuard AI" in response.data

    # Submit an HTML form with a mocked FRAUD verdict and check that the message appears.
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

    # Expect HTTP 400 for blank JSON input and HTTP 200 with the mocked verdict for valid input.
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

    # After three scans with a limit of two, retain the latest two; the clear endpoint must empty the store.
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

    # Health checks must report status without loading the model.
    def test_health_does_not_load_model(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.get_json()["model_loaded"] is False
