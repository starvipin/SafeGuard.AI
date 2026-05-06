import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from unittest.mock import patch, MagicMock
with patch.dict('sys.modules', {'transformers': MagicMock()}):
    from app import predict

class TestPredictFunction:
    """Test the predict function from app.py"""

    def test_predict_fraud(self):
        """Test prediction of fraud message."""
        import torch
        from app import predict

        with patch('app.tokenizer') as mock_tokenizer, \
             patch('app.model') as mock_model:
            
            mock_tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}
            mock_model.return_value.logits = torch.tensor([[0.1, 0.9]])

            # Test fraud prediction
            status, reason, alert_class = predict("Fake lottery win message")

            assert status == "FRAUD"
            assert "AI Model" in reason
            assert alert_class == "danger"

    def test_predict_legit(self):
        """Test prediction of legitimate message."""
        import torch
        from app import predict

        with patch('app.tokenizer') as mock_tokenizer, \
             patch('app.model') as mock_model:
             
            mock_tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}
            mock_model.return_value.logits = torch.tensor([[0.9, 0.1]])

            # Test legit prediction
            status, reason, alert_class = predict("Hello, how are you?")

            assert status == "LEGIT"
            assert reason == "Safe Message"
            assert alert_class == "success"

    def test_predict_warning(self):
        """Test prediction with suspicious keywords."""
        import torch
        from app import predict

        with patch('app.tokenizer') as mock_tokenizer, \
             patch('app.model') as mock_model:

            mock_tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}
            mock_model.return_value.logits = torch.tensor([[0.9, 0.1]])

            # Test with suspicious keywords
            status, reason, alert_class = predict("Click here to update your account immediately")

            assert status == "WARNING"
            assert "suspicious words found" in reason
            assert alert_class == "warning"



class TestFlaskApp:
    """Test Flask application routes"""

    def setup_method(self, method):
        """Setup test client."""
        from app import app as flask_app
        self.app = flask_app.test_client()
        self.app.testing = True

    @patch('app.predict')
    def test_index_get(self, mock_predict):
        """Test GET request to index."""
        response = self.app.get('/')
        assert response.status_code == 200
        assert b'SafeGuard AI' in response.data

    @patch('app.predict')
    def test_index_post(self, mock_predict):
        """Test POST request to index with message."""
        mock_predict.return_value = ("FRAUD", "Detected by AI", "danger")

        response = self.app.post('/', data={'message': 'Test fraud message'})
        assert response.status_code == 200
        assert b'Test fraud message' in response.data

    @patch('app.predict')
    def test_clear_history(self, mock_predict):
        """Test clear history functionality."""
        # First add a message
        mock_predict.return_value = ("FRAUD", "Detected", "danger")
        self.app.post('/', data={'message': 'Test message'})

        # Clear history
        response = self.app.post('/clear_history')
        assert response.status_code == 302  # Redirect

        # Check history is cleared
        response = self.app.get('/')
        # Should not contain the test message anymore
        assert b'Test message' not in response.data