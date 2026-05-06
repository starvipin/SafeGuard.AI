import pytest
import os
import tempfile
import json
import pandas as pd
import joblib
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestClassifier

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestEvaluateModel:
    """Test evaluation functionality"""

    def setup_method(self):
        """Setup test data and model."""
        self.temp_dir = tempfile.mkdtemp()

        # Create sample data
        self.data_path = os.path.join(self.temp_dir, "data", "raw_data", "dataset.parquet")
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

        # Create sample parquet data (simplified for testing)
        data = {
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'resp': [1, -1, 1, -1, 1, -1],
            'date': [1, 1, 1, 1, 1, 1],
            'ts_id': [1, 2, 3, 4, 5, 6],
            'resp_1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'resp_2': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            'resp_3': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'resp_4': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }
        df = pd.DataFrame(data)
        df.to_parquet(self.data_path)

        # Create sample model
        self.model_path = os.path.join(self.temp_dir, "models", "model.pkl")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_sample = df.drop(columns=['resp', 'date', 'ts_id', 'resp_1', 'resp_2', 'resp_3', 'resp_4'])
        y_sample = (df['resp'] > 0).astype(int)
        model.fit(X_sample, y_sample)
        joblib.dump(model, self.model_path)

        # Change to temp directory for testing
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def teardown_method(self):
        """Cleanup."""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('stage_03_evaluate.train_test_split')
    def test_evaluate_model(self, mock_split):
        """Test model evaluation."""
        # Mock train_test_split to return predictable splits
        X_test = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3]
        })
        y_test = pd.Series([0, 1, 0])

        mock_split.return_value = (None, X_test, None, y_test)

        from stage_03_evaluate import evaluate_model

        # Run evaluation
        evaluate_model()

        # Check if metrics file was created
        metrics_path = "metrics.json"
        assert os.path.exists(metrics_path)

        # Check metrics content
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_evaluate_model_file_not_found(self):
        """Test evaluation when files don't exist."""
        # Remove files
        os.remove(self.data_path)
        os.remove(self.model_path)

        from stage_03_evaluate import evaluate_model

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            evaluate_model()