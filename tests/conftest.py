import pytest
import tempfile
import os
import yaml
from unittest.mock import MagicMock

@pytest.fixture
def temp_config():
    """Create a temporary config file for testing."""
    config = {
        "data_source": {
            "local_path": "sample_data.csv",
            "raw_data_dir": "data/raw_data",
            "dataset_name": "fraud_dataset.csv"
        },
        "train": {
            "model_name": "distilbert-base-uncased",
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 5e-5,
            "save_steps": 500
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    os.unlink(config_path)

@pytest.fixture
def sample_data():
    """Create sample fraud data for testing."""
    # Lazy import to avoid access violation on Windows
    import pandas as pd

    data = {
        'text': [
            "Congratulations! You won a lottery. Claim now!",
            "Your account has been suspended. Click here to verify.",
            "Hello, how are you doing today?",
            "Update your payment information immediately.",
            "Thank you for your purchase."
        ],
        'label': [1, 1, 0, 1, 0]  # 1 for fraud, 0 for legit
    }

    df = pd.DataFrame(data)
    return df

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    mock = MagicMock()
    mock.return_value = {
        'input_ids': [[101, 2023, 2003, 1037, 5604, 6251, 102]],
        'attention_mask': [[1, 1, 1, 1, 1, 1, 1]]
    }
    return mock

@pytest.fixture
def mock_model():
    """Mock model for testing."""
    mock = MagicMock()
    mock.return_value.logits = MagicMock()
    mock.return_value.logits.argmax.return_value.item.return_value = 1
    mock.return_value.logits.softmax.return_value = [[0.2, 0.8]]
    return mock