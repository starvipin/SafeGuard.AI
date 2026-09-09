# Pytest ke reusable fixtures: temporary config, sample messages aur fake model/tokenizer yahan milte hain.
import pytest
import tempfile
import os
import importlib.abc
import sys
import yaml
from unittest.mock import MagicMock

# Tests mein pandas ke Python string storage ko prefer karte hain.
os.environ.setdefault("PANDAS_STRING_STORAGE", "python")
os.environ.setdefault("PANDAS_FUTURE_INFER_STRING", "0")


# Windows test environment mein pyarrow native-import issue se bachne ke liye uske imports block hote hain.
class _BlockPyArrow(importlib.abc.MetaPathFinder):
    # PyArrow import aaye to ModuleNotFoundError do; baaki imports ko normal process hone do.
    def find_spec(self, fullname, path, target=None):
        if fullname == "pyarrow" or fullname.startswith("pyarrow."):
            raise ModuleNotFoundError("pyarrow is disabled during tests")
        return None


# Yeh import hook test process mein pehle check hota hai; production app mein nahi lagta.
sys.meta_path.insert(0, _BlockPyArrow())

@pytest.fixture
# Fixture ek temporary YAML banata hai; test ko path deta hai aur test ke baad file delete karta hai.
def temp_config():
    """Create a temporary config file for testing."""
    config = {
        "data_source": {
            "local_path": "sample_data.csv",
            "raw_data_dir": "data/raw_data",
            "dataset_name": "dataset.csv"
        },
        "train": {
            "model_name": "distilbert-base-uncased",
            "model_output_dir": "models/fraud_model_final",
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 5e-5,
            "save_steps": 500
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    # yield se pehle setup, uske baad cleanup; pytest test complete hone par execution yahan resume karta hai.
    yield config_path

    # Cleanup
    os.unlink(config_path)

@pytest.fixture
# Chhote sample dataset mein 1 fraud aur 0 legit label hai; real user data ki zaroorat nahi.
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
# MagicMock fake token IDs/masks deta hai, isliye tokenizer download nahi karna padta.
def mock_tokenizer():
    """Mock tokenizer for testing."""
    mock = MagicMock()
    mock.return_value = {
        'input_ids': [[101, 2023, 2003, 1037, 5604, 6251, 102]],
        'attention_mask': [[1, 1, 1, 1, 1, 1, 1]]
    }
    return mock

@pytest.fixture
# Fake model output tests ko predictable banata hai; asli model inference yahan nahi chalti.
def mock_model():
    """Mock model for testing."""
    mock = MagicMock()
    mock.return_value.logits = MagicMock()
    mock.return_value.logits.argmax.return_value.item.return_value = 1
    mock.return_value.logits.softmax.return_value = [[0.2, 0.8]]
    return mock
