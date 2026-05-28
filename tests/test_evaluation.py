import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
import yaml

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestEvaluateModel:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data", "raw_data")
        self.model_dir = os.path.join(self.temp_dir, "models", "fraud_model_final")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.data_path = os.path.join(self.data_dir, "dataset.csv")
        pd.DataFrame({
            "text": ["safe hello", "claim now", "meeting today", "urgent otp"],
            "label": [0, 1, 0, 1],
        }).to_csv(self.data_path, index=False)

        self.config_path = os.path.join(self.temp_dir, "params.yaml")
        with open(self.config_path, "w") as f:
            yaml.safe_dump({
                "data_source": {
                    "raw_data_dir": self.data_dir,
                    "dataset_name": "dataset.csv",
                },
                "train": {
                    "model_output_dir": self.model_dir,
                    "batch_size": 2,
                },
            }, f)

        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def teardown_method(self):
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch("stage_03_evaluate.train_test_split")
    @patch("stage_03_evaluate.DistilBertTokenizerFast.from_pretrained")
    @patch("stage_03_evaluate.DistilBertForSequenceClassification.from_pretrained")
    def test_evaluate_model(self, mock_model_class, mock_tokenizer_class, mock_split):
        mock_split.return_value = (
            None,
            pd.Series(["safe hello", "claim now"]),
            None,
            pd.Series([0, 1]),
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
        }
        mock_tokenizer_class.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.return_value.logits = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        mock_model_class.return_value = mock_model

        from stage_03_evaluate import evaluate_model

        evaluate_model(self.config_path)

        with open("metrics.json", "r") as f:
            metrics = json.load(f)

        assert metrics == {
            "accuracy": 1.0,
            "f1_score": 1.0,
            "roc_auc": 1.0,
        }

    def test_evaluate_model_file_not_found(self):
        os.remove(self.data_path)

        from stage_03_evaluate import evaluate_model

        with pytest.raises(FileNotFoundError):
            evaluate_model(self.config_path)
