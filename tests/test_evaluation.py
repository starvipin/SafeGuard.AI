import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
import yaml

from src.model_training.step_03_evaluate_model import evaluate_model


def evaluation_config(tmp_path):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "model"
    data_dir.mkdir()
    model_dir.mkdir()
    pd.DataFrame(
        {
            "text": ["safe hello", "claim now", "meeting today", "urgent otp"],
            "label": [0, 1, 0, 1],
        }
    ).to_csv(data_dir / "dataset.csv", index=False)
    path = tmp_path / "params.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data_source": {
                    "raw_data_dir": str(data_dir),
                    "dataset_name": "dataset.csv",
                },
                "train": {"model_output_dir": str(model_dir), "batch_size": 2},
            }
        ),
        encoding="utf-8",
    )
    return path


@patch("src.model_training.step_03_evaluate_model.train_test_split")
@patch("src.model_training.step_03_evaluate_model.DistilBertTokenizerFast.from_pretrained")
@patch("src.model_training.step_03_evaluate_model.DistilBertForSequenceClassification.from_pretrained")
def test_evaluate_model(mock_model_class, mock_tokenizer_class, mock_split, tmp_path, monkeypatch):
    config_path = evaluation_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    mock_split.return_value = (
        None,
        pd.Series(["safe hello", "claim now"]),
        None,
        pd.Series([0, 1]),
    )
    mock_tokenizer_class.return_value.return_value = {
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 1]],
    }
    model = MagicMock()
    model.return_value.logits = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    mock_model_class.return_value = model

    metrics = evaluate_model(config_path)

    assert metrics == {"accuracy": 1.0, "f1_score": 1.0, "roc_auc": 1.0}
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics


def test_evaluate_model_reports_missing_dataset(tmp_path):
    config_path = evaluation_config(tmp_path)
    (tmp_path / "data" / "dataset.csv").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_model(config_path)
