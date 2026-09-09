import pandas as pd
import torch

from model_training.pipeline_helpers import classification_metrics, load_config
from model_training.step_02_train_model import FraudDataset, cleanup_checkpoints


def test_training_config_can_be_loaded(temp_config):
    config = load_config(temp_config)
    assert config["train"]["model_name"] == "distilbert-base-uncased"


def test_fraud_dataset_builds_tensors():
    dataset = FraudDataset(
        {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
        },
        pd.Series([0, 1], index=[8, 9]),
    )

    assert len(dataset) == 2
    assert torch.equal(dataset[1]["input_ids"], torch.tensor([4, 5, 6]))
    assert dataset[1]["labels"].item() == 1


def test_classification_metrics_handles_single_class():
    metrics = classification_metrics([1, 1], [1, 1], [0.9, 0.8])
    assert metrics == {"accuracy": 1.0, "f1_score": 1.0, "roc_auc": 0.0}


def test_cleanup_checkpoints_only_removes_checkpoints(tmp_path):
    checkpoint_one = tmp_path / "checkpoint-100"
    checkpoint_two = tmp_path / "checkpoint-200"
    final_model = tmp_path / "fraud_model_final"
    for directory in (checkpoint_one, checkpoint_two, final_model):
        directory.mkdir()

    cleanup_checkpoints(tmp_path)

    assert not checkpoint_one.exists()
    assert not checkpoint_two.exists()
    assert final_model.exists()
