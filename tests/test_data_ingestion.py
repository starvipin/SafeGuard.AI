from pathlib import Path

import pandas as pd
import pytest
import yaml

from model_training.pipeline_helpers import load_config, read_dataset
from model_training.step_01_prepare_data import ingest_data


def write_config(path: Path, source: Path, target_dir: Path, name="dataset.csv") -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "data_source": {
                    "local_path": str(source),
                    "raw_data_dir": str(target_dir),
                    "dataset_name": name,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_config(temp_config):
    config = load_config(temp_config)
    assert config["data_source"]["dataset_name"] == "dataset.csv"


def test_ingest_data_copies_csv(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame({"text": ["hello", "claim now"], "label": [0, 1]}).to_csv(
        source, index=False
    )
    target_dir = tmp_path / "pipeline-data"
    config_path = write_config(tmp_path / "params.yaml", source, target_dir)

    target = ingest_data(config_path)

    assert target == target_dir / "dataset.csv"
    assert read_dataset(target).to_dict("list") == {
        "text": ["hello", "claim now"],
        "label": [0, 1],
    }


def test_ingest_data_raises_for_missing_source(tmp_path):
    config_path = write_config(
        tmp_path / "params.yaml", tmp_path / "missing.csv", tmp_path / "data"
    )
    with pytest.raises(FileNotFoundError, match="Source dataset not found"):
        ingest_data(config_path)


def test_dataset_validation_rejects_invalid_labels(tmp_path):
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame({"text": ["one", "two"], "label": [0, 9]}).to_csv(
        dataset, index=False
    )
    with pytest.raises(ValueError, match="Labels must contain only"):
        read_dataset(dataset)
