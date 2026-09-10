# Notebook cells ko temporary project aur tiny model se test karo; real training/HF upload nahi hota.
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import nbformat
import pandas as pd
import pytest
import torch
import yaml
import huggingface_hub
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "src" / "model_training"


def execute_notebook(filename):
    # Notebook source mein outputs/token/data save nahi hone chahiye.
    notebook = nbformat.read(NOTEBOOKS / filename, as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "__notebook__"}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
            exec(compile(cell.source, f"{filename}:cell-{index + 1}", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("start_in_notebook_folder", [False, True])
def test_notebooks_train_save_and_evaluate_cell_by_cell(tmp_path, monkeypatch, start_in_notebook_folder):
    # Root aur nested notebook folder, dono launch locations se path detection verify hoti hai.
    notebook_dir = tmp_path / "src" / "model_training"
    notebook_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({
        "text": [f"example {i}" for i in range(20)],
        "label": [0, 1] * 10,
    }).to_csv(data_dir / "dataset.csv", index=False)
    (tmp_path / "params.yaml").write_text(yaml.safe_dump({
        "data_source": {"raw_data_dir": "data", "dataset_name": "dataset.csv"},
        "train": {
            "model_name": "test-base-model", "model_output_dir": "models/custom-model",
            "batch_size": 2, "epochs": 1, "learning_rate": 0.0007, "save_steps": 1,
        },
    }), encoding="utf-8")
    monkeypatch.chdir(notebook_dir if start_in_notebook_folder else tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("HF_TOKEN", "test-token-not-a-real-secret")
    hub = MagicMock(side_effect=AssertionError("Notebook must not upload"))
    monkeypatch.setattr(huggingface_hub, "HfApi", hub)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))
            self.training_batches = 0

        def forward(self, input_ids, attention_mask, labels=None):
            logits = self.weight.expand(len(input_ids), 2)
            loss = None
            if labels is not None:
                self.training_batches += 1
                loss = torch.nn.functional.cross_entropy(logits, labels)
            return SimpleNamespace(logits=logits, loss=loss)

        def save_pretrained(self, directory):
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "config.json").write_text("{}", encoding="utf-8")

    model = TinyModel()
    model_loads = []

    def load_model(location, **kwargs):
        model_loads.append((location, kwargs))
        if kwargs.get("local_files_only"):
            assert (Path(location) / "config.json").is_file()
        return model

    tokenizer = MagicMock(side_effect=lambda texts, **kwargs: {
        "input_ids": [[1, 2, 3]] * len(texts),
        "attention_mask": [[1, 1, 1]] * len(texts),
    })
    tokenizer.save_pretrained.side_effect = lambda directory: (
        Path(directory) / "tokenizer_config.json"
    ).write_text("{}", encoding="utf-8")

    def load_tokenizer(location, **kwargs):
        if kwargs.get("local_files_only"):
            assert (Path(location) / "tokenizer_config.json").is_file()
        return tokenizer

    monkeypatch.setattr(DistilBertForSequenceClassification, "from_pretrained", load_model)
    monkeypatch.setattr(DistilBertTokenizerFast, "from_pretrained", load_tokenizer)

    trained = execute_notebook("step_02_train_model.ipynb")
    assert trained["PROJECT_ROOT"] == tmp_path
    assert trained["optimizer"].param_groups[0]["lr"] == 0.0007
    assert model.training_batches == 8  # 80% of 20 rows / batch size 2.
    assert len(trained["epoch_losses"]) == 1
    assert (tmp_path / "models" / "custom-model" / "config.json").is_file()
    assert not list((tmp_path / "models").glob("checkpoint-*"))

    # Doosri notebook bhi nested folder se independently setup kar sake.
    monkeypatch.chdir(notebook_dir if start_in_notebook_folder else tmp_path)
    evaluated = execute_notebook("step_03_evaluate_model.ipynb")
    assert evaluated["PROJECT_ROOT"] == tmp_path
    assert len(evaluated["results"]) == 4
    assert evaluated["metrics"] == trained["metrics"]
    assert json.loads((tmp_path / "metrics.json").read_text()) == evaluated["metrics"]
    assert model_loads[-1] == (
        tmp_path / "models" / "custom-model", {"local_files_only": True},
    )
    assert model.training_batches == 8  # Evaluation mein koi training batch nahi chala.
    hub.assert_not_called()
