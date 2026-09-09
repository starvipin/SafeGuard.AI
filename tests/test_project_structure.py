# Integration checks: moved assets/model path, real data CLI, tiny training loop aur mocked HF upload.
"""Regression checks for relocated code, assets and explicit publishing."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import torch
import yaml

from src.web_app import create_app


# Test file se project root nikalte hain, taaki temporary working directory mein bhi source mil sake.
ROOT = Path(__file__).resolve().parents[1]


# Dusre working folder se app banao; HTML, CSS/JS bytes, root model path aur health verify karo.
def test_web_assets_and_model_root_work_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    application = create_app({"TESTING": True})
    client = application.test_client()
    page = client.get("/")
    assert page.status_code == 200
    for relative in ("css/app.css", "js/app.js"):
        url = f"/static/{relative}"
        assert url.encode() in page.data
        response = client.get(url)
        assert response.status_code == 200
        assert response.data == (ROOT / "src" / "web_app" / "static" / relative).read_bytes()
    from src.web_app.settings import PROJECT_ROOT

    assert PROJECT_ROOT == ROOT
    assert client.get("/health").get_json()["model_loaded"] is False


# Nayi Python process mein documented step 01 command chalao, par input/output temporary folders mein rakho.
def test_data_preparation_command(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("text,label\nhello,0\nclaim now,1\n", encoding="utf-8")
    (tmp_path / "params.yaml").write_text(yaml.safe_dump({
        "data_source": {
            "local_path": str(source),
            "raw_data_dir": str(tmp_path / "prepared"),
            "dataset_name": "dataset.csv",
        },
    }), encoding="utf-8")
    # Child process ko PYTHONPATH se project root dikhate hain, taaki src package import ho sake.
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-m", "src.model_training.step_01_prepare_data"], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "prepared" / "dataset.csv").read_bytes() == source.read_bytes()


# Fake HF token hone par bhi training local save kare; HfApi call aaye to test fail ho.
def test_training_saves_locally_without_upload_even_with_token(tmp_path, monkeypatch):
    from src.model_training import step_02_train_model as training
    import huggingface_hub

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "test-token-not-a-real-secret")
    hub = MagicMock(side_effect=AssertionError("Training must not publish"))
    monkeypatch.setattr(huggingface_hub, "HfApi", hub)
    pd.DataFrame({"text": [f"message {i}" for i in range(10)],
                  "label": [0, 1] * 5}).to_csv("dataset.csv", index=False)
    (tmp_path / "params.yaml").write_text(yaml.safe_dump({
        "data_source": {"raw_data_dir": ".", "dataset_name": "dataset.csv"},
        "train": {"model_name": "test-model", "model_output_dir": "saved-model",
                  "batch_size": 2, "epochs": 1, "learning_rate": 0.001,
                  "save_steps": 0},
    }), encoding="utf-8")

    # Asli DistilBERT ki jagah do trainable weights ka chhota PyTorch model; training loop phir bhi real chalta hai.
    class TinyModel(torch.nn.Module):
        # Tiny learnable parameter banao; save_pretrained mock disk par real weights nahi likhta.
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))
            self.save_pretrained = MagicMock()

        # Har sample ke do class scores aur labels present hon to differentiable cross-entropy loss do.
        def forward(self, input_ids, attention_mask, labels=None):
            logits = self.weight.expand(len(input_ids), 2)
            loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
            return SimpleNamespace(logits=logits, loss=loss)

    # Deterministic token IDs se network/model downloads avoid karte hain.
    tokenizer = MagicMock(side_effect=lambda texts, **kwargs: {
        "input_ids": [[1, 2]] * len(texts),
        "attention_mask": [[1, 1]] * len(texts),
    })
    model = TinyModel()
    # CPU force karke test ko GPU availability se independent rakho.
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training.DistilBertTokenizerFast, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(training.DistilBertForSequenceClassification, "from_pretrained", lambda *a, **k: model)

    metrics = training.train_model()

    assert set(metrics) == {"accuracy", "f1_score", "roc_auc"}
    assert (tmp_path / "metrics.json").is_file()
    # Final output path model aur tokenizer dono ko milna chahiye; hub bilkul call nahi hona chahiye.
    model.save_pretrained.assert_called_once_with(Path("saved-model"))
    tokenizer.save_pretrained.assert_called_once_with(Path("saved-model"))
    hub.assert_not_called()


# Step 04 custom local folder/repository aur .env loading use kare; actual HF network call mock hai.
def test_upload_command_uses_configured_model_and_repository(tmp_path, monkeypatch):
    from src.model_training import step_04_upload_to_hf as publishing
    import dotenv
    import huggingface_hub

    # dotenv loader mock hai; real credentials padhne ki zaroorat nahi.
    load_env = MagicMock()
    monkeypatch.setattr(dotenv, "load_dotenv", load_env)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "test-token-not-a-real-secret")
    monkeypatch.setenv("HF_MODEL_REPO", "test-owner/test-model")
    model_dir = tmp_path / "custom-model"
    model_dir.mkdir()
    (tmp_path / "params.yaml").write_text(yaml.safe_dump({
        "train": {"model_output_dir": str(model_dir)},
    }), encoding="utf-8")
    # HfApi ko MagicMock se replace karte hain; publish behavior verify hota hai, real upload nahi.
    api = MagicMock()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: api)

    publishing.main()

    load_env.assert_called_once_with()
    api.create_repo.assert_called_once_with(
        repo_id="test-owner/test-model", repo_type="model", exist_ok=True,
    )
    assert api.upload_folder.call_args.kwargs["folder_path"] == model_dir
    assert api.upload_folder.call_args.kwargs["repo_id"] == "test-owner/test-model"
