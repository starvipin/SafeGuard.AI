---
title: SafeGuard AI
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 5000
---

# SafeGuard AI

SafeGuard AI is a Flask web app that checks suspicious messages using a DistilBERT fraud classifier plus a keyword fallback. It includes a responsive UI, in-memory analysis history, a clear-history action, tests, and a DVC-friendly ML pipeline.

## Features

- DistilBERT-based fraud/legit classification
- Keyword fallback when the model is unavailable or returns legit with suspicious terms
- AJAX message analysis without a full page refresh
- History panel with clear-history button
- Lazy model loading so imports and tests stay fast
- CSV-based data ingestion, training, and evaluation pipeline
- Pytest test suite and GitHub Actions workflows

## Project Structure

```text
SafeGuard.AI/
├── app.py
├── main.py
├── params.yaml
├── pyproject.toml
├── uv.lock
├── dvc.yaml
├── metrics.json
├── src/
│   ├── stage_01_get_data.py
│   ├── stage_02_train.py
│   ├── stage_03_evaluate.py
│   └── upload_to_hf.py
├── templates/
│   └── index.html
├── tests/
└── .github/workflows/
```

Local datasets, downloaded models, virtual environments, caches, and secrets are ignored by Git.

## Quick Start

Prerequisites:

- Python 3.12+
- uv

Install dependencies:

```bash
uv sync
```

Run the app:

```bash
uv run python app.py
```

Open:

```text
http://localhost:5000
```

## Testing

Run all tests:

```bash
uv run --frozen pytest tests
```

Current verified result:

```text
19 passed
```

## ML Pipeline

The pipeline is configured in `dvc.yaml` and `params.yaml`.

Stage 01 ingests data:

```bash
uv run python src/stage_01_get_data.py
```

It reads `data/raw_data/fraud_dataset.csv` and writes the canonical pipeline dataset to `data/raw_data/dataset.csv`.

Stage 02 trains the model:

```bash
uv run python src/stage_02_train.py
```

It trains DistilBERT and saves the final model to `models/fraud_model_final`. If `HF_TOKEN` is available, it uploads the model through `src/upload_to_hf.py`; otherwise upload is skipped.

Stage 03 evaluates the model:

```bash
uv run python src/stage_03_evaluate.py
```

It loads `models/fraud_model_final`, evaluates on the held-out split, and writes `metrics.json`.

Run the full DVC pipeline if DVC is installed:

```bash
uv run dvc repro
```

## Configuration

```yaml
data_source:
  local_path: data/raw_data/fraud_dataset.csv
  raw_data_dir: data/raw_data
  dataset_name: dataset.csv

train:
  model_name: distilbert-base-uncased
  model_output_dir: models/fraud_model_final
  batch_size: 4
  epochs: 3
  learning_rate: 5e-5
  save_steps: 500
```

## App Routes

- `GET /` renders the UI and history
- `POST /` analyzes a submitted message
- `POST /clear_history` clears in-memory history

## Recent Updates

- Fixed the history delete button so it appears immediately after AJAX analysis, without refreshing the page.
- Moved app model loading to lazy loading.
- Aligned ingestion, training, evaluation, and DVC paths around the DistilBERT model flow.
- Removed the duplicate root Hugging Face upload script; `src/upload_to_hf.py` is the maintained version.
- Removed the direct `pyarrow` dependency and kept the pipeline CSV-based for local Windows stability.

## Notes

- History is stored in memory and clears when the server restarts.
- Production deployments should configure HTTPS and secure secret handling.
- The model directory and raw data are intentionally ignored because they can be large or environment-specific.
