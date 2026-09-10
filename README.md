---
title: SafeGuard AI
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 5000
---

# SafeGuard AI

A Flask website for scanning potentially fraudulent messages. A trained DistilBERT model produces predictions, with keyword fallback when the model is unavailable.

## Start here

| Section | Purpose | Guide |
| --- | --- | --- |
| `src/model_training/` | Prepare data → train → evaluate → upload the model to HF | [Training sequence](src/model_training/README.md) |
| `src/web_app/` | Website, API, predictions, and recent history | [Website flow](src/web_app/README.md) |
| `.github/workflows/` | GitHub Actions for tests and website deployment | [Actions flow](.github/workflows/README.md) |

Starting the website does not train the model. Step 04 uploads a trained model to an HF **model repository**; the deployment workflow sends website code to an HF **Space**.

## Reading the code

Each source file starts with an overview, followed by English comments explaining its functions and major blocks. For the website, read `app.py` → `src/web_app/__init__.py` → `settings.py` → `routes.py` → `fraud_detector.py` → prediction/history types → HTML/CSS/JavaScript. For training, read Steps 01–04 and the shared `pipeline_helpers.py`. Test comments explain which behavior is being verified.

`metrics.json` stores generated scores: accuracy measures overall correctness, F1 balances fraud precision and recall, and ROC AUC measures score ranking quality. JSON does not support comments. `uv.lock` is uv's generated dependency record; `.python-version` selects the Python version. Their machine-readable formats are preserved. Model weights, datasets, and `.env` are not annotated.

## File structure

```text
SafeGuard.AI/
├── README.md                            # Start here
├── app.py                               # Website / WSGI entrypoint
├── src/
│   ├── __init__.py                      # Source package marker
│   ├── model_training/
│   │   ├── README.md                    # Inputs, outputs, and commands
│   │   ├── step_01_prepare_data.py      # Prepare the pipeline dataset
│   │   ├── step_02_train_model.py       # Train and save the model locally
│   │   ├── step_03_evaluate_model.py    # Evaluate the saved model
│   │   ├── step_04_upload_to_hf.py      # Explicitly publish the model to HF
│   │   ├── pipeline_helpers.py         # Shared config, data, and metrics helpers
│   │   └── __init__.py                  # Training package marker
│   └── web_app/
│       ├── README.md                    # Request flow and editing guide
│       ├── __init__.py                  # Flask application factory
│       ├── settings.py                  # Environment settings and model path
│       ├── routes.py                    # Page, API, health, and history endpoints
│       ├── fraud_detector.py            # Model loading and predictions
│       ├── prediction_result.py         # Prediction result format
│       ├── scan_history.py              # Recent results in memory
│       ├── templates/index.html         # Page template
│       └── static/
│           ├── css/app.css              # Layout, themes, and animations
│           └── js/app.js                # Browser interactions
├── .github/workflows/
│   ├── README.md                        # CI and deployment explained
│   ├── ci.yml                           # Syntax checks and automated tests
│   └── deploy.yml                       # Sync website code to the HF Space
├── params.yaml                          # Training settings and data/model paths
├── dvc.yaml                             # Pipeline for Steps 01 → 02 → 03
├── data/raw_data/                       # Local source and prepared datasets
├── models/fraud_model_final/            # Saved model and tokenizer
├── metrics.json                         # Training/evaluation scores
├── tests/                               # Regression tests
├── Dockerfile                           # Website container build
├── docker-compose.yml                   # Local container configuration
├── pyproject.toml                       # Dependencies and test settings
└── uv.lock                              # Locked dependency versions
```

Application source lives under `src/`: website code in `src/web_app/` and model-building code in `src/model_training/`. Start the website with the root `app.py`.

`.venv/` contains installed dependencies and `__pycache__/` contains generated Python bytecode; neither needs editing to understand the application. Python executes `__init__.py` when importing a package. The website package exposes the app factory, while the training package's initializer only documents the package.

## Run the website

Python 3.12+ and `uv` are required. Run all commands from the project root:

```bash
uv sync --frozen
uv run python app.py
```

Open `http://localhost:5000`. On the first real analysis, a missing local model is downloaded from `sainivipin/fraud-model-final`. Startup and `/health` do not load the model.

## Training sequence

The current checkout contains the Python scripts. The Step 02 and Step 03 companion notebooks were removed; the [notebook reference](src/model_training/README.md#cell-by-cell-notebooks) applies only if they are restored. The existing `tests/test_notebooks.py` still expects those notebook files.

Place a CSV at `data/raw_data/fraud_dataset.csv` with `text` and `label` columns (`0` = legitimate, `1` = fraud). Configure training in `params.yaml`.

```bash
uv run python -m src.model_training.step_01_prepare_data
uv run python -m src.model_training.step_02_train_model
uv run python -m src.model_training.step_03_evaluate_model
```

After inspecting the metrics, set `HF_TOKEN` in the environment or `.env` and publish explicitly:

```bash
uv run python -m src.model_training.step_04_upload_to_hf
```

**Step 02 does not automatically upload, even when a token is present.** Training can overwrite the configured local model; Step 04 updates the remote model repository.

If DVC is installed, `dvc repro` runs Steps 01–03. Upload is not part of that pipeline. See the [detailed inputs and outputs](src/model_training/README.md).

## API and settings

| Request | Purpose |
| --- | --- |
| `GET /` | Render the website |
| `POST /` | Analyze an HTML form submission |
| `POST /api/v1/analyze` | Analyze a JSON message |
| `GET /health` | Return lightweight health status |
| `POST /clear_history` | Clear this process's history |

Send `Content-Type: application/json` with a body such as `{"message":"Verify your account immediately"}`. Responses contain `text`, `status`, `reason`, `alert_class`, `source`, and `confidence`.

| Setting | Location | Default / purpose |
| --- | --- | --- |
| Data paths, epochs, batch size, learning rate | `params.yaml` | Training configuration |
| `HOST` / `PORT` | Environment / `.env` | `0.0.0.0` / `5000` |
| `FLASK_DEBUG` | Environment / `.env` | `false` |
| `HF_MODEL_REPO` | Environment / `.env` | `sainivipin/fraud-model-final` |
| `MODEL_DIR` | Environment / `.env` | Project's `models/fraud_model_final` |
| `MODEL_FILENAME` | Environment / `.env` | `model.safetensors` |
| `HISTORY_LIMIT` | Environment / `.env` | `10` |
| `MAX_CONTENT_LENGTH` | Environment / `.env` | `65536` bytes |
| `HF_TOKEN` | Environment / `.env`; GitHub secret for deployment | HF access token |

Do not commit `.env`, tokens, local datasets, or model artifacts. If you change the training output directory, set the website's `MODEL_DIR` accordingly.

## Verify and deploy

```bash
uv run --frozen pytest -q
```

Docker starts the root `app.py` on port `5000`. Pushing to GitHub `main` triggers the existing HF Space deployment workflow. See the [Actions guide](.github/workflows/README.md).

Use HTTPS and a production WSGI server in production. History is process-local; shared multi-worker history requires an external store. Fraud verdicts are decision support, so independently verify sensitive requests.
