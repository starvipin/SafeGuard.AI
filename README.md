---
title: SafeGuard AI
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 5000
---

# SafeGuard AI

Fraud messages scan karne wali Flask website. Trained DistilBERT model prediction deta hai; model unavailable ho to keyword fallback kaam karta hai.

## Yahan se samajhna shuru karo

| Section | Kya kaam hai? | Guide |
| --- | --- | --- |
| `src/model_training/` | Data → training → evaluation → HF model upload | [Training sequence](src/model_training/README.md) |
| `src/web_app/` | Website, API, prediction aur history | [Website flow](src/web_app/README.md) |
| `.github/workflows/` | GitHub Actions: tests aur website deployment | [Actions flow](.github/workflows/README.md) |

Website chalane par training nahi hoti. Step 04 trained model HF **model repository** mein bhejta hai; GitHub deploy website code HF **Space** mein bhejta hai.

## Simple file structure

```text
SafeGuard.AI/
├── README.md                         ← Pehle yeh padho
├── app.py                            ← Website / WSGI entrypoint
├── src/                             ← Project ka actual source code
│   ├── __init__.py                  ← Source package marker
│   ├── model_training/              ← Sirf model banane ka kaam
│   │   ├── README.md                ← Inputs, outputs aur commands
│   │   ├── step_01_prepare_data.py  ← Dataset pipeline mein lao
│   │   ├── step_02_train_model.py   ← Model train aur local save karo
│   │   ├── step_03_evaluate_model.py ← Saved model ke metrics nikalo
│   │   ├── step_04_upload_to_hf.py  ← Model explicitly HF par bhejo
│   │   ├── pipeline_helpers.py     ← Shared config/data/metrics helpers
│   │   └── __init__.py             ← Training package marker
│   └── web_app/                     ← Website ka poora code
│       ├── README.md                ← Request ka step-by-step flow
│       ├── __init__.py             ← create_app(): Flask app setup
│       ├── settings.py             ← Environment settings aur model path
│       ├── routes.py               ← Page, API, health aur history endpoints
│       ├── fraud_detector.py       ← Model loading + fraud prediction
│       ├── prediction_result.py    ← Prediction result ka format
│       ├── scan_history.py         ← Recent results memory mein
│       ├── templates/index.html    ← Page ka HTML
│       └── static/
│           ├── css/app.css         ← Design aur colors
│           └── js/app.js           ← Browser interactions
├── .github/workflows/                ← GitHub Actions ka section
│   ├── README.md                     ← CI aur deploy ka difference
│   ├── ci.yml                        ← Syntax aur automated tests
│   └── deploy.yml                    ← Website code HF Space par sync
├── params.yaml                       ← Training settings aur data/model paths
├── dvc.yaml                          ← Steps 01 → 02 → 03 ka pipeline
├── data/raw_data/                    ← Local input aur copied dataset
├── models/fraud_model_final/         ← Saved model + tokenizer
├── metrics.json                      ← Training/evaluation scores
├── tests/                            ← Regression tests
├── Dockerfile                        ← Live website ka container
├── docker-compose.yml                ← Local Docker run
├── pyproject.toml                    ← Dependencies + test settings
└── uv.lock                           ← Locked dependency versions
```

Har kaam ka code `src/` ke andar hai: website ke liye `src/web_app/`, model banane ke liye `src/model_training/`. Website start karne ke liye root `app.py` use karo; training ke numbered commands neeche hain.

Editor mein `.venv/` dikhe to woh installed Python dependencies hain; `__pycache__/` Python ka generated cache hai. Project samajhne ke liye inhe padhne/edit karne ki zaroorat nahi. `__init__.py` package import hone par chalti hai; website wali file Flask app banati hai, training wali sirf package ka introduction deti hai.

## Website kaise chalayen

Python 3.12+ aur `uv` chahiye. Saare commands project root se chalao:

```bash
uv sync --frozen
uv run python app.py
```

Browser mein `http://localhost:5000` kholo. First real analysis par local model missing ho to `sainivipin/fraud-model-final` se download hota hai. Startup aur `/health` model load nahi karte.

## Training ka sequence

CSV rakho: `data/raw_data/fraud_dataset.csv`. Columns: `text`, `label` (`0` = legit, `1` = fraud). Settings `params.yaml` mein hain.

```bash
uv run python -m src.model_training.step_01_prepare_data
uv run python -m src.model_training.step_02_train_model
uv run python -m src.model_training.step_03_evaluate_model
```

Metrics check karne ke baad, jab publish karna ho, `.env` ya environment mein `HF_TOKEN` set karke:

```bash
uv run python -m src.model_training.step_04_upload_to_hf
```

**Step 02 ab token present hone par bhi automatic upload nahi karta.** Upload explicit step 04 / upload command se hota hai. Training configured local model overwrite kar sakti hai; step 04 remote model update karta hai.

DVC installed ho to `dvc repro` steps 01–03 chalata hai. Upload DVC ka part nahi hai. [Detailed inputs/outputs](src/model_training/README.md).

## API aur settings

| Request | Kaam |
| --- | --- |
| `GET /` | Website |
| `POST /` | HTML form se scan |
| `POST /api/v1/analyze` | JSON message scan |
| `GET /health` | Lightweight health status |
| `POST /clear_history` | Current process ki history clear |

JSON API ko `Content-Type: application/json` aur body `{"message":"Verify your account immediately"}` bhejo. Response mein `text`, `status`, `reason`, `alert_class`, `source`, `confidence` milte hain.

| Setting | Kahan? | Default / purpose |
| --- | --- | --- |
| Data paths, epochs, batch size, learning rate | `params.yaml` | Training settings |
| `HOST` / `PORT` | Environment / `.env` | `0.0.0.0` / `5000` |
| `FLASK_DEBUG` | Environment / `.env` | `false` |
| `HF_MODEL_REPO` | Environment / `.env` | `sainivipin/fraud-model-final` |
| `MODEL_DIR` | Environment / `.env` | Project ka `models/fraud_model_final` |
| `MODEL_FILENAME` | Environment / `.env` | `model.safetensors` |
| `HISTORY_LIMIT` | Environment / `.env` | `10` |
| `MAX_CONTENT_LENGTH` | Environment / `.env` | `65536` bytes |
| `HF_TOKEN` | Environment / `.env`; deploy ke liye GitHub secret | HF token |

`.env`, tokens, local data aur model artifacts Git mein commit mat karo. Training output path badalne par website ka `MODEL_DIR` bhi uske hisaab se set karo.

## Verify aur deploy

```bash
uv run --frozen pytest -q
```

Docker ab bhi root `app.py` chalata hai, port `5000` par. GitHub `main` par push se existing deploy workflow live HF Space update karta hai. [Actions guide](.github/workflows/README.md).

Production mein HTTPS aur production WSGI server use karo. History process-local hai; shared multi-worker history ke liye external store chahiye. Fraud verdict decision support hai; sensitive requests independently verify karo.
