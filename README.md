---
title: SafeGuard AI
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 5000
---

# SafeGuard AI

SafeGuard AI is an explainable fraud-message scanner. It combines a fine-tuned DistilBERT classifier with a deterministic phrase check, so the app remains useful when the model is unavailable and can explain risky signals to the user.

## Highlights

- Lazy model loading keeps startup and health checks fast.
- Hybrid verdicts distinguish model, keyword, and combined results.
- Responsive, accessible interface with dark/light themes and no artificial scan delay.
- Versioned JSON API for integrations.
- Bounded, thread-safe in-memory history.
- Reusable ingestion, training, evaluation, metrics, and Hugging Face publishing modules.
- Strict pytest CI and DVC-compatible ML stages.

## Architecture

```text
SafeGuard.AI/
├── app.py                         # WSGI/deployment entrypoint
├── main.py                        # Local development entrypoint
├── src/
│   └── safeguard_ai/
│       ├── __init__.py            # Flask application factory
│       ├── config.py              # Environment-driven configuration
│       ├── domain.py              # Typed domain results
│       ├── services/
│       │   ├── detector.py        # Model lifecycle + hybrid detection
│       │   └── history.py         # Thread-safe bounded history
│       ├── web/
│       │   └── routes.py          # HTML, JSON API, and health routes
│       └── ml/
│           ├── common.py          # Dataset/config/metrics utilities
│           ├── ingestion.py
│           ├── training.py
│           ├── evaluation.py
│           └── hub.py
├── static/                        # Versioned CSS and JavaScript
├── templates/                     # Jinja templates
├── tests/                         # Unit and integration tests
├── params.yaml                    # ML pipeline configuration
└── dvc.yaml                       # Reproducible pipeline stages
```

The root entrypoint creates the Flask app through an application factory. Routes depend on services registered in `app.extensions`, keeping HTTP concerns separate from detection and state management. ML stages reuse validated I/O and metric helpers instead of duplicating pipeline code.

## Quick start

Prerequisites: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen
uv run python app.py
```

Open `http://localhost:5000`.

The app downloads `sainivipin/fraud-model-final` on the first real analysis if the model is not already present. If loading fails, the scanner safely falls back to its phrase detector.

## API

Analyze a message:

```bash
curl -X POST http://localhost:5000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"message":"Verify your account immediately"}'
```

Example response:

```json
{
  "alert_class": "warning",
  "confidence": null,
  "reason": "Suspicious phrases found: immediately, account",
  "source": "keywords",
  "status": "WARNING",
  "text": "Verify your account immediately"
}
```

Other endpoints:

- `GET /` — web interface
- `GET /health` — lightweight service health
- `POST /clear_history` — clear current process history

## Configuration

Runtime settings are read from environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | Web bind address |
| `PORT` | `5000` | Web port |
| `FLASK_DEBUG` | `false` | Development debug mode |
| `HF_MODEL_REPO` | `sainivipin/fraud-model-final` | Model repository |
| `MODEL_DIR` | `models/fraud_model_final` | Local model directory |
| `HISTORY_LIMIT` | `10` | Maximum recent results |
| `HF_TOKEN` | unset | Optional model publishing token |

Never commit `.env` or access tokens.

## Testing

```bash
uv run --frozen pytest -q
```

Verified after the architecture refactor: `20 passed`.

## ML pipeline

If DVC is installed, run the full pipeline from the synced project environment:

```bash
dvc repro
```

Or run individual modules:

```bash
uv run python -m src.safeguard_ai.ml.ingestion
uv run python -m src.safeguard_ai.ml.training
uv run python -m src.safeguard_ai.ml.evaluation
```

The local source dataset is expected at `data/raw_data/fraud_dataset.csv`; generated data and model artifacts remain outside Git.

To explicitly publish a trained model:

```bash
HF_TOKEN=... uv run python src/upload_to_hf.py
```

## Production notes

- Run behind HTTPS and a production WSGI server.
- The bundled history store is process-local; use Redis or a database for shared multi-worker history.
- Treat the verdict as decision support. Users should independently verify payment, credential, and account-security requests.
