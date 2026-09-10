# Website request flow

This package contains the website and inference code. Training lives in `src/model_training/`.

1. Root `app.py` calls `create_app()` from `src/web_app/__init__.py`.
2. `settings.py` reads configuration; the factory registers the detector and history store.
3. The browser requests `GET /`; `routes.py` renders `templates/index.html`.
4. The browser loads `/static/css/app.css` and `/static/js/app.js`.
5. When the user scans a message, JavaScript calls the JSON API. Regular HTML form submissions are also supported.
6. `routes.py` calls `predict()` in `fraud_detector.py`. The model loads on the first real scan and downloads from HF if missing locally.
7. Model predictions and suspicious phrases determine the verdict. Keyword fallback handles unavailable models.
8. `prediction_result.py` defines the result format; `scan_history.py` stores it. The route responds, and the browser displays the result.

| What to change | File |
| --- | --- |
| Page content / HTML | `templates/index.html` |
| Colors, spacing, and layout | `static/css/app.css` |
| Scan button / browser behavior | `static/js/app.js` |
| Endpoints / requests / responses | `routes.py` |
| Predictions / keywords / model loading | `fraud_detector.py` |
| Recent history | `scan_history.py` |
| Result fields | `prediction_result.py` |
| Environment settings | `settings.py` |
| Flask application setup | `__init__.py` |

Run `uv run python app.py` from the project root. The `/health` endpoint does not load the model. Static asset URLs remain `/static/...`.

Runtime settings come from the environment or `.env`. The default model location is `models/fraud_model_final/` under the project root. History is held in the current process's memory and clears when it restarts.
