# Website ka flow

Yahan website aur trained model se prediction ka code hai. Training `model_training/` mein hai.

1. Root `app.py` → `web_app/__init__.py` ka `create_app()` call karta hai.
2. `settings.py` settings padhta hai; factory detector aur history objects register karti hai.
3. Browser `GET /` bhejta hai → `routes.py` → `templates/index.html` render hota hai.
4. Browser `/static/css/app.css` aur `/static/js/app.js` load karta hai.
5. User scan karta hai → JavaScript JSON API ko call karta hai; HTML form POST bhi supported hai.
6. `routes.py` → `fraud_detector.py` ka `predict()` call karta hai. First real scan par model load hota hai; local model missing ho to HF se download hota hai.
7. AI prediction aur suspicious phrases se verdict banta hai. Model unavailable ho to keyword fallback chalta hai.
8. `prediction_result.py` ka `Prediction` result format deta hai; `scan_history.py` result store karta hai; route response bhejta hai aur browser result dikhata hai.

| Kya badalna hai? | File |
| --- | --- |
| Page content / HTML | `templates/index.html` |
| Colors, spacing, design | `static/css/app.css` |
| Scan button / browser behavior | `static/js/app.js` |
| Endpoint / request / response | `routes.py` |
| Prediction / keywords / model loading | `fraud_detector.py` |
| Recent history | `scan_history.py` |
| Result fields | `prediction_result.py` |
| Environment settings | `settings.py` |
| Flask setup | `__init__.py` |

Run: project root se `uv run python app.py`. `main.py` alternate entrypoint hai. `/health` model load nahi karta. Files move hui hain, browser URLs ab bhi `/static/...` hain.

Runtime settings `.env`/environment se aati hain. Default model path project root ka `models/fraud_model_final/` hai. History current process ki memory mein rehti hai; restart par clear ho jati hai.
