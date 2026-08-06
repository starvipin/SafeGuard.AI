"""HTML and JSON routes for the SafeGuard AI web application."""

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for


web = Blueprint("web", __name__)


def _detector():
    return current_app.extensions["fraud_detector"]


def _history():
    return current_app.extensions["analysis_history"]


def _analyze(message: str) -> dict:
    prediction = _detector().predict(message)
    result = {"text": message, **prediction.to_dict()}
    _history().add(result)
    return result


@web.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if message:
            _analyze(message)
    return render_template("index.html", history=_history().snapshot())


@web.post("/api/v1/analyze")
def analyze_api():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"error": "A non-empty message is required."}), 400
    return jsonify(_analyze(message))


@web.get("/health")
def health():
    return jsonify(
        status="ok",
        model_loaded=_detector().model_available,
        history_size=len(_history().snapshot()),
    )


@web.post("/clear_history")
def clear_history():
    _history().clear()
    if request.accept_mimetypes.best == "application/json":
        return jsonify({"status": "cleared"})
    return redirect(url_for("web.index"))
