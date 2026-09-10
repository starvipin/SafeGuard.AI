# HTTP flow: select the handler by URL, call the detector, and return HTML or JSON.
"""HTML and JSON routes for the SafeGuard AI web application."""

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for


# Group related URLs in a blueprint that the application factory registers.
web = Blueprint("web", __name__)


# Retrieve the detector registered on the Flask application handling this request.
def _detector():
    return current_app.extensions["fraud_detector"]


# Retrieve the shared history store instead of creating a new one for each request.
def _history():
    return current_app.extensions["analysis_history"]


# Predict, combine the verdict with the original message, store the result, and return it.
def _analyze(message: str) -> dict:
    prediction = _detector().predict(message)
    result = {"text": message, **prediction.to_dict()}
    _history().add(result)
    return result


# GET renders the page; POST analyzes the HTML form input and displays updated history.
@web.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if message:
            _analyze(message)
    # Pass history to Jinja so it can render result cards as HTML.
    return render_template("index.html", history=_history().snapshot())


# Browser JavaScript sends JSON to this endpoint and receives a JSON response.
@web.post("/api/v1/analyze")
def analyze_api():
    # Parse the JSON body; missing or invalid JSON falls back to an empty dictionary.
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    # Reject blank messages with HTTP 400; send valid messages through the shared analysis function.
    if not message:
        return jsonify({"error": "A non-empty message is required."}), 400
    return jsonify(_analyze(message))


# Report current model and history status without triggering model loading.
@web.get("/health")
def health():
    return jsonify(
        status="ok",
        model_loaded=_detector().model_available,
        history_size=len(_history().snapshot()),
    )


# Clear history, then return JSON to API clients or redirect an HTML form back to the home page.
@web.post("/clear_history")
def clear_history():
    _history().clear()
    if request.accept_mimetypes.best == "application/json":
        return jsonify({"status": "cleared"})
    return redirect(url_for("web.index"))
