# HTTP request ka flow: URL se function chunta hai, detector ko call karta hai, HTML/JSON bhejta hai.
"""HTML and JSON routes for the SafeGuard AI web application."""

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, url_for


# Related web URLs ko ek Blueprint group mein rakha hai; factory ise app se jodti hai.
web = Blueprint("web", __name__)


# Current request ke Flask app mein registered detector lo.
def _detector():
    return current_app.extensions["fraud_detector"]


# Current app ka history store lo; har request par naya store nahi banta.
def _history():
    return current_app.extensions["analysis_history"]


# Prediction lo, original message ke saath result banao, history mein rakho aur caller ko do.
def _analyze(message: str) -> dict:
    prediction = _detector().predict(message)
    result = {"text": message, **prediction.to_dict()}
    _history().add(result)
    return result


# GET par page dikhao; POST par HTML form ka message scan karke updated history dikhao.
@web.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if message:
            _analyze(message)
    # Jinja template ko history do; template cards ko HTML mein badalta hai.
    return render_template("index.html", history=_history().snapshot())


# Browser JavaScript yahan JSON bhejta hai; response bhi JSON hota hai.
@web.post("/api/v1/analyze")
def analyze_api():
    # JSON body padho; missing/invalid JSON par empty dictionary milti hai.
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    # Blank message ko HTTP 400 do; valid message ko common _analyze() flow mein bhejo.
    if not message:
        return jsonify({"error": "A non-empty message is required."}), 400
    return jsonify(_analyze(message))


# Fast health check: model load karaye bina current load/history status batata hai.
@web.get("/health")
def health():
    return jsonify(
        status="ok",
        model_loaded=_detector().model_available,
        history_size=len(_history().snapshot()),
    )


# History clear karne ke baad JSON client ko JSON, browser form ko home-page redirect do.
@web.post("/clear_history")
def clear_history():
    _history().clear()
    if request.accept_mimetypes.best == "application/json":
        return jsonify({"status": "cleared"})
    return redirect(url_for("web.index"))
