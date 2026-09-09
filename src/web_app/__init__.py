# Website ka setup: settings padho, detector/history banao, phir URL routes register karo.
"""SafeGuard AI Flask application factory."""

from flask import Flask

from .settings import AppConfig
from .fraud_detector import FraudDetector
from .scan_history import AnalysisHistory
from .routes import web


# Har call par naya Flask app banta hai; tests apni alag settings de sakte hain.
def create_app(config: dict | None = None) -> Flask:
    """Create and configure a SafeGuard AI application instance."""
    # HTML aur CSS/JS ke folders isi package ke andar hain; browser URL /static/... rehta hai.
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    # Pehle default/environment settings lo; diya gaya config unhe override kar sakta hai.
    app.config.from_object(AppConfig)
    if config:
        app.config.update(config)

    # Objects app.extensions mein rakhe hain, taaki sab routes wahi detector aur history use karein.
    app.extensions["fraud_detector"] = FraudDetector.from_config(app.config)
    app.extensions["analysis_history"] = AnalysisHistory(
        max_items=app.config["HISTORY_LIMIT"]
    )
    # Blueprint ke /, /api/v1/analyze, /health aur /clear_history URLs app se jodo.
    app.register_blueprint(web)
    return app


# Package se publicly export hone wale naam ki list.
__all__ = ["create_app"]
