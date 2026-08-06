"""SafeGuard AI Flask application factory."""

from flask import Flask

from .config import AppConfig
from .services.detector import FraudDetector
from .services.history import AnalysisHistory
from .web.routes import web


def create_app(config: dict | None = None) -> Flask:
    """Create and configure a SafeGuard AI application instance."""
    app = Flask(
        __name__,
        static_folder="../../static",
        template_folder="../../templates",
    )
    app.config.from_object(AppConfig)
    if config:
        app.config.update(config)

    app.extensions["fraud_detector"] = FraudDetector.from_config(app.config)
    app.extensions["analysis_history"] = AnalysisHistory(
        max_items=app.config["HISTORY_LIMIT"]
    )
    app.register_blueprint(web)
    return app


__all__ = ["create_app"]
