# Application setup: load settings, create the detector and history store, then register URL routes.
"""SafeGuard AI Flask application factory."""

from flask import Flask

from .settings import AppConfig
from .fraud_detector import FraudDetector
from .scan_history import AnalysisHistory
from .routes import web


# Each call creates a fresh Flask application; tests can supply their own configuration.
def create_app(config: dict | None = None) -> Flask:
    """Create and configure a SafeGuard AI application instance."""
    # Templates and static assets live inside this package; browser asset URLs remain /static/...
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    # Load default and environment settings first, then apply any supplied overrides.
    app.config.from_object(AppConfig)
    if config:
        app.config.update(config)

    # Store shared services in app.extensions so routes use the same detector and history store.
    app.extensions["fraud_detector"] = FraudDetector.from_config(app.config)
    app.extensions["analysis_history"] = AnalysisHistory(
        max_items=app.config["HISTORY_LIMIT"]
    )
    # Register the blueprint containing the home, analysis, health, and clear-history endpoints.
    app.register_blueprint(web)
    return app


# List the names publicly exported by this package.
__all__ = ["create_app"]
