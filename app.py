"""WSGI entrypoint for SafeGuard AI.

The application implementation lives in ``web_app``. Keeping this
small module makes local execution, Docker, and WSGI servers use the same app.
"""

from web_app import create_app


app = create_app()


def predict(text: str) -> tuple[str, str, str]:
    """Compatibility helper for callers that used the original module API."""
    return app.extensions["fraud_detector"].predict(text).as_tuple()


# Compatibility alias for the original public module attribute.
message_history = app.extensions["analysis_history"].items


if __name__ == "__main__":
    app.run(
        debug=app.config["DEBUG"],
        host=app.config["HOST"],
        port=app.config["PORT"],
    )
