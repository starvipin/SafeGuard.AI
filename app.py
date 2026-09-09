"""WSGI entrypoint for SafeGuard AI.

The application implementation lives in ``src.web_app``. Keeping this
small module makes local execution, Docker, and WSGI servers use the same app.
"""

from src.web_app import create_app


app = create_app()


def main() -> None:
    """Start the website using its configured host, port and debug setting."""
    app.run(
        debug=app.config["DEBUG"],
        host=app.config["HOST"],
        port=app.config["PORT"],
    )


if __name__ == "__main__":
    main()
