# The website starts here: create_app() configures the application, and main() runs the server.
"""WSGI entrypoint for SafeGuard AI.

The application implementation lives in ``src.web_app``. Keeping this
small module makes local execution, Docker, and WSGI servers use the same app.
"""

# Import the application factory from the source package.
from src.web_app import create_app


# WSGI servers can use this app object; creating it does not train or load the model.
app = create_app()


# Run the local web server with the configured host, port, and debug setting.
def main() -> None:
    """Start the website using its configured host, port and debug setting."""
    app.run(
        debug=app.config["DEBUG"],
        host=app.config["HOST"],
        port=app.config["PORT"],
    )


# Run this block only for python app.py; importing the module does not start a server.
if __name__ == "__main__":
    main()
