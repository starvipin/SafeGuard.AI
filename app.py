# Website yahan se start hoti hai: create_app() setup karta hai, main() server chalata hai.
"""WSGI entrypoint for SafeGuard AI.

The application implementation lives in ``src.web_app``. Keeping this
small module makes local execution, Docker, and WSGI servers use the same app.
"""

# Flask app banane wala function source package se la rahe hain.
from src.web_app import create_app


# WSGI server bhi isi app object ko use kar sakta hai; yahan model train/load nahi hota.
app = create_app()


# Settings ke host, port aur debug option ke saath local web server start karo.
def main() -> None:
    """Start the website using its configured host, port and debug setting."""
    app.run(
        debug=app.config["DEBUG"],
        host=app.config["HOST"],
        port=app.config["PORT"],
    )


# Yeh block sirf python app.py chalane par chalega; import karne par server start nahi hoga.
if __name__ == "__main__":
    main()
