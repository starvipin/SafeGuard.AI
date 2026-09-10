# Test the website entrypoint without starting an actual Flask server.
from unittest.mock import patch

from app import app, main


# Mock app.run and check that main() forwards the configured host, port, and debug setting.
def test_main_runs_configured_app():
    with patch.object(app, "run") as run:
        main()

    # Verify both the exact arguments and that the function is called once.
    run.assert_called_once_with(
        debug=app.config["DEBUG"],
        host=app.config["HOST"],
        port=app.config["PORT"],
    )
