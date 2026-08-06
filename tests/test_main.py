from unittest.mock import patch

from main import app, main


def test_main_runs_configured_app():
    with patch.object(app, "run") as run:
        main()

    run.assert_called_once_with(
        debug=app.config["DEBUG"],
        host=app.config["HOST"],
        port=app.config["PORT"],
    )
