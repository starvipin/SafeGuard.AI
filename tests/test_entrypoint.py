# Website ke entrypoint ko test karo, bina actual Flask server start kiye.
from unittest.mock import patch

from app import app, main


# app.run ko mock karo; main() configured debug, host aur port hi pass kare.
def test_main_runs_configured_app():
    with patch.object(app, "run") as run:
        main()

    # Arguments aur exactly ek call dono verify hote hain.
    run.assert_called_once_with(
        debug=app.config["DEBUG"],
        host=app.config["HOST"],
        port=app.config["PORT"],
    )
