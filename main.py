"""Local development entrypoint."""

from app import app


def main() -> None:
    app.run(
        debug=app.config["DEBUG"],
        host=app.config["HOST"],
        port=app.config["PORT"],
    )


if __name__ == "__main__":
    main()
