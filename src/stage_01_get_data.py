"""Backward-compatible entrypoint for data ingestion."""

try:
    from src.safeguard_ai.ml.common import load_config as read_params
    from src.safeguard_ai.ml.ingestion import ingest_data
except ModuleNotFoundError:  # Direct execution: python src/stage_01_get_data.py
    from safeguard_ai.ml.common import load_config as read_params
    from safeguard_ai.ml.ingestion import ingest_data


def get_data(config_path="params.yaml"):
    try:
        return ingest_data(config_path)
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"Stage 01 failed: {error}")
        return None


if __name__ == "__main__":
    result = get_data()
    if result is None:
        raise SystemExit(1)
    print(f"Stage 01 complete: dataset written to '{result}'")
