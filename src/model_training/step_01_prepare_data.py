# STEP 01: copy or convert the source dataset into the configured pipeline location; no model training happens here.
"""Data-ingestion stage."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .pipeline_helpers import load_config, pipeline_data_path


# Read configuration from params.yaml and return the prepared dataset's Path.
def ingest_data(config_path: str | Path = "params.yaml") -> Path:
    config = load_config(config_path)
    # local_path identifies the source; the target combines raw_data_dir and dataset_name.
    source = Path(config["data_source"]["local_path"])
    target = pipeline_data_path(config)
    # Stop with an error if the source file does not exist.
    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    # Create the output directory if needed, allowing an existing directory.
    target.parent.mkdir(parents=True, exist_ok=True)
    source_suffix = source.suffix.lower()
    target_suffix = target.suffix.lower()
    # Use pandas when conversion between CSV and Parquet is requested.
    if source_suffix == ".csv" and target_suffix == ".parquet":
        pd.read_csv(source).to_parquet(target, index=False)
    elif source_suffix == ".parquet" and target_suffix == ".csv":
        pd.read_parquet(source).to_csv(target, index=False)
    else:
        # Otherwise copy the file with its metadata; text and label validation happens in later stages.
        shutil.copy2(source, target)
    return target


# Print the prepared dataset location when the stage runs from the command line.
def main() -> None:
    target = ingest_data()
    print(f"Stage 01 complete: dataset written to '{target}'")


# Run: uv run python -m src.model_training.step_01_prepare_data
if __name__ == "__main__":
    main()
