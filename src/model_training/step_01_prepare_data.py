"""Data-ingestion stage."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .pipeline_helpers import load_config, pipeline_data_path


def ingest_data(config_path: str | Path = "params.yaml") -> Path:
    config = load_config(config_path)
    source = Path(config["data_source"]["local_path"])
    target = pipeline_data_path(config)
    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    source_suffix = source.suffix.lower()
    target_suffix = target.suffix.lower()
    if source_suffix == ".csv" and target_suffix == ".parquet":
        pd.read_csv(source).to_parquet(target, index=False)
    elif source_suffix == ".parquet" and target_suffix == ".csv":
        pd.read_parquet(source).to_csv(target, index=False)
    else:
        shutil.copy2(source, target)
    return target


def main() -> None:
    target = ingest_data()
    print(f"Stage 01 complete: dataset written to '{target}'")


if __name__ == "__main__":
    main()
