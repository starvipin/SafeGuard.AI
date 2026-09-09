# STEP 01: original data ko configured pipeline location par copy/format-convert karo; yahan model train nahi hota.
"""Data-ingestion stage."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .pipeline_helpers import load_config, pipeline_data_path


# Input params.yaml hai; output prepared dataset ka Path hai.
def ingest_data(config_path: str | Path = "params.yaml") -> Path:
    config = load_config(config_path)
    # local_path original dataset hai; target raw_data_dir + dataset_name se banta hai.
    source = Path(config["data_source"]["local_path"])
    target = pipeline_data_path(config)
    # Source missing ho to aage badhne ke bajay error do.
    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    # Output folder pehle se na ho to banao; existing folder se error nahi aata.
    target.parent.mkdir(parents=True, exist_ok=True)
    source_suffix = source.suffix.lower()
    target_suffix = target.suffix.lower()
    # CSV se Parquet ya Parquet se CSV chahiye to pandas conversion karta hai.
    if source_suffix == ".csv" and target_suffix == ".parquet":
        pd.read_csv(source).to_parquet(target, index=False)
    elif source_suffix == ".parquet" and target_suffix == ".csv":
        pd.read_parquet(source).to_csv(target, index=False)
    else:
        # Baaki cases mein file ko metadata ke saath copy karo; label/text validation agle steps mein hoti hai.
        shutil.copy2(source, target)
    return target


# Command line se step chalne par output file ki location print karo.
def main() -> None:
    target = ingest_data()
    print(f"Stage 01 complete: dataset written to '{target}'")


# Run: uv run python -m src.model_training.step_01_prepare_data
if __name__ == "__main__":
    main()
