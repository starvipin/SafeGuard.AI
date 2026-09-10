# Shared training and evaluation helpers for configuration, dataset validation, and metrics.
"""Shared configuration, dataset, and metrics helpers for ML stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# Read params.yaml into a dictionary and reject a non-mapping top-level value.
def load_config(config_path: str | Path = "params.yaml") -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


# Join raw_data_dir and dataset_name to locate the prepared dataset.
def pipeline_data_path(config: Mapping[str, Any]) -> Path:
    data_config = config["data_source"]
    return Path(data_config["raw_data_dir"]) / data_config["dataset_name"]


# Read CSV or Parquet based on its extension, then apply common validation.
def read_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(dataset_path, encoding="latin-1")
    elif suffix == ".parquet":
        frame = pd.read_parquet(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix or '<none>'}")
    return validate_dataset(frame)


# Require text and label columns and report missing columns first.
def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"text", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    # Keep the required columns, remove null rows, and trim surrounding text whitespace.
    cleaned = frame.loc[:, ["text", "label"]].dropna().copy()
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned = cleaned[cleaned["text"] != ""]
    # Convert labels to integers, then accept only 0 (legitimate) and 1 (fraud).
    cleaned["label"] = cleaned["label"].astype(int)
    invalid_labels = set(cleaned["label"].unique()).difference({0, 1})
    if invalid_labels:
        raise ValueError("Labels must contain only 0 (legit) or 1 (fraud)")
    # Require at least two valid rows; stratified splitting may need more examples in each class.
    if len(cleaned) < 2:
        raise ValueError("Dataset must contain at least two valid rows")
    # Reset row indices after filtering so positional access remains consistent.
    return cleaned.reset_index(drop=True)


# Accuracy measures overall correctness, F1 balances fraud precision and recall, and ROC AUC measures ranking quality.
def classification_metrics(labels, predictions, probabilities) -> dict[str, float]:
    # ROC AUC requires both true classes; this helper returns a 0.0 placeholder for single-class data.
    unique_labels = set(int(label) for label in labels)
    return {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "f1_score": round(f1_score(labels, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(labels, probabilities), 4)
        if len(unique_labels) > 1
        else 0.0,
    }


# Write scores to metrics.json, replacing the previous metrics file.
def write_metrics(metrics: Mapping[str, float], path: str | Path = "metrics.json") -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(dict(metrics), stream, indent=2)
        stream.write("\n")
