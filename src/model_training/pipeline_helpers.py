# Training aur evaluation ke shared helpers: config, dataset validation aur metrics ka kaam.
"""Shared configuration, dataset, and metrics helpers for ML stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# params.yaml ko dictionary mein padho; top-level mapping na ho to clear error do.
def load_config(config_path: str | Path = "params.yaml") -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


# Config ke raw_data_dir aur dataset_name ko jodkar prepared dataset ka path banao.
def pipeline_data_path(config: Mapping[str, Any]) -> Path:
    data_config = config["data_source"]
    return Path(data_config["raw_data_dir"]) / data_config["dataset_name"]


# File extension ke hisaab se CSV/Parquet padho, phir common validation lagao.
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


# Sirf text aur label columns chahiye; missing columns ko pehle report karo.
def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"text", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    # Extra columns hatao, null rows drop karo aur surrounding spaces saaf karo.
    cleaned = frame.loc[:, ["text", "label"]].dropna().copy()
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned = cleaned[cleaned["text"] != ""]
    # Labels ko integers mein badlo; uske baad sirf 0 (legit) aur 1 (fraud) allow hain.
    cleaned["label"] = cleaned["label"].astype(int)
    invalid_labels = set(cleaned["label"].unique()).difference({0, 1})
    if invalid_labels:
        raise ValueError("Labels must contain only 0 (legit) or 1 (fraud)")
    # Kam se kam 2 valid rows chahiye; stratified split ke liye har class mein aur examples lag sakte hain.
    if len(cleaned) < 2:
        raise ValueError("Dataset must contain at least two valid rows")
    # Rows drop hone ke baad index ko 0,1,2... karo, taaki positional access sahi rahe.
    return cleaned.reset_index(drop=True)


# Accuracy = overall sahi labels; F1 = fraud precision/recall ka balance; ROC AUC = ranking quality.
def classification_metrics(labels, predictions, probabilities) -> dict[str, float]:
    # ROC AUC ko dono true classes chahiye; single-class data par yahan placeholder 0.0 diya jata hai.
    unique_labels = set(int(label) for label in labels)
    return {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "f1_score": round(f1_score(labels, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(labels, probabilities), 4)
        if len(unique_labels) > 1
        else 0.0,
    }


# Scores ko metrics.json mein likho; nayi run pe purani metrics file replace hoti hai.
def write_metrics(metrics: Mapping[str, float], path: str | Path = "metrics.json") -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(dict(metrics), stream, indent=2)
        stream.write("\n")
