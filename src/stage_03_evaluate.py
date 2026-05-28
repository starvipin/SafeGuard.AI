import json
import os

import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


def read_params(config_path):
    with open(config_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)


def read_evaluation_data(data_path):
    extension = os.path.splitext(data_path)[1].lower()

    if extension == ".parquet":
        return pd.read_parquet(data_path)
    if extension == ".csv":
        return pd.read_csv(data_path, encoding="latin-1")

    raise ValueError(f"Unsupported evaluation data format: {extension}")


class EvaluationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)


def evaluate_model(config_path="params.yaml"):
    config = read_params(config_path)
    data_path = os.path.join(
        config["data_source"]["raw_data_dir"],
        config["data_source"]["dataset_name"],
    )
    model_dir = config["train"].get("model_output_dir", "models/fraud_model_final")
    batch_size = config["train"].get("batch_size", 4)
    metrics_path = "metrics.json"

    print("Loading data for evaluation...")
    df = read_evaluation_data(data_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Evaluation data must contain 'text' and 'label' columns")

    _, test_texts, _, test_labels = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"] if df["label"].nunique() > 1 else None,
    )

    print("Loading trained model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    encodings = tokenizer(list(test_texts), truncation=True, padding=True)
    dataset = EvaluationDataset(encodings, test_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    prediction_probs = []
    true_labels = []

    print("Calculating metrics...")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            prediction_probs.extend(probs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    metrics = {
        "accuracy": round(accuracy_score(true_labels, predictions), 4),
        "f1_score": round(f1_score(true_labels, predictions), 4),
        "roc_auc": round(roc_auc_score(true_labels, prediction_probs), 4)
        if len(set(true_labels)) > 1
        else 0.0,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Stage 03 Success: Evaluation complete. Metrics saved to '{metrics_path}'")
    print(metrics)


if __name__ == "__main__":
    evaluate_model()
