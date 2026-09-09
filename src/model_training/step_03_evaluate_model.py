"""Model evaluation stage."""

from __future__ import annotations

from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from .pipeline_helpers import (
    classification_metrics,
    load_config,
    pipeline_data_path,
    read_dataset,
    write_metrics,
)


class EvaluationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    def __getitem__(self, index):
        item = {name: torch.tensor(values[index]) for name, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[index])
        return item

    def __len__(self):
        return len(self.labels)


def evaluate_model(config_path: str | Path = "params.yaml") -> dict[str, float]:
    config = load_config(config_path)
    frame = read_dataset(pipeline_data_path(config))
    train_config = config["train"]
    model_dir = Path(train_config.get("model_output_dir", "models/fraud_model_final"))
    batch_size = int(train_config.get("batch_size", 4))

    _, test_texts, _, test_labels = train_test_split(
        frame["text"],
        frame["label"],
        test_size=0.2,
        random_state=42,
        stratify=frame["label"] if frame["label"].nunique() > 1 else None,
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    encodings = tokenizer(list(test_texts), truncation=True, padding=True)
    loader = DataLoader(
        EvaluationDataset(encodings, test_labels), batch_size=batch_size, shuffle=False
    )
    predictions, probabilities, labels = [], [], []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    metrics = classification_metrics(labels, predictions, probabilities)
    write_metrics(metrics)
    return metrics


def main() -> None:
    metrics = evaluate_model()
    print(f"Stage 03 complete: {metrics}")


if __name__ == "__main__":
    main()
