# STEP 03: evaluate the saved model on the same seeded hold-out split used during training, not a new benchmark.
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


# Convert tokenized test messages and true labels into tensor samples for DataLoader.
class EvaluationDataset(Dataset):
    # Reset label indices to keep them aligned with the tokenized rows.
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    # Return one test sample's input IDs, attention mask, and expected label.
    def __getitem__(self, index):
        item = {name: torch.tensor(values[index]) for name, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[index])
        return item

    # Return the number of evaluation samples.
    def __len__(self):
        return len(self.labels)


# Read configuration and prepared data; return metrics and update metrics.json.
def evaluate_model(config_path: str | Path = "params.yaml") -> dict[str, float]:
    config = load_config(config_path)
    frame = read_dataset(pipeline_data_path(config))
    train_config = config["train"]
    model_dir = Path(train_config.get("model_output_dir", "models/fraud_model_final"))
    batch_size = int(train_config.get("batch_size", 4))

    # Recreate the training script's random_state=42 and 20% test split; discard the unused training portions.
    _, test_texts, _, test_labels = train_test_split(
        frame["text"],
        frame["label"],
        test_size=0.2,
        random_state=42,
        stratify=frame["label"] if frame["label"].nunique() > 1 else None,
    )

    # Load the tokenizer and classifier saved by Step 02.
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    # Choose GPU or CPU, move the model there, and disable training-only behavior with eval().
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Tokenize test messages and group them into batches with DataLoader.
    encodings = tokenizer(list(test_texts), truncation=True, padding=True)
    loader = DataLoader(
        EvaluationDataset(encodings, test_labels), batch_size=batch_size, shuffle=False
    )
    predictions, probabilities, labels = [], [], []
    # Generate predictions without tracking gradients or updating weights.
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            # Convert raw scores into predicted labels and fraud scores from softmax column 1.
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    # Calculate accuracy, F1, and ROC AUC from the collected results, then update the metrics file.
    metrics = classification_metrics(labels, predictions, probabilities)
    write_metrics(metrics)
    return metrics


# Run: uv run python -m src.model_training.step_03_evaluate_model
def main() -> None:
    metrics = evaluate_model()
    print(f"Stage 03 complete: {metrics}")


if __name__ == "__main__":
    main()
