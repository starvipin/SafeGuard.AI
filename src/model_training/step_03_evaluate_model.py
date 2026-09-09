# STEP 03: saved model ko evaluate karo; yeh training ke same seed/split wala hold-out hai, naya unseen benchmark nahi.
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


# Tokenized test messages aur true labels ko DataLoader ke liye tensor samples mein badlo.
class EvaluationDataset(Dataset):
    # Labels ka index reset karo, taaki tokenized rows se positional matching rahe.
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    # Ek test sample ke input IDs, attention mask aur expected label return karo.
    def __getitem__(self, index):
        item = {name: torch.tensor(values[index]) for name, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[index])
        return item

    # Evaluation dataset mein kitne samples hain, woh batao.
    def __len__(self):
        return len(self.labels)


# Config aur prepared dataset padho; final output metrics dictionary aur metrics.json hai.
def evaluate_model(config_path: str | Path = "params.yaml") -> dict[str, float]:
    config = load_config(config_path)
    frame = read_dataset(pipeline_data_path(config))
    train_config = config["train"]
    model_dir = Path(train_config.get("model_output_dir", "models/fraud_model_final"))
    batch_size = int(train_config.get("batch_size", 4))

    # Training jaisa random_state=42 aur 20% test split; unused training portions ko _ mein ignore kiya hai.
    _, test_texts, _, test_labels = train_test_split(
        frame["text"],
        frame["label"],
        test_size=0.2,
        random_state=42,
        stratify=frame["label"] if frame["label"].nunique() > 1 else None,
    )

    # Step 02 ke saved folder se tokenizer aur classifier load karo.
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    # GPU/CPU choose karke model ko wahan bhejo; eval mode training-only behavior band karta hai.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Saare test texts tokenize karo; DataLoader unhe batch_size ke groups mein deta hai.
    encodings = tokenizer(list(test_texts), truncation=True, padding=True)
    loader = DataLoader(
        EvaluationDataset(encodings, test_labels), batch_size=batch_size, shuffle=False
    )
    predictions, probabilities, labels = [], [], []
    # Sirf predictions chahiye; gradients aur optimizer updates nahi hote.
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            # Har batch ke raw scores nikalo; argmax label aur softmax ka column 1 fraud score deta hai.
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    # Collected true labels aur predictions se accuracy/F1/AUC nikalo, phir file update karo.
    metrics = classification_metrics(labels, predictions, probabilities)
    write_metrics(metrics)
    return metrics


# Run: uv run python -m src.model_training.step_03_evaluate_model
def main() -> None:
    metrics = evaluate_model()
    print(f"Stage 03 complete: {metrics}")


if __name__ == "__main__":
    main()
