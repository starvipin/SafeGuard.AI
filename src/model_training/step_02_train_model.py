# STEP 02: fine-tune DistilBERT on prepared data, calculate metrics, and save the model and tokenizer locally.
"""DistilBERT training stage for fraud-message classification."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from .pipeline_helpers import (
    classification_metrics,
    load_config,
    pipeline_data_path,
    read_dataset,
    write_metrics,
)


# Implement the Dataset interface so DataLoader can retrieve tokenized samples and their true labels.
class FraudDataset(Dataset):
    # Reset label indices after splitting to match the positional indices of the encodings.
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    # Return one sample's input IDs, attention mask, and label as PyTorch tensors.
    def __getitem__(self, index):
        item = {name: torch.tensor(values[index]) for name, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[index])
        return item

    # Tell DataLoader how many samples are available.
    def __len__(self):
        return len(self.labels)


# A fixed seed improves repeatability of random operations; exact GPU determinism is not guaranteed.
def set_reproducible_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# After the final save, remove checkpoint-* directories from the model output's parent directory.
def cleanup_checkpoints(checkpoint_root: str | Path) -> None:
    root = Path(checkpoint_root)
    if not root.exists():
        return
    for path in root.glob("checkpoint-*"):
        if path.is_dir():
            shutil.rmtree(path)


# Measure held-out performance without updating model weights.
def _evaluate(model, loader, device) -> dict[str, float]:
    # Disable training-mode dropout; inference_mode below also disables gradient tracking.
    model.eval()
    predictions, probabilities, labels = [], [], []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            # Collect predicted labels and fraud-class scores (column 1) as CPU arrays.
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    return classification_metrics(labels, predictions, probabilities)


# Read configuration and data; return a metrics dictionary and save model files to disk.
def train_model(config_path: str | Path = "params.yaml") -> dict[str, float]:
    config = load_config(config_path)
    frame = read_dataset(pipeline_data_path(config))
    # Read the model, output path, batch size, epochs, learning rate, and checkpoint interval from configuration.
    settings = config["train"]
    model_name = settings["model_name"]
    model_output_dir = Path(settings.get("model_output_dir", "models/fraud_model_final"))
    batch_size = int(settings["batch_size"])
    epochs = int(settings["epochs"])
    learning_rate = float(settings["learning_rate"])
    save_steps = int(settings["save_steps"])

    # Use an 80/20 training/evaluation split; stratify approximately preserves class proportions.
    set_reproducible_seed()
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        frame["text"],
        frame["label"],
        test_size=0.2,
        random_state=42,
        stratify=frame["label"] if frame["label"].nunique() > 1 else None,
    )

    # Tokenize text with the base model's tokenizer, padding lengths and truncating to the supported limit.
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)
    # Shuffle training batches while keeping evaluation batches in a fixed order.
    train_loader = DataLoader(
        FraudDataset(train_encodings, train_labels), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        FraudDataset(test_encodings, test_labels), batch_size=batch_size, shuffle=False
    )

    # Choose GPU when available; num_labels=2 creates a legitimate/fraud classifier.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    # AdamW updates model weights using gradients and the configured learning rate.
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Store intermediate checkpoints beside the final model directory.
    checkpoint_root = model_output_dir.parent
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    global_step = 0
    # Each epoch processes all training batches once.
    for epoch in range(epochs):
        # Enable training mode and reset the epoch's loss counters.
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            # Clear gradients from the previous batch so they do not accumulate.
            optimizer.zero_grad()
            # Forward pass: supply inputs and true labels to obtain logits and loss.
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            # Backpropagate the loss to compute gradients, then update weights with optimizer.step().
            outputs.loss.backward()
            optimizer.step()
            epoch_loss += float(outputs.loss.item())
            batch_count += 1
            global_step += 1
            # Save a temporary checkpoint every save_steps batches; zero disables intermediate checkpoints.
            if save_steps > 0 and global_step % save_steps == 0:
                model.save_pretrained(checkpoint_root / f"checkpoint-{global_step}")
        # Print the average batch loss to show training progress.
        average_loss = epoch_loss / max(batch_count, 1)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {average_loss:.4f}")

    # Evaluate the held-out split after training and write the metrics to metrics.json.
    metrics = _evaluate(model, test_loader, device)
    write_metrics(metrics)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    # Save both model weights/configuration and the tokenizer; the website requires both.
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    # Clean temporary checkpoints after saving the final files; this stage does not upload to HF.
    cleanup_checkpoints(checkpoint_root)

    return metrics


# Run: uv run python -m src.model_training.step_02_train_model
def main() -> None:
    metrics = train_model()
    print(f"Stage 02 complete: model trained with metrics {metrics}")


# Importing this module does not train the model; command-line execution calls main().
if __name__ == "__main__":
    main()
