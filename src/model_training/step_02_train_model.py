# STEP 02: prepared data se DistilBERT fine-tune karo, metrics nikalo aur model/tokenizer local save karo.
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


# DataLoader ko ek sample dene ke liye tokenized text aur true labels ko Dataset interface dete hain.
class FraudDataset(Dataset):
    # Split ke baad labels ke purane indices hatao; encodings ke positional indices se match karao.
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    # Index par ek sample ke input_ids, attention_mask aur label ko PyTorch tensors mein do.
    def __getitem__(self, index):
        item = {name: torch.tensor(values[index]) for name, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[index])
        return item

    # DataLoader ko total samples ki sankhya chahiye.
    def __len__(self):
        return len(self.labels)


# Same seed se random split/training repeat karna aasaan hota hai; GPU par exact determinism guaranteed nahi.
def set_reproducible_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Final save ke baad checkpoint-* folders hatao; model output ke parent folder mein search hoti hai.
def cleanup_checkpoints(checkpoint_root: str | Path) -> None:
    root = Path(checkpoint_root)
    if not root.exists():
        return
    for path in root.glob("checkpoint-*"):
        if path.is_dir():
            shutil.rmtree(path)


# Training ke held-out samples par bina weights update kiye metrics nikalo.
def _evaluate(model, loader, device) -> dict[str, float]:
    # Training-mode dropout band karo; inference_mode neeche gradients bhi disable karta hai.
    model.eval()
    predictions, probabilities, labels = [], [], []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            # Winning labels aur fraud class (column 1) ke scores CPU arrays mein collect karo.
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    return classification_metrics(labels, predictions, probabilities)


# Config/data padho; return value metrics dictionary hai, model files disk par save hoti hain.
def train_model(config_path: str | Path = "params.yaml") -> dict[str, float]:
    config = load_config(config_path)
    frame = read_dataset(pipeline_data_path(config))
    # Model name, output path, batch size, epochs, learning rate aur checkpoint interval config se lo.
    settings = config["train"]
    model_name = settings["model_name"]
    model_output_dir = Path(settings.get("model_output_dir", "models/fraud_model_final"))
    batch_size = int(settings["batch_size"])
    epochs = int(settings["epochs"])
    learning_rate = float(settings["learning_rate"])
    save_steps = int(settings["save_steps"])

    # 80% training aur 20% evaluation; stratify dono classes ka ratio roughly preserve karta hai.
    set_reproducible_seed()
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        frame["text"],
        frame["label"],
        test_size=0.2,
        random_state=42,
        stratify=frame["label"] if frame["label"].nunique() > 1 else None,
    )

    # Base model ka tokenizer text ko numbers mein badalta hai; padding lengths match aur truncation length limit rakhta hai.
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)
    # Training batches shuffle hote hain; evaluation batches ka order fixed rehta hai.
    train_loader = DataLoader(
        FraudDataset(train_encodings, train_labels), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        FraudDataset(test_encodings, test_labels), batch_size=batch_size, shuffle=False
    )

    # GPU mile to use karo; num_labels=2 ka matlab legit/fraud classifier.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    # AdamW gradients ke basis par model weights ko learning_rate ke hisaab se update karta hai.
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Intermediate checkpoints final model directory ke parent mein rakhe jate hain.
    checkpoint_root = model_output_dir.parent
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    global_step = 0
    # Ek epoch mein poore training dataset ke batches ek baar process hote hain.
    for epoch in range(epochs):
        # Har epoch mein training mode on karo aur us epoch ka loss counter reset karo.
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            # Pichhle batch ke gradients clear karo; warna gradients add hote rahenge.
            optimizer.zero_grad()
            # Forward pass: text inputs aur true labels do; model logits aur loss deta hai.
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            # Backward pass gradients nikalta hai; optimizer.step() unse weights update karta hai.
            outputs.loss.backward()
            optimizer.step()
            epoch_loss += float(outputs.loss.item())
            batch_count += 1
            global_step += 1
            # Har save_steps batches par temporary checkpoint; 0 hone par intermediate saving band.
            if save_steps > 0 and global_step % save_steps == 0:
                model.save_pretrained(checkpoint_root / f"checkpoint-{global_step}")
        # Batch losses ka average terminal par dikhao; training progress samajhne mein madad milti hai.
        average_loss = epoch_loss / max(batch_count, 1)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {average_loss:.4f}")

    # Training ke baad held-out split evaluate karo aur metrics.json mein scores likho.
    metrics = _evaluate(model, test_loader, device)
    write_metrics(metrics)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    # Model weights/config aur tokenizer dono save karo; website ko prediction ke liye dono chahiye.
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    # Final files save hone ke baad temporary checkpoints clean karo; HF upload is step mein nahi hota.
    cleanup_checkpoints(checkpoint_root)

    return metrics


# Run: uv run python -m src.model_training.step_02_train_model
def main() -> None:
    metrics = train_model()
    print(f"Stage 02 complete: model trained with metrics {metrics}")


# Module import par training nahi chalegi; command line execution par hi main() chalega.
if __name__ == "__main__":
    main()
