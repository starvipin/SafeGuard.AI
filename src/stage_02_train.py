import os
import shutil
import tempfile
import yaml
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import json



def read_params(config_path):
    with open(config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

class FraudDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_model(config_path="params.yaml"):
    try:
        config = read_params(config_path)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found")
        return
    except Exception as e:
        print(f"Error reading config: {e}")
        return
    
    # Parameters
    data_path = os.path.join(config["data_source"]["raw_data_dir"], config["data_source"]["dataset_name"])
    model_name = config["train"]["model_name"]
    batch_size = config["train"]["batch_size"]
    epochs = config["train"]["epochs"]
    lr = float(config["train"]["learning_rate"])
    save_steps = config["train"]["save_steps"]
    
    print("Loading data...")
    try:
        df = pd.read_csv(data_path, encoding="latin-1")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    except Exception as e:
        print(f"Error reading data: {e}")
        return
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    print("Tokenizing...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

    train_dataset = FraudDataset(train_encodings, train_labels)
    test_dataset = FraudDataset(test_encodings, test_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)

    print("Starting training...")
    model.train()
    global_step = 0
    
    os.makedirs("models", exist_ok=True)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            # Checkpoint saving mechanism
            if global_step % save_steps == 0:
                checkpoint_dir = f"models/checkpoint-{global_step}"
                model.save_pretrained(checkpoint_dir)
                print(f"Checkpoint saved at step {global_step}")
                
        print(f"Epoch {epoch+1} Loss: {loss.item()}")



    # Evaluate the model on test set
    print("Evaluating model on test set...")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    predictions = []
    prediction_probs = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of positive class
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            prediction_probs.extend(probs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, prediction_probs)
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4)
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Evaluation complete. Metrics saved to metrics.json: {metrics}")



    # Final Model Save to temporary directory and upload
    with tempfile.TemporaryDirectory() as tmp_model_dir:
        model.save_pretrained(tmp_model_dir)
        tokenizer.save_pretrained(tmp_model_dir)
        print(f"Training complete! Final model saved temporarily at {tmp_model_dir}.")

        # Automatically upload to Hugging Face after training
        print("Uploading model to Hugging Face Hub...")
        from upload_to_hf import upload_model
        upload_model(tmp_model_dir)
        print("Model upload complete.")

    # --- NEW CODE: To delete all checkpoints ---
    print("Cleaning up intermediate checkpoints...")
    models_dir = "models"
    if os.path.exists(models_dir):
        for folder_name in os.listdir(models_dir):
            # It will only delete folders whose name starts with 'checkpoint-'
            if folder_name.startswith("checkpoint-"):
                folder_path = os.path.join(models_dir, folder_name)
                if os.path.isdir(folder_path):
                    shutil.rmtree(folder_path)
    print("All intermediate checkpoints deleted successfully!")



if __name__ == "__main__":
    train_model()