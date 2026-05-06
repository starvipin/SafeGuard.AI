from flask import Flask, render_template, request, redirect

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

# Hugging Face Hub config
HF_USERNAME = "sainivipin"
MODEL_REPO = "fraud-model-final"
MODEL_FILENAME = "model.safetensors"

# Download model from Hugging Face if not present
model_dir = "models/fraud_model_final"
model_path = os.path.join(model_dir, MODEL_FILENAME)
if not os.path.exists(model_path):
    os.makedirs(model_dir, exist_ok=True)
    print("Downloading model from Hugging Face Hub...")
    model_path = hf_hub_download(repo_id=f"{HF_USERNAME}/{MODEL_REPO}", filename=MODEL_FILENAME)
    # Move downloaded file to model_dir if needed
    if not model_path.startswith(model_dir):
        import shutil
        shutil.copy(model_path, os.path.join(model_dir, MODEL_FILENAME))
        model_path = os.path.join(model_dir, MODEL_FILENAME)
    print(f"Model downloaded to {model_path}")


app = Flask(__name__)

# Model load karna
print("Loading model and tokenizer...")
if os.path.exists(model_dir):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}!")
else:
    print("Model not found, running in test mode or without model.")
    tokenizer = None
    model = None
    device = None

all_fraud_keywords = [
    "immediately", "24 hours", "tonight", "blocked", "suspended", "asap", 
    "click here", "update kyc", "verify now", "call this number", "log in", "link below",
    "lottery", "cashback", "winner", "work from home", "easy money", 
    "prize", "bonus", "gift", "claim now","account", 
    "police", "arrest", "fir", "court case", "legal notice", "warrant", 
    "seized", "customs", "disconnected", "terminated", "jail",
    "income tax", "rbi", "govt", "support team", "admin", "manager", 
    "ceo", "officer", "cbi", "narcotics",
    "unusual login", "device detected", "password changed", "security alert", 
    "unauthorized access", "validation code", "otp", "reset pin", "account frozen",
    "limited time", "expires today", "last chance", "offer ends", "urgent attention",
    "wire transfer", "refund", "credit", "debit", "pending transaction", 
    "invoice", "bill payment", "processing fee", "bitcoin", "wallet address","wallet",
]

# History store karne ke liye ek list
message_history = []

def predict(text):
    if model is None or tokenizer is None:
        # Fallback to keyword-based detection if model not loaded
        text_lower = text.lower()
        found_keywords = [word for word in all_fraud_keywords if word in text_lower]
        if len(found_keywords) > 0:
            return "WARNING", f"Suspicious words found: {', '.join(found_keywords)}", "warning"
        else:
            return "LEGIT", "Safe Message", "success"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs).item()
    print("DEBUG LABEL:", label, type(label))

    if label == 1:
        return "FRAUD", "Detected by AI Model", "danger"
    else:
        text_lower = text.lower()
        found_keywords = [word for word in all_fraud_keywords if word in text_lower]
        
        if len(found_keywords) > 0:
            return "WARNING", f"Model said LEGIT, but suspicious words found: {', '.join(found_keywords)}", "warning"
        else:
            return "LEGIT", "Safe Message", "success"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        msg = request.form.get("message", "").strip()
        if msg:
            status, reason, alert_class = predict(msg)
            # Naya result history list mein sabse upar (index 0) daalna
            message_history.insert(0, {
                "text": msg,
                "status": status,
                "reason": reason,
                "alert_class": alert_class
            })

    # History ko template mein bhej rahe hain
    return render_template("index.html", history=message_history)

@app.route("/clear_history", methods=["POST"])
def clear_history():
    message_history.clear()
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)