"""
Script to load model from Hugging Face Hub
"""
from huggingface_hub import hf_hub_download
import os

HF_USERNAME = "sainivipin"
MODEL_REPO = "fraud-model-final"
MODEL_FILENAME = "model.safetensors"  # Change if your model file name is different

# Download model file from Hugging Face Hub
def download_model():
    model_path = hf_hub_download(repo_id=f"{HF_USERNAME}/{MODEL_REPO}", filename=MODEL_FILENAME)
    print(f"Model downloaded to: {model_path}")
    return model_path

if __name__ == "__main__":
    download_model()
