from huggingface_hub import HfApi, login
import os
from dotenv import load_dotenv

# This line will load your token from the .env file into Python
load_dotenv()

# Now os.getenv will correctly fetch the token
HF_TOKEN = os.getenv("HF_TOKEN")

# A quick check to see if the token was loaded
if HF_TOKEN is None:
    print("Error: Token not loaded from .env file! Check if the variable name is exactly HF_TOKEN.")
    exit()

# Log in using the token
login(token=HF_TOKEN)

HF_USERNAME = "sainivipin"
MODEL_REPO = "fraud-model-final"  
MODEL_DIR = "models/fraud_model_final"

api = HfApi()

repo_id = f"{HF_USERNAME}/{MODEL_REPO}"

def upload_model():
    print(f"Checking/creating repository: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    print("Repository found! Files are uploading now, please wait...")
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial model upload"
    )
    print("Congratulations! Model successfully uploaded to Hugging Face Hub.")

if __name__ == "__main__":
    upload_model()