from huggingface_hub import HfApi, login
import os
from dotenv import load_dotenv

# Ye line .env file se tumhara token Python me load karegi
load_dotenv()

# Ab os.getenv sahi se token nikal lega
HF_TOKEN = os.getenv("HF_TOKEN")

# Ek chota sa check taaki pata chal jaye token load hua ya nahi
if HF_TOKEN is None:
    print("Error: .env file se token load nahi hua! Check karo ki variable ka naam exact HF_TOKEN hai.")
    exit()

# Token se login kar lo
login(token=HF_TOKEN)

HF_USERNAME = "sainivipin"
MODEL_REPO = "fraud-model-final"  

DEFAULT_MODEL_DIR = "models/fraud_model_final"

api = HfApi()

repo_id = f"{HF_USERNAME}/{MODEL_REPO}"


def upload_model(model_dir=DEFAULT_MODEL_DIR):
    print(f"Repository check/create ho rahi hai: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print("Repository mil gayi! Ab files upload ho rahi hain, wait karo...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial model upload"
    )
    print("Badhai ho! Model successfully Hugging Face Hub par upload ho gaya hai.")

if __name__ == "__main__":
    upload_model()