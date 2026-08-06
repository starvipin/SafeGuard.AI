"""Backward-compatible entrypoint for Hugging Face model publishing."""

try:
    from src.safeguard_ai.ml.hub import upload_model
except ModuleNotFoundError:  # Direct execution: python src/upload_to_hf.py
    from safeguard_ai.ml.hub import upload_model


if __name__ == "__main__":
    repository = upload_model()
    print(f"Model uploaded to '{repository}'")
