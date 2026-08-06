"""Backward-compatible entrypoint for model training."""

try:
    from src.safeguard_ai.ml.common import (
        load_config as read_params,
        read_dataset as read_training_data,
    )
    from src.safeguard_ai.ml.training import FraudDataset, cleanup_checkpoints, train_model
except ModuleNotFoundError:  # Direct execution: python src/stage_02_train.py
    from safeguard_ai.ml.common import load_config as read_params, read_dataset as read_training_data
    from safeguard_ai.ml.training import FraudDataset, cleanup_checkpoints, train_model


if __name__ == "__main__":
    print(f"Stage 02 complete: {train_model()}")
