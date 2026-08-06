"""Backward-compatible entrypoint for model evaluation."""

try:
    from src.safeguard_ai.ml.common import (
        load_config as read_params,
        read_dataset as read_evaluation_data,
    )
    from src.safeguard_ai.ml.evaluation import EvaluationDataset, evaluate_model
except ModuleNotFoundError:  # Direct execution: python src/stage_03_evaluate.py
    from safeguard_ai.ml.common import load_config as read_params, read_dataset as read_evaluation_data
    from safeguard_ai.ml.evaluation import EvaluationDataset, evaluate_model


if __name__ == "__main__":
    print(f"Stage 03 complete: {evaluate_model()}")
