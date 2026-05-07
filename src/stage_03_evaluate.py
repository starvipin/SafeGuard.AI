import os
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_model():
    # 1. Define paths
    data_path = os.path.join("data", "raw_data", "dataset.parquet")
    model_path = os.path.join("models", "model.pkl")
    metrics_path = "metrics.json"

    # 2. Load data and apply the same preprocessing
    print("Loading data for evaluation...")
    df = pd.read_parquet(data_path)
    
    if 'action' not in df.columns and 'resp' in df.columns:
        df['action'] = (df['resp'] > 0).astype(int)
        
    y = df['action']
    cols_to_drop = ['action', 'date', 'ts_id', 'resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # It is important to use the same random_state (42) so we get exactly the same test set as in training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Load the trained model
    print("Loading the trained model...")
    model = joblib.load(model_path)

    # 4. Calculate Predictions and Metrics
    print("Calculating metrics...")
    predictions = model.predict(X_test)
    prediction_probs = model.predict_proba(X_test)[:, 1] # Probabilities are needed for ROC-AUC

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, prediction_probs)

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4)
    }

    # 5. Save metrics to JSON (CI/CD and DVC will track this file)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Stage 03 Success: Evaluation complete. Metrics saved to '{metrics_path}'")
    print(metrics)

if __name__ == "__main__":
    evaluate_model()