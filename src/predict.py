import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_artifact


def parse_features(feature_string: str) -> np.ndarray:
    values = [float(value.strip()) for value in feature_string.split(",") if value.strip()]
    return np.array(values, dtype=float)


def build_feature_dataframe(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    if values.shape[0] != len(feature_names):
        raise ValueError(
            f"Expected {len(feature_names)} feature values, received {values.shape[0]}"
        )
    return pd.DataFrame([values], columns=feature_names)


def predict_transaction(feature_string: str, model_path: Path):
    artifact = load_artifact(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_columns = artifact["feature_columns"]

    values = parse_features(feature_string)
    X = build_feature_dataframe(values, feature_columns)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_columns)

    probability = model.predict_proba(X_scaled)[0, 1]
    label = model.predict(X_scaled)[0]
    print(f"Fraud probability: {probability:.6f}")
    print(f"Predicted class: {label} (0 = not fraud, 1 = fraud)")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict fraud for a single transaction")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/xgb_fraud_detector.joblib"),
        help="Path to the saved model artifact",
    )
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Comma-separated feature vector matching the training dataset columns",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_transaction(args.features, args.model_path)
