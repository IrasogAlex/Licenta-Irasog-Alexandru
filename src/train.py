import argparse
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

from config import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, RANDOM_STATE
from preprocess import load_dataset, prepare_train_test
from utils import save_artifact


def compute_scale_pos_weight(y):
    fraud = np.sum(y == 1)
    nonfraud = np.sum(y == 0)
    if fraud == 0:
        return 1.0
    return nonfraud / fraud


def train_model(
    data_path: Path,
    model_path: Path,
    use_smote: bool = True,
    test_size: float = 0.2,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
):
    df = load_dataset(data_path)
    X_train, X_test, y_train, y_test, scaler = prepare_train_test(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    if use_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = compute_scale_pos_weight(y_train)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("\n=== Evaluation on test set ===")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_fraud = precision_score(y_test, y_pred, zero_division=0)
    recall_fraud = recall_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print("ROC AUC:", roc_auc)
    
    # Create metrics dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision_fraud,
        "recall": recall_fraud,
        "roc_auc": roc_auc,
    }

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": X_train.columns.tolist(),
        "metrics": metrics,
    }
    save_artifact(model_path, artifact)
    print(f"Saved trained model and scaler to {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the raw credit card dataset CSV file",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Output path for the trained model artifact",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE resampling and rely on class weighting",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        data_path=args.data_path,
        model_path=args.model_path,
        use_smote=not args.no_smote,
        test_size=args.test_size,
    )
