import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from config import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH
from preprocess import engineer_features, load_dataset
from utils import load_artifact


def plot_confusion_matrix(matrix, labels, output_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")

    plt.title("Confusion matrix")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved confusion matrix to {output_path}")
    plt.close(fig)


def evaluate_model(data_path: Path, model_path: Path):
    artifact = load_artifact(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_columns = artifact["feature_columns"]

    df = load_dataset(data_path)
    df = engineer_features(df)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    if list(X.columns) != feature_columns:
        raise ValueError("Feature column order does not match saved model artifact")

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n=== Evaluation report ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("Average precision (PR AUC):", average_precision_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, ["Not Fraud", "Fraud"], output_path=Path("models/confusion_matrix.png"))

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/precision_recall_curve.png", dpi=150)
    plt.close()
    print("Saved precision-recall curve to models/precision_recall_curve.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fraud detection model")
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
        help="Path to the saved model artifact",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args.data_path, args.model_path)
