"""
utility functions module
purpose: common helpers for data loading and metrics computation
"""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_data(file_path, target_col='quality_binary'):
    df = pd.read_csv(file_path)

    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found in data.")

    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x, y


def calc_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary')
    }


def print_metrics(metrics, prefix=""):
    print(f"\n{prefix}metrics:")
    print("=" * 40)
    for key, value in metrics.items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")
    print("=" * 40)


def save_metrics(metrics, file_path):
    pd.DataFrame([metrics]).to_csv(file_path, index=False)
    print(f"metrics saved to: {file_path}")
