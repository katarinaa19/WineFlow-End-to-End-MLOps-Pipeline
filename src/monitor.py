"""
drift monitoring module
purpose: compare model performance before and after data drift
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from utils_py import load_data, calc_metrics
import os


def monitor_data_drift(run_id=None):
    """monitor effect of data drift on model performance."""
    print("starting drift monitoring")

    if run_id is None:
        if os.path.exists('outputs/latest_run_id.txt'):
            with open('outputs/latest_run_id.txt', 'r') as f:
                run_id = f.read().strip()
        else:
            print("error: run_id not found")
            return

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    x_test_original, y_test_original = load_data('data/test.csv', target_col='quality_binary')
    x_test_drift, y_test_drift = load_data('data/test_changed.csv', target_col='quality_binary')

    y_pred_original = model.predict(x_test_original)
    y_pred_drift = model.predict(x_test_drift)

    metrics_original = calc_metrics(y_test_original, y_pred_original)
    metrics_drift = calc_metrics(y_test_drift, y_pred_drift)

    feature_stats = []
    for col in x_test_original.columns:
        om = x_test_original[col].mean()
        dm = x_test_drift[col].mean()
        osd = x_test_original[col].std()
        dsd = x_test_drift[col].std()

        mean_change = ((dm - om) / om * 100) if om != 0 else 0
        std_change = ((dsd - osd) / osd * 100) if osd != 0 else 0

        feature_stats.append({
            'feature': col,
            'original_mean': om,
            'drift_mean': dm,
            'mean_change_%': mean_change,
            'original_std': osd,
            'drift_std': dsd,
            'std_change_%': std_change
        })

    stats_df = pd.DataFrame(feature_stats)
    stats_df['abs_mean_change'] = stats_df['mean_change_%'].abs()
    stats_df = stats_df.sort_values('abs_mean_change', ascending=False)

    os.makedirs('outputs', exist_ok=True)
    stats_df.to_csv('outputs/feature_drift_stats.csv', index=False)

    report_path = 'outputs/drift_monitoring_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("drift monitoring report\n\n")
        f.write(f"run id: {run_id}\n\n")

        f.write("original test metrics:\n")
        for k, v in metrics_original.items():
            f.write(f"  {k}: {v:.4f}\n")

        f.write("\ndrift test metrics:\n")
        for k, v in metrics_drift.items():
            f.write(f"  {k}: {v:.4f}\n")

        f.write("\nmetric changes:\n")
        for k in ['accuracy', 'f1_score', 'precision', 'recall']:
            change = metrics_drift[k] - metrics_original[k]
            f.write(f"  {k}: {change:+.4f}\n")

        f.write("\ntop feature changes:\n")
        f.write(stats_df.head(5).to_string(index=False))

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            'drift_accuracy': metrics_drift['accuracy'],
            'drift_f1_score': metrics_drift['f1_score'],
            'drift_precision': metrics_drift['precision'],
            'drift_recall': metrics_drift['recall']
        })
        mlflow.log_artifact('outputs/feature_drift_stats.csv')
        mlflow.log_artifact(report_path)

    print("drift monitoring complete")

    return {
        'metrics_original': metrics_original,
        'metrics_drift': metrics_drift,
        'feature_stats': stats_df
    }


if __name__ == "__main__":
    if not os.path.exists('data/test.csv'):
        print("error: data/test.csv not found")
    elif not os.path.exists('data/test_changed.csv'):
        print("error: data/test_changed.csv not found")
    elif not os.path.exists('outputs/latest_run_id.txt'):
        print("error: run record not found")
    else:
        monitor_data_drift()
