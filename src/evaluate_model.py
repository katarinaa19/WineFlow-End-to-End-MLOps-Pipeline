"""
model evaluation module
purpose: load mlflow model and evaluate on test set
"""
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from utils_py import load_data, calc_metrics
import os


def evaluate_model(run_id=None):
    """evaluate mlflow model on test set."""
    print("starting model evaluation")

    if run_id is None:
        if os.path.exists('outputs/latest_run_id.txt'):
            with open('outputs/latest_run_id.txt', 'r') as f:
                run_id = f.read().strip()
        else:
            print("error: run_id not found")
            return

    x_test, y_test = load_data('data/test.csv', target_col='quality_binary')

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)

    metrics = calc_metrics(y_test, y_pred)
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"f1_score: {metrics['f1_score']:.4f}")
    print("model evaluation complete")

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            'test_accuracy': metrics['accuracy'],
            'test_f1_score': metrics['f1_score'],
            'test_precision': metrics['precision'],
            'test_recall': metrics['recall']
        })

        results_df = pd.DataFrame({
            'true_label': y_test.values,
            'predicted_label': y_pred,
            'prob_class_0': y_pred_proba[:, 0],
            'prob_class_1': y_pred_proba[:, 1]
        })
        os.makedirs('outputs', exist_ok=True)
        results_df.to_csv('outputs/test_predictions.csv', index=False)
        mlflow.log_artifact('outputs/test_predictions.csv')

        cm_df = pd.DataFrame(
            cm,
            index=['true_0', 'true_1'],
            columns=['pred_0', 'pred_1']
        )
        cm_df.to_csv('outputs/confusion_matrix.csv', index=False)
        mlflow.log_artifact('outputs/confusion_matrix.csv')

    report_path = 'outputs/evaluation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("model evaluation report\n\n")
        f.write(f"run_id: {run_id}\n\n")
        f.write(f"accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"f1_score:  {metrics['f1_score']:.4f}\n")
        f.write(f"precision: {metrics['precision']:.4f}\n")
        f.write(f"recall:    {metrics['recall']:.4f}\n\n")
        f.write("confusion matrix:\n")
        f.write(cm_df.to_string())
        f.write("\n\nclassification report:\n")
        f.write(report)

    return metrics


if __name__ == "__main__":
    if not os.path.exists('data/test.csv'):
        print("error: data/test.csv not found")
    elif not os.path.exists('outputs/latest_run_id.txt'):
        print("error: training run not found")
    else:
        evaluate_model()
