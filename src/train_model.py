"""
model training module
purpose: train gradient boosting model with mlflow tracking
"""
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from utils_py import load_data, calc_metrics
import os


def train_model(config_path='config/params.yaml'):
    """train gradient boosting classifier and log with mlflow."""
    print("starting model training")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_params = config['model']
    train_params = config['training']

    x_train, y_train = load_data('data/train.csv', target_col='quality_binary')

    experiment_name = train_params.get('experiment_name', 'wine_quality_classification')
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=train_params.get('run_name', 'gbc_baseline')) as run:

        mlflow.log_params(model_params)
        mlflow.log_params({
            'train_samples': x_train.shape[0],
            'n_features': x_train.shape[1]
        })

        model = GradientBoostingClassifier(**model_params)
        model.fit(x_train, y_train)

        y_train_pred = model.predict(x_train)
        train_metrics = calc_metrics(y_train, y_train_pred)

        mlflow.log_metrics({
            'train_accuracy': train_metrics['accuracy'],
            'train_f1_score': train_metrics['f1_score']
        })

        feature_importance = pd.DataFrame({
            'feature': x_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        os.makedirs('outputs', exist_ok=True)
        importance_path = 'outputs/feature_importance.csv'
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="wine_quality_gb_classifier"
        )

        with open('outputs/latest_run_id.txt', 'w') as f:
            f.write(run.info.run_id)

    print("model training complete")
    return run.info.run_id


if __name__ == "__main__":
    if not os.path.exists('data/train.csv'):
        print("error: data/train.csv not found")
    else:
        train_model()
