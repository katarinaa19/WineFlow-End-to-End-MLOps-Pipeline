"""
model training module - IMPROVED VERSION
purpose: train model using enhanced AutoML with hyperparameter tuning
"""
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from utils_py import load_data, calc_metrics
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def train_model(config_path='config/params.yaml'):
    """train model using improved AutoML with grid search and preprocessing."""
    print("=" * 60)
    print("IMPROVED AUTOML MODEL TRAINING")
    print("with hyperparameter tuning and feature scaling")
    print("=" * 60)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    automl_params = config.get('automl', {})
    train_params = config['training']

    print("\nloading training data...")
    x_train, y_train = load_data('data/train.csv', target_col='quality_binary')
    print(f"training set: {x_train.shape[0]} samples, {x_train.shape[1]} features")

    # Check class balance
    class_counts = y_train.value_counts()
    print(f"\nclass distribution:")
    print(f"  class 0: {class_counts[0]} ({class_counts[0]/len(y_train)*100:.1f}%)")
    print(f"  class 1: {class_counts[1]} ({class_counts[1]/len(y_train)*100:.1f}%)")

    experiment_name = train_params.get('experiment_name', 'wine_quality_classification')
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=train_params.get('run_name', 'improved_automl')) as run:
        print(f"\nmlflow run id: {run.info.run_id}")

        mlflow.log_params({
            'train_samples': x_train.shape[0],
            'n_features': x_train.shape[1],
            'automl_tool': 'Enhanced_GridSearch_CV',
            'cv_folds': automl_params.get('cv', 5),
            'use_scaling': True
        })

        print("\n" + "=" * 60)
        print("PHASE 1: QUICK ALGORITHM SCREENING")
        print("=" * 60)

        # Define candidate algorithms with better configurations
        algorithms = {
            'RandomForest': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            'GradientBoosting': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=7,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    subsample=0.8,
                    random_state=42
                ))
            ]),
            'ExtraTrees': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', ExtraTreesClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            'AdaBoost': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', AdaBoostClassifier(
                    n_estimators=150,
                    learning_rate=0.5,
                    random_state=42
                ))
            ]),
            'LogisticRegression': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    C=1.0,
                    class_weight='balanced',
                    max_iter=2000,
                    random_state=42,
                    n_jobs=-1
                ))
            ])
        }

        cv_folds = automl_params.get('cv', 5)
        scoring_metric = automl_params.get('scoring', 'f1')
        
        results = {}
        
        print(f"\nevaluating {len(algorithms)} algorithms with {cv_folds}-fold CV...")
        print(f"optimization metric: {scoring_metric}")
        print("-" * 60)

        for name, pipeline in algorithms.items():
            print(f"\ntesting {name}...")
            try:
                scores = cross_val_score(
                    pipeline, x_train, y_train,
                    cv=cv_folds,
                    scoring=scoring_metric,
                    n_jobs=-1
                )
                mean_score = scores.mean()
                std_score = scores.std()
                results[name] = {
                    'pipeline': pipeline,
                    'mean_score': mean_score,
                    'std_score': std_score
                }
                
                print(f"  cv {scoring_metric}: {mean_score:.4f} (+/- {std_score:.4f})")
                mlflow.log_metric(f"cv_{name}_{scoring_metric}", mean_score)
                
            except Exception as e:
                print(f"  error: {str(e)}")
                continue

        # Get top 2 performers
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_score'], reverse=True)
        
        print("\n" + "=" * 60)
        print("PHASE 1 RESULTS - Algorithm Ranking")
        print("=" * 60)
        for rank, (name, result) in enumerate(sorted_results, 1):
            print(f"{rank}. {name}: {result['mean_score']:.4f} (+/- {result['std_score']:.4f})")

        # Select top 2 for hyperparameter tuning
        top_algorithms = sorted_results[:2]

        print("\n" + "=" * 60)
        print("PHASE 2: HYPERPARAMETER TUNING")
        print(f"tuning top {len(top_algorithms)} algorithms")
        print("=" * 60)

        best_overall_score = 0
        best_overall_model = None
        best_overall_name = None

        for name, result in top_algorithms:
            print(f"\n>>> tuning {name}...")
            
            # Define hyperparameter grids
            if name == 'RandomForest':
                param_grid = {
                    'clf__n_estimators': [150, 200, 250],
                    'clf__max_depth': [12, 15, 18],
                    'clf__min_samples_split': [3, 5, 7],
                    'clf__min_samples_leaf': [1, 2, 3]
                }
            elif name == 'GradientBoosting':
                param_grid = {
                    'clf__n_estimators': [150, 200, 250],
                    'clf__learning_rate': [0.03, 0.05, 0.1],
                    'clf__max_depth': [5, 7, 9],
                    'clf__subsample': [0.7, 0.8, 0.9]
                }
            elif name == 'ExtraTrees':
                param_grid = {
                    'clf__n_estimators': [150, 200, 250],
                    'clf__max_depth': [12, 15, 18],
                    'clf__min_samples_split': [3, 5, 7]
                }
            elif name == 'AdaBoost':
                param_grid = {
                    'clf__n_estimators': [100, 150, 200],
                    'clf__learning_rate': [0.3, 0.5, 0.7, 1.0]
                }
            else:  # LogisticRegression
                param_grid = {
                    'clf__C': [0.1, 0.5, 1.0, 2.0, 5.0],
                    'clf__penalty': ['l2']
                }

            grid_search = GridSearchCV(
                result['pipeline'],
                param_grid,
                cv=cv_folds,
                scoring=scoring_metric,
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(x_train, y_train)
            
            tuned_score = grid_search.best_score_
            print(f"  before tuning: {result['mean_score']:.4f}")
            print(f"  after tuning:  {tuned_score:.4f}")
            print(f"  improvement:   {tuned_score - result['mean_score']:+.4f}")
            print(f"  best params: {grid_search.best_params_}")
            
            mlflow.log_metric(f"tuned_{name}_{scoring_metric}", tuned_score)
            mlflow.log_params({f"{name}_best_{k}": v for k, v in grid_search.best_params_.items()})
            
            if tuned_score > best_overall_score:
                best_overall_score = tuned_score
                best_overall_model = grid_search.best_estimator_
                best_overall_name = name

        print("\n" + "=" * 60)
        print("FINAL SELECTION")
        print("=" * 60)
        print(f"best algorithm: {best_overall_name}")
        print(f"best cv {scoring_metric}: {best_overall_score:.4f}")
        print("=" * 60)

        mlflow.log_param('best_algorithm', best_overall_name)
        mlflow.log_param('best_cv_score', best_overall_score)
        mlflow.log_param('algorithms_tested', len(algorithms))
        mlflow.log_param('tuning_performed', True)

        print(f"\ntraining final {best_overall_name} model on full training set...")
        # Model is already trained via GridSearchCV
        
        # Evaluate on training set
        print("\nevaluating on training set...")
        y_train_pred = best_overall_model.predict(x_train)
        train_metrics = calc_metrics(y_train, y_train_pred)

        print(f"training accuracy:  {train_metrics['accuracy']:.4f}")
        print(f"training f1 score:  {train_metrics['f1_score']:.4f}")
        print(f"training precision: {train_metrics['precision']:.4f}")
        print(f"training recall:    {train_metrics['recall']:.4f}")

        mlflow.log_metrics({
            'train_accuracy': train_metrics['accuracy'],
            'train_f1_score': train_metrics['f1_score'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall']
        })

        # Save feature importance if available
        os.makedirs('outputs', exist_ok=True)
        
        classifier = best_overall_model.named_steps['clf']
        if hasattr(classifier, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': x_train.columns,
                'importance': classifier.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\ntop 5 important features:")
            for idx, row in feature_importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

            importance_path = 'outputs/feature_importance.csv'
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)

        # Save tuning results
        tuning_results = pd.DataFrame([
            {
                'algorithm': name,
                'base_cv_score': result['mean_score'],
                'tuned': name in [n for n, _ in top_algorithms]
            }
            for name, result in sorted_results
        ])
        
        tuning_path = 'outputs/tuning_results.csv'
        tuning_results.to_csv(tuning_path, index=False)
        mlflow.log_artifact(tuning_path)

        # Log model
        print("\nsaving model to mlflow...")
        mlflow.sklearn.log_model(
            best_overall_model,
            "model",
            registered_model_name="wine_quality_automl_classifier"
        )

        with open('outputs/latest_run_id.txt', 'w') as f:
            f.write(run.info.run_id)

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print(f"best model: {best_overall_name}")
    print(f"cv score: {best_overall_score:.4f}")
    print("=" * 60)
    
    return run.info.run_id


if __name__ == "__main__":
    if not os.path.exists('data/train.csv'):
        print("error: data/train.csv not found")
        print("please run: python src/preprocess.py")
    else:
        train_model()