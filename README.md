# Wine Quality Classification Project - Overview

## Project Description

This project implements a complete MLOps pipeline for wine quality classification using the UCI Wine Quality Dataset. The pipeline includes data preprocessing, model training with MLflow tracking, evaluation, inference, drift monitoring, and an interactive dashboard for visualization and analysis.


---
## Project Structure

```
mlop_final/
├── data/                     
├── src/                     
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── inference.py
│   ├── monitor.py
│   ├── dashboard.py
│   └── utils.py
├── config/
│   └── params.yaml
├── outputs/                  
├── mlruns/                  
├── requirements.txt
├── run_all.py
└── PROJECT_OVERVIEW.md        
```

---

## Python Files Description

### 1. src/preprocess.py

**Purpose**: Data preprocessing and dataset preparation

**Key Functions**:
- Loads the raw wine quality dataset from CSV
- Performs binary classification transformation (quality >= 6 → 1, otherwise → 0)
- Splits data into training and testing sets (80/20 split)
- Generates a drift test set by adding noise to features for monitoring purposes
- Saves processed datasets to the data directory

**Input**:
- `data/winequality-red.csv` (raw dataset)

**Output**:
- `data/train.csv` (training set)
- `data/test.csv` (testing set)
- `data/test_changed.csv` (drift test set with added noise)

**Usage**:
```bash
python src/preprocess.py
```

---

### 2. src/train_model.py

**Purpose**: Train the machine learning model with MLflow tracking

**Key Functions**:
- Reads hyperparameters from configuration file (`config/params.yaml`)
- Trains a RandomForestClassifier on the training data
- Logs all parameters and metrics to MLflow
- Calculates and logs feature importance
- Saves the trained model to MLflow Model Registry
- Stores the run ID for later use

**Input**:
- `data/train.csv`
- `config/params.yaml`

**Output**:
- MLflow logged model in `mlruns/` directory
- `outputs/latest_run_id.txt` (saved run ID)
- `outputs/feature_importance.csv`

**Usage**:
```bash
python src/train_model.py
```

---

### 3. src/evaluate_model.py

**Purpose**: Evaluate model performance on test set

**Key Functions**:
- Loads the trained model from MLflow using run ID
- Evaluates model on the test dataset
- Calculates performance metrics (Accuracy, F1 Score, Precision, Recall)
- Generates confusion matrix and classification report
- Logs evaluation metrics back to MLflow
- Saves prediction results and evaluation reports

**Input**:
- `data/test.csv`
- `outputs/latest_run_id.txt`
- MLflow model from `mlruns/`

**Output**:
- `outputs/test_predictions.csv`
- `outputs/confusion_matrix.csv`
- `outputs/evaluation_report.txt`
- Updated MLflow metrics

**Usage**:
```bash
python src/evaluate_model.py
```

---

### 4. src/inference.py

**Purpose**: Perform inference using the deployed model service

**Key Functions**:
- Sends HTTP requests to the MLflow model serving endpoint
- Tests model predictions on sample data
- Supports both batch and single sample predictions
- Validates model service availability
- Displays prediction results with confidence scores

**Input**:
- `data/test.csv` (for testing)
- MLflow model service running on port 8000

**Output**:
- Console output with prediction results
- Accuracy metrics for test samples

**Usage**:
```bash
# First, start the model serving service in another terminal
# Then run:
python src/inference.py
```

---

### 5. src/monitor.py

**Purpose**: Monitor data drift and model performance degradation

**Key Functions**:
- Loads the trained model and both test datasets
- Evaluates model on original test set
- Evaluates model on drift test set (with simulated distribution shift)
- Compares performance metrics between both datasets
- Performs statistical tests (KS test) on feature distributions
- Identifies features with significant drift
- Generates drift warnings when performance drops exceed threshold
- Logs drift metrics to MLflow

**Input**:
- `data/test.csv`
- `data/test_changed.csv`
- `outputs/latest_run_id.txt`
- MLflow model

**Output**:
- `outputs/feature_drift_stats.csv`
- `outputs/drift_monitoring_report.txt`
- Updated MLflow metrics with drift information

**Usage**:
```bash
python src/monitor.py
```

---

### 6. src/dashboard.py

**Purpose**: Interactive web-based dashboard for data exploration and model monitoring

**Key Functions**:
- Provides four main modules:
  - Data Exploration (EDA): Dataset overview, distributions, correlations
  - Model Monitoring: Performance metrics, confusion matrix, feature importance
  - Data Drift Detection: Compare original vs drift datasets, statistical tests
  - Feature Analysis: Feature relationships, statistical tests, outlier detection
- Uses Streamlit for web interface
- Uses Plotly for interactive visualizations
- Real-time data loading and analysis
- Supports multiple dataset switching

**Input**:
- `data/train.csv`
- `data/test.csv`
- `data/test_changed.csv`
- `outputs/latest_run_id.txt`
- MLflow model

**Output**:
- Interactive web interface on http://localhost:8501

**Usage**:
```bash
streamlit run src/dashboard.py
```

---

### 7. src/utils.py

**Purpose**: Utility functions shared across the project

**Key Functions**:
- `load_data()`: Load CSV data and separate features from target
- `calc_metrics()`: Calculate classification metrics (Accuracy, F1, Precision, Recall)
- `print_metrics()`: Format and display metrics
- `save_metrics()`: Save metrics to CSV file

**Usage**:
This file is imported by other modules and not run directly.

---

## Configuration File

### config/params.yaml

**Purpose**: Centralized hyperparameter and configuration management

**Contents**:
- Model parameters (n_estimators, max_depth, min_samples_split, etc.)
- Training parameters (test_size, experiment_name, run_name)
- Data parameters (file paths, target threshold)
- Monitoring parameters (drift thresholds)

**Usage**:
Edit this file to adjust hyperparameters before training.

---

# How to Run Everything  

(Optional) Rerun to update the model and ID
```
conda activate mlops_env
cd C:\Users\wxiny\Desktop\mlop_final
python runall.py
```

Open one terminal and run the following commands to start the model service:
```
conda activate mlops_env
cd C:\Users\wxiny\Desktop\mlop_final

# Start model service (keep this terminal open)
mlflow models serve -m runs:/<RUN_ID>/model -p 8000 --no-conda
```

Open another terminal, then run the following commands to execute the entire pipeline:

```
conda activate mlops_env
cd C:\Users\wxiny\Desktop\mlop_final

python runall.py
```