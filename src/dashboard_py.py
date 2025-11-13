"""
Interactive Dashboard
Functions: Model monitoring, data drift detection, EDA analysis
Built using Streamlit
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys
from utils_py import load_data, calc_metrics
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Wine Quality ML Dashboard",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #8B0000;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #8B0000;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🍷 Wine Quality ML Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("🎛️ Control Panel")
page = st.sidebar.selectbox(
    "Select a page",
    ["📊 EDA", "🤖 Model Monitoring", "⚠️ Data Drift Detection", "📈 Feature Analysis"]
)

# Check data
@st.cache_data
def check_data_exists():
    files = {
        'train': 'data/train.csv',
        'test': 'data/test.csv',
        'test_drift': 'data/test_changed.csv',
        'original': 'data/winequality-red.csv'
    }
    
    exists = {key: os.path.exists(path) for key, path in files.items()}
    return exists, files

data_exists, data_files = check_data_exists()

# Missing data
if not all(data_exists.values()):
    st.warning("⚠️ Some required data files are missing. Please run the preprocessing script first.")
    missing_files = [path for key, path in data_files.items() if not data_exists[key]]
    st.code(f"Missing files: {', '.join(missing_files)}")
    st.info("Run: `python src/preprocess.py`")
    st.stop()

# Load all data
@st.cache_data
def load_all_data():
    X_train, y_train = load_data('data/train.csv', target_col='quality_binary')
    X_test, y_test = load_data('data/test.csv', target_col='quality_binary')
    X_test_drift, y_test_drift = load_data('data/test_changed.csv', target_col='quality_binary')
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    test_drift_df = pd.concat([X_test_drift, y_test_drift], axis=1)
    
    if os.path.exists('data/winequality-red.csv'):
        original_df = pd.read_csv('data/winequality-red.csv', sep=';')
    else:
        original_df = None
    
    return train_df, test_df, test_drift_df, original_df

train_df, test_df, test_drift_df, original_df = load_all_data()

# Load model
@st.cache_resource
def load_model():
    try:
        if os.path.exists('outputs/latest_run_id.txt'):
            with open('outputs/latest_run_id.txt', 'r') as f:
                run_id = f.read().strip()
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            return model, run_id
        return None, None
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

model, run_id = load_model()

# ---------------------------------------------------------------------
# PAGE 1 — EDA
# ---------------------------------------------------------------------
if page == "📊 EDA":
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    dataset_option = st.selectbox("Select dataset", ["Train", "Test", "Original"])
    
    if dataset_option == "Train":
        df = train_df
    elif dataset_option == "Test":
        df = test_df
    else:
        if original_df is not None:
            df = original_df.copy()
            df['quality_binary'] = (df['quality'] >= 6).astype(int)
        else:
            st.error("Original dataset is missing.")
            st.stop()

    # Overview
    st.subheader("📋 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Rows", df.shape[0])
    with col2: st.metric("Features", df.shape[1] - 1)
    with col3: st.metric("High Quality", (df['quality_binary'] == 1).sum())
    with col4: st.metric("Low Quality", (df['quality_binary'] == 0).sum())

    # Preview
    st.subheader("👀 Head")
    st.dataframe(df.head(10), use_container_width=True)

    # Summary
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    # Target distribution
    st.subheader("🎯 Target Distribution")
    fig = px.pie(values=df['quality_binary'].value_counts().values,
                 names=['0', '1'],
                 title='Quality Distribution')
    st.plotly_chart(fig)

    # Feature distribution
    st.subheader("📈 Feature Distribution")
    feature_cols = [c for c in df.columns if c not in ['quality', 'quality_binary']]
    selected = st.selectbox("Select feature", feature_cols)

    fig = px_hist = px.histogram(df, x=selected, color='quality_binary')
    st.plotly_chart(fig)

    # Correlation
    st.subheader("🔗 Correlation Matrix")
    corr_matrix = df[feature_cols].corr()
    fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto")
    st.plotly_chart(fig)


# ---------------------------------------------------------------------
# PAGE 2 — MODEL MONITORING
# ---------------------------------------------------------------------
elif page == "🤖 Model Monitoring":
    st.header("🤖 Model Performance Monitoring")

    if model is None:
        st.error("Model not loaded. Run training script first.")
        st.stop()

    st.success(f"Model Loaded (Run ID: {run_id})")

    # choose dataset
    eval_dataset = st.selectbox("Select dataset", ["Train", "Test", "Drift Test"])

    if eval_dataset == "Train":
        X_eval, y_eval = train_df.drop('quality_binary', axis=1), train_df['quality_binary']
    elif eval_dataset == "Test":
        X_eval, y_eval = test_df.drop('quality_binary', axis=1), test_df['quality_binary']
    else:
        X_eval, y_eval = test_drift_df.drop('quality_binary', axis=1), test_drift_df['quality_binary']

    # Predictions
    y_pred = model.predict(X_eval)
    y_pred_proba = model.predict_proba(X_eval)

    # Metrics
    metrics = calc_metrics(y_eval, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    with col2: st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
    with col3: st.metric("Precision", f"{metrics['precision']:.4f}")
    with col4: st.metric("Recall", f"{metrics['recall']:.4f}")

    # Confusion Matrix
    st.subheader("📈 Confusion Matrix")
    cm = confusion_matrix(y_eval, y_pred)
    fig = px.imshow(cm, text_auto=True)
    st.plotly_chart(fig)


# ---------------------------------------------------------------------
# PAGE 3 — DATA DRIFT
# ---------------------------------------------------------------------
elif page == "⚠️ Data Drift Detection":
    st.header("⚠️ Data Drift Detection")

    X_orig = test_df.drop('quality_binary', axis=1)
    y_orig = test_df['quality_binary']

    X_drift = test_drift_df.drop('quality_binary', axis=1)
    y_drift = test_drift_df['quality_binary']

    y_pred_orig = model.predict(X_orig)
    y_pred_drift = model.predict(X_drift)

    metrics_orig = calc_metrics(y_orig, y_pred_orig)
    metrics_drift = calc_metrics(y_drift, y_pred_drift)

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Original Test Set")
        for m, v in metrics_orig.items():
            st.metric(m.title(), f"{v:.4f}")
    with col2:
        st.write("### Drift Test Set")
        for m, v in metrics_drift.items():
            delta = v - metrics_orig[m]
            st.metric(m.title(), f"{v:.4f}", delta=f"{delta:+.4f}")

    # Drift summary table
    st.subheader("📋 Feature Drift Summary")
    feature_cols = list(X_orig.columns)

    drift_summary = []
    for col in feature_cols:
        ks, p = stats.ks_2samp(X_orig[col], X_drift[col])
        drift_summary.append({
            'Feature': col,
            'KS Statistic': ks,
            'p-value': p,
            'Drift?': "Yes" if p < 0.05 else "No"
        })
    st.dataframe(pd.DataFrame(drift_summary))


# ---------------------------------------------------------------------
# PAGE 4 — FEATURE ANALYSIS
# ---------------------------------------------------------------------
elif page == "📈 Feature Analysis":
    st.header("📈 Feature Analysis")

    df_analysis = st.selectbox("Dataset", ["Train", "Test"])
    df_analysis = train_df if df_analysis == "Train" else test_df

    feature_cols = [c for c in df_analysis.columns if c not in ['quality', 'quality_binary']]

    selected = st.selectbox("Select feature", feature_cols)

    fig = px.violin(df_analysis, x="quality_binary", y=selected, box=True, points="all")
    st.plotly_chart(fig)

# ---------------------------------------------------------------------
# Sidebar dataset and model info
# ---------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Info")
st.sidebar.write(f"Train samples: {train_df.shape[0]}")
st.sidebar.write(f"Test samples:  {test_df.shape[0]}")
st.sidebar.write(f"Feature count: {len(train_df.columns) - 1}")

if model is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Model Info")
    st.sidebar.write(f"Run ID: {run_id[:8]}...")
    st.sidebar.write("Model Type: RandomForest")

st.sidebar.markdown("---")
st.sidebar.info("Use `streamlit run src/dashboard.py` to launch the dashboard.")
