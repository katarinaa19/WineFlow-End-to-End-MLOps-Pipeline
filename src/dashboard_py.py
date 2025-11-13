"""
Interactive Dashboard
Functions: Model monitoring, data drift detection, EDA analysis, manual inference
Built using Streamlit
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.metrics import confusion_matrix
import os
from utils_py import load_data, calc_metrics  
import plotly.express as px
from scipy import stats

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Wine Quality ML Dashboard",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #8B0000;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🍷 Wine Quality ML Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("🎛️ Control Panel")

page = st.sidebar.selectbox(
    "Select a page",
    ["📊 EDA", "🤖 Model Monitoring", "⚠️ Data Drift Detection", "📈 Feature Analysis", "🧪 Manual Inference"]
)

# -----------------------------------------------------------
# CHECK DATA
# -----------------------------------------------------------
@st.cache_data
def check_data_exists():
    files = {
        'train': 'data/train.csv',
        'test': 'data/test.csv',
        'test_drift': 'data/test_changed.csv',
        'original': 'data/winequality-red.csv'
    }
    exists = {k: os.path.exists(v) for k, v in files.items()}
    return exists, files

data_exists, data_files = check_data_exists()

if not all(data_exists.values()):
    st.warning("⚠️ Some required data files are missing. Please run preprocessing.")
    missing = [f for k, f in data_files.items() if not data_exists[k]]
    st.code("Missing files:\n" + "\n".join(missing))
    st.stop()

# -----------------------------------------------------------
# LOAD ALL DATA
# -----------------------------------------------------------
@st.cache_data
def load_all_data():
    X_train, y_train = load_data('data/train.csv', target_col='quality_binary')
    X_test, y_test = load_data('data/test.csv', target_col='quality_binary')
    X_drift, y_drift = load_data('data/test_changed.csv', target_col='quality_binary')

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    drift_df = pd.concat([X_drift, y_drift], axis=1)

    original = pd.read_csv('data/winequality-red.csv', sep=';')
    original['quality_binary'] = (original['quality'] >= 6).astype(int)

    return train_df, test_df, drift_df, original

train_df, test_df, drift_df, original_df = load_all_data()

# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open('outputs/latest_run_id.txt', 'r') as f:
            run_id = f.read().strip()
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        return model, run_id
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

model, run_id = load_model()

# ===========================================================
# PAGE 1 — EDA
# ===========================================================
if page == "📊 EDA":
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    option = st.selectbox("Dataset", ["Train", "Test", "Original"])

    df = train_df if option == "Train" else test_df if option == "Test" else original_df

    st.subheader("📋 Overview")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Rows", df.shape[0])
    with col2: st.metric("Features", df.shape[1] - 1)
    with col3: st.metric("High Quality", int(df['quality_binary'].sum()))

    st.subheader("👀 Preview")
    st.dataframe(df.head(10))

    st.subheader("📊 Statistics")
    st.dataframe(df.describe())

    st.subheader("🎯 Target Distribution")
    fig = px.pie(df, names='quality_binary', title="Quality Distribution")
    st.plotly_chart(fig)

    st.subheader("📈 Feature Distribution")
    feature = st.selectbox("Feature", [c for c in df.columns if c not in ['quality', 'quality_binary']])
    fig = px.histogram(df, x=feature, color="quality_binary")
    st.plotly_chart(fig)


# ===========================================================
# PAGE 2 — MODEL MONITORING
# ===========================================================
elif page == "🤖 Model Monitoring":
    st.header("🤖 Model Performance Monitoring")

    if model is None:
        st.error("Model not loaded.")
        st.stop()

    dataset = st.selectbox("Dataset", ["Train", "Test", "Drift Test"])

    df = train_df if dataset == "Train" else test_df if dataset == "Test" else drift_df
    X_eval = df.drop("quality_binary", axis=1)
    y_eval = df["quality_binary"]

    y_pred = model.predict(X_eval)

    metrics = calc_metrics(y_eval, y_pred)

    st.subheader("📈 Metrics")
    for k, v in metrics.items():
        st.metric(k.title(), f"{v:.4f}")

    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_eval, y_pred)
    fig = px.imshow(cm, text_auto=True)
    st.plotly_chart(fig)


# ===========================================================
# PAGE 3 — DATA DRIFT
# ===========================================================
elif page == "⚠️ Data Drift Detection":
    st.header("⚠️ Data Drift Detection")

    X_orig = test_df.drop('quality_binary', axis=1)
    X_drift = drift_df.drop('quality_binary', axis=1)

    drift_table = []

    for col in X_orig.columns:
        ks, p = stats.ks_2samp(X_orig[col], X_drift[col])
        drift_table.append({
            "Feature": col,
            "KS Statistic": ks,
            "p-value": p,
            "Drift?": "Yes" if p < 0.05 else "No"
        })

    st.dataframe(pd.DataFrame(drift_table))

# ===========================================================
# PAGE 4 — FEATURE ANALYSIS
# ===========================================================
elif page == "📈 Feature Analysis":
    st.header("📈 Feature Behavior by Quality Class")

    df = st.selectbox("Dataset", ["Train", "Test"])
    df = train_df if df == "Train" else test_df

    feature = st.selectbox("Select feature", [c for c in df.columns if c not in ['quality', 'quality_binary']])

    fig = px.violin(df, x='quality_binary', y=feature, box=True)
    st.plotly_chart(fig)


# ===========================================================
# PAGE 5 — MANUAL INFERENCE
# ===========================================================
elif page == "🧪 Manual Inference":
    st.header("🧪 Manual Inference – Predict Wine Quality")

    if model is None:
        st.error("Model not loaded.")
        st.stop()

    feature_cols = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
        'density', 'pH', 'sulphates', 'alcohol'
    ]

    st.subheader("Input Values")
    user_inputs = {}
    cols = st.columns(3)

    for i, feat in enumerate(feature_cols):
        with cols[i % 3]:
            user_inputs[feat] = st.number_input(
                feat,
                value=float(train_df[feat].median()),
                format="%.4f"
            )

    input_df = pd.DataFrame([user_inputs])
    st.write(input_df)

    if st.button("Predict"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.success(f"Predicted Quality: **{pred}**")
        st.metric("Probability of High Quality", f"{prob:.4f}")

# -----------------------------------------------------------
# SIDEBAR INFORMATION (BOTTOM)
# -----------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Summary")
st.sidebar.write(f"Train samples: {train_df.shape[0]}")
st.sidebar.write(f"Test samples: {test_df.shape[0]}")
st.sidebar.write(f"Features: {len(train_df.columns) - 1}")

if model is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Model Info")
<<<<<<< HEAD
    st.sidebar.write(f"Run ID: {run_id}")
    st.sidebar.write("Model Type: GradientBoostingClassifier")
=======
    st.sidebar.write(f"Run ID: {run_id[:8]}...")
    st.sidebar.write("Model Type: RandomForest")


>>>>>>> 4739b51509afa9d7c62a68bfba5213bd2fdfa7af
