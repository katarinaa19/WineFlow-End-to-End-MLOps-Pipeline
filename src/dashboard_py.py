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
# PAGE 1 – EDA
# ===========================================================
if page == "📊 EDA":
    st.header("📊 Exploratory Data Analysis (EDA)")
    
    option = st.selectbox("Dataset", ["Train", "Test", "Original"])
    df = train_df if option == "Train" else test_df if option == "Test" else original_df

    # ===========================================================
    # SECTION 1: OVERVIEW
    # ===========================================================
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        st.metric("Total Samples", df.shape[0])
    with col2: 
        st.metric("Features", df.shape[1] - 1)
    with col3: 
        high_quality = int(df['quality_binary'].sum())
        st.metric("High Quality (1)", f"{high_quality} ({high_quality/len(df)*100:.1f}%)")
    with col4: 
        low_quality = len(df) - high_quality
        st.metric("Low Quality (0)", f"{low_quality} ({low_quality/len(df)*100:.1f}%)")

    # Data preview
    with st.expander("👀 View Raw Data"):
        st.dataframe(df.head(20), use_container_width=True)

    # ===========================================================
    # SECTION 2: TARGET DISTRIBUTION
    # ===========================================================
    st.subheader("🎯 Target Variable Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig = px.pie(
            df, 
            names='quality_binary', 
            title="Binary Quality Distribution",
            color='quality_binary',
            color_discrete_map={0: '#ff6b6b', 1: '#51cf66'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Original quality distribution (if available)
        if 'quality' in df.columns:
            fig = px.histogram(
                df, 
                x='quality',
                title='Original Quality Score Distribution',
                color_discrete_sequence=['#8B0000'],
                labels={'quality': 'Quality Score', 'count': 'Frequency'}
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Bar chart for binary
            counts = df['quality_binary'].value_counts().sort_index()
            fig = px.bar(
                x=['Low Quality', 'High Quality'],
                y=counts.values,
                title='Quality Class Counts',
                color=['Low Quality', 'High Quality'],
                color_discrete_map={'Low Quality': '#ff6b6b', 'High Quality': '#51cf66'}
            )
            st.plotly_chart(fig, use_container_width=True)

    # Class balance check
    balance_ratio = min(df['quality_binary'].value_counts()) / max(df['quality_binary'].value_counts())
    if balance_ratio < 0.5:
        st.warning(f"⚠️ Class imbalance detected! Ratio: {balance_ratio:.2f}. Consider using class weights or SMOTE.")
    else:
        st.success(f"✅ Classes are reasonably balanced. Ratio: {balance_ratio:.2f}")

    # ===========================================================
    # SECTION 3: STATISTICAL SUMMARY
    # ===========================================================
    st.subheader("📊 Statistical Summary")
    
    feature_cols = [c for c in df.columns if c not in ['quality', 'quality_binary']]
    
    summary_stats = df[feature_cols].describe().T
    summary_stats['missing'] = df[feature_cols].isnull().sum()
    summary_stats['missing_pct'] = (summary_stats['missing'] / len(df) * 100).round(2)
    
    st.dataframe(summary_stats, use_container_width=True)

    # ===========================================================
    # SECTION 4: MISSING VALUES
    # ===========================================================
    with st.expander("🔍 Missing Values Analysis"):
        missing = df[feature_cols].isnull().sum()
        if missing.sum() == 0:
            st.success("✅ No missing values detected in the dataset!")
        else:
            st.warning(f"⚠️ Total missing values: {missing.sum()}")
            missing_df = pd.DataFrame({
                'Feature': missing.index,
                'Missing Count': missing.values,
                'Missing %': (missing.values / len(df) * 100).round(2)
            }).sort_values('Missing Count', ascending=False)
            st.dataframe(missing_df[missing_df['Missing Count'] > 0])

    # ===========================================================
    # SECTION 5: FEATURE DISTRIBUTIONS
    # ===========================================================
    st.subheader("📈 Feature Distributions")
    
    feature = st.selectbox("Select Feature to Analyze", feature_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram with quality overlay
        fig = px.histogram(
            df, 
            x=feature, 
            color="quality_binary",
            title=f"{feature} Distribution by Quality",
            color_discrete_map={0: '#ff6b6b', 1: '#51cf66'},
            barmode='overlay',
            opacity=0.7,
            labels={'quality_binary': 'Quality'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot
        fig = px.box(
            df, 
            x='quality_binary', 
            y=feature,
            title=f"{feature} Box Plot by Quality",
            color='quality_binary',
            color_discrete_map={0: '#ff6b6b', 1: '#51cf66'},
            labels={'quality_binary': 'Quality Class'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Statistics by Quality Class**")
        stats_by_class = df.groupby('quality_binary')[feature].describe()
        st.dataframe(stats_by_class)
    
    with col2:
        st.markdown("**Statistical Test (t-test)**")
        low_quality = df[df['quality_binary'] == 0][feature]
        high_quality = df[df['quality_binary'] == 1][feature]
        
        from scipy import stats as sp_stats
        t_stat, p_val = sp_stats.ttest_ind(low_quality, high_quality)
        
        st.metric("t-statistic", f"{t_stat:.4f}")
        st.metric("p-value", f"{p_val:.4f}")
        
        if p_val < 0.05:
            st.success(f"✅ Significant difference between classes (p < 0.05)")
        else:
            st.info(f"ℹ️ No significant difference between classes (p >= 0.05)")

    # ===========================================================
    # SECTION 6: CORRELATION ANALYSIS
    # ===========================================================
    st.subheader("🔗 Correlation Analysis")
    
    # Correlation matrix
    corr_matrix = df[feature_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect="auto",
        title='Feature Correlation Heatmap',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top correlations
    with st.expander("🔝 Top Feature Correlations"):
        # Get upper triangle of correlation matrix
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
        corr_df = corr_df.sort_values('Abs Correlation', ascending=False)
        
        st.dataframe(corr_df.head(10), use_container_width=True)

    # Correlation with target
    st.markdown("**Feature Correlation with Target (Quality)**")
    target_corr = df[feature_cols].corrwith(df['quality_binary']).sort_values(ascending=False)
    
    fig = px.bar(
        x=target_corr.values,
        y=target_corr.index,
        orientation='h',
        title='Feature Correlation with Quality',
        labels={'x': 'Correlation', 'y': 'Feature'},
        color=target_corr.values,
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ===========================================================
    # SECTION 7: OUTLIER DETECTION
    # ===========================================================
    st.subheader("⚠️ Outlier Detection")
    
    outlier_feature = st.selectbox("Select Feature for Outlier Analysis", feature_cols, key='outlier')
    
    # Calculate IQR
    Q1 = df[outlier_feature].quantile(0.25)
    Q3 = df[outlier_feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[outlier_feature] < lower_bound) | (df[outlier_feature] > upper_bound)]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Outliers Count", len(outliers))
    with col2:
        st.metric("Outliers %", f"{len(outliers)/len(df)*100:.2f}%")
    with col3:
        st.metric("Valid Range", f"[{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Box plot with outliers
    fig = px.box(
        df, 
        y=outlier_feature,
        title=f"{outlier_feature} - Outlier Detection",
        points='outliers'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if len(outliers) > 0:
        with st.expander("View Outlier Samples"):
            st.dataframe(outliers.head(20), use_container_width=True)

    # ===========================================================
    # SECTION 8: MULTIVARIATE ANALYSIS
    # ===========================================================
    st.subheader("🔍 Multivariate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature_x = st.selectbox("X-axis Feature", feature_cols, key='scatter_x')
    with col2:
        feature_y = st.selectbox("Y-axis Feature", feature_cols, index=1, key='scatter_y')
    
    # Scatter plot
    fig = px.scatter(
        df,
        x=feature_x,
        y=feature_y,
        color='quality_binary',
        title=f'{feature_x} vs {feature_y}',
        color_discrete_map={0: '#ff6b6b', 1: '#51cf66'},
        opacity=0.6,
        labels={'quality_binary': 'Quality'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter matrix
    with st.expander("📊 Scatter Matrix (Select Features)"):
        selected_features = st.multiselect(
            "Choose 2-4 features for scatter matrix",
            feature_cols,
            default=feature_cols[:3]
        )
        
        if len(selected_features) >= 2:
            fig = px.scatter_matrix(
                df,
                dimensions=selected_features,
                color='quality_binary',
                title='Scatter Matrix',
                color_discrete_map={0: '#ff6b6b', 1: '#51cf66'},
                height=800
            )
            fig.update_traces(diagonal_visible=False, showupperhalf=False)
            st.plotly_chart(fig, use_container_width=True)

    # ===========================================================
    # SECTION 9: DISTRIBUTION COMPARISON
    # ===========================================================
    with st.expander("📊 All Features Distribution Comparison"):
        n_cols = 3
        n_features = len(feature_cols)
        
        for i in range(0, n_features, n_cols):
            cols = st.columns(n_cols)
            for j in range(n_cols):
                if i + j < n_features:
                    feat = feature_cols[i + j]
                    with cols[j]:
                        fig = px.violin(
                            df,
                            x='quality_binary',
                            y=feat,
                            box=True,
                            title=feat,
                            color='quality_binary',
                            color_discrete_map={0: '#ff6b6b', 1: '#51cf66'}
                        )
                        fig.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig, use_container_width=True)


# ===========================================================
# PAGE 2 – MODEL MONITORING
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
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    with col2: st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
    with col3: st.metric("Precision", f"{metrics['precision']:.4f}")
    with col4: st.metric("Recall", f"{metrics['recall']:.4f}")

    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_eval, y_pred)
    fig = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual"))
    st.plotly_chart(fig)


# ===========================================================
# PAGE 3 – DATA DRIFT
# ===========================================================
elif page == "⚠️ Data Drift Detection":
    st.header("⚠️ Data Drift Detection")

    if model is None:
        st.warning("Model not loaded, showing distribution comparison only.")
    
    X_orig = test_df.drop('quality_binary', axis=1)
    y_orig = test_df['quality_binary']
    X_drift = drift_df.drop('quality_binary', axis=1)
    y_drift = drift_df['quality_binary']

    # Performance comparison if model exists
    if model is not None:
        st.subheader("📊 Performance Comparison")
        
        try:
            y_pred_orig = model.predict(X_orig)
            y_pred_drift = model.predict(X_drift)
            
            metrics_orig = calc_metrics(y_orig, y_pred_orig)
            metrics_drift = calc_metrics(y_drift, y_pred_drift)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Test Set")
                st.metric("Accuracy", f"{metrics_orig['accuracy']:.4f}")
                st.metric("F1 Score", f"{metrics_orig['f1_score']:.4f}")
            
            with col2:
                st.markdown("### Drift Test Set")
                acc_change = metrics_drift['accuracy'] - metrics_orig['accuracy']
                f1_change = metrics_drift['f1_score'] - metrics_orig['f1_score']
                st.metric("Accuracy", f"{metrics_drift['accuracy']:.4f}", delta=f"{acc_change:.4f}")
                st.metric("F1 Score", f"{metrics_drift['f1_score']:.4f}", delta=f"{f1_change:.4f}")
            
            # Warning
            if abs(acc_change) > 0.05 or abs(f1_change) > 0.05:
                st.error("⚠️ Significant performance change detected!")
            else:
                st.success("✅ Model performance is stable")
        
        except ValueError as e:
            if "feature names" in str(e).lower():
                st.error("Feature mismatch: Model and data have different features.")
                st.info("Cannot perform drift detection due to feature mismatch.")
            else:
                st.error(f"Error during prediction: {e}")

    # Feature drift analysis (only on common features)
    st.subheader("📊 Feature Distribution Drift")
    
    # Get only original features (non-engineered)
    original_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
                        'density', 'pH', 'sulphates', 'alcohol']
    
    # Filter to features present in both datasets
    common_features = [f for f in original_features if f in X_orig.columns and f in X_drift.columns]
    
    drift_table = []
    for col in common_features:
        ks, p = stats.ks_2samp(X_orig[col], X_drift[col])
        drift_table.append({
            "Feature": col,
            "KS Statistic": f"{ks:.4f}",
            "p-value": f"{p:.4f}",
            "Drift?": "Yes" if p < 0.05 else "No"
        })

    drift_df_table = pd.DataFrame(drift_table)
    st.dataframe(drift_df_table, use_container_width=True)
    
    # Visualize drift for selected feature
    st.subheader("🔍 Feature Distribution Comparison")
    selected_feat = st.selectbox("Select feature", common_features)
    
    fig = px.histogram(
        pd.DataFrame({
            'Original': X_orig[selected_feat],
            'Drift': X_drift[selected_feat]
        }).melt(var_name='Dataset', value_name='Value'),
        x='Value',
        color='Dataset',
        barmode='overlay',
        title=f"{selected_feat} Distribution Comparison",
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================
# PAGE 4 – FEATURE ANALYSIS
# ===========================================================
elif page == "📈 Feature Analysis":
    st.header("📈 Feature Behavior by Quality Class")

    dataset_choice = st.selectbox("Dataset", ["Train", "Test"])
    df = train_df if dataset_choice == "Train" else test_df

    feature = st.selectbox("Select feature", [c for c in df.columns if c not in ['quality', 'quality_binary']])

    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.violin(df, x='quality_binary', y=feature, box=True, 
                       title=f"{feature} by Quality Class")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, x='quality_binary', y=feature,
                    title=f"{feature} Box Plot")
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("📊 Statistical Summary")
    summary = df.groupby('quality_binary')[feature].describe()
    st.dataframe(summary)


# ===========================================================
# PAGE 5 – MANUAL INFERENCE
# ===========================================================
elif page == "🧪 Manual Inference":
    st.header("🧪 Manual Inference – Predict Wine Quality")

    if model is None:
        st.error("Model not loaded. Please train a model first.")
        st.stop()

    # Check if model expects engineered features
    try:
        # Try to get feature names from the model
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps['clf'], 'feature_names_in_'):
            expected_features = list(model.named_steps['clf'].feature_names_in_)
        else:
            expected_features = list(train_df.columns[:-1])  # All except target
        
        has_engineered_features = 'acidity_ratio' in expected_features
        
        if has_engineered_features:
            st.info("ℹ️ Model was trained with feature engineering. Engineered features will be calculated automatically.")
    except:
        expected_features = None
        has_engineered_features = False

    # Original feature inputs
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
    
    st.subheader("Input Summary")
    st.dataframe(input_df)

    if st.button("🔮 Predict Quality", type="primary"):
        # Add engineered features if needed
        if has_engineered_features:
            st.markdown("**Adding Engineered Features...**")
            
            # Create engineered features
            input_df['acidity_ratio'] = input_df['fixed acidity'] / (input_df['volatile acidity'] + 0.001)
            input_df['sulfur_ratio'] = input_df['free sulfur dioxide'] / (input_df['total sulfur dioxide'] + 0.001)
            input_df['alcohol_density'] = input_df['alcohol'] / input_df['density']
            input_df['total_acidity'] = input_df['fixed acidity'] + input_df['volatile acidity']
            input_df['alcohol_squared'] = input_df['alcohol'] ** 2
            input_df['ph_alcohol'] = input_df['pH'] * input_df['alcohol']
            input_df['sulphates_alcohol'] = input_df['sulphates'] * input_df['alcohol']
            
            with st.expander("View All Features (including engineered)"):
                st.dataframe(input_df)
        
        try:
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]

            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                quality_label = "High Quality (1)" if pred == 1 else "Low Quality (0)"
                st.metric("Predicted Quality", quality_label)
            
            with col2:
                st.metric("High Quality Probability", f"{prob[1]:.2%}")
            
            # Probability bar
            st.progress(prob[1])
            
            if prob[1] > 0.7:
                st.success("High confidence in high quality prediction!")
            elif prob[1] < 0.3:
                st.info("High confidence in low quality prediction.")
            else:
                st.warning("Moderate confidence - borderline case.")
        
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.error("This may be due to feature mismatch. Check if preprocessing used feature engineering.")

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
    st.sidebar.write(f"Run ID: {run_id[:8]}...")
    st.sidebar.write("Model Type: AutoML Selected")

# st.sidebar.markdown("---")
# st.sidebar.info("💡 Run: streamlit run src/dashboard_py.py")