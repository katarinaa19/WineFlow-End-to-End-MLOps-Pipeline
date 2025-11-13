"""
inference module
purpose: send requests to mlflow model service and return predictions
includes feature engineering for compatibility with improved model
"""
import requests
import pandas as pd
import numpy as np
from utils_py import load_data
import os


def add_engineered_features(df):
    """
    add engineered features to match training data
    ONLY USE THIS IF YOU USED FEATURE ENGINEERING IN PREPROCESSING
    """
    df = df.copy()
    
    # 1. Acidity ratio
    df['acidity_ratio'] = df['fixed acidity'] / (df['volatile acidity'] + 0.001)
    
    # 2. Sulfur dioxide ratio
    df['sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 0.001)
    
    # 3. Alcohol-to-density ratio
    df['alcohol_density'] = df['alcohol'] / df['density']
    
    # 4. Total acidity
    df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
    
    # 5. Alcohol squared
    df['alcohol_squared'] = df['alcohol'] ** 2
    
    # 6. pH-alcohol interaction
    df['ph_alcohol'] = df['pH'] * df['alcohol']
    
    # 7. Sulphates-alcohol interaction
    df['sulphates_alcohol'] = df['sulphates'] * df['alcohol']
    
    return df


def predict_api(data, endpoint='http://127.0.0.1:8000/invocations', use_feature_engineering=True):
    """
    call mlflow model service api
    
    args:
        data: dataframe or dict with 11 original features
        endpoint: model service url
        use_feature_engineering: set to True if model was trained with engineered features
    """
    
    # Convert dict to dataframe if needed
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a dataframe or dict")
    
    # Add engineered features if enabled
    if use_feature_engineering:
        data = add_engineered_features(data)
    
    # Prepare payload
    payload = {
        "dataframe_split": {
            "columns": data.columns.tolist(),
            "data": data.values.tolist()
        }
    }

    headers = {'content-type': 'application/json'}

    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()

        # safely parse json
        try:
            predictions = response.json()
        except ValueError:
            print("error: invalid json response from model service")
            return None

        # handle mlflow output naming conventions
        preds = predictions.get('predictions', predictions.get('outputs'))
        if preds is None:
            print("error: no predictions field in response")
            return None

        return preds

    except Exception as e:
        print(f"error: unable to call model service - {e}")
        return None


def test_inference(sample_size=5, use_feature_engineering=True):
    """
    test inference on random test samples
    
    args:
        sample_size: number of samples to test
        use_feature_engineering: must match preprocessing setting
    """
    print("=" * 60)
    print("starting inference test")
    print(f"feature engineering: {'enabled' if use_feature_engineering else 'disabled'}")
    print("=" * 60)

    x_test, y_test = load_data('data/test.csv', target_col='quality_binary')
    
    # If test data already has engineered features, don't add them again
    has_engineered = 'acidity_ratio' in x_test.columns
    
    if has_engineered:
        print("\ntest data already contains engineered features")
        use_fe_in_api = False
    else:
        print("\ntest data contains only original features")
        use_fe_in_api = use_feature_engineering
    
    print(f"number of features in test data: {x_test.shape[1]}")

    sample_indices = np.random.choice(len(x_test), size=sample_size, replace=False)
    x_sample = x_test.iloc[sample_indices].reset_index(drop=True)
    y_sample = y_test.iloc[sample_indices].reset_index(drop=True)

    print(f"\ntesting on {sample_size} samples...")
    preds = predict_api(x_sample, use_feature_engineering=use_fe_in_api)
    
    if preds is None:
        print("\ninference test failed")
        print("\ntroubleshooting:")
        print("1. check if model service is running: mlflow models serve ...")
        print("2. verify the model expects the same features as test data")
        print("3. check if feature engineering matches between train and test")
        return

    preds = np.array(preds)
    correct = np.sum(preds == y_sample.values)
    accuracy = correct / len(preds)

    print(f"\nresults:")
    print(f"  samples tested: {len(preds)}")
    print(f"  correct predictions: {correct}")
    print(f"  accuracy: {accuracy:.2%}")
    
    print("\n" + "=" * 60)
    print("inference test complete")
    print("=" * 60)


def predict_single_sample(features_dict, use_feature_engineering=True):
    """
    predict label for a single sample dictionary
    
    args:
        features_dict: dict with 11 original features
        use_feature_engineering: must match preprocessing setting
    
    returns:
        prediction (0 or 1)
    """
    print("\nsingle sample prediction:")
    print(f"input features: {list(features_dict.keys())}")
    
    preds = predict_api(features_dict, use_feature_engineering=use_feature_engineering)
    
    if preds is None:
        print("prediction failed")
        return None
    
    pred = preds[0]
    quality = "high quality (1)" if pred == 1 else "low quality (0)"
    print(f"prediction: {pred} ({quality})")
    
    return pred


if __name__ == "__main__":
    import sys

    if not os.path.exists('data/test.csv'):
        print("error: data/test.csv not found")
        print("please run: python src/preprocess.py")
        sys.exit(1)

    # Check if test data has engineered features
    test_df = pd.read_csv('data/test.csv')
    feature_engineering_used = 'acidity_ratio' in test_df.columns
    
    print("=" * 60)
    print("INFERENCE MODULE")
    print("=" * 60)
    print(f"\ndetected feature engineering: {feature_engineering_used}")
    print(f"test data features: {test_df.shape[1] - 1}")  # -1 for target
    
    # Run test
    test_inference(sample_size=10, use_feature_engineering=feature_engineering_used)

    # Single sample test
    print("\n" + "=" * 60)
    print("SINGLE SAMPLE TEST")
    print("=" * 60)
    
    example_features = {
        'fixed acidity': 7.4,
        'volatile acidity': 0.7,
        'citric acid': 0.0,
        'residual sugar': 1.9,
        'chlorides': 0.076,
        'free sulfur dioxide': 11.0,
        'total sulfur dioxide': 34.0,
        'density': 0.9978,
        'pH': 3.51,
        'sulphates': 0.56,
        'alcohol': 9.4
    }

    result = predict_single_sample(example_features, use_feature_engineering=feature_engineering_used)