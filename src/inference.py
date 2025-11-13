"""
inference module
purpose: send requests to mlflow model service and return predictions
"""
import requests
import pandas as pd
import numpy as np
from utils_py import load_data
import os


def predict_api(data, endpoint='http://127.0.0.1:8000/invocations'):
    """call mlflow model service api."""

    if isinstance(data, pd.DataFrame):
        payload = {
            "dataframe_split": {
                "columns": data.columns.tolist(),
                "data": data.values.tolist()
            }
        }
    elif isinstance(data, dict):
        # convert dict to dataframe_split format
        payload = {
            "dataframe_split": {
                "columns": list(data.keys()),
                "data": [list(data.values())]
            }
        }
    else:
        raise ValueError("data must be a dataframe or dict")

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

    except Exception:
        print("error: unable to call model service")
        return None


def test_inference(sample_size=5):
    """test inference on random test samples."""
    print("starting inference test")

    x_test, y_test = load_data('data/test.csv', target_col='quality_binary')

    sample_indices = np.random.choice(len(x_test), size=sample_size, replace=False)
    x_sample = x_test.iloc[sample_indices].reset_index(drop=True)
    y_sample = y_test.iloc[sample_indices].reset_index(drop=True)

    preds = predict_api(x_sample)
    if preds is None:
        print("inference test failed")
        return

    preds = np.array(preds)
    correct = np.sum(preds == y_sample.values)
    accuracy = correct / len(preds)

    print(f"inference test accuracy: {accuracy:.4f}")
    print("inference test complete")


def predict_single_sample(features_dict):
    """predict label for a single sample dictionary."""
    df = pd.DataFrame([features_dict])
    preds = predict_api(df)
    if preds is None:
        return None
    return preds[0]


if __name__ == "__main__":
    import sys

    if not os.path.exists('data/test.csv'):
        print("error: data/test.csv not found")
        sys.exit(1)

    test_inference(sample_size=10)

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

    result = predict_single_sample(example_features)
    print(f"single sample prediction: {result}")
