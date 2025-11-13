"""
data preprocessing module
purpose: load raw data, create binary labels, split train and test
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def preprocess_wine_data(input_path='data/winequality-red.csv',
                         test_size=0.2,
                         random_state=42):
    """preprocess wine dataset and create train/test splits."""
    print("starting data preprocessing")

    df = pd.read_csv(input_path, sep=';')

    df['quality_binary'] = (df['quality'] >= 6).astype(int)

    x = df.drop(['quality', 'quality_binary'], axis=1)
    y = df['quality_binary']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = pd.concat([x_train, y_train], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)

    os.makedirs('data', exist_ok=True)

    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    test_changed_df = test_df.copy()
    feature_cols = list(x_test.columns)
    drift_features = feature_cols[:2]

    for feat in drift_features:
        noise = np.random.normal(0, test_changed_df[feat].std() * 0.3, size=len(test_changed_df))
        test_changed_df[feat] = test_changed_df[feat] + noise

    test_changed_df.to_csv('data/test_changed.csv', index=False)

    print("data preprocessing complete")
    return train_df, test_df, test_changed_df


if __name__ == "__main__":
    if not os.path.exists('data/winequality-red.csv'):
        print("error: data/winequality-red.csv not found")
    else:
        preprocess_wine_data()
