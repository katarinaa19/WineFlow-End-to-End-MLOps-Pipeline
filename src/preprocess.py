"""
data preprocessing module - IMPROVED VERSION
purpose: enhanced preprocessing with feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def preprocess_wine_data(input_path='data/winequality-red.csv', 
                         test_size=0.2, 
                         random_state=42,
                         feature_engineering=True):
    """
    preprocess wine quality dataset with optional feature engineering
    """
    print("=" * 60)
    print("IMPROVED DATA PREPROCESSING")
    print("=" * 60)
    
    print(f"\nreading data: {input_path}")
    df = pd.read_csv(input_path, sep=';')
    print(f"original shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    
    # Binary classification
    print("\nbinary classification: quality >= 6 -> 1, else -> 0")
    df['quality_binary'] = (df['quality'] >= 6).astype(int)
    
    value_counts = df['quality_binary'].value_counts()
    print(f"class distribution:")
    print(f"  high quality (1): {value_counts.get(1, 0)} ({value_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  low quality (0):  {value_counts.get(0, 0)} ({value_counts.get(0, 0)/len(df)*100:.1f}%)")
    
    # Feature engineering (optional)
    if feature_engineering:
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)
        
        # 1. Acidity ratios
        df['acidity_ratio'] = df['fixed acidity'] / (df['volatile acidity'] + 0.001)
        print("created: acidity_ratio = fixed acidity / volatile acidity")
        
        # 2. Sulfur dioxide ratio
        df['sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 0.001)
        print("created: sulfur_ratio = free SO2 / total SO2")
        
        # 3. Alcohol-to-density ratio (alcohol concentration indicator)
        df['alcohol_density'] = df['alcohol'] / df['density']
        print("created: alcohol_density = alcohol / density")
        
        # 4. Total acidity
        df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
        print("created: total_acidity = fixed acidity + volatile acidity")
        
        # 5. Alcohol categories (polynomial feature)
        df['alcohol_squared'] = df['alcohol'] ** 2
        print("created: alcohol_squared = alcohol^2")
        
        # 6. pH-alcohol interaction
        df['ph_alcohol'] = df['pH'] * df['alcohol']
        print("created: ph_alcohol = pH * alcohol")
        
        # 7. Sulphates-alcohol interaction (quality indicator)
        df['sulphates_alcohol'] = df['sulphates'] * df['alcohol']
        print("created: sulphates_alcohol = sulphates * alcohol")
        
        print(f"\nnew shape after feature engineering: {df.shape}")
        print(f"added {df.shape[1] - 13} new features")
    
    # Prepare features and target
    X = df.drop(['quality', 'quality_binary'], axis=1)
    y = df['quality_binary']
    
    print(f"\nfinal feature count: {X.shape[1]}")
    print(f"features: {list(X.columns)}")
    
    # Check for missing values
    if X.isnull().any().any():
        print("\nwarning: missing values detected, filling with median...")
        X = X.fillna(X.median())
    
    # Train-test split with stratification
    print(f"\ntrain-test split (test_size={test_size}, stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"training set: {X_train.shape[0]} samples")
    print(f"test set:     {X_test.shape[0]} samples")
    
    # Verify stratification
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    print(f"\ntrain distribution: class 0={train_dist[0]:.1%}, class 1={train_dist[1]:.1%}")
    print(f"test distribution:  class 0={test_dist[0]:.1%}, class 1={test_dist[1]:.1%}")
    
    # Save datasets
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    os.makedirs('data', exist_ok=True)
    
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nsaved: {train_path}")
    print(f"saved: {test_path}")
    
    # Create drift test set
    test_changed_df = test_df.copy()
    
    # Add noise to first 2 original features
    original_features = ['fixed acidity', 'volatile acidity']
    
    print(f"\ncreating drift test set...")
    print(f"adding noise to: {original_features}")
    
    for feat in original_features:
        if feat in test_changed_df.columns:
            noise = np.random.normal(0, test_changed_df[feat].std() * 0.3, size=len(test_changed_df))
            test_changed_df[feat] = test_changed_df[feat] + noise
    
    test_changed_path = 'data/test_changed.csv'
    test_changed_df.to_csv(test_changed_path, index=False)
    print(f"saved: {test_changed_path}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    return train_df, test_df, test_changed_df


if __name__ == "__main__":
    if not os.path.exists('data/winequality-red.csv'):
        print("error: data/winequality-red.csv not found")
        print("\nplease download from:")
        print("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
        print("and place in data/ directory")
    else:
        # set feature_engineering=True for improved performance
        preprocess_wine_data(feature_engineering=True)