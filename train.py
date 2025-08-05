#!/usr/bin/env python3
"""
Training script for Rocile and AdaRocile models.

This script demonstrates how to train and evaluate the Rocile and AdaRocile
ensemble reconciliation models on sample datasets.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import time

from mypackage.model import Rocile, AdaRocile
from utils import preprocess_data, split_data, evaluate_model


def generate_sample_data(n_samples=1000, n_features=20, task_type='classification', random_state=42):
    """
    Generate sample data for training.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate.
    n_features : int, default=20
        Number of features to generate.
    task_type : str, default='classification'
        Type of task ('classification' or 'regression').
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    X, y : tuple
        Generated features and target values.
    """
    if task_type == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=2,
            random_state=random_state
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,
            random_state=random_state
        )
    
    return X, y


def train_and_evaluate_models(X, y, task_type='classification', random_state=42):
    """
    Train and evaluate Rocile and AdaRocile models.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features.
    y : array-like of shape (n_samples,)
        Target values.
    task_type : str, default='classification'
        Type of task ('classification' or 'regression').
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    results : dict
        Dictionary containing training results and metrics.
    """
    print(f"\n{'='*60}")
    print(f"Training {task_type.capitalize()} Models")
    print(f"{'='*60}")
    
    # Preprocess data
    X_processed, y_processed = preprocess_data(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X_processed, y_processed, test_size=0.2, random_state=random_state
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    
    results = {}
    
    # Train Rocile model
    print(f"\n--- Training Rocile Model ---")
    start_time = time.time()
    
    rocile_model = Rocile(
        batch_size=2,
        momentum=0.9,
        learning_rate=0.05,
        max_iter=1000,
        random_state=random_state
    )
    
    rocile_model.fit(X_train, y_train)
    rocile_train_time = time.time() - start_time
    
    # Predict with Rocile
    start_time = time.time()
    rocile_pred = rocile_model.predict(X_test)
    rocile_pred_time = time.time() - start_time
    
    # Evaluate Rocile
    if task_type == 'classification':
        rocile_accuracy = accuracy_score(y_test, rocile_pred.round())
        rocile_metrics = {'accuracy': rocile_accuracy}
    else:
        rocile_mse = mean_squared_error(y_test, rocile_pred)
        rocile_metrics = {'mse': rocile_mse}
    
    results['rocile'] = {
        'model': rocile_model,
        'train_time': rocile_train_time,
        'pred_time': rocile_pred_time,
        'metrics': rocile_metrics
    }
    
    print(f"Rocile training time: {rocile_train_time:.3f}s")
    print(f"Rocile prediction time: {rocile_pred_time:.3f}s")
    if task_type == 'classification':
        print(f"Rocile accuracy: {rocile_accuracy:.4f}")
    else:
        print(f"Rocile MSE: {rocile_mse:.4f}")
    
    # Show initial disagreement metrics
    if hasattr(rocile_model, 'initial_disagreement_') and rocile_model.initial_disagreement_ is not None:
        print(f"Rocile initial variance: {rocile_model.initial_disagreement_['initial_variance']:.4f}")
        print(f"Rocile initial disagreement: {rocile_model.initial_disagreement_['initial_disagreement']:.4f}")
    
    # Train AdaRocile model
    print(f"\n--- Training AdaRocile Model ---")
    start_time = time.time()
    
    adarocile_model = AdaRocile(
        batch_size=2,
        momentum=0.9,
        learning_rate=0.05,
        max_iter=1000,
        bias_threshold=0.6,
        base_k=30,
        random_state=random_state
    )
    
    adarocile_model.fit(X_train, y_train)
    adarocile_train_time = time.time() - start_time
    
    # Predict with AdaRocile
    start_time = time.time()
    adarocile_pred = adarocile_model.predict(X_test)
    adarocile_pred_time = time.time() - start_time
    
    # Evaluate AdaRocile
    if task_type == 'classification':
        adarocile_accuracy = accuracy_score(y_test, adarocile_pred.round())
        adarocile_metrics = {'accuracy': adarocile_accuracy}
    else:
        adarocile_mse = mean_squared_error(y_test, adarocile_pred)
        adarocile_metrics = {'mse': adarocile_mse}
    
    results['adarocile'] = {
        'model': adarocile_model,
        'train_time': adarocile_train_time,
        'pred_time': adarocile_pred_time,
        'metrics': adarocile_metrics
    }
    
    print(f"AdaRocile training time: {adarocile_train_time:.3f}s")
    print(f"AdaRocile prediction time: {adarocile_pred_time:.3f}s")
    if task_type == 'classification':
        print(f"AdaRocile accuracy: {adarocile_accuracy:.4f}")
    else:
        print(f"AdaRocile MSE: {adarocile_mse:.4f}")
    
    # Show initial disagreement metrics
    if hasattr(adarocile_model, 'initial_disagreement_') and adarocile_model.initial_disagreement_ is not None:
        print(f"AdaRocile initial variance: {adarocile_model.initial_disagreement_['initial_variance']:.4f}")
        print(f"AdaRocile initial disagreement: {adarocile_model.initial_disagreement_['initial_disagreement']:.4f}")
    
    return results


def main():
    """Main function to run the training script."""
    print("Rocile and AdaRocile Training Script")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random_state = 42
    np.random.seed(random_state)
    
    # Train on classification data
    print("\n1. Classification Task")
    X_clf, y_clf = generate_sample_data(
        n_samples=1000, 
        n_features=20, 
        task_type='classification', 
        random_state=random_state
    )
    
    clf_results = train_and_evaluate_models(
        X_clf, y_clf, task_type='classification', random_state=random_state
    )
    
    # Train on regression data
    print("\n2. Regression Task")
    X_reg, y_reg = generate_sample_data(
        n_samples=1000, 
        n_features=20, 
        task_type='regression', 
        random_state=random_state
    )
    
    reg_results = train_and_evaluate_models(
        X_reg, y_reg, task_type='regression', random_state=random_state
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print("\nClassification Results:")
    print(f"  Rocile Accuracy: {clf_results['rocile']['metrics']['accuracy']:.4f}")
    print(f"  AdaRocile Accuracy: {clf_results['adarocile']['metrics']['accuracy']:.4f}")
    
    print("\nRegression Results:")
    print(f"  Rocile MSE: {reg_results['rocile']['metrics']['mse']:.4f}")
    print(f"  AdaRocile MSE: {reg_results['adarocile']['metrics']['mse']:.4f}")
    
    print("\nTraining completed successfully!")
    print("Models are ready for use.")


if __name__ == "__main__":
    main() 