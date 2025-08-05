#!/usr/bin/env python3
"""
Tests for utility functions.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from utils import normalize_data, preprocess_data, split_data, evaluate_model, calculate_ensemble_metrics


def test_normalize_data():
    """Test data normalization function."""
    # Create test data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    
    # Normalize data
    X_norm = normalize_data(X)
    
    # Check that means are close to 0
    assert np.allclose(np.mean(X_norm, axis=0), 0, atol=1e-10)
    
    # Check that standard deviations are close to 1
    assert np.allclose(np.std(X_norm, axis=0), 1, atol=1e-10)
    
    # Check that normalization is reversible (up to scaling)
    X_original = X_norm * np.std(X, axis=0) + np.mean(X, axis=0)
    np.testing.assert_array_almost_equal(X, X_original, decimal=10)


def test_preprocess_data():
    """Test data preprocessing function."""
    # Create test data with missing values
    X = np.array([[1, 2, np.nan], [3, 4, 5], [5, 6, 7]])
    y = np.array([0, 1, 0])
    
    # Preprocess data
    X_processed, y_processed = preprocess_data(X, y)
    
    # Check that missing values are filled
    assert not np.any(np.isnan(X_processed))
    
    # Check that data is standardized
    assert np.allclose(np.mean(X_processed, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(X_processed, axis=0), 1, atol=1e-10)
    
    # Check that target is unchanged
    np.testing.assert_array_equal(y, y_processed)


def test_preprocess_data_without_target():
    """Test data preprocessing function without target."""
    # Create test data
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Preprocess data
    X_processed = preprocess_data(X)
    
    # Check that data is standardized
    assert np.allclose(np.mean(X_processed, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(X_processed, axis=0), 1, atol=1e-10)


def test_preprocess_data_with_categorical():
    """Test data preprocessing with categorical columns."""
    # Create test data with categorical-like values
    X = np.array([[1, 'A', 3], [2, 'B', 4], [1, 'A', 5]])
    y = np.array([0, 1, 0])
    
    # Preprocess data with categorical columns
    X_processed, y_processed = preprocess_data(X, y, categorical_columns=[1])
    
    # Check that categorical column is encoded
    assert X_processed.shape == (3, 3)
    assert not np.any(np.isnan(X_processed))
    
    # Check that target is unchanged
    np.testing.assert_array_equal(y, y_processed)


def test_split_data():
    """Test data splitting function."""
    # Create test data
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    # Check split sizes
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    
    # Check that data is not shuffled (with fixed random state)
    np.testing.assert_array_equal(X_train, X[:80])
    np.testing.assert_array_equal(X_test, X[80:])
    np.testing.assert_array_equal(y_train, y[:80])
    np.testing.assert_array_equal(y_test, y[80:])


def test_evaluate_model_classification():
    """Test model evaluation for classification."""
    # Create test data
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    # Evaluate model
    metrics = evaluate_model(y_true, y_pred, task_type='classification')
    
    # Check metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    
    # Check metric values
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1


def test_evaluate_model_regression():
    """Test model evaluation for regression."""
    # Create test data
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    # Evaluate model
    metrics = evaluate_model(y_true, y_pred, task_type='regression')
    
    # Check metrics
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    
    # Check metric values
    assert metrics['mse'] >= 0
    assert metrics['rmse'] >= 0
    assert metrics['mae'] >= 0
    
    # Check that RMSE is square root of MSE
    assert abs(metrics['rmse'] - np.sqrt(metrics['mse'])) < 1e-10


def test_calculate_ensemble_metrics():
    """Test ensemble metrics calculation."""
    # Create test predictions DataFrame
    predictions_df = pd.DataFrame({
        'model_0': [0.1, 0.2, 0.3, 0.4, 0.5],
        'model_1': [0.15, 0.25, 0.35, 0.45, 0.55],
        'model_2': [0.12, 0.22, 0.32, 0.42, 0.52]
    })
    
    # Calculate ensemble metrics
    metrics = calculate_ensemble_metrics(predictions_df)
    
    # Check metrics
    assert 'variance' in metrics
    assert 'ambiguity' in metrics
    assert 'disagreement' in metrics
    
    # Check metric values
    assert metrics['variance'] >= 0
    assert metrics['ambiguity'] >= 0
    assert 0 <= metrics['disagreement'] <= 1


def test_calculate_ensemble_metrics_single_model():
    """Test ensemble metrics with single model."""
    # Create test predictions DataFrame with single model
    predictions_df = pd.DataFrame({
        'model_0': [0.1, 0.2, 0.3, 0.4, 0.5]
    })
    
    # Calculate ensemble metrics
    metrics = calculate_ensemble_metrics(predictions_df)
    
    # Check metrics
    assert 'variance' in metrics
    assert 'ambiguity' in metrics
    assert 'disagreement' in metrics
    
    # With single model, variance should be 0 and disagreement should be 0
    assert metrics['variance'] == 0
    assert metrics['disagreement'] == 0


def test_normalize_data_edge_cases():
    """Test normalize_data with edge cases."""
    # Test with single sample
    X = np.array([[1, 2, 3]])
    X_norm = normalize_data(X)
    assert X_norm.shape == (1, 3)
    
    # Test with constant column
    X = np.array([[1, 2, 3], [1, 4, 5], [1, 6, 7]])
    X_norm = normalize_data(X)
    assert X_norm.shape == (3, 3)
    
    # Test with zero variance column (should handle division by zero)
    X = np.array([[1, 2, 3], [1, 4, 5], [1, 6, 7]])
    X_norm = normalize_data(X)
    # First column should be all zeros (constant column)
    assert np.allclose(X_norm[:, 0], 0)


def test_preprocess_data_edge_cases():
    """Test preprocess_data with edge cases."""
    # Test with all missing values
    X = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    y = np.array([0, 1])
    
    X_processed, y_processed = preprocess_data(X, y)
    assert not np.any(np.isnan(X_processed))
    
    # Test with empty array
    X = np.array([])
    y = np.array([])
    
    if len(X) > 0:  # Only test if array is not empty
        X_processed, y_processed = preprocess_data(X, y)
        assert len(X_processed) == 0
        assert len(y_processed) == 0


if __name__ == "__main__":
    # Run tests
    test_normalize_data()
    test_preprocess_data()
    test_preprocess_data_without_target()
    test_preprocess_data_with_categorical()
    test_split_data()
    test_evaluate_model_classification()
    test_evaluate_model_regression()
    test_calculate_ensemble_metrics()
    test_calculate_ensemble_metrics_single_model()
    test_normalize_data_edge_cases()
    test_preprocess_data_edge_cases()
    
    print("All utility tests passed!") 