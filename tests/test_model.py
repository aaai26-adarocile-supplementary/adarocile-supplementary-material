#!/usr/bin/env python3
"""
Tests for Rocile and AdaRocile models.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from mypackage.model import Rocile, AdaRocile


def test_rocile_initialization():
    """Test Rocile model initialization."""
    model = Rocile(
        batch_size=2,
        momentum=0.9,
        learning_rate=0.05,
        max_iter=1000,
        random_state=42
    )
    
    assert model.batch_size == 2
    assert model.momentum == 0.9
    assert model.learning_rate == 0.05
    assert model.max_iter == 1000
    assert model.random_state == 42
    assert not model.is_fitted_


def test_adarocile_initialization():
    """Test AdaRocile model initialization."""
    model = AdaRocile(
        batch_size=2,
        momentum=0.9,
        learning_rate=0.05,
        max_iter=1000,
        bias_threshold=0.6,
        base_k=30,
        random_state=42
    )
    
    assert model.batch_size == 2
    assert model.momentum == 0.9
    assert model.learning_rate == 0.05
    assert model.max_iter == 1000
    assert model.bias_threshold == 0.6
    assert model.base_k == 30
    assert model.random_state == 42
    assert not model.is_fitted_


def test_rocile_classification():
    """Test Rocile model on classification task."""
    # Generate classification data
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = Rocile(random_state=42)
    model.fit(X_train, y_train)
    
    # Check that model is fitted
    assert model.is_fitted_
    assert model.models_ is not None
    assert len(model.models_) >= 2
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Check predictions
    assert len(y_pred) == len(y_test)
    assert np.all((y_pred >= 0) & (y_pred <= 1))  # Classification probabilities
    assert not np.any(np.isnan(y_pred))
    
    # Calculate accuracy
    y_pred_class = y_pred.round().astype(int)
    accuracy = accuracy_score(y_test, y_pred_class)
    assert 0 <= accuracy <= 1


def test_rocile_regression():
    """Test Rocile model on regression task."""
    # Generate regression data
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = Rocile(random_state=42)
    model.fit(X_train, y_train)
    
    # Check that model is fitted
    assert model.is_fitted_
    assert model.models_ is not None
    assert len(model.models_) >= 2
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Check predictions
    assert len(y_pred) == len(y_test)
    assert not np.any(np.isnan(y_pred))
    
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    assert mse >= 0


def test_adarocile_classification():
    """Test AdaRocile model on classification task."""
    # Generate classification data
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = AdaRocile(random_state=42)
    model.fit(X_train, y_train)
    
    # Check that model is fitted
    assert model.is_fitted_
    assert model.models_ is not None
    assert len(model.models_) >= 2
    assert model.X_train_ is not None
    assert model.y_train_ is not None
    assert model.nn_ is not None
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Check predictions
    assert len(y_pred) == len(y_test)
    assert np.all((y_pred >= 0) & (y_pred <= 1))  # Classification probabilities
    assert not np.any(np.isnan(y_pred))
    
    # Calculate accuracy
    y_pred_class = y_pred.round().astype(int)
    accuracy = accuracy_score(y_test, y_pred_class)
    assert 0 <= accuracy <= 1


def test_adarocile_regression():
    """Test AdaRocile model on regression task."""
    # Generate regression data
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = AdaRocile(random_state=42)
    model.fit(X_train, y_train)
    
    # Check that model is fitted
    assert model.is_fitted_
    assert model.models_ is not None
    assert len(model.models_) >= 2
    assert model.X_train_ is not None
    assert model.y_train_ is not None
    assert model.nn_ is not None
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Check predictions
    assert len(y_pred) == len(y_test)
    assert not np.any(np.isnan(y_pred))
    
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    assert mse >= 0


def test_model_fit_predict_consistency():
    """Test that model predictions are consistent across multiple calls."""
    # Generate data
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Test Rocile
    rocile_model = Rocile(random_state=42)
    rocile_model.fit(X_train, y_train)
    
    pred1 = rocile_model.predict(X_test)
    pred2 = rocile_model.predict(X_test)
    
    np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)
    
    # Test AdaRocile
    adarocile_model = AdaRocile(random_state=42)
    adarocile_model.fit(X_train, y_train)
    
    pred1 = adarocile_model.predict(X_test)
    pred2 = adarocile_model.predict(X_test)
    
    np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)


def test_model_with_insufficient_data():
    """Test that models handle insufficient data gracefully."""
    # Very small dataset
    X = np.random.rand(5, 3)
    y = np.random.randint(0, 2, 5)
    
    # Should raise error for insufficient data
    with pytest.raises(ValueError):
        model = Rocile(random_state=42)
        model.fit(X, y)
    
    with pytest.raises(ValueError):
        model = AdaRocile(random_state=42)
        model.fit(X, y)


def test_model_without_fitting():
    """Test that prediction without fitting raises error."""
    X = np.random.rand(10, 5)
    
    rocile_model = Rocile(random_state=42)
    with pytest.raises(ValueError):
        rocile_model.predict(X)
    
    adarocile_model = AdaRocile(random_state=42)
    with pytest.raises(ValueError):
        adarocile_model.predict(X)


if __name__ == "__main__":
    # Run tests
    test_rocile_initialization()
    test_adarocile_initialization()
    test_rocile_classification()
    test_rocile_regression()
    test_adarocile_classification()
    test_adarocile_regression()
    test_model_fit_predict_consistency()
    
    print("All tests passed!") 