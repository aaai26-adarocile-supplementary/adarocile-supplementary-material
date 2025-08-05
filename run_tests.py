#!/usr/bin/env python3
"""
Simple test runner for Rocile and AdaRocile package.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_model_tests():
    """Run model tests without pytest."""
    print("Running model tests...")
    
    try:
        from mypackage.model import Rocile, AdaRocile
        from sklearn.datasets import make_classification, make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, mean_squared_error
        import numpy as np
        
        # Test Rocile initialization
        print("  ✓ Testing Rocile initialization...")
        rocile = Rocile(batch_size=2, momentum=0.9, learning_rate=0.05, max_iter=1000, random_state=42)
        assert rocile.batch_size == 2
        assert rocile.momentum == 0.9
        assert not rocile.is_fitted_
        
        # Test AdaRocile initialization
        print("  ✓ Testing AdaRocile initialization...")
        adarocile = AdaRocile(batch_size=2, momentum=0.9, learning_rate=0.05, max_iter=1000, bias_threshold=0.6, base_k=30, random_state=42)
        assert adarocile.batch_size == 2
        assert adarocile.bias_threshold == 0.6
        assert not adarocile.is_fitted_
        
        # Test classification
        print("  ✓ Testing classification...")
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rocile.fit(X_train, y_train)
        assert rocile.is_fitted_
        assert len(rocile.models_) >= 2
        
        y_pred = rocile.predict(X_test)
        assert len(y_pred) == len(y_test)
        assert np.all((y_pred >= 0) & (y_pred <= 1))
        
        # Test regression
        print("  ✓ Testing regression...")
        X, y = make_regression(n_samples=100, n_features=10, n_informative=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rocile_reg = Rocile(random_state=42)
        rocile_reg.fit(X_train, y_train)
        assert rocile_reg.is_fitted_
        
        y_pred = rocile_reg.predict(X_test)
        assert len(y_pred) == len(y_test)
        
        print("  ✓ All model tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Model tests failed: {e}")
        return False

def run_utils_tests():
    """Run utility tests without pytest."""
    print("Running utility tests...")
    
    try:
        from utils import normalize_data, preprocess_data, split_data, evaluate_model
        import numpy as np
        
        # Test normalize_data
        print("  ✓ Testing normalize_data...")
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_norm = normalize_data(X)
        assert np.allclose(np.mean(X_norm, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_norm, axis=0), 1, atol=1e-10)
        
        # Test preprocess_data
        print("  ✓ Testing preprocess_data...")
        X = np.array([[1, 2, np.nan], [3, 4, 5], [5, 6, 7]])
        y = np.array([0, 1, 0])
        X_processed, y_processed = preprocess_data(X, y)
        assert not np.any(np.isnan(X_processed))
        assert np.allclose(np.mean(X_processed, axis=0), 0, atol=1e-10)
        
        # Test split_data
        print("  ✓ Testing split_data...")
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        assert len(X_train) == 80
        assert len(X_test) == 20
        
        # Test evaluate_model
        print("  ✓ Testing evaluate_model...")
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        metrics = evaluate_model(y_true, y_pred, task_type='classification')
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        
        print("  ✓ All utility tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Utility tests failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Rocile and AdaRocile Package Tests")
    print("=" * 50)
    
    model_tests_passed = run_model_tests()
    utils_tests_passed = run_utils_tests()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if model_tests_passed and utils_tests_passed:
        print("✓ All tests passed!")
        print("Package is working correctly.")
        return 0
    else:
        print("✗ Some tests failed.")
        print("Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 