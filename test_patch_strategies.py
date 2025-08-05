#!/usr/bin/env python3
"""
Quick test script to verify all patch strategies work correctly.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from mypackage.model import AdaRocile

# Generate sample data
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test all patch strategies
patch_strategies = [
    'BiasCorrected',
    'DistanceWeighted',
    'ModelSpecific',
    'CertaintyWeighted',
    'EnsembleLevel'
]

print("Testing all patch strategies...")
print("=" * 50)

for strategy in patch_strategies:
    print(f"\nTesting {strategy}...")
    
    try:
        model = AdaRocile(
            patch_strategy=strategy,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred.round())
        
        print(f"  ✓ {strategy} completed successfully")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Initial variance: {model.initial_disagreement_['initial_variance']:.4f}")
        print(f"  Initial disagreement: {model.initial_disagreement_['initial_disagreement']:.4f}")
        
    except Exception as e:
        print(f"  ✗ {strategy} failed: {e}")

print(f"\n{'='*50}")
print("All patch strategies tested!") 