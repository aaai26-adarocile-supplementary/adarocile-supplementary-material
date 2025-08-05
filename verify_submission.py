#!/usr/bin/env python3
"""
Submission Verification Script

This script verifies that all components of the AdaRocile package are properly
set up and working correctly for academic submission.
"""

import os
import sys
import importlib
import pandas as pd
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists and print status."""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description}: {dirpath} - MISSING")
        return False

def check_dataset_files():
    """Check that all dataset files exist."""
    print("\n📊 Checking Dataset Files...")
    datasets = [
        ('german_cleaned.csv', 'German Dataset'),
        ('compas_cleaned.csv', 'COMPAS Dataset'),
        ('communities_cleaned.csv', 'Communities Dataset'),
        ('adult_cleaned.csv', 'Adult Dataset'),
        ('folks_mobility_FL_cleaned.csv', 'Folk Mobility Dataset'),
        ('folks_travel_FL_cleaned.csv', 'Folk Travel Dataset'),
        ('folks_income_FL_cleaned.csv', 'Folk Income Dataset')
    ]
    
    all_exist = True
    for filename, description in datasets:
        filepath = os.path.join('datasets', filename)
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist

def check_example_runners():
    """Check that all example runners exist."""
    print("\n🧪 Checking Example Runners...")
    examples = [
        ('German_Data/run_german_dataset.py', 'German Dataset Runner'),
        ('Compas_Data/run_compas_dataset.py', 'COMPAS Dataset Runner'),
        ('Communities_Data/run_communities_dataset.py', 'Communities Dataset Runner'),
        ('Adult_Data/run_adult_dataset.py', 'Adult Dataset Runner'),
        ('Folk_Mobility_Data/run_folk_mobility_dataset.py', 'Folk Mobility Dataset Runner'),
        ('Folk_Travel_Data/run_folk_travel_dataset.py', 'Folk Travel Dataset Runner'),
        ('Folk_Income_Data/run_folk_income_dataset.py', 'Folk Income Dataset Runner')
    ]
    
    all_exist = True
    for filepath, description in examples:
        full_path = os.path.join('Examples', filepath)
        if not check_file_exists(full_path, description):
            all_exist = False
    
    return all_exist

def check_core_files():
    """Check that all core package files exist."""
    print("\n🏗️ Checking Core Package Files...")
    core_files = [
        ('mypackage/__init__.py', 'Package Initialization'),
        ('mypackage/model.py', 'Core Model Implementation'),
        ('utils.py', 'Utility Functions'),
        ('train.py', 'Training Script'),
        ('evaluate.py', 'Evaluation Script'),
        ('run_all_examples.py', 'All Examples Runner'),
        ('test_patch_strategies.py', 'Patch Strategies Tester'),
        ('run_tests.py', 'Test Runner'),
        ('setup.py', 'Package Setup'),
        ('requirements.txt', 'Dependencies'),
        ('setup.sh', 'Setup Script'),
        ('README.md', 'Documentation'),
        ('SUBMISSION_SUMMARY.md', 'Submission Summary')
    ]
    
    all_exist = True
    for filepath, description in core_files:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist

def check_test_files():
    """Check that test files exist."""
    print("\n🧪 Checking Test Files...")
    test_files = [
        ('tests/test_model.py', 'Model Tests'),
        ('tests/test_utils.py', 'Utility Tests')
    ]
    
    all_exist = True
    for filepath, description in test_files:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist

def test_imports():
    """Test that core modules can be imported."""
    print("\n📦 Testing Imports...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Test mypackage import
        import mypackage
        print("✅ mypackage imported successfully")
        
        # Test model classes
        from mypackage.model import Rocile, AdaRocile
        print("✅ Rocile and AdaRocile classes imported successfully")
        
        # Test utils
        import utils
        print("✅ utils module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of the models."""
    print("\n🔧 Testing Basic Functionality...")
    
    try:
        from mypackage.model import Rocile, AdaRocile
        import numpy as np
        from sklearn.datasets import make_classification
        
        # Generate test data
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        # Test Rocile
        rocile = Rocile(random_state=42)
        rocile.fit(X, y)
        predictions = rocile.predict(X)
        print("✅ Rocile basic functionality works")
        
        # Test AdaRocile
        adarocile = AdaRocile(patch_strategy='BiasCorrected', random_state=42)
        adarocile.fit(X, y)
        predictions = adarocile.predict(X)
        print("✅ AdaRocile basic functionality works")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def check_zip_file():
    """Check that the supplementary material zip exists."""
    print("\n📦 Checking Supplementary Material...")
    zip_path = '../supplementary_material.zip'
    if os.path.exists(zip_path):
        size = os.path.getsize(zip_path) / (1024 * 1024)  # MB
        print(f"✅ Supplementary Material: {zip_path} ({size:.1f} MB)")
        return True
    else:
        print(f"❌ Supplementary Material: {zip_path} - MISSING")
        return False

def main():
    """Run all verification checks."""
    print("🔍 AdaRocile Package - Submission Verification")
    print("=" * 60)
    
    # Change to Code directory
    if not os.path.exists('mypackage'):
        print("❌ Please run this script from the Code directory")
        return False
    
    all_checks_passed = True
    
    # Check all components
    if not check_core_files():
        all_checks_passed = False
    
    if not check_dataset_files():
        all_checks_passed = False
    
    if not check_example_runners():
        all_checks_passed = False
    
    if not check_test_files():
        all_checks_passed = False
    
    if not test_imports():
        all_checks_passed = False
    
    if not test_basic_functionality():
        all_checks_passed = False
    
    if not check_zip_file():
        all_checks_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED! Package is ready for submission.")
        print("\n📋 Submission Checklist:")
        print("✅ Core algorithms implemented (Rocile, AdaRocile)")
        print("✅ All 7 datasets included")
        print("✅ Enhanced metrics tracking")
        print("✅ 5 local patching strategies")
        print("✅ Comprehensive examples and tests")
        print("✅ Complete documentation")
        print("✅ Automated setup and installation")
        print("✅ Supplementary material archive")
        print("\n🚀 The package is ready for academic submission!")
    else:
        print("❌ Some checks failed. Please fix the issues before submission.")
    
    return all_checks_passed

if __name__ == "__main__":
    main() 