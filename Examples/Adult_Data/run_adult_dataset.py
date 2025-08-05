#!/usr/bin/env python3
"""
Folk Mobility Dataset Experiment Runner

This script runs experiments on the Folk Mobility dataset for 5 seeds using the BiasCorrected patch strategy.
It saves all local and global metrics for analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, brier_score_loss, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mypackage.model import Rocile, AdaRocile
from utils import preprocess_data, split_data, evaluate_model, calculate_ensemble_metrics

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None

def load_adult_dataset():
    """Load and preprocess the Adult dataset."""
    print("Loading Adult dataset...")
    
    # Load dataset
    dataset_path = os.path.join('..', '..', 'datasets', 'adult_cleaned.csv')
    df = pd.read_csv(dataset_path)
    print(f"  Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Check if target column exists
    target_column = 'income'
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()
    
    # Handle categorical variables
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].nunique() < 20:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    # Fill missing values
    X = X.fillna(X.median())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=X.columns, 
        index=X.index
    )
    
    # Determine task type
    unique_values = y.dropna().unique()
    n_unique = len(unique_values)
    is_binary = n_unique == 2 and set(unique_values).issubset({0, 1, '0', '1', True, False})
    is_categorical = n_unique <= 10 and all(str(val).isdigit() or val in ['0', '1', True, False] for val in unique_values)
    is_classification = is_binary or is_categorical
    
    print(f"  Task type: {'Classification' if is_classification else 'Regression'}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    return X_scaled, y, is_classification

def calculate_initial_voting_metrics(models, X_test, y_test, task_type):
    """Calculate metrics for initial ensemble voting (before reconciliation)."""
    metrics = {
        'Method': 'Initial_Voting',
        'Task_Type': task_type
    }
    
    # Get initial predictions from all models
    preds_df = pd.DataFrame()
    for i, model in enumerate(models):
        if hasattr(model, 'predict_proba'):
            preds = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        else:
            preds = model.predict(X_test)
        preds_df[f'model_{i}'] = preds
    
    # Calculate ensemble prediction (simple average)
    y_pred = preds_df.mean(axis=1)
    
    # Basic performance metrics
    if task_type == 'classification':
        y_pred_class = y_pred.round().astype(int)
        metrics['Accuracy'] = accuracy_score(y_test, y_pred_class)
        metrics['Brier_Score'] = brier_score_loss(y_test, y_pred)
        metrics['Local_Calibration_Error'] = (y_pred - y_test).abs().mean()
    else:
        metrics['MSE'] = mean_squared_error(y_test, y_pred)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['MAE'] = np.mean(np.abs(y_pred - y_test))
        metrics['R2_Score'] = r2_score(y_test, y_pred)
        metrics['Local_Calibration_Error'] = np.mean(np.abs(y_pred - y_test))
    
    # Initial disagreement metrics
    metrics['Variance'] = preds_df.var(axis=1).mean()
    metrics['Ambiguity'] = (preds_df.max(axis=1) - preds_df.min(axis=1)).mean()
    
    # Maximum disagreement between any pair of models
    num_models = preds_df.shape[1]
    if num_models < 2:
        metrics['Max_Disagreement'] = 0.0
    else:
        max_disagreement = 0
        for i in range(num_models):
            for j in range(i + 1, num_models):
                disagreement = np.mean(np.abs(preds_df.iloc[:, i] - preds_df.iloc[:, j]))
                if disagreement > max_disagreement:
                    max_disagreement = disagreement
        metrics['Max_Disagreement'] = max_disagreement
    
    # Mean disagreement (fraction of points where models differ significantly)
    epsilon = 0.05
    ensemble_mean = preds_df.mean(axis=1)
    disagreements = ((preds_df > (ensemble_mean.values[:, None] + epsilon)).sum(axis=1) +
                     (preds_df < (ensemble_mean.values[:, None] - epsilon)).sum(axis=1))
    metrics['Disagreement'] = disagreements.mean() / preds_df.shape[1]
    
    return metrics

def calculate_comprehensive_metrics(model, X_test, y_test, method_name, task_type):
    """Calculate comprehensive metrics including final disagreement metrics."""
    metrics = {
        'Method': method_name,
        'Task_Type': task_type
    }
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Basic performance metrics
    if task_type == 'classification':
        y_pred_class = y_pred.round().astype(int)
        metrics['Accuracy'] = accuracy_score(y_test, y_pred_class)
        metrics['Brier_Score'] = brier_score_loss(y_test, y_pred)
        metrics['Local_Calibration_Error'] = (y_pred - y_test).abs().mean()
    else:
        metrics['MSE'] = mean_squared_error(y_test, y_pred)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['MAE'] = np.mean(np.abs(y_pred - y_test))
        metrics['R2_Score'] = r2_score(y_test, y_pred)
        metrics['Local_Calibration_Error'] = np.mean(np.abs(y_pred - y_test))
    
    # Final disagreement metrics (if available)
    if hasattr(model, 'final_disagreement_') and model.final_disagreement_ is not None:
        metrics['Variance'] = model.final_disagreement_['final_variance']
        metrics['Ambiguity'] = model.final_disagreement_['final_ambiguity']
        metrics['Max_Disagreement'] = model.final_disagreement_['final_max_disagreement']
        metrics['Disagreement'] = model.final_disagreement_['final_disagreement']
    
    return metrics

def run_adult_experiment():
    """Run experiments on Adult dataset for 5 seeds."""
    print("Adult Dataset Experiment Runner")
    print("=" * 60)
    print(f"Starting experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load dataset
    X, y, is_classification = load_adult_dataset()
    task_type = 'classification' if is_classification else 'regression'
    
    # Initialize results storage
    all_results = []
    
    # Run experiments for 5 seeds
    seeds = 5
    for seed in range(seeds):
        print(f"\n--- Seed {seed + 1}/{seeds} ---")
        
        # Set random seed
        np.random.seed(seed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, 
            stratify=y if is_classification else None
        )
        
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
        # Run Initial Voting (baseline ensemble)
        try:
            print(f"    Running Initial_Voting...")
            # Create a temporary Rocile model to get the base models
            temp_model = Rocile(random_state=seed)
            temp_model.fit(X_train, y_train)
            
            initial_voting_metrics = calculate_initial_voting_metrics(
                temp_model.models_, X_test, y_test, task_type
            )
            initial_voting_metrics['Dataset'] = 'Adult_Data'
            initial_voting_metrics['Seed'] = seed
            
            all_results.append(initial_voting_metrics)
            print(f"      ✓ Initial_Voting completed successfully")
            
        except Exception as e:
            print(f"      ✗ Initial_Voting failed: {e}")
        
        # Run Rocile
        try:
            print(f"    Running Rocile...")
            rocile_model = Rocile(
                batch_size=2,
                momentum=0.9,
                learning_rate=0.05,
                max_iter=1000,
                random_state=seed
            )
            
            rocile_model.fit(X_train, y_train)
            rocile_metrics = calculate_comprehensive_metrics(
                rocile_model, X_test, y_test, 'Rocile', task_type
            )
            rocile_metrics['Dataset'] = 'Adult_Data'
            rocile_metrics['Seed'] = seed
            
            all_results.append(rocile_metrics)
            print(f"      ✓ Rocile completed successfully")
            
        except Exception as e:
            print(f"      ✗ Rocile failed: {e}")
        
        # Run AdaRocile with BiasCorrected
        try:
            print(f"    Running AdaRocile_BiasCorrected...")
            adarocile_model = AdaRocile(
                batch_size=2,
                momentum=0.9,
                learning_rate=0.05,
                max_iter=1000,
                bias_threshold=0.6,
                base_k=30,
                patch_strategy='BiasCorrected',
                random_state=seed
            )
            
            adarocile_model.fit(X_train, y_train)
            adarocile_metrics = calculate_comprehensive_metrics(
                adarocile_model, X_test, y_test, 'AdaRocile_BiasCorrected', task_type
            )
            adarocile_metrics['Dataset'] = 'Adult_Data'
            adarocile_metrics['Seed'] = seed
            
            all_results.append(adarocile_metrics)
            print(f"      ✓ AdaRocile_BiasCorrected completed successfully")
            
        except Exception as e:
            print(f"      ✗ AdaRocile_BiasCorrected failed: {e}")
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save detailed results
        results_file = 'Adult_Data_results.csv'
        results_df.to_csv(results_file, index=False, float_format='%.6f')
        
        # Save summary statistics
        summary_file = 'Adult_Data_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("Adult Dataset - Experiment Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Task Type: {task_type}\n")
            f.write(f"Number of Seeds: {seeds}\n")
            f.write(f"Total Results: {len(results_df)}\n\n")
            
            # Performance summary
            f.write("Performance Summary:\n")
            f.write("-" * 30 + "\n")
            for method in results_df['Method'].unique():
                method_data = results_df[results_df['Method'] == method]
                if task_type == 'classification':
                    mean_acc = method_data['Accuracy'].mean()
                    std_acc = method_data['Accuracy'].std()
                    f.write(f"{method}: Accuracy = {mean_acc:.4f} ± {std_acc:.4f}\n")
                else:
                    mean_mse = method_data['MSE'].mean()
                    std_mse = method_data['MSE'].std()
                    f.write(f"{method}: MSE = {mean_mse:.4f} ± {std_mse:.4f}\n")
            
            # Disagreement summary
            f.write("\nDisagreement Summary:\n")
            f.write("-" * 30 + "\n")
            for method in results_df['Method'].unique():
                method_data = results_df[results_df['Method'] == method]
                if 'Variance' in method_data.columns:
                    mean_var = method_data['Variance'].mean()
                    mean_disag = method_data['Disagreement'].mean()
                    f.write(f"{method}: Variance={mean_var:.4f}, Disagreement={mean_disag:.4f}\n")
        
        print(f"\nResults saved:")
        print(f"  - {results_file} (detailed results)")
        print(f"  - {summary_file} (summary statistics)")
        
        # Print summary to console
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        for method in results_df['Method'].unique():
            method_data = results_df[results_df['Method'] == method]
            if task_type == 'classification':
                mean_acc = method_data['Accuracy'].mean()
                std_acc = method_data['Accuracy'].std()
                print(f"{method}: {mean_acc:.4f} ± {std_acc:.4f}")
            else:
                mean_mse = method_data['MSE'].mean()
                std_mse = method_data['MSE'].std()
                print(f"{method}: {mean_mse:.4f} ± {std_mse:.4f}")
        
        # Print disagreement metrics
        print(f"\n{'='*60}")
        print("DISAGREEMENT METRICS")
        print(f"{'='*60}")
        for method in results_df['Method'].unique():
            method_data = results_df[results_df['Method'] == method]
            if 'Variance' in method_data.columns:
                mean_var = method_data['Variance'].mean()
                mean_disag = method_data['Disagreement'].mean()
                print(f"{method}: Variance={mean_var:.4f}, Disagreement={mean_disag:.4f}")
        
        return results_df
    
    return None

if __name__ == "__main__":
    run_adult_experiment() 