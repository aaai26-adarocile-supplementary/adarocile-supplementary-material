#!/usr/bin/env python3
"""
Comprehensive Example Runner for Rocile and AdaRocile Package

This script runs experiments on all available datasets with multiple seeds
and saves detailed results including initial disagreement metrics and improvement analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
import traceback
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

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mypackage.model import Rocile, AdaRocile
from utils import preprocess_data, split_data, evaluate_model, calculate_ensemble_metrics

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None

# Define datasets with their target columns
DATASETS = {
    'Adult_Data': {
        'file': 'datasets/adult_cleaned.csv',
        'target': 'income',
        'type': 'classification'
    },
    'Folk_Income_Data': {
        'file': 'datasets/folks_income_FL_cleaned.csv',
        'target': 'income',
        'type': 'classification'
    },
    'German_Data': {
        'file': 'datasets/german_cleaned.csv',
        'target': 'Creditability',
        'type': 'classification'
    },
    'Compas_Data': {
        'file': 'datasets/compas_cleaned.csv',
        'target': 'two_year_recid',
        'type': 'classification'
    },
    'Folk_Travel_Data': {
        'file': 'datasets/folks_travel_FL_cleaned.csv',
        'target': 'travel_time',
        'type': 'classification'
    },
    'Folk_Mobility_Data': {
        'file': 'datasets/folks_mobility_FL_cleaned.csv',
        'target': 'mobility',
        'type': 'classification'
    },
    'Communities_Data': {
        'file': 'datasets/communities_cleaned.csv',
        'target': '127',
        'type': 'regression'
    }
}

# Define patch strategies
PATCH_STRATEGIES = [
    'BiasCorrected',
    'DistanceWeighted',
    'ModelSpecific',
    'CertaintyWeighted',
    'EnsembleLevel'
]

# Define methods to run
METHODS = {
    'Rocile': Rocile,
    'AdaRocile_BiasCorrected': lambda **kwargs: AdaRocile(patch_strategy='BiasCorrected', **kwargs),
    'AdaRocile_DistanceWeighted': lambda **kwargs: AdaRocile(patch_strategy='DistanceWeighted', **kwargs),
    'AdaRocile_ModelSpecific': lambda **kwargs: AdaRocile(patch_strategy='ModelSpecific', **kwargs),
    'AdaRocile_CertaintyWeighted': lambda **kwargs: AdaRocile(patch_strategy='CertaintyWeighted', **kwargs),
    'AdaRocile_EnsembleLevel': lambda **kwargs: AdaRocile(patch_strategy='EnsembleLevel', **kwargs),
}

class DatasetProcessor:
    """Process and prepare datasets for experiments."""
    
    def __init__(self, file_path, target_column):
        self.file_path = file_path
        self.target_column = target_column
        print(f"Loading dataset from {self.file_path}...")
    
    def load_and_preprocess(self):
        """Load and preprocess the dataset."""
        try:
            df = pd.read_csv(self.file_path)
            print(f"  Loaded {len(df)} samples with {len(df.columns)} features")
            
            # Check if target column exists
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in dataset")
            
            y = df[self.target_column].copy()
            X = df.drop(columns=[self.target_column]).copy()
            
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
            
        except Exception as e:
            print(f"  Error loading dataset: {e}")
            raise

def calculate_comprehensive_metrics(model, X_test, y_test, method_name, task_type):
    """Calculate comprehensive metrics including initial disagreement metrics."""
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
    
    # Initial disagreement metrics (if available)
    if hasattr(model, 'initial_disagreement_') and model.initial_disagreement_ is not None:
        metrics['Initial_Variance'] = model.initial_disagreement_['initial_variance']
        metrics['Initial_Ambiguity'] = model.initial_disagreement_['initial_ambiguity']
        metrics['Initial_Max_Disagreement'] = model.initial_disagreement_['initial_max_disagreement']
        metrics['Initial_Disagreement'] = model.initial_disagreement_['initial_disagreement']
    
    return metrics

def run_dataset_experiment(dataset_name, dataset_info, seeds=5):
    """Run experiments on a single dataset with multiple seeds."""
    print(f"\n{'='*80}")
    print(f"Running experiments on {dataset_name}")
    print(f"{'='*80}")
    
    # Create dataset-specific directory
    dataset_dir = os.path.join('Examples', dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Initialize results storage
    all_results = []
    
    # Process dataset
    processor = DatasetProcessor(dataset_info['file'], dataset_info['target'])
    X, y, is_classification = processor.load_and_preprocess()
    
    task_type = 'classification' if is_classification else 'regression'
    
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
        
        # Run all methods
        for method_name, method_class in METHODS.items():
            try:
                print(f"    Running {method_name}...")
                
                # Initialize model
                if method_name == 'Rocile':
                    model = method_class(
                        batch_size=2,
                        momentum=0.9,
                        learning_rate=0.05,
                        max_iter=1000,
                        random_state=seed
                    )
                else:
                    model = method_class(
                        batch_size=2,
                        momentum=0.9,
                        learning_rate=0.05,
                        max_iter=1000,
                        bias_threshold=0.6,
                        base_k=30,
                        random_state=seed
                    )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Calculate metrics
                metrics = calculate_comprehensive_metrics(
                    model, X_test, y_test, method_name, task_type
                )
                metrics['Dataset'] = dataset_name
                metrics['Seed'] = seed
                
                all_results.append(metrics)
                print(f"      ✓ Completed successfully")
                
            except Exception as e:
                print(f"      ✗ Failed: {e}")
                traceback.print_exc()
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save detailed results
        results_file = os.path.join(dataset_dir, f'{dataset_name}_results.csv')
        results_df.to_csv(results_file, index=False, float_format='%.6f')
        
        # Save summary statistics
        summary_file = os.path.join(dataset_dir, f'{dataset_name}_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Task Type: {task_type}\n")
            f.write(f"Number of Seeds: {seeds}\n")
            f.write(f"Total Results: {len(results_df)}\n\n")
            
            # Performance summary
            if task_type == 'classification':
                f.write("Performance Summary (Accuracy):\n")
                f.write("-" * 50 + "\n")
                for method in results_df['Method'].unique():
                    method_data = results_df[results_df['Method'] == method]
                    mean_acc = method_data['Accuracy'].mean()
                    std_acc = method_data['Accuracy'].std()
                    f.write(f"{method}: {mean_acc:.4f} ± {std_acc:.4f}\n")
            else:
                f.write("Performance Summary (MSE):\n")
                f.write("-" * 50 + "\n")
                for method in results_df['Method'].unique():
                    method_data = results_df[results_df['Method'] == method]
                    mean_mse = method_data['MSE'].mean()
                    std_mse = method_data['MSE'].std()
                    f.write(f"{method}: {mean_mse:.4f} ± {std_mse:.4f}\n")
            
            # Initial disagreement summary
            f.write("\nInitial Disagreement Summary:\n")
            f.write("-" * 50 + "\n")
            for method in results_df['Method'].unique():
                method_data = results_df[results_df['Method'] == method]
                if 'Initial_Variance' in method_data.columns:
                    mean_var = method_data['Initial_Variance'].mean()
                    mean_disag = method_data['Initial_Disagreement'].mean()
                    f.write(f"{method}: Variance={mean_var:.4f}, Disagreement={mean_disag:.4f}\n")
        
        print(f"\nResults saved to {dataset_dir}/")
        print(f"  - {dataset_name}_results.csv (detailed results)")
        print(f"  - {dataset_name}_summary.txt (summary statistics)")
        
        return results_df
    
    return None

def main():
    """Main function to run all experiments."""
    print("Comprehensive AdaRocile Package Example Runner")
    print("=" * 60)
    print(f"Starting experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create main results directory
    os.makedirs('Examples', exist_ok=True)
    
    # Initialize overall results
    all_dataset_results = []
    
    # Run experiments on each dataset
    for dataset_name, dataset_info in DATASETS.items():
        try:
            results_df = run_dataset_experiment(dataset_name, dataset_info, seeds=5)
            if results_df is not None:
                all_dataset_results.append(results_df)
        except Exception as e:
            print(f"Failed to process dataset {dataset_name}: {e}")
            traceback.print_exc()
    
    # Create overall summary
    if all_dataset_results:
        combined_results = pd.concat(all_dataset_results, ignore_index=True)
        
        # Save combined results
        combined_file = os.path.join('Examples', 'all_datasets_results.csv')
        combined_results.to_csv(combined_file, index=False, float_format='%.6f')
        
        # Create overall summary
        summary_file = os.path.join('Examples', 'overall_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("AdaRocile Package - Overall Experiment Summary\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Results: {len(combined_results)}\n")
            f.write(f"Datasets Processed: {len(combined_results['Dataset'].unique())}\n")
            f.write(f"Methods Tested: {len(combined_results['Method'].unique())}\n\n")
            
            # Performance comparison
            f.write("Performance Comparison Across All Datasets:\n")
            f.write("-" * 60 + "\n")
            
            for task_type in ['classification', 'regression']:
                task_data = combined_results[combined_results['Task_Type'] == task_type]
                if len(task_data) > 0:
                    f.write(f"\n{task_type.capitalize()} Tasks:\n")
                    if task_type == 'classification':
                        for method in task_data['Method'].unique():
                            method_data = task_data[task_data['Method'] == method]
                            mean_acc = method_data['Accuracy'].mean()
                            std_acc = method_data['Accuracy'].std()
                            f.write(f"  {method}: {mean_acc:.4f} ± {std_acc:.4f}\n")
                    else:
                        for method in task_data['Method'].unique():
                            method_data = task_data[task_data['Method'] == method]
                            mean_mse = method_data['MSE'].mean()
                            std_mse = method_data['MSE'].std()
                            f.write(f"  {method}: {mean_mse:.4f} ± {std_mse:.4f}\n")
            
            # Improvement analysis
            f.write("\nImprovement Analysis:\n")
            f.write("-" * 60 + "\n")
            
            # Compare AdaRocile variants to Rocile
            rocile_data = combined_results[combined_results['Method'] == 'Rocile']
            for method in combined_results['Method'].unique():
                if method.startswith('AdaRocile_'):
                    method_data = combined_results[combined_results['Method'] == method]
                    
                    # Merge with Rocile data for comparison
                    merged_data = pd.merge(
                        rocile_data[['Dataset', 'Seed', 'Accuracy' if 'Accuracy' in rocile_data.columns else 'MSE']], 
                        method_data[['Dataset', 'Seed', 'Accuracy' if 'Accuracy' in method_data.columns else 'MSE']], 
                        on=['Dataset', 'Seed'], 
                        suffixes=('_Rocile', f'_{method}')
                    )
                    
                    if len(merged_data) > 0:
                        if 'Accuracy' in merged_data.columns:
                            improvement = (merged_data[f'Accuracy_{method}'] - merged_data['Accuracy_Rocile']).mean()
                            f.write(f"  {method} vs Rocile (Accuracy): {improvement:+.4f}\n")
                        else:
                            improvement = (merged_data['MSE_Rocile'] - merged_data[f'MSE_{method}']).mean()
                            f.write(f"  {method} vs Rocile (MSE): {improvement:+.4f}\n")
        
        print(f"\n{'='*60}")
        print("EXPERIMENTS COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Combined results saved to: Examples/all_datasets_results.csv")
        print(f"Overall summary saved to: Examples/overall_summary.txt")
        print(f"Individual dataset results saved to: Examples/{dataset_name}/")
        
    else:
        print("No results were generated. Please check the error messages above.")

if __name__ == "__main__":
    main() 