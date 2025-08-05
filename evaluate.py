#!/usr/bin/env python3
"""
Evaluation script for Rocile and AdaRocile models.

This script demonstrates how to evaluate the performance of trained
Rocile and AdaRocile models on test datasets.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from mypackage.model import Rocile, AdaRocile
from utils import preprocess_data, split_data, evaluate_model, calculate_ensemble_metrics


def evaluate_model_performance(model, X_test, y_test, model_name, task_type='classification'):
    """
    Evaluate model performance with detailed metrics.
    
    Parameters
    ----------
    model : fitted model
        Trained Rocile or AdaRocile model.
    X_test : array-like
        Test features.
    y_test : array-like
        Test targets.
    model_name : str
        Name of the model for reporting.
    task_type : str, default='classification'
        Type of task ('classification' or 'regression').
        
    Returns
    -------
    results : dict
        Dictionary containing evaluation results.
    """
    print(f"\n--- Evaluating {model_name} ---")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    if task_type == 'classification':
        # Round predictions for classification
        y_pred_class = y_pred.round().astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_class)
        
        # Detailed classification report
        report = classification_report(y_test, y_pred_class, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_class)
        
        results = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'confusion_matrix': cm,
            'predictions': y_pred,
            'predictions_class': y_pred_class
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        
    else:
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred
        }
        
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    
    return results


def plot_results(clf_results, reg_results):
    """
    Create visualization plots for the evaluation results.
    
    Parameters
    ----------
    clf_results : dict
        Classification results.
    reg_results : dict
        Regression results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Rocile and AdaRocile Evaluation Results', fontsize=16)
    
    # Classification accuracy comparison
    if clf_results:
        ax1 = axes[0, 0]
        models = ['Rocile', 'AdaRocile']
        accuracies = [clf_results['rocile']['accuracy'], clf_results['adarocile']['accuracy']]
        
        bars = ax1.bar(models, accuracies, color=['skyblue', 'lightcoral'])
        ax1.set_title('Classification Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
    
    # Regression MSE comparison
    if reg_results:
        ax2 = axes[0, 1]
        models = ['Rocile', 'AdaRocile']
        mses = [reg_results['rocile']['mse'], reg_results['adarocile']['mse']]
        
        bars = ax2.bar(models, mses, color=['skyblue', 'lightcoral'])
        ax2.set_title('Regression MSE Comparison')
        ax2.set_ylabel('MSE')
        
        # Add value labels on bars
        for bar, mse in zip(bars, mses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{mse:.3f}', ha='center', va='bottom')
    
    # Classification confusion matrix (AdaRocile)
    if clf_results and 'confusion_matrix' in clf_results['adarocile']:
        ax3 = axes[1, 0]
        cm = clf_results['adarocile']['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title('AdaRocile Confusion Matrix')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
    
    # Regression predictions vs actual (AdaRocile)
    if reg_results:
        ax4 = axes[1, 1]
        y_test_reg = reg_results['y_test']
        y_pred_adarocile = reg_results['adarocile']['predictions']
        
        ax4.scatter(y_test_reg, y_pred_adarocile, alpha=0.6)
        ax4.plot([y_test_reg.min(), y_test_reg.max()], 
                [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
        ax4.set_xlabel('Actual Values')
        ax4.set_ylabel('Predicted Values')
        ax4.set_title('AdaRocile: Predicted vs Actual')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to run the evaluation script."""
    print("Rocile and AdaRocile Evaluation Script")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random_state = 42
    np.random.seed(random_state)
    
    results = {}
    
    # Evaluate on classification data
    print("\n1. Classification Task Evaluation")
    print("-" * 40)
    
    # Generate classification data
    X_clf, y_clf = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        random_state=random_state
    )
    
    # Preprocess and split
    X_clf_processed, y_clf_processed = preprocess_data(X_clf, y_clf)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = split_data(
        X_clf_processed, y_clf_processed, test_size=0.2, random_state=random_state
    )
    
    # Train and evaluate Rocile
    rocile_clf = Rocile(random_state=random_state)
    rocile_clf.fit(X_train_clf, y_train_clf)
    rocile_clf_results = evaluate_model_performance(
        rocile_clf, X_test_clf, y_test_clf, "Rocile", "classification"
    )
    
    # Train and evaluate AdaRocile
    adarocile_clf = AdaRocile(random_state=random_state)
    adarocile_clf.fit(X_train_clf, y_train_clf)
    adarocile_clf_results = evaluate_model_performance(
        adarocile_clf, X_test_clf, y_test_clf, "AdaRocile", "classification"
    )
    
    results['classification'] = {
        'rocile': rocile_clf_results,
        'adarocile': adarocile_clf_results,
        'y_test': y_test_clf
    }
    
    # Evaluate on regression data
    print("\n2. Regression Task Evaluation")
    print("-" * 40)
    
    # Generate regression data
    X_reg, y_reg = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        random_state=random_state
    )
    
    # Preprocess and split
    X_reg_processed, y_reg_processed = preprocess_data(X_reg, y_reg)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(
        X_reg_processed, y_reg_processed, test_size=0.2, random_state=random_state
    )
    
    # Train and evaluate Rocile
    rocile_reg = Rocile(random_state=random_state)
    rocile_reg.fit(X_train_reg, y_train_reg)
    rocile_reg_results = evaluate_model_performance(
        rocile_reg, X_test_reg, y_test_reg, "Rocile", "regression"
    )
    
    # Train and evaluate AdaRocile
    adarocile_reg = AdaRocile(random_state=random_state)
    adarocile_reg.fit(X_train_reg, y_train_reg)
    adarocile_reg_results = evaluate_model_performance(
        adarocile_reg, X_test_reg, y_test_reg, "AdaRocile", "regression"
    )
    
    results['regression'] = {
        'rocile': rocile_reg_results,
        'adarocile': adarocile_reg_results,
        'y_test': y_test_reg
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    print("\nClassification Results:")
    print(f"  Rocile Accuracy: {results['classification']['rocile']['accuracy']:.4f}")
    print(f"  AdaRocile Accuracy: {results['classification']['adarocile']['accuracy']:.4f}")
    
    print("\nRegression Results:")
    print(f"  Rocile MSE: {results['regression']['rocile']['mse']:.4f}")
    print(f"  AdaRocile MSE: {results['regression']['adarocile']['mse']:.4f}")
    
    # Create visualizations
    try:
        plot_results(results['classification'], results['regression'])
        print("\nVisualization saved as 'evaluation_results.png'")
    except Exception as e:
        print(f"\nCould not create visualization: {e}")
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main() 