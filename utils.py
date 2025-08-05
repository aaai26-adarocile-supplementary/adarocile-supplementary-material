import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report


def normalize_data(X):
    """
    Normalize data using z-score standardization.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data to normalize.
        
    Returns
    -------
    X_normalized : array-like of shape (n_samples, n_features)
        Normalized data.
    """
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def preprocess_data(X, y=None, categorical_columns=None):
    """
    Preprocess data for machine learning models.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features.
    y : array-like of shape (n_samples,), optional
        Target values.
    categorical_columns : list, optional
        List of categorical column indices or names.
        
    Returns
    -------
    X_processed : array-like of shape (n_samples, n_features)
        Processed features.
    y_processed : array-like of shape (n_samples,), optional
        Processed target values.
    """
    X = np.asarray(X)
    
    # Handle categorical variables
    if categorical_columns is not None:
        for col in categorical_columns:
            if col < X.shape[1]:
                le = LabelEncoder()
                X[:, col] = le.fit_transform(X[:, col].astype(str))
    
    # Fill missing values
    X = np.nan_to_num(X, nan=np.nanmedian(X))
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_processed = scaler.fit_transform(X)
    
    if y is not None:
        y = np.asarray(y)
        return X_processed, y
    
    return X_processed


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features.
    y : array-like of shape (n_samples,)
        Target values.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Training and testing splits.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(y_true, y_pred, task_type='classification'):
    """
    Evaluate model performance.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.
    task_type : str, default='classification'
        Type of task ('classification' or 'regression').
        
    Returns
    -------
    metrics : dict
        Dictionary containing evaluation metrics.
    """
    if task_type == 'classification':
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        return {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score']
        }
    else:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }


def calculate_ensemble_metrics(predictions_df):
    """
    Calculate ensemble diversity metrics.
    
    Parameters
    ----------
    predictions_df : pandas.DataFrame
        DataFrame with predictions from multiple models.
        
    Returns
    -------
    metrics : dict
        Dictionary containing ensemble metrics.
    """
    # Variance across models
    variance = predictions_df.var(axis=1).mean()
    
    # Ambiguity (max - min predictions)
    ambiguity = (predictions_df.max(axis=1) - predictions_df.min(axis=1)).mean()
    
    # Disagreement (fraction of points where models differ significantly)
    from itertools import combinations
    disagreements = []
    for col1, col2 in combinations(predictions_df.columns, 2):
        diff = np.abs(predictions_df[col1] - predictions_df[col2])
        disagreements.append(np.mean(diff > 0.05))  # 5% threshold
    
    disagreement = np.mean(disagreements) if disagreements else 0.0
    
    return {
        'variance': variance,
        'ambiguity': ambiguity,
        'disagreement': disagreement
    } 