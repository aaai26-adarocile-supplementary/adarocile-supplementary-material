from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from itertools import combinations


class Rocile(BaseEstimator, ClassifierMixin):
    """
    Rocile: Momentum-Batch Sequential Reconciliation
    
    A scikit-learn compatible implementation of the Rocile algorithm for ensemble reconciliation.
    Uses momentum-based batch updates to efficiently reconcile model predictions.
    
    Parameters
    ----------
    batch_size : int, default=2
        Number of model pairs to update in each batch.
    momentum : float, default=0.9
        Momentum coefficient for gradient updates.
    learning_rate : float, default=0.05
        Learning rate for the reconciliation updates.
    max_iter : int, default=1000
        Maximum number of iterations for convergence.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, batch_size=2, momentum=0.9, learning_rate=0.05, max_iter=1000, random_state=None):
        self.batch_size = batch_size
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.models_ = None
        self.is_fitted_ = False
        self.initial_disagreement_ = None
        self.final_disagreement_ = None
    """
    Rocile: Momentum-Batch Sequential Reconciliation
    
    A scikit-learn compatible implementation of the Rocile algorithm for ensemble reconciliation.
    Uses momentum-based batch updates to efficiently reconcile model predictions.
    
    Parameters
    ----------
    batch_size : int, default=2
        Number of model pairs to update in each batch.
    momentum : float, default=0.9
        Momentum coefficient for gradient updates.
    learning_rate : float, default=0.05
        Learning rate for the reconciliation updates.
    max_iter : int, default=1000
        Maximum number of iterations for convergence.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, batch_size=2, momentum=0.9, learning_rate=0.05, max_iter=1000, random_state=None):
        self.batch_size = batch_size
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.models_ = None
        self.is_fitted_ = False
        self.initial_disagreement_ = None
        
    def _create_rashomon_set(self, X, y):
        """Create a Rashomon set of diverse models."""
        # Define model prototypes based on task type
        is_classification = len(np.unique(y)) < 25
        
        if is_classification:
            model_prototypes = [
                ('RandomForest', RandomForestClassifier(n_estimators=50, random_state=self.random_state)),
                ('GradientBoosting', GradientBoostingClassifier(n_estimators=50, random_state=self.random_state)),
                ('KNN', KNeighborsClassifier()),
                ('DecisionTree', DecisionTreeClassifier(max_depth=8, random_state=self.random_state)),
                ('LogisticRegression', LogisticRegression(random_state=self.random_state, max_iter=500)),
                ('AdaBoost', AdaBoostClassifier(random_state=self.random_state)),
                ('GaussianNB', GaussianNB()),
            ]
        else:
            model_prototypes = [
                ('RandomForest', RandomForestRegressor(n_estimators=50, random_state=self.random_state)),
                ('GradientBoosting', GradientBoostingRegressor(n_estimators=50, random_state=self.random_state)),
                ('KNN', KNeighborsRegressor()),
                ('DecisionTree', DecisionTreeRegressor(max_depth=8, random_state=self.random_state)),
                ('LinearRegression', LinearRegression()),
                ('Ridge', Ridge(random_state=self.random_state)),
                ('AdaBoost', AdaBoostRegressor(random_state=self.random_state)),
            ]
        
        # Evaluate models and select top performers
        performance_scores = {}
        trained_models = {}
        
        for name, model in model_prototypes:
            pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
            scoring = 'accuracy' if is_classification else 'neg_mean_squared_error'
            scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring)
            performance_scores[name] = np.mean(scores)
            trained_models[name] = model.fit(X, y)
        
        # Select models within performance threshold
        top_score = max(performance_scores.values())
        if is_classification:
            threshold = max(top_score - 0.05, top_score * 0.95)
        else:
            # For regression, use a more lenient threshold since MSE is lower-is-better
            threshold = min(top_score + 0.05, top_score * 1.05)
        
        if is_classification:
            selected_models = [trained_models[name] for name, score in performance_scores.items() 
                              if score >= threshold]
        else:
            selected_models = [trained_models[name] for name, score in performance_scores.items() 
                              if score <= threshold]
        
        return selected_models, is_classification
    
    def _get_predictions(self, models, X):
        """Get predictions from all models."""
        preds = {}
        for i, model in enumerate(models):
            if hasattr(model, 'predict_proba'):
                preds[f'model_{i}'] = model.predict_proba(X)[:, 1]
            else:
                preds[f'model_{i}'] = model.predict(X)
        return pd.DataFrame(preds, index=X.index if hasattr(X, 'index') else range(len(X)))
    
    def _calculate_max_disagreement(self, preds):
        """Calculate maximum disagreement between any pair of models."""
        num_models = preds.shape[1]
        if num_models < 2:
            return 0.0
        
        max_disagreement = 0
        for i in range(num_models):
            for j in range(i + 1, num_models):
                disagreement = np.mean(np.abs(preds.iloc[:, i] - preds.iloc[:, j]))
                if disagreement > max_disagreement:
                    max_disagreement = disagreement
        return max_disagreement
    
    def _calculate_initial_metrics(self, preds_df):
        """Calculate initial disagreement metrics before reconciliation."""
        metrics = {}
        
        # Variance across models
        metrics['initial_variance'] = preds_df.var(axis=1).mean()
        
        # Ambiguity (max - min predictions)
        metrics['initial_ambiguity'] = (preds_df.max(axis=1) - preds_df.min(axis=1)).mean()
        
        # Maximum disagreement between any pair of models
        metrics['initial_max_disagreement'] = self._calculate_max_disagreement(preds_df)
        
        # Mean disagreement (fraction of points where models differ significantly)
        epsilon = 0.05
        ensemble_mean = preds_df.mean(axis=1)
        disagreements = ((preds_df > (ensemble_mean.values[:, None] + epsilon)).sum(axis=1) +
                         (preds_df < (ensemble_mean.values[:, None] - epsilon)).sum(axis=1))
        metrics['initial_disagreement'] = disagreements.mean() / preds_df.shape[1]
        
        return metrics
    
    def _calculate_final_metrics(self, preds_df):
        """Calculate final disagreement metrics after reconciliation."""
        metrics = {}
        
        # Variance across models
        metrics['final_variance'] = preds_df.var(axis=1).mean()
        
        # Ambiguity (max - min predictions)
        metrics['final_ambiguity'] = (preds_df.max(axis=1) - preds_df.min(axis=1)).mean()
        
        # Maximum disagreement between any pair of models
        metrics['final_max_disagreement'] = self._calculate_max_disagreement(preds_df)
        
        # Mean disagreement (fraction of points where models differ significantly)
        epsilon = 0.05
        ensemble_mean = preds_df.mean(axis=1)
        disagreements = ((preds_df > (ensemble_mean.values[:, None] + epsilon)).sum(axis=1) +
                         (preds_df < (ensemble_mean.values[:, None] - epsilon)).sum(axis=1))
        metrics['final_disagreement'] = disagreements.mean() / preds_df.shape[1]
        
        return metrics
    
    def fit(self, X, y):
        """
        Fit the Rocile model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Create Rashomon set
        self.models_, self.is_classification_ = self._create_rashomon_set(X, y)
        
        if len(self.models_) < 2:
            raise ValueError("Need at least 2 models for reconciliation")
        
        # Calculate initial disagreement metrics
        initial_preds = self._get_predictions(self.models_, X)
        self.initial_disagreement_ = self._calculate_initial_metrics(initial_preds)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Predict using the reconciled ensemble.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y : array-like of shape (n_samples,)
            Predicted values.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        
        # Get initial predictions
        preds_df = self._get_predictions(self.models_, X)
        preds = preds_df.to_numpy().T
        num_models = preds.shape[0]
        
        if num_models < 2:
            return preds_df.mean(axis=1).values
        
        # Store initial predictions for final disagreement calculation
        initial_preds = preds.copy()
        
        # Rocile reconciliation with momentum
        momentum = np.zeros_like(preds)
        
        for t in range(self.max_iter):
            # Calculate disagreements
            diffs = preds[:, np.newaxis, :] - preds
            mean_abs_diffs = np.mean(np.abs(diffs), axis=2)
            np.fill_diagonal(mean_abs_diffs, -1)
            
            if np.max(mean_abs_diffs) < 1e-6:
                break
            
            # Select top disagreeing pairs
            pairs_indices = np.dstack(np.unravel_index(
                np.argsort(mean_abs_diffs.ravel()), mean_abs_diffs.shape))[0]
            top_pairs = [p for p in pairs_indices if p[0] < p[1]][-self.batch_size:]
            
            # Adaptive learning rate
            learning_rate = self.learning_rate * np.exp(-t / 100)
            
            # Update predictions with momentum
            for i, j in top_pairs:
                update = learning_rate * (preds[i] - preds[j])
                momentum[i] = self.momentum * momentum[i] - update
                momentum[j] = self.momentum * momentum[j] + update
                
                preds[i] = preds[i] + momentum[i]
                preds[j] = preds[j] + momentum[j]
                
                # Clip for classification
                if self.is_classification_:
                    preds[i] = np.clip(preds[i], 0, 1)
                    preds[j] = np.clip(preds[j], 0, 1)
        
        # Calculate final disagreement metrics
        final_preds_df = pd.DataFrame(preds.T, index=preds_df.index, columns=preds_df.columns)
        self.final_disagreement_ = self._calculate_final_metrics(final_preds_df)
        
        # Return ensemble mean
        return np.mean(preds, axis=0)


class AdaRocile(BaseEstimator, ClassifierMixin):
    """
    AdaRocile: Adaptive Local Patching with Rocile Reconciliation
    
    A scikit-learn compatible implementation of the AdaRocile algorithm that combines
    local bias correction with Rocile reconciliation for improved ensemble performance.
    
    Parameters
    ----------
    batch_size : int, default=2
        Number of model pairs to update in each batch for Rocile.
    momentum : float, default=0.9
        Momentum coefficient for gradient updates.
    learning_rate : float, default=0.05
        Learning rate for the reconciliation updates.
    max_iter : int, default=1000
        Maximum number of iterations for convergence.
    bias_threshold : float, default=0.6
        Threshold for detecting bias in local patches.
    base_k : int, default=30
        Base number of neighbors for local patching.
    patch_strategy : str, default='BiasCorrected'
        Local patching strategy to use. Options: 'BiasCorrected', 'DistanceWeighted', 
        'ModelSpecific', 'CertaintyWeighted', 'EnsembleLevel'.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, batch_size=2, momentum=0.9, learning_rate=0.05, max_iter=1000, 
                 bias_threshold=0.6, base_k=30, patch_strategy='BiasCorrected', random_state=None):
        self.batch_size = batch_size
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.bias_threshold = bias_threshold
        self.base_k = base_k
        self.patch_strategy = patch_strategy
        self.random_state = random_state
        self.models_ = None
        self.is_fitted_ = False
        self.X_train_ = None
        self.y_train_ = None
        self.nn_ = None
        self.initial_disagreement_ = None
        
    def _create_rashomon_set(self, X, y):
        """Create a Rashomon set of diverse models."""
        # Same as Rocile
        is_classification = len(np.unique(y)) < 25
        
        if is_classification:
            model_prototypes = [
                ('RandomForest', RandomForestClassifier(n_estimators=50, random_state=self.random_state)),
                ('GradientBoosting', GradientBoostingClassifier(n_estimators=50, random_state=self.random_state)),
                ('KNN', KNeighborsClassifier()),
                ('DecisionTree', DecisionTreeClassifier(max_depth=8, random_state=self.random_state)),
                ('LogisticRegression', LogisticRegression(random_state=self.random_state, max_iter=500)),
                ('AdaBoost', AdaBoostClassifier(random_state=self.random_state)),
                ('GaussianNB', GaussianNB()),
            ]
        else:
            model_prototypes = [
                ('RandomForest', RandomForestRegressor(n_estimators=50, random_state=self.random_state)),
                ('GradientBoosting', GradientBoostingRegressor(n_estimators=50, random_state=self.random_state)),
                ('KNN', KNeighborsRegressor()),
                ('DecisionTree', DecisionTreeRegressor(max_depth=8, random_state=self.random_state)),
                ('LinearRegression', LinearRegression()),
                ('Ridge', Ridge(random_state=self.random_state)),
                ('AdaBoost', AdaBoostRegressor(random_state=self.random_state)),
            ]
        
        performance_scores = {}
        trained_models = {}
        
        for name, model in model_prototypes:
            pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
            scoring = 'accuracy' if is_classification else 'neg_mean_squared_error'
            scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring)
            performance_scores[name] = np.mean(scores)
            trained_models[name] = model.fit(X, y)
        
        top_score = max(performance_scores.values())
        if is_classification:
            threshold = max(top_score - 0.05, top_score * 0.95)
        else:
            # For regression, use a more lenient threshold since MSE is lower-is-better
            threshold = min(top_score + 0.05, top_score * 1.05)
        
        if is_classification:
            selected_models = [trained_models[name] for name, score in performance_scores.items() 
                              if score >= threshold]
        else:
            selected_models = [trained_models[name] for name, score in performance_scores.items() 
                              if score <= threshold]
        
        return selected_models, is_classification
    
    def _get_predictions(self, models, X):
        """Get predictions from all models."""
        preds = {}
        for i, model in enumerate(models):
            if hasattr(model, 'predict_proba'):
                preds[f'model_{i}'] = model.predict_proba(X)[:, 1]
            else:
                preds[f'model_{i}'] = model.predict(X)
        return pd.DataFrame(preds, index=X.index if hasattr(X, 'index') else range(len(X)))
    
    def _get_adaptive_k(self, test_point, distances):
        """Adaptively determine k based on local density."""
        if len(distances) < 3:
            return len(distances)
        
        distances_safe = np.maximum(distances, 1e-10)
        dist_ratios = distances_safe[1:] / distances_safe[:-1]
        
        valid_ratios = dist_ratios[np.isfinite(dist_ratios)]
        if len(valid_ratios) == 0:
            return self.base_k
        
        mean_growth = np.mean(valid_ratios)
        
        if mean_growth > 1.5:  # Sparse area
            adaptive_k = min(int(self.base_k * 1.3), len(distances))
        elif mean_growth < 1.2:  # Dense area
            adaptive_k = max(int(self.base_k * 0.7), 5)
        else:  # Medium density
            adaptive_k = self.base_k
        
        return min(adaptive_k, len(distances))
    
    def _calculate_local_patches(self, neighbor_preds, neighbor_labels, test_point_features=None, neighbor_features=None):
        """Calculate local bias correction patches using the specified strategy."""
        final_patches = pd.Series(0.0, index=neighbor_preds.columns)
        
        # Handle both pandas DataFrame and numpy array inputs
        if hasattr(neighbor_preds, 'values'):
            preds_array = neighbor_preds.values
        else:
            preds_array = neighbor_preds
            
        if hasattr(neighbor_labels, 'values'):
            labels_array = neighbor_labels.values
        else:
            labels_array = neighbor_labels
        
        if self.patch_strategy == 'BiasCorrected':
            # BiasCorrected: Analyzes error directions for systematic correction
            errors = labels_array[:, np.newaxis] - preds_array
            bias_patches = np.zeros(errors.shape[1])
            
            for i in range(errors.shape[1]):
                if np.mean(errors[:, i] < 0) > self.bias_threshold:
                    bias_patches[i] = errors[:, i][errors[:, i] < 0].mean()
                elif np.mean(errors[:, i] > 0) > self.bias_threshold:
                    bias_patches[i] = errors[:, i][errors[:, i] > 0].mean()
            
            final_patches += bias_patches
            
        elif self.patch_strategy == 'DistanceWeighted':
            # DistanceWeighted: Weights by inverse distance
            if neighbor_features is not None and test_point_features is not None:
                # Handle both pandas and numpy inputs
                if hasattr(neighbor_features, 'values'):
                    neighbor_array = neighbor_features.values
                else:
                    neighbor_array = neighbor_features
                
                if hasattr(test_point_features, 'values'):
                    test_array = test_point_features.values
                else:
                    test_array = test_point_features
                
                distances = np.linalg.norm(neighbor_array - test_array, axis=1)
                weights = 1 / (distances + 1e-6)
                weights /= weights.sum()
                weighted_errors = (labels_array[:, np.newaxis] - preds_array) * weights[:, np.newaxis]
                final_patches += weighted_errors.sum(axis=0)
            else:
                # Fallback to simple averaging
                errors = labels_array[:, np.newaxis] - preds_array
                final_patches += errors.mean(axis=0)
                
        elif self.patch_strategy == 'ModelSpecific':
            # ModelSpecific: Computes model-specific averages
            errors = labels_array[:, np.newaxis] - preds_array
            final_patches += errors.mean(axis=0)
            
        elif self.patch_strategy == 'CertaintyWeighted':
            # CertaintyWeighted: Weights by inverse variance
            neighbor_variance = neighbor_preds.var(axis=1)
            weights = 1 / (neighbor_variance + 1e-6)
            weights /= weights.sum()
            certainty_errors = (labels_array[:, np.newaxis] - preds_array) * weights.values[:, np.newaxis]
            final_patches += certainty_errors.sum(axis=0)
            
        elif self.patch_strategy == 'EnsembleLevel':
            # EnsembleLevel: Uses ensemble-level correction
            ensemble_neighbor_mean_preds = neighbor_preds.mean(axis=1)
            ensemble_correction = (neighbor_labels - ensemble_neighbor_mean_preds).mean()
            final_patches += ensemble_correction
            
        else:
            # Default to BiasCorrected
            errors = labels_array[:, np.newaxis] - preds_array
            bias_patches = np.zeros(errors.shape[1])
            
            for i in range(errors.shape[1]):
                if np.mean(errors[:, i] < 0) > self.bias_threshold:
                    bias_patches[i] = errors[:, i][errors[:, i] < 0].mean()
                elif np.mean(errors[:, i] > 0) > self.bias_threshold:
                    bias_patches[i] = errors[:, i][errors[:, i] > 0].mean()
            
            final_patches += bias_patches
        
        return final_patches
    
    def _apply_local_patching(self, X):
        """Apply local bias correction patches."""
        patched_preds = self._get_predictions(self.models_, X)
        
        for idx, row in enumerate(X):
            distances, neighbors_idx = self.nn_.kneighbors(row.reshape(1, -1))
            distances = distances[0]
            neighbors_idx = neighbors_idx[0]
            
            # Adaptive k selection
            adaptive_k = self._get_adaptive_k(row, distances)
            selected_neighbors_idx = neighbors_idx[:adaptive_k]
            
            # Get neighbor information
            neighbor_features = self.X_train_[selected_neighbors_idx]
            neighbor_train_preds = self._get_predictions(self.models_, neighbor_features)
            neighbor_true_labels = self.y_train_[selected_neighbors_idx]
            
            # Calculate and apply patches
            final_patches = self._calculate_local_patches(
                neighbor_train_preds, neighbor_true_labels, 
                pd.Series(row, name=idx), neighbor_features
            )
            new_preds = patched_preds.iloc[idx] + final_patches
            
            if self.is_classification_:
                new_preds = np.clip(new_preds, 0, 1)
            
            patched_preds.iloc[idx] = new_preds
        
        return patched_preds
    
    def _rocile_reconciliation(self, patched_preds):
        """Apply Rocile reconciliation to patched predictions."""
        preds = patched_preds.to_numpy().T
        num_models = preds.shape[0]
        
        if num_models < 2:
            return patched_preds.mean(axis=1).values
        
        momentum = np.zeros_like(preds)
        
        for t in range(self.max_iter):
            diffs = preds[:, np.newaxis, :] - preds
            mean_abs_diffs = np.mean(np.abs(diffs), axis=2)
            np.fill_diagonal(mean_abs_diffs, -1)
            
            if np.max(mean_abs_diffs) < 1e-6:
                break
            
            pairs_indices = np.dstack(np.unravel_index(
                np.argsort(mean_abs_diffs.ravel()), mean_abs_diffs.shape))[0]
            top_pairs = [p for p in pairs_indices if p[0] < p[1]][-self.batch_size:]
            
            learning_rate = self.learning_rate * np.exp(-t / 100)
            
            for i, j in top_pairs:
                update = learning_rate * (preds[i] - preds[j])
                momentum[i] = self.momentum * momentum[i] - update
                momentum[j] = self.momentum * momentum[j] + update
                
                preds[i] = preds[i] + momentum[i]
                preds[j] = preds[j] + momentum[j]
                
                if self.is_classification_:
                    preds[i] = np.clip(preds[i], 0, 1)
                    preds[j] = np.clip(preds[j], 0, 1)
        
        return np.mean(preds, axis=0)
    
    def _calculate_initial_metrics(self, preds_df):
        """Calculate initial disagreement metrics before reconciliation."""
        metrics = {}
        
        # Variance across models
        metrics['initial_variance'] = preds_df.var(axis=1).mean()
        
        # Ambiguity (max - min predictions)
        metrics['initial_ambiguity'] = (preds_df.max(axis=1) - preds_df.min(axis=1)).mean()
        
        # Maximum disagreement between any pair of models
        num_models = preds_df.shape[1]
        if num_models < 2:
            metrics['initial_max_disagreement'] = 0.0
        else:
            max_disagreement = 0
            for i in range(num_models):
                for j in range(i + 1, num_models):
                    disagreement = np.mean(np.abs(preds_df.iloc[:, i] - preds_df.iloc[:, j]))
                    if disagreement > max_disagreement:
                        max_disagreement = disagreement
            metrics['initial_max_disagreement'] = max_disagreement
        
        # Mean disagreement (fraction of points where models differ significantly)
        epsilon = 0.05
        ensemble_mean = preds_df.mean(axis=1)
        disagreements = ((preds_df > (ensemble_mean.values[:, None] + epsilon)).sum(axis=1) +
                         (preds_df < (ensemble_mean.values[:, None] - epsilon)).sum(axis=1))
        metrics['initial_disagreement'] = disagreements.mean() / preds_df.shape[1]
        
        return metrics
    
    def _calculate_final_metrics(self, preds_df):
        """Calculate final disagreement metrics after reconciliation."""
        metrics = {}
        
        # Variance across models
        metrics['final_variance'] = preds_df.var(axis=1).mean()
        
        # Ambiguity (max - min predictions)
        metrics['final_ambiguity'] = (preds_df.max(axis=1) - preds_df.min(axis=1)).mean()
        
        # Maximum disagreement between any pair of models
        num_models = preds_df.shape[1]
        if num_models < 2:
            metrics['final_max_disagreement'] = 0.0
        else:
            max_disagreement = 0
            for i in range(num_models):
                for j in range(i + 1, num_models):
                    disagreement = np.mean(np.abs(preds_df.iloc[:, i] - preds_df.iloc[:, j]))
                    if disagreement > max_disagreement:
                        max_disagreement = disagreement
            metrics['final_max_disagreement'] = max_disagreement
        
        # Mean disagreement (fraction of points where models differ significantly)
        epsilon = 0.05
        ensemble_mean = preds_df.mean(axis=1)
        disagreements = ((preds_df > (ensemble_mean.values[:, None] + epsilon)).sum(axis=1) +
                         (preds_df < (ensemble_mean.values[:, None] - epsilon)).sum(axis=1))
        metrics['final_disagreement'] = disagreements.mean() / preds_df.shape[1]
        
        return metrics
    
    def fit(self, X, y):
        """
        Fit the AdaRocile model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Store training data for local patching
        self.X_train_ = X
        self.y_train_ = y
        
        # Create Rashomon set
        self.models_, self.is_classification_ = self._create_rashomon_set(X, y)
        
        if len(self.models_) < 2:
            raise ValueError("Need at least 2 models for reconciliation")
        
        # Fit nearest neighbors for local patching
        max_k = min(self.base_k * 2, len(X) - 1)
        self.nn_ = NearestNeighbors(n_neighbors=max_k).fit(X)
        
        # Calculate initial disagreement metrics
        initial_preds = self._get_predictions(self.models_, X)
        self.initial_disagreement_ = self._calculate_initial_metrics(initial_preds)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Predict using the AdaRocile ensemble.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y : array-like of shape (n_samples,)
            Predicted values.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        
        # Step 1: Apply local patching
        patched_preds = self._apply_local_patching(X)
        
        # Step 2: Apply Rocile reconciliation
        final_preds = self._rocile_reconciliation(patched_preds)
        
        # Calculate final disagreement metrics
        final_preds_df = pd.DataFrame(final_preds.reshape(-1, 1), index=patched_preds.index, columns=['final_pred'])
        self.final_disagreement_ = self._calculate_final_metrics(patched_preds)
        
        return final_preds 