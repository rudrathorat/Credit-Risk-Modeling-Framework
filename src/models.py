"""
Model training and comparison framework.
Supports multiple algorithms with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    # Test if it actually works
    xgb.__version__
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False
    # Silently handle - XGBoost is optional


class ModelTrainer:
    """Train and compare multiple credit risk models."""
    
    def __init__(self, config=None):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.best_model = None
        self.best_model_name = None
        self.cv_results = {}
        
    def _get_model(self, model_name: str, params: Dict = None):
        """Get model instance."""
        if params is None:
            params = {}
        
        if model_name == "logistic_regression":
            default_params = {
                "max_iter": 1000,
                "random_state": 42,
                "C": 1.0,
                "penalty": "l2"
            }
            default_params.update(params)
            return LogisticRegression(**default_params)
        
        elif model_name == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1
            }
            default_params.update(params)
            return RandomForestClassifier(**default_params)
        
        elif model_name == "gradient_boosting":
            default_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42
            }
            default_params.update(params)
            return GradientBoostingClassifier(**default_params)
        
        elif model_name == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost not available. Install with: pip install xgboost. On Mac, also run: brew install libomp")
            default_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "eval_metric": "logloss",
                "use_label_encoder": False
            }
            default_params.update(params)
            return xgb.XGBClassifier(**default_params)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   model_name: str, params: Dict = None,
                   scale_features: bool = False) -> Any:
        """Train a single model."""
        X_train_processed = X_train.copy()
        
        # Scale features if needed
        if scale_features:
            scaler = StandardScaler()
            X_train_processed = pd.DataFrame(
                scaler.fit_transform(X_train_processed),
                columns=X_train_processed.columns,
                index=X_train_processed.index
            )
            self.scalers[model_name] = scaler
        
        # Get and train model
        model = self._get_model(model_name, params)
        model.fit(X_train_processed, y_train)
        
        self.models[model_name] = model
        return model
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                      model_name: str, cv_folds: int = 5,
                      scoring: str = "roc_auc") -> Dict:
        """Perform cross-validation."""
        model = self._get_model(model_name)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        results = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        self.cv_results[model_name] = results
        return results
    
    def train_multiple_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                             model_list: List[str] = None,
                             scale_features: bool = False) -> Dict:
        """Train multiple models and compare."""
        if model_list is None:
            model_list = ["logistic_regression", "random_forest", "gradient_boosting"]
            if XGBOOST_AVAILABLE:
                model_list.append("xgboost")
        
        results = {}
        
        for model_name in model_list:
            print(f"Training {model_name}...")
            
            # Get config params if available
            params = None
            if self.config and hasattr(self.config, 'model_config'):
                model_config = getattr(self.config, 'model_config', {})
                params = model_config.get(model_name, {})
            
            model = self.train_model(
                X_train, y_train, model_name, params, scale_features
            )
            
            # Cross-validation
            cv_results = self.cross_validate(X_train, y_train, model_name)
            results[model_name] = {
                'model': model,
                'cv_mean': cv_results['mean'],
                'cv_std': cv_results['std']
            }
            
            print(f"  CV Score: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
        
        return results
    
    def select_best_model(self, X_val: pd.DataFrame, y_val: pd.Series,
                         metric: str = "roc_auc") -> Tuple[str, Any]:
        """Select best model based on validation performance."""
        if not self.models:
            raise ValueError("No models trained yet. Call train_multiple_models first.")
        
        best_score = -np.inf
        best_name = None
        
        for model_name, model in self.models.items():
            # Get predictions
            X_val_processed = X_val.copy()
            if model_name in self.scalers:
                X_val_processed = pd.DataFrame(
                    self.scalers[model_name].transform(X_val_processed),
                    columns=X_val_processed.columns,
                    index=X_val_processed.index
                )
            
            y_pred_proba = model.predict_proba(X_val_processed)[:, 1]
            
            # Calculate metric
            if metric == "roc_auc":
                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(y_val, y_pred_proba)
            else:
                from sklearn.metrics import accuracy_score
                y_pred = (y_pred_proba >= 0.5).astype(int)
                score = accuracy_score(y_val, y_pred)
            
            if score > best_score:
                best_score = score
                best_name = model_name
        
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        
        return best_name, self.best_model
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """Make predictions."""
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No model specified and no best model selected.")
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models[model_name]
        
        X_processed = X.copy()
        if model_name in self.scalers:
            X_processed = pd.DataFrame(
                self.scalers[model_name].transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        
        return model.predict_proba(X_processed)[:, 1]


def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series,
                         model_name: str, param_grid: Dict,
                         cv: int = 5, scoring: str = "roc_auc") -> Dict:
    """Perform grid search for hyperparameter tuning."""
    from sklearn.model_selection import GridSearchCV
    
    trainer = ModelTrainer()
    base_model = trainer._get_model(model_name)
    
    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv, scoring=scoring,
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_
    }

