"""
Configuration management for the credit risk modeling framework.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model training."""
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    cv_folds: int = 5
    scoring_metric: str = "roc_auc"
    
    # Model hyperparameters
    logistic_regression: Dict[str, Any] = None
    xgboost: Dict[str, Any] = None
    random_forest: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.logistic_regression is None:
            self.logistic_regression = {
                "max_iter": 1000,
                "C": 1.0,
                "penalty": "l2"
            }
        if self.xgboost is None:
            self.xgboost = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            }
        if self.random_forest is None:
            self.random_forest = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            }


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Features to use
    numeric_features: List[str] = None
    categorical_features: List[str] = None
    
    # Feature engineering options
    use_woe: bool = True
    use_interactions: bool = True
    use_binning: bool = True
    iv_threshold: float = 0.02  # Minimum IV for feature selection
    
    # Binning configuration
    binning_method: str = "optimal"  # optimal, quantile, uniform
    n_bins: int = 5
    
    def __post_init__(self):
        if self.numeric_features is None:
            self.numeric_features = [
                "duration", "credit_amount", "installment_rate",
                "age", "existing_credits", "residence_since"
            ]
        if self.categorical_features is None:
            self.categorical_features = [
                "credit_history", "purpose", "savings",
                "employment", "personal_status", "property",
                "housing", "job"
            ]


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring."""
    psi_threshold: float = 0.25  # Alert if PSI > 0.25
    score_bins: int = 10
    monitoring_window_days: int = 30


@dataclass
class DecisionConfig:
    """Configuration for decisioning."""
    risk_bands: Dict[str, Dict[str, float]] = None
    approval_threshold: float = 0.3
    
    def __post_init__(self):
        if self.risk_bands is None:
            self.risk_bands = {
                "Very Low": {"min_score": 0.0, "max_score": 0.15},
                "Low": {"min_score": 0.15, "max_score": 0.25},
                "Medium": {"min_score": 0.25, "max_score": 0.40},
                "High": {"min_score": 0.40, "max_score": 0.60},
                "Very High": {"min_score": 0.60, "max_score": 1.0}
            }


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.json"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}

