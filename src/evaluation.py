"""
Comprehensive model evaluation metrics for credit risk modeling.
Includes KS, Gini, PSI, and other industry-standard metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')


class CreditRiskMetrics:
    """Comprehensive evaluation metrics for credit risk models."""
    
    @staticmethod
    def calculate_ks(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic."""
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            ks = np.max(tpr - fpr)
            return ks
        except:
            return 0.0
    
    @staticmethod
    def calculate_gini(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate Gini coefficient."""
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
            gini = 2 * auc - 1
            return gini
        except:
            return 0.0
    
    @staticmethod
    def calculate_psi(expected: pd.Series, actual: pd.Series, 
                     bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        PSI < 0.1: No significant change
        PSI 0.1-0.25: Some minor change
        PSI > 0.25: Significant change - model may need retraining
        """
        try:
            # Create bins based on expected distribution
            breakpoints = np.linspace(0, 1, bins + 1)
            
            expected_percent = pd.cut(expected, bins=breakpoints, include_lowest=True).value_counts(normalize=True, sort=False)
            actual_percent = pd.cut(actual, bins=breakpoints, include_lowest=True).value_counts(normalize=True, sort=False)
            
            # Handle zero values
            expected_percent = expected_percent.replace(0, 0.0001)
            actual_percent = actual_percent.replace(0, 0.0001)
            
            psi = np.sum((actual_percent - expected_percent) * 
                        np.log(actual_percent / expected_percent))
            
            return psi
        except Exception as e:
            print(f"Error calculating PSI: {e}")
            return np.nan
    
    @staticmethod
    def calculate_iv_woe(df: pd.DataFrame, feature: str, target: str) -> Tuple[float, pd.DataFrame]:
        """Calculate Information Value and WOE for a feature."""
        total_good = (df[target] == 0).sum()
        total_bad = (df[target] == 1).sum()
        
        if isinstance(df[feature].dtype, pd.CategoricalDtype) or df[feature].dtype == 'object':
            grouped = df.groupby(feature)[target].agg(['count', 'sum'])
        else:
            # Bin numeric features
            grouped = pd.cut(df[feature], bins=10, duplicates='drop')
            grouped = df.groupby(grouped)[target].agg(['count', 'sum'])
        
        grouped['good'] = grouped['count'] - grouped['sum']
        grouped['bad'] = grouped['sum']
        
        grouped['good_pct'] = grouped['good'] / (total_good + 1e-6)
        grouped['bad_pct'] = grouped['bad'] / (total_bad + 1e-6)
        
        grouped['woe'] = np.log((grouped['bad_pct'] + 1e-6) / (grouped['good_pct'] + 1e-6))
        grouped['iv'] = (grouped['bad_pct'] - grouped['good_pct']) * grouped['woe']
        
        iv_total = grouped['iv'].sum()
        
        return iv_total, grouped
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                             y_pred: Optional[np.ndarray] = None,
                             threshold: float = 0.5) -> Dict:
        """Calculate all evaluation metrics."""
        if y_pred is None:
            y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'ks': CreditRiskMetrics.calculate_ks(y_true, y_pred_proba),
            'gini': CreditRiskMetrics.calculate_gini(y_true, y_pred_proba),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        })
        
        return metrics
    
    @staticmethod
    def calculate_score_distribution(y_pred_proba: np.ndarray, bins: int = 20) -> pd.DataFrame:
        """Calculate score distribution statistics."""
        score_df = pd.DataFrame({
            'score': y_pred_proba
        })
        
        score_df['bin'] = pd.cut(score_df['score'], bins=bins)
        
        distribution = score_df.groupby('bin').agg({
            'score': ['count', 'mean', 'std', 'min', 'max']
        }).reset_index()
        
        distribution.columns = ['score_range', 'count', 'mean_score', 'std_score', 'min_score', 'max_score']
        distribution['pct'] = distribution['count'] / len(score_df) * 100
        
        return distribution
    
    @staticmethod
    def calculate_risk_band_metrics(df: pd.DataFrame, score_col: str, 
                                   target_col: str, n_bands: int = 5) -> pd.DataFrame:
        """Calculate metrics by risk band."""
        df = df.copy()
        df['risk_band'] = pd.qcut(df[score_col], q=n_bands, 
                                  labels=[f'Band_{i+1}' for i in range(n_bands)], 
                                  duplicates='drop')
        
        band_metrics = df.groupby('risk_band').agg({
            score_col: ['count', 'mean', 'min', 'max'],
            target_col: ['mean', 'sum']
        }).reset_index()
        
        band_metrics.columns = ['risk_band', 'count', 'avg_score', 'min_score', 'max_score',
                               'default_rate', 'total_defaults']
        
        band_metrics['cumulative_default_rate'] = (
            df.groupby('risk_band')[target_col].sum().cumsum() / 
            df.groupby('risk_band')[target_col].count().cumsum()
        ).values
        
        return band_metrics
    
    @staticmethod
    def calculate_feature_importance(model, feature_names: list) -> pd.DataFrame:
        """Extract feature importance from model."""
        importance_df = pd.DataFrame({
            'feature': feature_names
        })
        
        # Try different methods based on model type
        if hasattr(model, 'feature_importances_'):
            importance_df['importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            importance_df['importance'] = np.abs(coef)
        else:
            importance_df['importance'] = 0
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['importance_pct'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        )
        
        return importance_df


def print_evaluation_report(y_true: np.ndarray, y_pred_proba: np.ndarray,
                           y_pred: Optional[np.ndarray] = None,
                           threshold: float = 0.5):
    """Print comprehensive evaluation report."""
    metrics = CreditRiskMetrics.calculate_all_metrics(y_true, y_pred_proba, y_pred, threshold)
    
    print("=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"\nDiscriminatory Power:")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"  Gini:        {metrics['gini']:.4f}")
    print(f"  KS Statistic: {metrics['ks']:.4f}")
    
    print(f"\nClassification Metrics (threshold={threshold}):")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positive']}")
    print(f"  True Negatives:  {metrics['true_negative']}")
    print(f"  False Positives: {metrics['false_positive']}")
    print(f"  False Negatives: {metrics['false_negative']}")
    print("=" * 60)

