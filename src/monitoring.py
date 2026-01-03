"""
Model monitoring utilities for production credit risk models.
Includes PSI calculation, score distribution tracking, and drift detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .evaluation import CreditRiskMetrics


class ModelMonitor:
    """Monitor model performance and stability in production."""
    
    def __init__(self, baseline_scores: pd.Series, 
                 baseline_date: Optional[datetime] = None):
        """
        Initialize monitor with baseline score distribution.
        
        Args:
            baseline_scores: Reference score distribution (training/validation set)
            baseline_date: Date of baseline (for time-based monitoring)
        """
        self.baseline_scores = baseline_scores.copy()
        self.baseline_date = baseline_date or datetime.now()
        self.monitoring_history = []
        
    def calculate_psi(self, current_scores: pd.Series, 
                     bins: int = 10) -> Dict:
        """Calculate Population Stability Index."""
        psi = CreditRiskMetrics.calculate_psi(
            self.baseline_scores, current_scores, bins
        )
        
        # Interpret PSI
        if psi < 0.1:
            status = "No significant change"
            alert = False
        elif psi < 0.25:
            status = "Minor change - monitor closely"
            alert = True
        else:
            status = "Significant change - model may need retraining"
            alert = True
        
        return {
            'psi': psi,
            'status': status,
            'alert': alert,
            'timestamp': datetime.now()
        }
    
    def detect_score_drift(self, current_scores: pd.Series,
                          threshold: float = 0.25) -> Dict:
        """Detect if score distribution has drifted."""
        psi_result = self.calculate_psi(current_scores)
        
        drift_detected = psi_result['psi'] > threshold
        
        # Additional statistics
        stats = {
            'baseline_mean': self.baseline_scores.mean(),
            'current_mean': current_scores.mean(),
            'mean_shift': current_scores.mean() - self.baseline_scores.mean(),
            'baseline_std': self.baseline_scores.std(),
            'current_std': current_scores.std(),
            'std_shift': current_scores.std() - self.baseline_scores.std()
        }
        
        result = {
            **psi_result,
            'drift_detected': drift_detected,
            'statistics': stats
        }
        
        # Store in history
        self.monitoring_history.append(result)
        
        return result
    
    def detect_feature_drift(self, baseline_features: pd.DataFrame,
                            current_features: pd.DataFrame,
                            threshold: float = 0.25) -> pd.DataFrame:
        """Detect drift in feature distributions."""
        drift_results = []
        
        for col in baseline_features.columns:
            if col not in current_features.columns:
                continue
            
            # Calculate PSI for this feature
            psi = CreditRiskMetrics.calculate_psi(
                baseline_features[col], current_features[col]
            )
            
            drift_results.append({
                'feature': col,
                'psi': psi,
                'drift_detected': psi > threshold,
                'baseline_mean': baseline_features[col].mean(),
                'current_mean': current_features[col].mean(),
                'baseline_std': baseline_features[col].std(),
                'current_std': current_features[col].std()
            })
        
        drift_df = pd.DataFrame(drift_results).sort_values('psi', ascending=False)
        return drift_df
    
    def monitor_performance(self, y_true: pd.Series, y_pred_proba: pd.Series,
                          window_days: int = 30) -> Dict:
        """Monitor model performance over time."""
        from .evaluation import CreditRiskMetrics
        
        metrics = CreditRiskMetrics.calculate_all_metrics(
            y_true.values, y_pred_proba.values
        )
        
        # Check if performance degraded
        # (In real scenario, compare with baseline performance)
        performance_status = "Good"
        if metrics['roc_auc'] < 0.6:
            performance_status = "Poor - investigate"
        elif metrics['roc_auc'] < 0.65:
            performance_status = "Degraded - monitor"
        
        result = {
            **metrics,
            'performance_status': performance_status,
            'timestamp': datetime.now()
        }
        
        return result
    
    def generate_monitoring_report(self, current_scores: pd.Series,
                                  current_features: Optional[pd.DataFrame] = None,
                                  baseline_features: Optional[pd.DataFrame] = None) -> Dict:
        """Generate comprehensive monitoring report."""
        report = {
            'timestamp': datetime.now(),
            'baseline_date': self.baseline_date,
            'days_since_baseline': (datetime.now() - self.baseline_date).days
        }
        
        # Score distribution drift
        score_drift = self.detect_score_drift(current_scores)
        report['score_drift'] = score_drift
        
        # Feature drift
        if current_features is not None and baseline_features is not None:
            feature_drift = self.detect_feature_drift(baseline_features, current_features)
            report['feature_drift'] = feature_drift.to_dict('records')
            report['features_with_drift'] = feature_drift[
                feature_drift['drift_detected']
            ]['feature'].tolist()
        
        # Score distribution statistics
        report['score_distribution'] = {
            'baseline': {
                'mean': float(self.baseline_scores.mean()),
                'std': float(self.baseline_scores.std()),
                'min': float(self.baseline_scores.min()),
                'max': float(self.baseline_scores.max()),
                'p25': float(self.baseline_scores.quantile(0.25)),
                'p50': float(self.baseline_scores.quantile(0.50)),
                'p75': float(self.baseline_scores.quantile(0.75))
            },
            'current': {
                'mean': float(current_scores.mean()),
                'std': float(current_scores.std()),
                'min': float(current_scores.min()),
                'max': float(current_scores.max()),
                'p25': float(current_scores.quantile(0.25)),
                'p50': float(current_scores.quantile(0.50)),
                'p75': float(current_scores.quantile(0.75))
            }
        }
        
        # Overall status
        alerts = []
        if score_drift['alert']:
            alerts.append(f"Score drift detected (PSI: {score_drift['psi']:.3f})")
        
        if current_features is not None and baseline_features is not None:
            feature_drift_df = self.detect_feature_drift(baseline_features, current_features)
            high_psi_features = feature_drift_df[feature_drift_df['psi'] > 0.25]
            if len(high_psi_features) > 0:
                alerts.append(f"{len(high_psi_features)} features with significant drift")
        
        report['alerts'] = alerts
        report['overall_status'] = "Healthy" if len(alerts) == 0 else "Alert"
        
        return report
    
    def plot_score_distribution(self, current_scores: pd.Series,
                               save_path: Optional[str] = None):
        """Plot score distribution comparison."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram
            ax1.hist(self.baseline_scores, bins=20, alpha=0.5, 
                    label='Baseline', density=True)
            ax1.hist(current_scores, bins=20, alpha=0.5, 
                    label='Current', density=True)
            ax1.set_xlabel('Risk Score')
            ax1.set_ylabel('Density')
            ax1.set_title('Score Distribution Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # CDF
            sorted_baseline = np.sort(self.baseline_scores)
            sorted_current = np.sort(current_scores)
            
            ax2.plot(sorted_baseline, np.arange(len(sorted_baseline)) / len(sorted_baseline),
                    label='Baseline', linewidth=2)
            ax2.plot(sorted_current, np.arange(len(sorted_current)) / len(sorted_current),
                    label='Current', linewidth=2)
            ax2.set_xlabel('Risk Score')
            ax2.set_ylabel('Cumulative Probability')
            ax2.set_title('Cumulative Distribution Function')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
        except ImportError:
            print("Matplotlib not available for plotting")


def calculate_monthly_monitoring_stats(df: pd.DataFrame, 
                                     score_col: str,
                                     date_col: str,
                                     target_col: Optional[str] = None) -> pd.DataFrame:
    """Calculate monthly monitoring statistics."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['year_month'] = df[date_col].dt.to_period('M')
    
    monthly_stats = df.groupby('year_month').agg({
        score_col: ['count', 'mean', 'std', 'min', 'max']
    })
    
    if target_col and target_col in df.columns:
        default_stats = df.groupby('year_month').agg({
            target_col: ['mean', 'sum']
        })
        monthly_stats = pd.concat([monthly_stats, default_stats], axis=1)
    
    monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns]
    monthly_stats = monthly_stats.reset_index()
    
    return monthly_stats

