"""
Model explainability utilities using SHAP values and feature importance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")


class ModelExplainer:
    """Generate model explanations using SHAP and other methods."""
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize explainer.
        
        Args:
            model: Trained model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = None
        
    def fit_shap_explainer(self, X_train: pd.DataFrame, 
                          explainer_type: str = "auto"):
        """Fit SHAP explainer on training data."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not installed")
        
        if explainer_type == "auto":
            # Auto-detect explainer type
            model_type = type(self.model).__name__.lower()
            if "tree" in model_type or "forest" in model_type or "xgboost" in model_type:
                explainer_type = "tree"
            else:
                explainer_type = "kernel"
        
        if explainer_type == "tree":
            self.shap_explainer = shap.TreeExplainer(self.model)
        elif explainer_type == "kernel":
            # Use sample of training data for kernel explainer
            X_sample = X_train.sample(min(100, len(X_train)))
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba, X_sample
            )
        elif explainer_type == "linear":
            self.shap_explainer = shap.LinearExplainer(self.model, X_train)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        
        return self.shap_explainer
    
    def explain_prediction(self, X: pd.DataFrame, 
                          idx: Optional[int] = None) -> Dict:
        """Explain a single prediction."""
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not fitted. Call fit_shap_explainer first.")
        
        if idx is not None:
            X_instance = X.iloc[[idx]]
        else:
            X_instance = X
        
        # Get SHAP values
        shap_values = self.shap_explainer.shap_values(X_instance)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        shap_df = pd.DataFrame(
            shap_values,
            columns=self.feature_names,
            index=X_instance.index
        )
        
        # Calculate feature contributions
        base_value = self.shap_explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]  # Positive class
        
        prediction = self.model.predict_proba(X_instance)[:, 1][0]
        
        explanation = {
            'prediction': float(prediction),
            'base_value': float(base_value),
            'feature_contributions': shap_df.iloc[0].to_dict(),
            'top_contributors': shap_df.iloc[0].abs().sort_values(ascending=False).head(10).to_dict()
        }
        
        return explanation
    
    def explain_batch(self, X: pd.DataFrame, 
                     max_samples: int = 100) -> pd.DataFrame:
        """Explain predictions for a batch of instances."""
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not fitted. Call fit_shap_explainer first.")
        
        X_sample = X.sample(min(max_samples, len(X)))
        
        shap_values = self.shap_explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap_df = pd.DataFrame(
            shap_values,
            columns=self.feature_names,
            index=X_sample.index
        )
        
        return shap_df
    
    def get_feature_importance_shap(self, X: pd.DataFrame,
                                   max_samples: int = 100) -> pd.DataFrame:
        """Calculate feature importance using mean absolute SHAP values."""
        shap_df = self.explain_batch(X, max_samples)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': shap_df.abs().mean().values,
            'std_abs_shap': shap_df.abs().std().values
        }).sort_values('mean_abs_shap', ascending=False)
        
        importance_df['importance_pct'] = (
            importance_df['mean_abs_shap'] / importance_df['mean_abs_shap'].sum() * 100
        )
        
        return importance_df
    
    def plot_shap_summary(self, X: pd.DataFrame, 
                         max_samples: int = 100,
                         plot_type: str = "bar",
                         save_path: Optional[str] = None):
        """Plot SHAP summary."""
        if not SHAP_AVAILABLE:
            print("SHAP not available for plotting")
            return
        
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not fitted. Call fit_shap_explainer first.")
        
        X_sample = X.sample(min(max_samples, len(X)))
        
        shap_values = self.shap_explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        if plot_type == "bar":
            shap.plots.bar(shap_values, feature_names=self.feature_names, show=False)
        elif plot_type == "waterfall":
            shap.plots.waterfall(shap_values[0], feature_names=self.feature_names, show=False)
        elif plot_type == "beeswarm":
            shap.plots.beeswarm(shap_values, feature_names=self.feature_names, show=False)
        
        if save_path:
            import matplotlib.pyplot as plt
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def get_feature_importance_from_model(self) -> pd.DataFrame:
        """Extract feature importance from model directly."""
        importance_df = pd.DataFrame({
            'feature': self.feature_names
        })
        
        # Try different methods based on model type
        if hasattr(self.model, 'feature_importances_'):
            importance_df['importance'] = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
            importance_df['importance'] = np.abs(coef)
        else:
            importance_df['importance'] = 0
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['importance_pct'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        )
        
        return importance_df
    
    def generate_explanation_report(self, X: pd.DataFrame,
                                   y_pred_proba: Optional[pd.Series] = None,
                                   use_shap: bool = True) -> Dict:
        """Generate comprehensive explanation report."""
        report = {}
        
        # Feature importance from model
        model_importance = self.get_feature_importance_from_model()
        report['feature_importance'] = model_importance.to_dict('records')
        
        # SHAP-based importance if available
        if use_shap and SHAP_AVAILABLE and self.shap_explainer is not None:
            try:
                shap_importance = self.get_feature_importance_shap(X)
                report['shap_importance'] = shap_importance.to_dict('records')
                
                # Top features from both methods
                top_model = model_importance.head(10)['feature'].tolist()
                top_shap = shap_importance.head(10)['feature'].tolist()
                report['top_features'] = {
                    'model_based': top_model,
                    'shap_based': top_shap,
                    'common': list(set(top_model) & set(top_shap))
                }
            except Exception as e:
                report['shap_error'] = str(e)
        
        # Prediction statistics
        if y_pred_proba is not None:
            report['prediction_stats'] = {
                'mean': float(y_pred_proba.mean()),
                'std': float(y_pred_proba.std()),
                'min': float(y_pred_proba.min()),
                'max': float(y_pred_proba.max()),
                'p25': float(y_pred_proba.quantile(0.25)),
                'p50': float(y_pred_proba.quantile(0.50)),
                'p75': float(y_pred_proba.quantile(0.75))
            }
        
        return report


def create_explanation_dashboard(explainer: ModelExplainer,
                                X: pd.DataFrame,
                                y_pred_proba: pd.Series,
                                save_path: Optional[str] = None):
    """Create a visual explanation dashboard."""
    try:
        import matplotlib.pyplot as plt
        
        # Get feature importance
        importance_df = explainer.get_feature_importance_from_model()
        top_features = importance_df.head(15)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Feature Importance
        axes[0, 0].barh(range(len(top_features)), top_features['importance_pct'])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['feature'])
        axes[0, 0].set_xlabel('Importance (%)')
        axes[0, 0].set_title('Top 15 Feature Importance')
        axes[0, 0].invert_yaxis()
        
        # 2. Score Distribution
        axes[0, 1].hist(y_pred_proba, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(y_pred_proba.mean(), color='r', linestyle='--', 
                          label=f'Mean: {y_pred_proba.mean():.3f}')
        axes[0, 1].set_xlabel('Predicted Default Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Score Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature Importance Comparison (if SHAP available)
        if SHAP_AVAILABLE and explainer.shap_explainer is not None:
            try:
                shap_importance = explainer.get_feature_importance_shap(X)
                top_shap = shap_importance.head(10)
                
                # Compare top features
                common_features = list(set(top_features.head(10)['feature']) & 
                                      set(top_shap['feature']))
                model_only = list(set(top_features.head(10)['feature']) - 
                                 set(top_shap['feature']))
                shap_only = list(set(top_shap['feature']) - 
                                set(top_features.head(10)['feature']))
                
                axes[1, 0].text(0.1, 0.8, f'Common: {len(common_features)}', 
                               fontsize=12, transform=axes[1, 0].transAxes)
                axes[1, 0].text(0.1, 0.7, f'Model Only: {len(model_only)}', 
                               fontsize=12, transform=axes[1, 0].transAxes)
                axes[1, 0].text(0.1, 0.6, f'SHAP Only: {len(shap_only)}', 
                               fontsize=12, transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Feature Importance Comparison')
                axes[1, 0].axis('off')
            except:
                axes[1, 0].text(0.5, 0.5, 'SHAP analysis unavailable', 
                               ha='center', transform=axes[1, 0].transAxes)
                axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'SHAP not available', 
                           ha='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
        
        # 4. Risk Score by Feature (example with top feature)
        if len(X.columns) > 0:
            top_feature = top_features.iloc[0]['feature']
            if top_feature in X.columns:
                scatter_data = pd.DataFrame({
                    'feature_value': X[top_feature],
                    'risk_score': y_pred_proba
                })
                # Sample for plotting if too many points
                if len(scatter_data) > 1000:
                    scatter_data = scatter_data.sample(1000)
                axes[1, 1].scatter(scatter_data['feature_value'], 
                                  scatter_data['risk_score'], 
                                  alpha=0.5, s=10)
                axes[1, 1].set_xlabel(top_feature)
                axes[1, 1].set_ylabel('Risk Score')
                axes[1, 1].set_title(f'Risk Score vs {top_feature}')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    except ImportError:
        print("Matplotlib not available for plotting")

