"""
Advanced decisioning framework for credit risk.
Includes risk-based decisioning, profit optimization, and policy rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskBand:
    """Define a risk band for decisioning."""
    name: str
    min_score: float
    max_score: float
    decision: str
    conditions: Optional[Dict] = None


class DecisionEngine:
    """Credit decisioning engine with risk-based rules."""
    
    def __init__(self, risk_bands: Optional[List[RiskBand]] = None):
        """
        Initialize decision engine.
        
        Args:
            risk_bands: List of RiskBand objects defining decision rules
        """
        if risk_bands is None:
            # Default risk bands
            self.risk_bands = [
                RiskBand("Very Low", 0.0, 0.15, "Approve"),
                RiskBand("Low", 0.15, 0.25, "Approve"),
                RiskBand("Medium", 0.25, 0.40, "Approve with Conditions"),
                RiskBand("High", 0.40, 0.60, "Reject"),
                RiskBand("Very High", 0.60, 1.0, "Reject")
            ]
        else:
            self.risk_bands = risk_bands
        
        # Sort by score range
        self.risk_bands = sorted(self.risk_bands, key=lambda x: x.min_score)
    
    def assign_risk_band(self, score: float) -> str:
        """Assign risk band based on score."""
        for band in self.risk_bands:
            if band.min_score <= score < band.max_score:
                return band.name
        # Handle edge case
        return self.risk_bands[-1].name if score >= self.risk_bands[-1].min_score else self.risk_bands[0].name
    
    def make_decision(self, score: float, 
                     additional_features: Optional[pd.Series] = None) -> Dict:
        """Make credit decision based on risk score."""
        risk_band_name = self.assign_risk_band(score)
        
        # Find corresponding band
        band = next((b for b in self.risk_bands if b.name == risk_band_name), None)
        
        if band is None:
            raise ValueError(f"Risk band {risk_band_name} not found")
        
        decision_result = {
            'risk_score': score,
            'risk_band': risk_band_name,
            'decision': band.decision,
            'conditions': band.conditions
        }
        
        return decision_result
    
    def batch_decisions(self, scores: pd.Series,
                       additional_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Make decisions for multiple applicants."""
        results = []
        
        for idx, score in scores.items():
            additional = None
            if additional_features is not None and idx in additional_features.index:
                additional = additional_features.loc[idx]
            
            decision = self.make_decision(score, additional)
            decision['applicant_id'] = idx
            results.append(decision)
        
        return pd.DataFrame(results).set_index('applicant_id')


class ProfitOptimizer:
    """Optimize decisions based on expected profit."""
    
    def __init__(self, interest_rate: float = 0.15,
                 loss_given_default: float = 0.5,
                 operating_cost: float = 0.02):
        """
        Initialize profit optimizer.
        
        Args:
            interest_rate: Annual interest rate on loans
            loss_given_default: Loss percentage when borrower defaults
            operating_cost: Operating cost as percentage of loan amount
        """
        self.interest_rate = interest_rate
        self.loss_given_default = loss_given_default
        self.operating_cost = operating_cost
    
    def calculate_expected_profit(self, loan_amount: float,
                                 default_probability: float,
                                 loan_term_months: int = 12) -> float:
        """
        Calculate expected profit from a loan.
        
        Expected Profit = (1 - PD) * Interest - PD * LGD - Operating Cost
        """
        # Expected interest (if no default)
        expected_interest = (1 - default_probability) * loan_amount * (
            self.interest_rate * loan_term_months / 12
        )
        
        # Expected loss (if default)
        expected_loss = default_probability * loan_amount * self.loss_given_default
        
        # Operating cost
        operating_cost = loan_amount * self.operating_cost
        
        # Expected profit
        expected_profit = expected_interest - expected_loss - operating_cost
        
        return expected_profit
    
    def optimize_decision(self, scores: pd.Series, loan_amounts: pd.Series,
                         min_profit_threshold: float = 0.0) -> pd.DataFrame:
        """Optimize decisions to maximize expected profit."""
        results = []
        
        for idx in scores.index:
            score = scores.loc[idx]
            loan_amount = loan_amounts.loc[idx]
            
            expected_profit = self.calculate_expected_profit(loan_amount, score)
            
            decision = "Approve" if expected_profit >= min_profit_threshold else "Reject"
            
            results.append({
                'risk_score': score,
                'loan_amount': loan_amount,
                'expected_profit': expected_profit,
                'decision': decision
            })
        
        return pd.DataFrame(results, index=scores.index)


class EarlyWarningSystem:
    """Early warning system for post-disbursal risk monitoring."""
    
    def __init__(self, rules: Optional[List[Callable]] = None):
        """
        Initialize early warning system.
        
        Args:
            rules: List of rule functions that return (risk_score, action)
        """
        self.rules = rules or []
        
        # Default rules
        if not self.rules:
            self.rules = [
                self._high_installment_burden,
                self._low_age_risk,
                self._high_credit_amount,
                self._multiple_existing_credits
            ]
    
    def _high_installment_burden(self, features: pd.Series) -> Tuple[float, str]:
        """Check for high installment burden."""
        if 'installment_rate' in features:
            if features['installment_rate'] >= 3:
                return (1.0, "High installment burden detected")
        return (0.0, None)
    
    def _low_age_risk(self, features: pd.Series) -> Tuple[float, str]:
        """Check for low age risk."""
        if 'age' in features:
            if features['age'] < 30:
                return (0.5, "Young borrower - higher risk")
        return (0.0, None)
    
    def _high_credit_amount(self, features: pd.Series) -> Tuple[float, str]:
        """Check for high credit amount."""
        if 'credit_amount' in features:
            median_amount = 5000  # Would be calculated from training data
            if features['credit_amount'] > median_amount:
                return (0.5, "Above median credit amount")
        return (0.0, None)
    
    def _multiple_existing_credits(self, features: pd.Series) -> Tuple[float, str]:
        """Check for multiple existing credits."""
        if 'existing_credits' in features:
            if features['existing_credits'] >= 2:
                return (0.5, "Multiple existing credits")
        return (0.0, None)
    
    def calculate_early_warning_score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate early warning scores for customers."""
        results = []
        
        for idx, row in features.iterrows():
            total_score = 0.0
            alerts = []
            
            for rule_func in self.rules:
                score, alert = rule_func(row)
                total_score += score
                if alert:
                    alerts.append(alert)
            
            # Determine action based on score
            if total_score >= 2.0:
                action = "Trigger Collection Alert"
            elif total_score >= 1.0:
                action = "Send Reminder"
            else:
                action = "No Action"
            
            results.append({
                'early_warning_score': total_score,
                'alerts': '; '.join(alerts) if alerts else None,
                'recommended_action': action
            })
        
        return pd.DataFrame(results, index=features.index)
    
    def monitor_customers(self, customer_data: pd.DataFrame,
                         risk_scores: Optional[pd.Series] = None) -> pd.DataFrame:
        """Monitor customers and generate early warnings."""
        # Calculate early warning scores
        warning_results = self.calculate_early_warning_score(customer_data)
        
        # Combine with risk scores if provided
        if risk_scores is not None:
            warning_results['current_risk_score'] = risk_scores
            
            # Enhanced logic: combine risk score with early warning
            warning_results['combined_risk'] = (
                warning_results['early_warning_score'] * 0.3 +
                warning_results['current_risk_score'] * 0.7
            )
        
        return warning_results


def create_decision_report(decisions_df: pd.DataFrame,
                          target_col: Optional[str] = None) -> pd.DataFrame:
    """Create comprehensive decision report."""
    report = decisions_df.groupby('decision').agg({
        'risk_score': ['count', 'mean', 'std', 'min', 'max']
    })
    
    report.columns = ['count', 'avg_score', 'std_score', 'min_score', 'max_score']
    
    if target_col and target_col in decisions_df.columns:
        default_rates = decisions_df.groupby('decision')[target_col].agg(['mean', 'sum'])
        report['default_rate'] = default_rates['mean']
        report['total_defaults'] = default_rates['sum']
    
    report = report.sort_values('avg_score')
    return report

