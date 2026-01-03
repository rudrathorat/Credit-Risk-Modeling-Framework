# Credit Risk Modeling and Decision Framework

Production ready credit risk assessment system implementing industry standard practices for consumer lending platforms.

## Overview

End-to-end credit risk modeling framework featuring advanced feature engineering, multi-algorithm model comparison, comprehensive evaluation metrics, model explainability, and production monitoring capabilities.

**Performance**: ROC-AUC improved from 0.683 to 0.78 (14% improvement) through advanced feature engineering and model optimization.

## Features

- **Feature Engineering**: WOE transformation, IV calculation, optimal binning, feature interactions
- **Model Development**: Logistic Regression, Random Forest, Gradient Boosting with cross-validation
- **Evaluation**: ROC-AUC, KS statistic, Gini coefficient, PSI, risk band analysis
- **Explainability**: SHAP values for feature importance and prediction explanations
- **Monitoring**: Score drift detection (PSI), feature distribution tracking, automated alerting
- **Decisioning**: Risk-based decision engine, profit optimization, early warning system

## Quick Start

**Option 1: Run Python Script (Recommended for Quick Results)**
```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python run.py
```
Generates complete results in `results/` folder (CSV files, metrics, reports).

**Option 2: Run Enhanced Notebook (Interactive with Visualizations)**
```bash
jupyter lab notebooks/02_enhanced_credit_risk_model.ipynb
```
The notebook is the main showcase with:
- Complete analysis workflow
- Visualizations (feature importance, ROC curve, risk bands)
- Interactive exploration
- Model explanations

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

## Project Structure

```
├── notebooks/
│   ├── 01_underwriting_model.ipynb         # Basic implementation
│   └── 02_enhanced_credit_risk_model.ipynb # ⭐ Enhanced model (main showcase)
├── src/                                     # Production-ready modules
│   ├── feature_engineering.py              # WOE, IV, binning
│   ├── models.py                           # Model training & comparison
│   ├── evaluation.py                       # Metrics & evaluation
│   ├── monitoring.py                       # Production monitoring
│   ├── decisioning.py                      # Decision engine
│   └── explainability.py                   # SHAP explanations
├── run.py                                  # Automated execution script
├── data/
│   └── german_credit.data                  # German Credit Dataset
├── results/                                # Generated results (created on run)
│   ├── predictions.csv
│   ├── metrics.json
│   ├── feature_importance.csv
│   ├── risk_bands.csv
│   └── report.txt
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## Usage

### Feature Engineering
```python
from src.feature_engineering import FeatureEngineer

fe = FeatureEngineer()
X_engineered = fe.fit_transform(X_train, y_train, use_woe=True)
```

### Model Training
```python
from src.models import ModelTrainer

trainer = ModelTrainer()
results = trainer.train_multiple_models(X_train, y_train)
best_model = trainer.select_best_model(X_val, y_val)
```

### Evaluation & Monitoring
```python
from src.evaluation import CreditRiskMetrics
from src.monitoring import ModelMonitor

metrics = CreditRiskMetrics.calculate_all_metrics(y_true, y_pred_proba)
monitor = ModelMonitor(baseline_scores)
report = monitor.generate_monitoring_report(current_scores)
```

## Results

| Metric | Baseline | Enhanced | Improvement |
|--------|----------|----------|-------------|
| ROC-AUC | 0.683 | 0.78 | +14% |
| Gini | 0.37 | 0.56 | +51% |
| KS Statistic | - | 0.44 | Strong separation |
| Features | 5 numeric | All features with WOE | Comprehensive |

**Key Achievements:**
- Achieved 0.78 ROC-AUC through advanced feature engineering
- Implemented comprehensive model evaluation (KS, Gini, PSI)
- Production-ready monitoring and explainability framework

## Technologies

- Python 3.8+
- scikit-learn, pandas, numpy
- XGBoost (optional), SHAP (optional)
- matplotlib, seaborn (for visualizations)

## Best Practices

- Data leakage prevention through proper train/validation/test splits
- IV-based feature selection (IV > 0.02 threshold)
- Cross-validation with stratified splits
- Industry-standard credit risk metrics (KS, Gini, PSI)
- Production-ready monitoring and explainability
