# ConnectTel: Predictive Customer Churn Model

An end-to-end machine-learning pipeline that identifies telecom customers at high risk of cancelling, explains *why* each one is at risk, and turns the predictions into a retention-campaign recommendation.

**Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 columns, 26.5% churn).

## Results at a glance

| | |
|---|---|
| Final model | XGBoost (tuned with GridSearchCV, `max_depth=2`, `learning_rate=0.02`, 400 trees) |
| Cross-validated ROC-AUC | **0.850** (stratified 5-fold) |
| Held-out test ROC-AUC | **0.846** (95% CI 0.824-0.867) |
| Decision threshold | **0.65**, chosen to maximise retention-campaign value |
| At that threshold (test set) | Contacts 30% of customers, reaches 66% of churners, 59% precision |
| Campaign value (test set, illustrative assumptions) | **+$26k** vs −$43k (contact everyone) and −$13k (random, same budget) |
| Biggest churn driver (SHAP) | **Contract type**, with about 7x the impact of the monthly charge |

## Project structure

```
├── data/raw/                     # Telco churn CSV (downloaded from IBM's public GitHub mirror)
├── notebooks/
│   └── ConnectTel_Churn_Analysis.ipynb   # The full report: EDA → models → SHAP → business insights
├── src/
│   ├── data_prep.py              # Cleaning, feature engineering, preprocessing, what-if scenarios
│   ├── modeling.py               # Pipelines + stratified cross-validation helpers
│   └── evaluation.py             # Test metrics, bootstrap CIs, threshold sweep, campaign value model
├── figures/                      # All numbered report figures (PNG)
├── models/
│   └── connecttel_churn_model.joblib     # Final pipeline + decision threshold
├── ConnectTel_Churn_Presentation_Updated.pptx    # Project presentation
└── requirements.txt
```

## Notebook contents

1. **Problem statement**: business goal and why accuracy is the wrong metric
2. **Data quality**: `TotalCharges` text column, 11 blank values (all brand-new customers, set to 0)
3. **EDA**: univariate and multivariate analysis, 8 annotated figures
4. **Hypothesis testing**: chi-square + Cramér's V, two-proportion z-tests, Mann-Whitney U
5. **Feature engineering**: `TotalChargesPerTenure`, `ServiceCount`, `TenureGroup`, `HasProtection`, `HasStreaming`, `IsAutoPay`
6. **Encoding & scaling**: One-Hot + StandardScaler inside a leak-free `Pipeline`
7. **Baseline models**: Logistic Regression (with odds ratios) and Random Forest, stratified 5-fold CV, class-imbalance strategy
8. **Boosting & tuning**: XGBoost and LightGBM, GridSearchCV over 194 combinations, held-out test evaluation (ROC, PR, confusion matrices), business threshold selection
9. **SHAP interpretation**: beeswarm summary, grouped global importance, Contract vs Monthly Charge, dependence plots, individual waterfalls
10. **Business Insight Report**: who is at risk, which retention levers work, recommendations
11. **Limitations & next steps**

## How to run

```bash
# Python 3.12 recommended (best wheel support for shap / xgboost)
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# Optional: register the kernel used by the notebook
python -m ipykernel install --user --name connecttel --display-name "Python 3 (ConnectTel venv)"

jupyter notebook notebooks/ConnectTel_Churn_Analysis.ipynb
```

The dataset is included in `data/raw/`. If it is missing, download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle, or from [IBM's GitHub mirror](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv), and save it under that name in `data/raw/`.

A full "Restart & Run All" takes about 8 minutes, most of it hyperparameter grid search.

## Using the saved model

```python
import joblib
from src.data_prep import load_raw, clean, engineer_features

artifact = joblib.load("models/connecttel_churn_model.joblib")
customers = engineer_features(clean(load_raw())).drop(columns=["Churn"])

scores = artifact["pipeline"].predict_proba(customers[artifact["feature_columns"]])[:, 1]
at_risk = customers[scores >= artifact["decision_threshold"]]
```

## Key assumptions

The campaign-value analysis uses **illustrative** figures: a $100 offer cost, a 30% save rate, and 12 months of revenue per saved customer. They are defined at the top of `src/evaluation.py`. Replace them with real business figures and re-run §8.5 to re-derive the optimal threshold.
