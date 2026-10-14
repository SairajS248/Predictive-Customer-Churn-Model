"""Model pipelines and cross-validation helpers for the ConnectTel churn project."""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data_prep import build_preprocessor

RANDOM_STATE = 42
N_FOLDS = 5

# ROC-AUC is the primary metric (brief requirement). Average precision (PR-AUC) is more sensitive
# to performance on the minority class, so we report it alongside.
SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "accuracy": "accuracy",
}


def make_cv(n_splits=N_FOLDS, random_state=RANDOM_STATE):
    """Stratified so every fold keeps the ~26.5% churn ratio."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def make_pipeline(model, numeric_cols, categorical_cols):
    """Preprocessing + model in one estimator, so the scaler/encoder are re-fit inside each CV fold."""
    return Pipeline([
        ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
        ("model", model),
    ])


def cross_validate_model(pipeline, X, y, cv=None, scoring=SCORING):
    """Run stratified K-fold CV and return the raw per-fold scores (train and validation).

    Folds run serially; tree models parallelise internally via their own n_jobs.
    """
    return cross_validate(pipeline, X, y, cv=cv or make_cv(), scoring=scoring, return_train_score=True)


def summarize_cv(cv_results, name):
    """Per-fold scores -> one row of 'mean ± std' validation metrics plus the train ROC-AUC."""
    row = {"model": name}
    for metric in SCORING:
        scores = cv_results[f"test_{metric}"]
        row[f"{metric}_mean"] = scores.mean()
        row[f"{metric}_std"] = scores.std()
    row["train_roc_auc"] = cv_results["train_roc_auc"].mean()
    row["overfit_gap"] = row["train_roc_auc"] - row["roc_auc_mean"]
    return row


def format_cv_table(rows):
    """Readable 'mean ± std' table from a list of summarize_cv rows."""
    df = pd.DataFrame(rows).set_index("model")
    out = pd.DataFrame(index=df.index)
    for metric in SCORING:
        out[metric] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(df[f"{metric}_mean"], df[f"{metric}_std"])]
    out["train_roc_auc"] = df["train_roc_auc"].map("{:.3f}".format)
    out["overfit_gap"] = df["overfit_gap"].map("{:+.3f}".format)
    return out
