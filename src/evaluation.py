"""Test-set evaluation, bootstrap confidence intervals, and the retention-campaign value model."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

# Illustrative retention-campaign assumptions. They are not ConnectTel figures and can be replaced with real ones.
OFFER_COST = 100.0          # $ cost of one retention offer (e.g. a discount, or an upgrade's cost to serve)
SAVE_RATE = 0.30            # share of contacted *true* churners the offer actually retains
VALUE_HORIZON_MONTHS = 12   # revenue protected by saving a churner = 12 x their monthly charge


def classification_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
        "flagged_share": pred.mean(),
    }


def bootstrap_auc_ci(y_true, proba, n_boot=1000, alpha=0.05, random_state=42):
    """Percentile bootstrap confidence interval for ROC-AUC."""
    rng = np.random.default_rng(random_state)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        aucs.append(roc_auc_score(y_true[idx], proba[idx]))
    return np.quantile(aucs, [alpha / 2, 1 - alpha / 2])


def bootstrap_auc_diff(y_true, proba_a, proba_b, n_boot=1000, random_state=42):
    """Paired bootstrap of AUC(a) - AUC(b) on the same resampled customers: (mean diff, 95% CI, P(diff <= 0))."""
    rng = np.random.default_rng(random_state)
    y_true, proba_a, proba_b = map(np.asarray, (y_true, proba_a, proba_b))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        diffs.append(roc_auc_score(y_true[idx], proba_a[idx]) - roc_auc_score(y_true[idx], proba_b[idx]))
    diffs = np.array(diffs)
    return diffs.mean(), np.quantile(diffs, [0.025, 0.975]), (diffs <= 0).mean()


def campaign_value(y_true, proba, monthly_charges, threshold,
                   offer_cost=OFFER_COST, save_rate=SAVE_RATE, horizon=VALUE_HORIZON_MONTHS):
    """Expected net value of sending a retention offer to every customer with proba >= threshold.

    Each contacted churner is saved with probability save_rate, protecting horizon x monthly charge.
    Every contacted customer (churner or not) costs offer_cost.
    """
    y_true, proba, monthly_charges = map(np.asarray, (y_true, proba, monthly_charges))
    flagged = proba >= threshold
    caught = flagged & (y_true == 1)
    revenue_saved = save_rate * horizon * monthly_charges[caught].sum()
    cost = offer_cost * flagged.sum()
    return {"threshold": threshold, "contacted": int(flagged.sum()), "churners_caught": int(caught.sum()),
            "revenue_saved": revenue_saved, "campaign_cost": cost, "net_value": revenue_saved - cost}


def threshold_sweep(y_true, proba, monthly_charges, thresholds=None):
    """Precision / recall / F1 / campaign value at each candidate threshold (0.05 to 0.95 in 0.01 steps)."""
    if thresholds is None:
        thresholds = np.round(np.arange(5, 96) / 100, 2)  # exact 2-dp values so they can be looked up by label
    rows = []
    for t in thresholds:
        m = classification_metrics(y_true, proba, t)
        v = campaign_value(y_true, proba, monthly_charges, t)
        rows.append({"threshold": t, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                     "flagged_share": m["flagged_share"], **{k: v[k] for k in ("contacted", "churners_caught",
                                                                               "net_value")}})
    return pd.DataFrame(rows)


def confusion_frame(y_true, pred):
    cm = confusion_matrix(y_true, pred)
    return pd.DataFrame(cm, index=["Actual: Stay", "Actual: Churn"], columns=["Pred: Stay", "Pred: Churn"])
