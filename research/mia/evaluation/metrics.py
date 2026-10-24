"""Membership-inference metrics: ROC-AUC, PR-AUC, TPR at fixed low FPR, thresholded scores."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

DEFAULT_FPRS = (0.001, 0.01, 0.05)


def fpr_key(fpr):
    return f"tpr_at_{fpr * 100:g}pct_fpr"


def tpr_at_fpr(y, scores, fpr):
    """Largest TPR among ROC operating points with FPR <= ``fpr`` (no interpolation).

    Returns None when the negatives cannot resolve that FPR (n_neg * fpr < 1).
    """
    y = np.asarray(y)
    n_neg = int((y == 0).sum())
    if n_neg * fpr < 1:
        return None
    fprs, tprs, _ = roc_curve(y, scores, drop_intermediate=False)
    return float(tprs[fprs <= fpr + 1e-12].max())


def threshold_metrics(y, scores, gamma):
    y = np.asarray(y).astype(int)
    pred = np.asarray(scores) >= gamma
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * tpr / max(precision + tpr, 1e-12)
    return {"accuracy": (tp + tn) / len(y), "balanced_accuracy": (tpr + tnr) / 2, "f1": f1,
            "tpr_at_gamma": tpr, "fpr_at_gamma": 1 - tnr, "gamma": float(gamma)}


def summarize(y, scores, fprs=DEFAULT_FPRS, gamma=None):
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(set(y)) != 2:
        raise ValueError("Need both members and non-members to compute MIA metrics")
    out = {"n": int(len(y)), "n_members": int(y.sum()), "n_nonmembers": int((y == 0).sum()),
           "roc_auc": float(roc_auc_score(y, scores)), "pr_auc": float(average_precision_score(y, scores)),
           "pr_auc_chance": float(y.mean()), "min_resolvable_fpr": 1 / int((y == 0).sum())}
    for fpr in fprs:
        out[fpr_key(fpr)] = tpr_at_fpr(y, scores, fpr)
    if gamma is not None:
        out.update(threshold_metrics(y, scores, gamma))
    return out


def roc_points(y, scores):
    fprs, tprs, _ = roc_curve(np.asarray(y).astype(int), scores)
    return fprs, tprs
