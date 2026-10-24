"""State-level (cluster) bootstrap confidence intervals for MIA metrics.

Rows of the same state are resampled together, and resampling is stratified by membership
so every replicate keeps the original member / non-member counts.
"""
import numpy as np

from mia.evaluation.metrics import DEFAULT_FPRS, summarize

CI_METRICS = ("roc_auc", "pr_auc")


def _group_index(y, groups):
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    if groups is None or len(groups) == 0:
        groups = np.arange(len(y))
    index = {}
    for i, g in enumerate(groups):
        index.setdefault(g, []).append(i)
    by_class = {0: [], 1: []}
    for g, rows in index.items():
        labels = set(y[rows])
        if len(labels) != 1:
            raise ValueError(f"Group {g} mixes members and non-members")
        by_class[labels.pop()].append(np.asarray(rows))
    return by_class


def _resample(by_class, rng):
    picks = []
    for label in (0, 1):
        groups = by_class[label]
        chosen = rng.integers(0, len(groups), size=len(groups))
        picks.extend(groups[c] for c in chosen)
    return np.concatenate(picks)


def bootstrap_summary(y, scores, groups=None, n_boot=1000, seed=0, fprs=DEFAULT_FPRS, gamma=None, alpha=0.05):
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    point = summarize(y, scores, fprs, gamma)
    by_class = _group_index(y, groups if groups is not None else np.arange(len(y)))
    rng = np.random.default_rng(seed)
    keys = [k for k, v in point.items() if (k in CI_METRICS or k.startswith("tpr_at_") or
                                            k in ("balanced_accuracy", "accuracy", "f1")) and v is not None]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = _resample(by_class, rng)
        rep = summarize(y[idx], scores[idx], fprs, gamma)
        for k in keys:
            if rep.get(k) is not None:
                samples[k].append(rep[k])
    ci = {k: {"low": float(np.quantile(v, alpha / 2)), "high": float(np.quantile(v, 1 - alpha / 2)),
              "n_boot": len(v)} for k, v in samples.items() if v}
    return {"point": point, "ci95": ci, "n_boot": n_boot, "bootstrap": "state-level, stratified by membership"}


def paired_difference(y, scores_a, scores_b, groups=None, metric="roc_auc", n_boot=1000, seed=0):
    """metric(a) - metric(b) on the same rows, with a paired state-level bootstrap CI and p-value."""
    y = np.asarray(y).astype(int)
    a, b = np.asarray(scores_a, dtype=np.float64), np.asarray(scores_b, dtype=np.float64)
    by_class = _group_index(y, groups if groups is not None else np.arange(len(y)))
    rng = np.random.default_rng(seed)

    def value(idx):
        return summarize(y[idx], a[idx])[metric] - summarize(y[idx], b[idx])[metric]

    point = value(np.arange(len(y)))
    reps = np.array([value(_resample(by_class, rng)) for _ in range(n_boot)])
    p_two_sided = float(min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean())))
    return {"metric": metric, "difference": float(point), "ci95": [float(np.quantile(reps, .025)),
                                                                   float(np.quantile(reps, .975))],
            "p_bootstrap_two_sided": p_two_sided, "n_boot": n_boot}
