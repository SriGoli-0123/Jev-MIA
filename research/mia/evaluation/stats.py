"""Two-sample tests used to decide whether RIFT features differ between members and non-members.

Nothing here assumes a direction: every test is two-sided, and the multivariate tests are
distribution-free (energy distance) or classifier-based with permutation nulls.
"""
import numpy as np
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def univariate_tests(x, y, n_perm=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y).astype(int)
    m, n = x[y == 1], x[y == 0]
    auc = roc_auc_score(y, x)
    null = np.array([roc_auc_score(rng.permutation(y), x) for _ in range(n_perm)])
    p_auc = (1 + np.sum(np.abs(null - 0.5) >= abs(auc - 0.5))) / (n_perm + 1)
    pooled = np.sqrt((m.var(ddof=1) + n.var(ddof=1)) / 2)
    return {"n_members": int(len(m)), "n_nonmembers": int(len(n)),
            "mean_members": float(m.mean()), "mean_nonmembers": float(n.mean()),
            "sd_members": float(m.std(ddof=1)), "sd_nonmembers": float(n.std(ddof=1)),
            "median_members": float(np.median(m)), "median_nonmembers": float(np.median(n)),
            "standardized_mean_difference": float((m.mean() - n.mean()) / pooled) if pooled > 0 else 0.0,
            "auc": float(auc), "p_auc_permutation": float(p_auc),
            "ks_stat": float(sps.ks_2samp(m, n).statistic), "p_ks": float(sps.ks_2samp(m, n).pvalue),
            "p_mannwhitney": float(sps.mannwhitneyu(m, n, alternative="two-sided").pvalue),
            "p_levene": float(sps.levene(m, n).pvalue)}


def energy_distance_test(X, y, n_perm=500, rng=None, max_n=3000):
    """Multivariate energy-distance two-sample statistic with a permutation p-value."""
    rng = rng or np.random.default_rng(0)
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y).astype(int)
    if len(y) > max_n:
        keep = rng.choice(len(y), max_n, replace=False)
        X, y = X[keep], y[keep]
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))

    def stat(labels):
        a, b = labels == 1, labels == 0
        return 2 * D[np.ix_(a, b)].mean() - D[np.ix_(a, a)].mean() - D[np.ix_(b, b)].mean()

    observed = stat(y)
    null = np.array([stat(rng.permutation(y)) for _ in range(n_perm)])
    return {"energy_distance": float(observed), "p_permutation": float((1 + np.sum(null >= observed)) / (n_perm + 1)),
            "n": int(len(y))}


def cv_auc(X, y, n_splits=5, seed=0, C=1.0):
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y).astype(int)
    oof = np.zeros(len(y))
    for train, test in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        model.fit(X[train], y[train])
        oof[test] = model.predict_proba(X[test])[:, 1]
    return float(roc_auc_score(y, oof))


def classifier_two_sample_test(X, y, n_perm=200, rng=None, n_splits=5, seed=0, C=1.0):
    """Cross-validated logistic AUC on the shadow rows, two-sided permutation p-value."""
    rng = rng or np.random.default_rng(0)
    observed = cv_auc(X, y, n_splits, seed, C)
    null = np.array([cv_auc(X, rng.permutation(y), n_splits, seed, C) for _ in range(n_perm)])
    p = (1 + np.sum(np.abs(null - 0.5) >= abs(observed - 0.5))) / (n_perm + 1)
    return {"cv_auc": observed, "p_permutation": float(p), "null_mean": float(null.mean()), "n_perm": n_perm}


def incremental_auc_test(X_base, X_add, y, n_perm=200, rng=None, n_splits=5, seed=0, C=1.0):
    """Does adding X_add to X_base raise CV AUC more than X_add rows shuffled across states would?"""
    rng = rng or np.random.default_rng(0)
    X_base, X_add = np.asarray(X_base, dtype=np.float64), np.asarray(X_add, dtype=np.float64)
    base = cv_auc(X_base, y, n_splits, seed, C)
    full = cv_auc(np.hstack([X_base, X_add]), y, n_splits, seed, C)
    null = np.array([cv_auc(np.hstack([X_base, X_add[rng.permutation(len(y))]]), y, n_splits, seed, C) - base
                     for _ in range(n_perm)])
    delta = full - base
    return {"cv_auc_base": base, "cv_auc_full": full, "delta_auc": float(delta),
            "p_permutation_one_sided": float((1 + np.sum(null >= delta)) / (n_perm + 1)), "n_perm": n_perm}


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, same order as input."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def caliper_match(scores, y, caliper_sd=0.05, rng=None):
    """Greedy 1:1 member/non-member matching on a score, without replacement, within caliper*SD.

    Evaluation-only (uses true labels) to build a subset where that score carries ~no signal.
    """
    rng = rng or np.random.default_rng(0)
    scores, y = np.asarray(scores, dtype=np.float64), np.asarray(y).astype(int)
    caliper = caliper_sd * scores.std()
    members = rng.permutation(np.flatnonzero(y == 1))
    pool = np.flatnonzero(y == 0)
    available = np.ones(len(pool), dtype=bool)
    pairs = []
    for m in members:
        if not available.any():
            break
        gaps = np.abs(scores[pool] - scores[m])
        gaps[~available] = np.inf
        best = int(np.argmin(gaps))
        if gaps[best] <= caliper:
            pairs.append((int(m), int(pool[best])))
            available[best] = False
    idx = np.array([i for pair in pairs for i in pair], dtype=int)
    return idx, {"pairs": len(pairs), "caliper": float(caliper), "caliper_sd": caliper_sd}
