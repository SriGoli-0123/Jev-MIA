"""Output-confidence membership features.

Paper-reproduction features (Hidden Risks of Jev, arXiv:2610.04985, Sec. IV-A):
  p(x) = max_{a in A} p_theta(a | x) for every primitive;
  h(x) = the returned Jev confidence for Choice and Score; Noul/Boolean has none.
The paper takes h(x) from the Jev response. NanoJev does not return a confidence field,
so h is computed with the formulas in TypeSafe's MIT system-one-adapter
(confidence_metrics.py, as recorded upstream in research/jev_public_sources_zh.md):
  choice: (max p - 1/K) / (1 - 1/K)
  score:  max(0, 1 - d / d_uniform),  d = sum_i p_i |i - argmax p|,  d_uniform = mean_i |i - (K-1)/2|

Exploratory features (not in the paper) are kept separate: entropy, top-1/top-2 margin,
p(y), loss = -log p(y), logit p(y), and p(true) for Boolean.
"""
import math

import numpy as np

PAPER_FEATURES = {"choice": ("h", "p_max"), "score": ("h", "p_max"), "boolean": ("p_max",)}
EXTENDED_FEATURES = {
    "choice": ("h", "p_max", "entropy", "margin", "p_y", "loss"),
    "score": ("h", "p_max", "entropy", "margin", "p_y", "loss"),
    "boolean": ("p_max", "p_true", "entropy", "margin", "p_y", "loss"),
}
FEATURE_SETS = {"paper": PAPER_FEATURES, "extended": EXTENDED_FEATURES}
DEFAULT_EPS = 1e-6


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def _normalize(p):
    p = np.asarray(p, dtype=np.float64)
    total = p.sum()
    return np.full(len(p), 1 / len(p)) if total == 0 else p / total


def choice_confidence(p):
    p = _normalize(p)
    k = len(p)
    if k == 1:
        return 1.0
    return float((p.max() - 1 / k) / (1 - 1 / k))


def score_confidence(p):
    p = _normalize(p)
    k = len(p)
    if k == 1:
        return 1.0
    idx = np.arange(k)
    mode = int(np.argmax(p))
    distance = float(np.sum(p * np.abs(idx - mode)))
    uniform = float(np.mean(np.abs(idx - (k - 1) / 2)))
    return max(0.0, 1 - distance / uniform)


def label_logit(logits, y, eps=DEFAULT_EPS):
    """f = logit(p(y)) computed stably as z_y - logsumexp(z_-y), then clipped to logit([eps, 1-eps])."""
    z = np.asarray(logits, dtype=np.float64)
    others = np.delete(z, y)
    m = others.max()
    value = z[y] - (m + math.log(np.exp(others - m).sum()))
    bound = math.log((1 - eps) / eps)
    return float(np.clip(value, -bound, bound))


def confidence_row(qtype, logits, y, eps=DEFAULT_EPS):
    p = softmax(logits)
    top = np.sort(p)[::-1]
    row = {
        "K": len(p),
        "p_max": float(top[0]),
        "h": choice_confidence(p) if qtype == "choice" else score_confidence(p) if qtype == "score" else math.nan,
        "entropy": float(-np.sum(p * np.log(np.clip(p, 1e-300, None)))),
        "margin": float(top[0] - top[1]),
        "p_y": float(p[y]),
        "loss": float(-math.log(max(p[y], 1e-300))),
        "logit_y": label_logit(logits, y, eps),
        "p_true": float(p[1]) if qtype == "boolean" else math.nan,
        "correct": int(int(np.argmax(p)) == y),
    }
    return row


def feature_names(feature_set, primitive):
    return list(FEATURE_SETS[feature_set][primitive])
