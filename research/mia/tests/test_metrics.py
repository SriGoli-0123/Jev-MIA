"""Metric, bootstrap, confidence-formula and statistics sanity checks."""
import numpy as np
import pytest

from mia.attacks.confidence import choice_confidence, confidence_row, score_confidence
from mia.evaluation.bootstrap import bootstrap_summary, paired_difference
from mia.evaluation.metrics import summarize, tpr_at_fpr
from mia.evaluation.stats import caliper_match, holm


def test_perfect_and_random_separation():
    y = np.repeat([1, 0], 500)
    assert summarize(y, y.astype(float))["roc_auc"] == 1.0
    assert summarize(y, np.random.default_rng(0).random(1000))["roc_auc"] == pytest.approx(0.5, abs=0.06)


def test_tpr_at_fpr_on_a_known_ranking():
    # 10 negatives, 10 positives. Top score is a negative, then 5 positives.
    y = np.array([0] + [1] * 5 + [0] * 9 + [1] * 5)
    s = np.arange(len(y))[::-1].astype(float)
    assert tpr_at_fpr(y, s, 0.1) == pytest.approx(0.5)
    assert tpr_at_fpr(y, s, 0.2) == pytest.approx(0.5)
    assert tpr_at_fpr(y, s, 0.05) is None  # 10 negatives cannot resolve a 5% FPR


def test_unresolvable_low_fpr_is_none():
    y = np.repeat([1, 0], 50)
    assert summarize(y, np.random.default_rng(1).random(100))["tpr_at_0.1pct_fpr"] is None


def test_bootstrap_interval_contains_point_and_respects_groups():
    rng = np.random.default_rng(2)
    y = np.repeat([1, 0], 300)
    s = rng.normal(y * 0.5, 1)
    out = bootstrap_summary(y, s, n_boot=200, seed=0)
    ci = out["ci95"]["roc_auc"]
    assert ci["low"] <= out["point"]["roc_auc"] <= ci["high"]
    groups = np.arange(600) % 300  # each group holds one member and one non-member: must be rejected
    with pytest.raises(ValueError):
        bootstrap_summary(y, s, groups=groups, n_boot=5)


def test_paired_difference_of_identical_scores_is_zero():
    y = np.repeat([1, 0], 100)
    s = np.random.default_rng(3).random(200)
    out = paired_difference(y, s, s, n_boot=50)
    assert out["difference"] == 0 and out["ci95"] == [0.0, 0.0]


def test_official_confidence_formulas():
    assert choice_confidence([0.25] * 4) == pytest.approx(0)
    assert choice_confidence([1, 0, 0]) == pytest.approx(1)
    assert choice_confidence([0.7, 0.3]) == pytest.approx(0.4)
    assert score_confidence([0.25] * 4) == pytest.approx(0)
    assert score_confidence([0, 0, 1, 0]) == pytest.approx(1)
    assert 0 <= score_confidence([0.1, 0.2, 0.3, 0.4]) <= 1
    assert choice_confidence([0, 0]) == pytest.approx(0)  # zero total -> uniform


def test_confidence_row_fields():
    row = confidence_row("boolean", np.array([0.0, 2.0]), 1)
    assert row["p_true"] == pytest.approx(1 / (1 + np.exp(-2)))
    assert row["p_max"] == pytest.approx(row["p_true"])
    assert np.isnan(row["h"]) and row["correct"] == 1
    row = confidence_row("choice", np.array([2.0, 0.0, 0.0]), 1)
    assert row["correct"] == 0 and row["p_y"] < row["p_max"]


def test_holm_and_caliper_matching():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    rng = np.random.default_rng(4)
    y = np.repeat([1, 0], 200)
    s = rng.normal(y * 2, 1)
    idx, info = caliper_match(s, y, 0.05, rng)
    assert info["pairs"] > 0 and len(idx) == 2 * info["pairs"]
    assert summarize(y[idx], s[idx])["roc_auc"] == pytest.approx(0.5, abs=0.1)
