"""Required checks 6 and 7: the discrete mixed second difference is 0 for additive functions and
recovers the interaction coefficient when one exists. Plus the logit and first-order identities."""
import itertools
import math

import numpy as np
import pytest

from mia.attacks.confidence import label_logit, softmax
from mia.attacks.first_order import single_fact_effects
from mia.attacks.rift import first_order_effects, grid_features, mixed_difference, pairwise_interaction

RNG = np.random.default_rng(0)


def grid(f, a, a_rep, b, b_rep):
    """f11, f10, f01, f00 for original (a, b) and replacements (a', b')."""
    return f(a, b), f(a, b_rep), f(a_rep, b), f(a_rep, b_rep)


@pytest.mark.parametrize("trial", range(20))
def test_additive_function_has_zero_interaction(trial):
    b0, bA, bB = RNG.normal(size=3)
    f = lambda A, B: b0 + bA * A + bB * B  # noqa: E731
    assert pairwise_interaction(*grid(f, 1, 0, 1, 0)) == pytest.approx(0, abs=1e-12)
    a, a_rep, b, b_rep = RNG.normal(size=4)
    assert pairwise_interaction(*grid(f, a, a_rep, b, b_rep)) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("trial", range(20))
def test_interaction_term_is_recovered(trial):
    b0, bA, bB, bAB = RNG.normal(size=4)
    f = lambda A, B: b0 + bA * A + bB * B + bAB * A * B  # noqa: E731
    assert pairwise_interaction(*grid(f, 1, 0, 1, 0)) == pytest.approx(bAB, abs=1e-12)
    a, a_rep, b, b_rep = RNG.normal(size=4)
    expected = bAB * (a - a_rep) * (b - b_rep)
    assert pairwise_interaction(*grid(f, a, a_rep, b, b_rep)) == pytest.approx(expected, abs=1e-12)


def test_pairwise_matches_definition_and_first_order_identity():
    f11, f10, f01, f00 = RNG.normal(size=4)
    assert pairwise_interaction(f11, f10, f01, f00) == pytest.approx(f11 - f10 - f01 + f00)
    d = first_order_effects(f11, f10, f01, f00)
    I = pairwise_interaction(f11, f10, f01, f00)
    assert d["d_j_i1"] - d["d_j_i0"] == pytest.approx(I)
    assert d["d_i_j1"] - d["d_i_j0"] == pytest.approx(I)
    g = grid_features({"11": f11, "10": f10, "01": f01, "00": f00})
    assert g["I"] == pytest.approx(I)


def test_third_order_difference_isolates_the_three_way_term():
    beta = RNG.normal(size=8)

    def f(A, B, C):
        return (beta[0] + beta[1] * A + beta[2] * B + beta[3] * C + beta[4] * A * B + beta[5] * A * C
                + beta[6] * B * C + beta[7] * A * B * C)

    values = {bits: f(*bits) for bits in itertools.product((0, 1), repeat=3)}
    assert mixed_difference(values, 3) == pytest.approx(beta[7], abs=1e-12)
    without = {bits: v - beta[7] * bits[0] * bits[1] * bits[2] for bits, v in values.items()}
    assert mixed_difference(without, 3) == pytest.approx(0, abs=1e-12)


def test_mixed_difference_requires_every_grid_point():
    with pytest.raises(KeyError):
        mixed_difference({(1, 1): 0.0, (1, 0): 0.0, (0, 1): 0.0}, 2)


def test_label_logit_is_logit_of_softmax_probability():
    for k in (2, 3, 5):
        z = RNG.normal(size=k) * 3
        for y in range(k):
            p = softmax(z)[y]
            assert label_logit(z, y) == pytest.approx(math.log(p / (1 - p)), rel=1e-9)


def test_label_logit_clips_with_epsilon():
    eps = 1e-6
    bound = math.log((1 - eps) / eps)
    assert label_logit(np.array([0.0, 500.0]), 1, eps) == pytest.approx(bound)
    assert label_logit(np.array([0.0, 500.0]), 0, eps) == pytest.approx(-bound)


def test_boolean_logit_is_the_raw_scalar():
    # Upstream Boolean logits are [0, z]; logit p(true) = z, logit p(false) = -z.
    z = 1.7
    assert label_logit(np.array([0.0, z]), 1) == pytest.approx(z)
    assert label_logit(np.array([0.0, z]), 0) == pytest.approx(-z)


def test_single_fact_effects_average_the_right_entries():
    rows = [{"i": 1, "j": 2, "d_i_j1": 1.0, "d_j_i1": 2.0}, {"i": 1, "j": 3, "d_i_j1": 3.0, "d_j_i1": 4.0},
            {"i": 2, "j": 3, "d_i_j1": 5.0, "d_j_i1": 6.0}]
    assert single_fact_effects(rows) == {"D1": 2.0, "D2": 3.5, "D3": 5.0}
