"""RIFT: relational-interaction features from discrete mixed differences of f = logit p(y).

Pairwise (the only order used by the initial experiments):
  I_ij = f(s11) - f(s10) - f(s01) + f(s00)
which equals beta_ij (a - a')(b - b') for f = b0 + bi*a + bj*b + beta_ij*a*b and is 0
for any additive f. ``mixed_difference`` implements the general order-k inclusion-exclusion
so I_123 = f111 - f110 - f101 - f011 + f100 + f010 + f001 - f000 can be added later.
"""
import itertools

RIFT_FEATURES = ("I12", "I13", "I23")


def mixed_difference(values, order):
    """Order-k discrete mixed difference over a {0,1}^k grid.

    ``values`` maps k-tuples of 0/1 (1 = original value, 0 = replacement) to f.
    Sum over T of (-1)^(k - |T|) f(T), where T is the set of coordinates left original.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    total = 0.0
    for bits in itertools.product((0, 1), repeat=order):
        if bits not in values:
            raise KeyError(f"Missing grid point {bits}")
        total += (-1) ** (order - sum(bits)) * values[bits]
    return total


def pairwise_interaction(f11, f10, f01, f00):
    return mixed_difference({(1, 1): f11, (1, 0): f10, (0, 1): f01, (0, 0): f00}, 2)


def first_order_effects(f11, f10, f01, f00):
    """Effects of changing one fact while the other is held at its original or replacement value.

    d_j_i1 = f11 - f10  (change Fj, Fi original)    d_j_i0 = f01 - f00  (change Fj, Fi replaced)
    d_i_j1 = f11 - f01  (change Fi, Fj original)    d_i_j0 = f10 - f00  (change Fi, Fj replaced)
    I_ij = d_j_i1 - d_j_i0 = d_i_j1 - d_i_j0.
    """
    return {"d_j_i1": f11 - f10, "d_j_i0": f01 - f00, "d_i_j1": f11 - f01, "d_i_j0": f10 - f00}


def grid_features(f):
    """All saved per-grid quantities for one (pair, replacement): f values, first-order effects, I."""
    out = {f"f{code}": f[code] for code in ("11", "10", "01", "00")}
    out.update(first_order_effects(f["11"], f["10"], f["01"], f["00"]))
    out["I"] = pairwise_interaction(f["11"], f["10"], f["01"], f["00"])
    return out
