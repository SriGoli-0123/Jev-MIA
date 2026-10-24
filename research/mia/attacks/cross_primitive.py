"""Cross-primitive membership transfer: train a state through one primitive, attack through another.

For training primitive P_train, member states contribute only their P_train question to SFT
(background states contribute all three so every primitive is still learned). Membership
still means "this state's P_train tuple was trained". At attack time the same state is
queried with P_query, whose question was never trained for that state unless P_query == P_train.
The diagonal of the 3x3 matrix is the same-primitive baseline.
"""
import math

from mia.data.schema import PRIMITIVES

XP_VARIANTS = {p: f"xp_{p}" for p in PRIMITIVES}


def variant_primitive(variant):
    if not variant.startswith("xp_") or variant[3:] not in PRIMITIVES:
        raise ValueError(f"{variant!r} is not a cross-primitive variant")
    return variant[3:]


def cell_kind(train_primitive, query_primitive):
    return "same-primitive (diagonal)" if train_primitive == query_primitive else "cross-primitive (off-diagonal)"


def transfer_matrix(cells, metric="roc_auc"):
    """``cells``: {(train, query): bootstrap summary}. Returns rows=train, cols=query (PRIMITIVES order)."""
    matrix = []
    for train in PRIMITIVES:
        row = []
        for query in PRIMITIVES:
            value = cells.get((train, query), {}).get("point", {}).get(metric)
            row.append(math.nan if value is None else float(value))
        matrix.append(row)
    return matrix
