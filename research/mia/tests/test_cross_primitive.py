"""Cross-primitive variants: members train exactly one primitive, membership still refers to the state."""
import math

import pytest

from mia.attacks.cross_primitive import XP_VARIANTS, cell_kind, transfer_matrix, variant_primitive
from mia.data.schema import PRIMITIVE_QID, PRIMITIVES, QUESTION_IDS
from mia.data.splits import build_sft_rows, membership_labels, membership_spec


@pytest.mark.parametrize("side", ["shadow", "target"])
@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_member_states_contribute_only_the_training_primitive(generated, side, primitive):
    records, splits = generated
    spec = membership_spec(side, XP_VARIANTS[primitive])
    rows = build_sft_rows(records, splits, spec)
    members = set(splits["pools"][f"{side}_member"])
    background = set(splits["pools"][f"xp_{side}_background"])
    for row in rows["train"]:
        if row["state_id"] in members:
            assert list(row["questions"]) == [PRIMITIVE_QID[primitive]]
            assert list(row["gold"]) == [PRIMITIVE_QID[primitive]]
        else:
            assert row["state_id"] in background
            assert list(row["questions"]) == list(QUESTION_IDS)
    assert members <= {r["state_id"] for r in rows["train"]}


@pytest.mark.parametrize("side", ["shadow", "target"])
def test_membership_labels_are_state_level_and_identical_across_training_primitives(generated, side):
    _, splits = generated
    labels = [membership_labels(splits, membership_spec(side, XP_VARIANTS[p])) for p in PRIMITIVES]
    assert labels[0] == labels[1] == labels[2]
    assert labels[0] == membership_labels(splits, membership_spec(side, "standard"))


def test_background_pools_are_disjoint_from_attack_pools(generated):
    _, splits = generated
    for side in ("shadow", "target"):
        background = set(splits["pools"][f"xp_{side}_background"])
        for role in ("member", "nonmember", "dev", "test"):
            assert not background & set(splits["pools"][f"{side}_{role}"])


def test_variant_names_and_cells():
    assert [variant_primitive(XP_VARIANTS[p]) for p in PRIMITIVES] == list(PRIMITIVES)
    with pytest.raises(ValueError):
        variant_primitive("standard")
    assert cell_kind("choice", "choice").startswith("same")
    assert cell_kind("choice", "score").startswith("cross")


def test_transfer_matrix_layout():
    cells = {(t, q): {"point": {"roc_auc": 0.5 + 0.1 * i + 0.01 * j}}
             for i, t in enumerate(PRIMITIVES) for j, q in enumerate(PRIMITIVES)}
    matrix = transfer_matrix(cells)
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    assert matrix[1][2] == pytest.approx(0.62)
    partial = transfer_matrix({("boolean", "boolean"): {"point": {"roc_auc": 0.7}}})
    assert partial[0][0] == 0.7 and math.isnan(partial[2][2])
