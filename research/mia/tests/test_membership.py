"""Required checks 2, 3 and 9: members are trained, non-members never are, and the attacker never
sees target membership labels."""
import numpy as np
import pandas as pd
import pytest

from mia.attacks.classifier import AttackClassifier, TargetLabelLeak
from mia.attacks.pipeline import attack_pipeline, evaluate_scores
from mia.data.splits import (VARIANTS, build_sft_rows, membership_labels, membership_spec, sft_state_ids,
                             write_sft_dir)
from mia.nanojev_bridge import resolve_steps, sft_command

ALL_VARIANTS = [("shadow", v) for v in VARIANTS] + [("target", v) for v in VARIANTS if not v.startswith("pio")]


@pytest.mark.parametrize("side,variant", ALL_VARIANTS)
def test_members_are_in_the_sft_train_split(generated, side, variant):
    records, splits = generated
    spec = membership_spec(side, variant)
    rows = build_sft_rows(records, splits, spec)
    train_ids = {r["state_id"] for r in rows["train"]}
    for pool in spec["member_pools"]:
        assert set(splits["pools"][pool]) <= train_ids


@pytest.mark.parametrize("side,variant", ALL_VARIANTS)
def test_nonmembers_never_appear_in_any_sft_file(generated, side, variant):
    records, splits = generated
    spec = membership_spec(side, variant)
    rows = build_sft_rows(records, splits, spec)
    everything = {r["state_id"] for split_rows in rows.values() for r in split_rows}
    for pool in spec["nonmember_pools"]:
        assert not set(splits["pools"][pool]) & everything


def test_shadow_sft_never_contains_target_states(generated):
    records, splits = generated
    rows = build_sft_rows(records, splits, membership_spec("shadow", "standard"))
    target = {sid for pool, ids in splits["pools"].items() if pool.startswith("target_") for sid in ids}
    assert not target & {r["state_id"] for split_rows in rows.values() for r in split_rows}


def test_written_sft_dir_is_accepted_by_upstream_and_keeps_membership(generated, tmp_path):
    records, splits = generated
    spec = membership_spec("shadow", "standard")
    write_sft_dir(build_sft_rows(records, splits, spec), tmp_path / "sft")  # runs upstream validation too
    on_disk = sft_state_ids(tmp_path / "sft")
    assert set(splits["pools"]["shadow_member"]) == set(on_disk["train"])
    assert not set(splits["pools"]["shadow_nonmember"]) & set().union(*map(set, on_disk.values()))


def test_membership_labels_cover_members_and_nonmembers_only(generated):
    _, splits = generated
    labels = membership_labels(splits, membership_spec("target", "standard"))
    assert {s for s, m in labels.items() if m == 1} == set(splits["pools"]["target_member"])
    assert {s for s, m in labels.items() if m == 0} == set(splits["pools"]["target_nonmember"])


def test_sft_command_uses_the_unmodified_upstream_entrypoint(tmp_path):
    config = {"base_model": "Qwen/Qwen3-0.6B-Base", "base_revision": "rev",
              "trainer": {"epochs": 2, "batch_questions": 12, "objective": "gold_distribution", "precision": "bf16"}}
    cmd = sft_command(config, tmp_path / "sft", tmp_path / "ckpt", 5, n_train_questions=300)
    assert cmd[1].endswith("scripts/train_pipeline_decisions.py")
    assert cmd[cmd.index("--steps") + 1] == str(resolve_steps(config["trainer"], 300)) == "50"
    assert cmd[cmd.index("--input") + 1] == str(tmp_path / "sft")


# ---------- check 9: attack training never sees target membership labels ----------

def _tables(rng, n=200):
    def table(role, shift):
        y = np.repeat([1, 0], n // 2)
        frame = pd.DataFrame({"state_id": [f"{role}-{i}" for i in range(n)], "qid": "route", "primitive": "choice",
                              "p_max": rng.normal(0.8 + shift * y, 0.1), "h": rng.normal(0.5 + shift * y, 0.1),
                              "model_role": role})
        return frame, dict(zip(frame["state_id"], y))
    shadow, shadow_labels = table("shadow", 0.05)
    shadow["member"] = shadow["state_id"].map(shadow_labels)
    target, target_labels = table("target", 0.05)
    return shadow, target, target_labels


def test_fitting_on_target_rows_is_refused():
    shadow, target, labels = _tables(np.random.default_rng(0))
    target = target.assign(member=target["state_id"].map(labels))
    with pytest.raises(TargetLabelLeak):
        AttackClassifier(["p_max", "h"]).fit(target)
    with pytest.raises(TargetLabelLeak):
        AttackClassifier(["p_max", "h"]).fit(pd.concat([shadow, target]))


def test_pipeline_refuses_a_labeled_target_table():
    shadow, target, labels = _tables(np.random.default_rng(1))
    with pytest.raises(TargetLabelLeak):
        attack_pipeline(shadow, target.assign(member=target["state_id"].map(labels)), ["p_max", "h"])


def test_target_labels_cannot_influence_the_attacker_or_its_scores():
    shadow, target, labels = _tables(np.random.default_rng(2))
    clf_a, scored_a, _ = attack_pipeline(shadow, target, ["p_max", "h"], seed=3)
    shuffled = dict(zip(labels, np.random.default_rng(9).permutation(list(labels.values()))))
    clf_b, scored_b, _ = attack_pipeline(shadow, target, ["p_max", "h"], seed=3)
    assert np.array_equal(clf_a.model.coef_, clf_b.model.coef_)
    assert np.array_equal(scored_a["score"], scored_b["score"])
    # Labels only change the evaluation, which happens after the attacker is frozen.
    assert clf_a.frozen
    with pytest.raises(RuntimeError, match="frozen"):
        clf_a.fit(shadow)
    real = evaluate_scores(scored_a, labels, n_boot=10)["point"]["roc_auc"]
    fake = evaluate_scores(scored_a, shuffled, n_boot=10)["point"]["roc_auc"]
    assert real != fake
