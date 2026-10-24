"""Required checks 1, 4 and 8: disjoint shadow/target states, grouped questions, deterministic data."""
import copy

import pytest

from mia.data import schema
from mia.data.generate_mia_data import generate
from mia.data.schema import QUESTION_IDS
from mia.data.splits import (build_sft_rows, check_disjoint, load_dataset, membership_spec, side_of_pool,
                             validate_sft_rows)


def test_shadow_and_target_state_ids_never_overlap(generated):
    _, splits = generated
    shadow = {sid for pool, ids in splits["pools"].items() if side_of_pool(pool) == "shadow" for sid in ids}
    target = {sid for pool, ids in splits["pools"].items() if side_of_pool(pool) == "target" for sid in ids}
    assert shadow and target
    assert not shadow & target


def test_every_state_belongs_to_exactly_one_pool(generated):
    records, splits = generated
    owner = check_disjoint(splits, records)
    assert set(owner) == set(records)


def test_check_disjoint_rejects_a_shared_state(generated):
    _, splits = generated
    broken = copy.deepcopy(splits)
    broken["pools"]["target_nonmember"].append(broken["pools"]["shadow_member"][0])
    with pytest.raises(ValueError, match="both"):
        check_disjoint(broken)


def test_rendered_states_are_unique(generated):
    records, _ = generated
    texts = [r["state"] for r in records.values()]
    assert len(texts) == len(set(texts))


@pytest.mark.parametrize("side", ["shadow", "target"])
def test_all_questions_of_a_state_stay_together(generated, side):
    records, splits = generated
    rows = build_sft_rows(records, splits, membership_spec(side, "standard"))
    seen = {}
    for split, split_rows in rows.items():
        for row in split_rows:
            assert row["state_id"] not in seen, "a state appears in two SFT splits"
            seen[row["state_id"]] = split
            assert list(row["questions"]) == list(QUESTION_IDS)
            assert set(row["gold"]) == set(QUESTION_IDS)


def test_validator_rejects_a_state_split_across_files(generated):
    records, splits = generated
    spec = membership_spec("shadow", "standard")
    rows = build_sft_rows(records, splits, spec)
    member = copy.deepcopy(rows["train"][0])
    member["split"] = "dev"
    rows["dev"].append(member)
    with pytest.raises(ValueError, match="appears in both"):
        validate_sft_rows(rows, splits, spec)


def test_validator_rejects_a_partial_member_outside_cross_primitive(generated):
    records, splits = generated
    spec = membership_spec("shadow", "standard")
    rows = build_sft_rows(records, splits, spec)
    rows["train"][0]["questions"].pop("route")
    with pytest.raises(ValueError, match="trained questions"):
        validate_sft_rows(rows, splits, spec)


def test_generation_is_deterministic(smoke_config, generated):
    records, splits = generated
    again_records, again_splits = generate(smoke_config, 1234)
    assert again_splits == splits
    assert again_records == records


def test_different_seed_changes_data(smoke_config, generated):
    records, _ = generated
    other, _ = generate(smoke_config, 4321)
    assert set(other) != set(records)


def test_written_dataset_round_trips_and_is_hash_checked(dataset_dir, generated):
    records, splits, manifest = load_dataset(dataset_dir)
    assert splits == generated[1]
    assert set(records) == set(generated[0])
    assert manifest["files"]["splits.json"]


def test_fact_combinations_are_balanced_within_each_pool(generated):
    """Balanced cycling: each of the 144 fact triples appears floor(n/144) or ceil(n/144) times."""
    import itertools
    from collections import Counter
    records, splits = generated
    combos = list(itertools.product(*(map(str, v) for v in schema.FACT_VALUES.values())))
    for pool in ("shadow_member", "shadow_nonmember", "target_member", "target_nonmember"):
        ids = splits["pools"][pool]
        counts = Counter(tuple(str(records[s]["metadata"]["facts"][k]) for k in schema.FACT_NAMES) for s in ids)
        low, high = len(ids) // len(combos), -(-len(ids) // len(combos))
        assert all(low <= counts.get(c, 0) <= high for c in combos)
