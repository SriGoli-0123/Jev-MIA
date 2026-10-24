"""Required check 5: counterfactual generation changes only the intended fact fields.
Also checks that our question/body builder reproduces upstream support_record exactly."""
import random

import pytest

from mia.data import schema
from mia.data.counterfactuals import PAIRS, all_grids, make_variant, pair_grid, replacement_pairs
from mia.paths import ensure_upstream_on_path

ensure_upstream_on_path()
import build_toy_decisions as upstream  # noqa: E402


def some_records(generated, n=12):
    records, splits = generated
    return [records[sid] for sid in splits["pools"]["shadow_member"][:n]]


def test_rerendering_original_facts_reproduces_the_state(generated):
    for record in some_records(generated):
        meta = record["metadata"]
        assert schema.render_state(meta["facts"], meta["nuisance"], meta["template_key"],
                                   meta["language"]) == record["state"]


@pytest.mark.parametrize("pair", PAIRS)
def test_grid_changes_only_the_intended_facts(generated, pair):
    i, j = pair
    fi, fj = schema.FACT_BY_INDEX[i], schema.FACT_BY_INDEX[j]
    for record in some_records(generated):
        original = record["metadata"]["facts"]
        for r, replacement in enumerate(replacement_pairs(record, pair, 3, seed=0)):
            grid = pair_grid(record, pair, replacement, r)
            expected_changed = {"11": set(), "10": {fj}, "01": {fi}, "00": {fi, fj}}
            for code, variant in grid.items():
                changed = {k for k in schema.FACT_NAMES if variant["facts"][k] != original[k]}
                assert changed == expected_changed[code]
                assert variant["questions"] == record["questions"]          # q and A fixed
                assert variant["evaluated_gold"] == record["gold"]          # y fixed
                header = schema.render_header(record["metadata"]["nuisance"], record["metadata"]["language"])
                assert variant["state"].startswith(header)                  # nuisance fixed
            assert grid["11"]["state"] == record["state"]
            assert grid["10"]["facts"][fi] == original[fi] and grid["10"]["facts"][fj] == replacement[1]
            assert grid["01"]["facts"][fi] == replacement[0] and grid["01"]["facts"][fj] == original[fj]


def test_only_fact_phrases_change_in_the_rendered_text(generated):
    record = some_records(generated, 1)[0]
    meta = record["metadata"]
    li = schema.language_index(meta["language"])
    new_value = next(v for v in schema.FACT_VALUES["refund_status"] if v != meta["facts"]["refund_status"])
    variant = make_variant(record, {"refund_status": new_value}, "v")
    old_phrase = upstream.REFUNDS[meta["facts"]["refund_status"]][li]
    new_phrase = upstream.REFUNDS[new_value][li]
    assert variant["state"] == record["state"].replace(old_phrase, new_phrase)


def test_replacements_differ_from_originals_and_are_deterministic(generated):
    for record in some_records(generated):
        for pair in PAIRS:
            fi, fj = schema.FACT_BY_INDEX[pair[0]], schema.FACT_BY_INDEX[pair[1]]
            reps = replacement_pairs(record, pair, 3, seed=5)
            assert reps == replacement_pairs(record, pair, 3, seed=5)
            assert len(set(reps)) == 3
            for a, b in reps:
                assert a != record["metadata"]["facts"][fi] and b != record["metadata"]["facts"][fj]
                assert a in schema.FACT_VALUES[fi] and b in schema.FACT_VALUES[fj]


def test_all_grids_shape(generated):
    record = some_records(generated, 1)[0]
    grids = all_grids(record, PAIRS, n_replacements=2, seed=0)
    assert len(grids) == 3 * 2
    assert all(set(item["grid"]) == {"11", "10", "01", "00"} for item in grids.values())


def test_invalid_fact_value_is_rejected(generated):
    record = some_records(generated, 1)[0]
    with pytest.raises(ValueError):
        make_variant(record, {"impact_level": 99}, "v")


@pytest.mark.parametrize("index", range(8))
def test_question_builder_matches_upstream_support_record(index):
    facts = ("billing", "approved", 2)
    for template_key in schema.TEMPLATE_KEYS:
        theirs = upstream.support_record(template_key, index, facts, random.Random(index))
        language = "zh" if index % 2 == 0 else "en"
        ours = schema.build_questions("billing", template_key, language, 2 + index % 4, random.Random(index))
        assert ours == theirs["questions"]
        named = dict(zip(schema.FACT_NAMES, facts))
        assert schema.render_body(named, template_key, language) == theirs["state"]
        assert schema.rule_gold(named) == theirs["gold"]
