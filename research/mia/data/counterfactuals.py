"""Counterfactual fact variants for RIFT, generated from the structured fact space.

For facts Fi, Fj of a state x and replacement values Fi', Fj' (same role, different value):
  s11 = (Fi,  Fj )   original state
  s10 = (Fi,  Fj')   original Fi, replacement Fj
  s01 = (Fi', Fj )   replacement Fi, original Fj
  s00 = (Fi', Fj')
All other facts, the nuisance header, the template, the language, the question q,
the candidates A and the evaluated answer y stay fixed. Every variant is re-rendered
by schema.render_state, the same renderer that produced the original. No text deletion
and no LLM rewriting is involved.
"""
import copy
import itertools
import random

from mia.data import schema
from mia.data.generate_mia_data import derive_seed

PAIRS = ((1, 2), (1, 3), (2, 3))
GRID_CODES = ("11", "10", "01", "00")


def variant_facts(facts, overrides):
    unknown = set(overrides) - set(schema.FACT_NAMES)
    if unknown:
        raise ValueError(f"Unknown facts {sorted(unknown)}")
    out = dict(facts)
    out.update(overrides)
    schema.validate_facts(out)
    return out


def make_variant(record, overrides, variant_id):
    """Re-render ``record`` with some facts replaced. q, A and the evaluated gold y are kept."""
    meta = record["metadata"]
    facts = variant_facts(meta["facts"], overrides)
    rule = schema.rule_gold(facts)
    variant = {
        "id": variant_id,
        "state": schema.render_state(facts, meta["nuisance"], meta["template_key"], meta["language"]),
        "questions": copy.deepcopy(record["questions"]),
        "base_state_id": record["state_id"],
        "facts": facts,
        "evaluated_gold": dict(record["gold"]),
        # What the task rule would answer for the variant; analysis only, never the evaluated y.
        "variant_rule_gold": rule,
        "variant_rule_gold_in_candidates": {
            qid: (rule[qid] in q["criteria"]) if q["type"] == "choice" else True
            for qid, q in record["questions"].items()},
    }
    return variant


def replacement_pairs(record, pair, n, seed):
    """``n`` distinct (Fi', Fj') with Fi' != Fi and Fj' != Fj, deterministic in (seed, state, pair)."""
    i, j = pair
    fi, fj = schema.FACT_BY_INDEX[i], schema.FACT_BY_INDEX[j]
    facts = record["metadata"]["facts"]
    options = [(a, b) for a, b in itertools.product(schema.FACT_VALUES[fi], schema.FACT_VALUES[fj])
               if a != facts[fi] and b != facts[fj]]
    if n > len(options):
        raise ValueError(f"Only {len(options)} replacement pairs exist for F{i},F{j}")
    rng = random.Random(derive_seed(seed, "replacement", record["state_id"], i, j))
    return rng.sample(options, n)


def pair_grid(record, pair, replacement, replicate):
    """The four states s11, s10, s01, s00 for one (Fi', Fj') replacement."""
    i, j = pair
    fi, fj = schema.FACT_BY_INDEX[i], schema.FACT_BY_INDEX[j]
    vi, vj = replacement
    overrides = {"11": {}, "10": {fj: vj}, "01": {fi: vi}, "00": {fi: vi, fj: vj}}
    base = record["state_id"]
    return {code: make_variant(record, overrides[code], f"{base}|F{i}F{j}|r{replicate}|{code}")
            for code in GRID_CODES}


def all_grids(record, pairs=PAIRS, n_replacements=3, seed=0):
    """Every grid for one state, as {(pair, replicate): {"replacement":..., "grid": {...}}}."""
    out = {}
    for pair in pairs:
        for r, replacement in enumerate(replacement_pairs(record, pair, n_replacements, seed)):
            out[(pair, r)] = {"replacement": replacement, "grid": pair_grid(record, pair, replacement, r)}
    return out


def fact_key(facts):
    return tuple(facts[name] for name in schema.FACT_NAMES)
