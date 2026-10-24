#!/usr/bin/env python3
"""Generate disjoint, deterministic shadow/target membership pools of structured support states.

Each record stores its latent facts (F1 main_topic, F2 refund_status, F3 impact_level),
nuisance fields, template and language *before* rendering, plus the rendered state s,
the three questions q with candidates A, and gold answers y. No LLM is involved.

Usage:
  python research/mia/data/generate_mia_data.py --config research/mia/configs/data.yaml --seed 20261024
"""
import argparse
import copy
import hashlib
import itertools
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mia.config import dump_config, load_config  # noqa: E402
from mia.data import schema  # noqa: E402
from mia.data.splits import (POOL_PATTERN, SIDES, build_sft_rows, check_disjoint, jsonl_text,  # noqa: E402
                             membership_spec, write_sft_dir)
from mia.paths import DEFAULT_ARTIFACTS  # noqa: E402
from mia.provenance import file_sha256, git_info, json_sha256, write_json  # noqa: E402

GENERATOR_VERSION = "mia.data.generate_mia_data/v1"


def derive_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def state_id(seed, pool, index):
    # Neutral IDs: a hash, so neither the id nor file order reveals the pool.
    return "mia-" + hashlib.sha256(f"{seed}|{pool}|{index}".encode()).hexdigest()[:16]


def balanced_sequence(values, n, rng):
    """Concatenated shuffled copies of ``values``: every value appears floor/ceil(n/len) times."""
    out = []
    while len(out) < n:
        block = list(values)
        rng.shuffle(block)
        out.extend(block)
    return out[:n]


def parse_pools(config):
    pools, seen = [], set()
    for entry in config["pools"]:
        name = entry["name"]
        if not POOL_PATTERN.match(name):
            raise ValueError(f"Unknown pool name {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate pool {name}")
        if ("size" in entry) == ("matched_to" in entry):
            raise ValueError(f"Pool {name} needs exactly one of size / matched_to")
        if "matched_to" in entry and entry["matched_to"] not in seen:
            raise ValueError(f"Pool {name} must be listed after {entry['matched_to']}")
        if "size" in entry and (type(entry["size"]) is not int or entry["size"] < 0):
            raise ValueError(f"Pool {name} size must be a non-negative integer")
        seen.add(name)
        pools.append(dict(entry))
    for side in SIDES:
        for role in ("member", "nonmember", "dev", "test"):
            if f"{side}_{role}" not in seen:
                raise ValueError(f"Missing required pool {side}_{role}")
    return pools


def apply_label_noise(record, rate, rng):
    """Positive-control knob: flip a question's label to another valid answer with probability ``rate``."""
    flipped = {}
    for qid, question in record["questions"].items():
        flip = rate > 0 and rng.random() < rate
        flipped[qid] = flip
        if not flip:
            continue
        gold = record["gold"][qid]
        if question["type"] == "boolean":
            record["gold"][qid] = not gold
        elif question["type"] == "choice":
            record["gold"][qid] = rng.choice([k for k in question["criteria"] if k != gold])
        else:
            record["gold"][qid] = rng.choice([i for i in range(len(question["criteria"])) if i != gold])
    record["metadata"]["label_noise"] = flipped
    record["gold_label_kind"] = {qid: ("unspecified_compatibility_label" if flipped[qid] else "deterministic_truth")
                                 for qid in record["questions"]}


def make_record(sid, facts, nuisance, template_key, language, questions):
    return {
        "id": sid, "state_id": sid, "family_id": schema.FAMILY_ID,
        "state": schema.render_state(facts, nuisance, template_key, language),
        "questions": questions,
        "gold": schema.rule_gold(facts),
        "metadata": {"source": "self_authored_programmatic", "license": "CC0-1.0", "source_group_id": sid,
                     "template_id": f"support_mia:{template_key}:{language}", "template_key": template_key,
                     "language": language, "facts": dict(facts), "nuisance": dict(nuisance),
                     "choice_k": len(questions["route"]["criteria"]), "rule_gold": schema.rule_gold(facts),
                     "generator": GENERATOR_VERSION},
    }


def generate(config, seed):
    pools = parse_pools(config)
    languages = list(config.get("languages", schema.LANGUAGES))
    templates = list(config.get("templates", schema.TEMPLATE_KEYS))
    k_min, k_max = config.get("choice_k", {}).get("min", 2), config.get("choice_k", {}).get("max", 5)
    if not 2 <= k_min <= k_max <= len(schema.FACT_VALUES["main_topic"]):
        raise ValueError("choice_k must satisfy 2 <= min <= max <= number of topics")
    noise = float(config.get("label_noise_rate", 0.0))
    if not 0 <= noise < 1:
        raise ValueError("label_noise_rate must be in [0, 1)")
    sizes = {}
    for entry in pools:
        sizes[entry["name"]] = entry["size"] if "size" in entry else sizes[entry["matched_to"]]
    total = sum(sizes.values())
    lo, hi = schema.TICKET_RANGE
    tickets = iter(random.Random(derive_seed(seed, "tickets")).sample(range(lo, hi + 1), total))
    combos = [dict(zip(schema.FACT_NAMES, values)) for values in itertools.product(*schema.FACT_VALUES.values())]
    records, pool_ids, matches, texts = {}, {}, {}, set()
    for entry in pools:
        name, n = entry["name"], sizes[entry["name"]]
        ids = []
        if "matched_to" in entry:
            source_ids = pool_ids[entry["matched_to"]]
            for index, source_id in enumerate(source_ids):
                rng = random.Random(derive_seed(seed, "state", name, index))
                src = records[source_id]
                meta = src["metadata"]
                nuisance = schema.draw_nuisance(rng, next(tickets))
                while (nuisance["first_name"], nuisance["last_name"]) == (meta["nuisance"]["first_name"],
                                                                           meta["nuisance"]["last_name"]):
                    nuisance = schema.draw_nuisance(rng, nuisance["ticket"])
                sid = state_id(seed, name, index)
                # Same facts, template, language, questions (q, A) and gold; only nuisance differs.
                record = make_record(sid, meta["facts"], nuisance, meta["template_key"], meta["language"],
                                     copy.deepcopy(src["questions"]))
                record["gold"] = dict(src["gold"])
                record["metadata"]["label_noise"] = dict(meta.get("label_noise", {}))
                if "gold_label_kind" in src:
                    record["gold_label_kind"] = dict(src["gold_label_kind"])
                records[sid] = record
                matches[sid] = source_id
                ids.append(sid)
        else:
            pool_rng = random.Random(derive_seed(seed, "pool", name))
            facts_seq = balanced_sequence(combos, n, pool_rng)
            lang_seq = balanced_sequence(languages, n, pool_rng)
            template_seq = balanced_sequence(templates, n, pool_rng)
            for index in range(n):
                rng = random.Random(derive_seed(seed, "state", name, index))
                facts, language, template_key = facts_seq[index], lang_seq[index], template_seq[index]
                questions = schema.build_questions(facts["main_topic"], template_key, language,
                                                   rng.randint(k_min, k_max), rng)
                nuisance = schema.draw_nuisance(rng, next(tickets))
                sid = state_id(seed, name, index)
                record = make_record(sid, facts, nuisance, template_key, language, questions)
                apply_label_noise(record, noise, random.Random(derive_seed(seed, "noise", name, index)))
                records[sid] = record
                ids.append(sid)
        for sid in ids:
            if records[sid]["state"] in texts:
                raise ValueError(f"Rendered state collision for {sid}")
            texts.add(records[sid]["state"])
        pool_ids[name] = sorted(ids)
    splits = {"schema": "mia-splits-v1", "seed": seed, "pools": pool_ids, "matches": matches}
    check_disjoint(splits, records)
    return records, splits


def pool_summary(records, ids):
    rows = [records[sid] for sid in ids]
    summary = {"states": len(rows), "questions": sum(len(r["questions"]) for r in rows)}
    for name in schema.FACT_NAMES:
        summary[f"facts.{name}"] = dict(sorted(Counter(str(r["metadata"]["facts"][name]) for r in rows).items()))
    summary["language"] = dict(Counter(r["metadata"]["language"] for r in rows))
    summary["template_key"] = dict(Counter(r["metadata"]["template_key"] for r in rows))
    summary["choice_k"] = dict(sorted(Counter(r["metadata"]["choice_k"] for r in rows).items()))
    summary["boolean_true"] = sum(bool(r["gold"]["completed"]) for r in rows)
    summary["noisy_labels"] = sum(sum(r["metadata"].get("label_noise", {}).values()) for r in rows)
    return summary


def write_dataset(config, seed, out_dir, config_path=None):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset {out_dir}")
    records, splits = generate(config, seed)
    out_dir.mkdir(parents=True)
    ordered = [records[sid] for sid in sorted(records)]
    (out_dir / "states.jsonl").write_text(jsonl_text(ordered), encoding="utf-8")
    write_json(out_dir / "splits.json", splits)
    dump_config({**config, "seed": seed}, out_dir / "config.yaml")
    sft = {}
    for side in SIDES:
        spec = membership_spec(side, "standard")
        rows = build_sft_rows(records, splits, spec)
        sft[f"{side}_standard"] = write_sft_dir(rows, out_dir / "sft" / f"{side}_standard")
    manifest = {
        "schema": "mia-dataset-v1", "generator": GENERATOR_VERSION, "seed": seed,
        "config_path": str(config_path) if config_path else None, "config_sha256": json_sha256(config),
        "files": {"states.jsonl": file_sha256(out_dir / "states.jsonl"),
                  "splits.json": file_sha256(out_dir / "splits.json")},
        "sft_dirs": sft,
        "pools": {name: pool_summary(records, ids) for name, ids in splits["pools"].items()},
        "fact_schema": {"F1": "main_topic", "F2": "refund_status", "F3": "impact_level",
                        "values": {k: list(v) for k, v in schema.FACT_VALUES.items()}},
        "git": git_info(),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", help="Output directory (default: <artifacts-root>/data/<config-stem>_seed<seed>)")
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS))
    args = parser.parse_args()
    config = load_config(args.config)
    out = Path(args.out) if args.out else Path(args.artifacts_root) / "data" / f"{Path(args.config).stem}_seed{args.seed}"
    manifest = write_dataset(config, args.seed, out, args.config)
    print(f"wrote {out}")
    for name, info in manifest["pools"].items():
        print(f"  {name:24s} {info['states']:6d} states  {info['questions']:6d} questions")


if __name__ == "__main__":
    main()
