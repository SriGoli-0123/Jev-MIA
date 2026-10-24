"""State-level pools, membership definitions and NanoJev SFT dataset writing.

A pool is a set of whole states. Every question of a state lives wherever the state
lives; the only exception is the cross-primitive variant, which deliberately trains a
member state through one primitive only (and still never splits it across pools).

Membership labels are kept in splits.json, never inside state records, so feature
extraction code can read records without seeing which pool a state belongs to.
"""
import copy
import json
import re
from pathlib import Path

from mia.data.schema import PRIMITIVE_QID, PRIMITIVES, QUESTION_IDS
from mia.paths import ensure_upstream_on_path
from mia.provenance import file_sha256

SIDES = ("shadow", "target")
SIDE_ROLES = ("member", "nonmember", "dev", "test")
VARIANTS = ("standard", *(f"xp_{p}" for p in PRIMITIVES), "pio_in", "pio_out")
POOL_PATTERN = re.compile(r"^(?:(shadow|target)_(member|nonmember|dev|test)|xp_(shadow|target)_background"
                          r"|pio_(base|canary|twin|control))$")
SFT_SPLITS = ("train", "dev", "test")


def side_pool(side, role):
    if side not in SIDES or role not in SIDE_ROLES:
        raise ValueError(f"Unknown side/role {side}/{role}")
    return f"{side}_{role}"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def jsonl_text(rows):
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    records = {}
    for row in read_jsonl(data_dir / "states.jsonl"):
        if row["id"] in records:
            raise ValueError(f"Duplicate state id {row['id']}")
        records[row["id"]] = row
    splits = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    for name in ("states.jsonl", "splits.json"):
        if file_sha256(data_dir / name) != manifest["files"][name]:
            raise ValueError(f"{name} does not match its manifest hash; regenerate instead of editing data")
    check_disjoint(splits, records)
    return records, splits, manifest


def check_disjoint(splits, records=None):
    """Every state belongs to exactly one pool; in particular shadow and target never share a state."""
    owner = {}
    for pool, ids in splits["pools"].items():
        if not POOL_PATTERN.match(pool):
            raise ValueError(f"Unknown pool name {pool!r}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Pool {pool} lists a state twice")
        for sid in ids:
            if sid in owner:
                raise ValueError(f"State {sid} is in both {owner[sid]} and {pool}")
            owner[sid] = pool
    if records is not None:
        missing = set(owner) - set(records)
        unassigned = set(records) - set(owner)
        if missing or unassigned:
            raise ValueError(f"{len(missing)} pooled ids lack records; {len(unassigned)} records lack a pool")
    return owner


def side_of_pool(pool):
    match = POOL_PATTERN.match(pool)
    if match is None:
        raise ValueError(pool)
    if match.group(1):
        return match.group(1)
    if match.group(3):
        return match.group(3)
    return "pio"


def membership_spec(side, variant):
    """Which pools train the model, and which define members and non-members for the attack."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")
    all_qids = list(QUESTION_IDS)
    if variant in ("pio_in", "pio_out"):
        trained, held_out = ("pio_canary", "pio_twin") if variant == "pio_in" else ("pio_twin", "pio_canary")
        return {"side": "pio", "variant": variant,
                "train": [{"pool": "pio_base", "qids": all_qids}, {"pool": trained, "qids": all_qids}],
                "dev": ["shadow_dev"], "test": ["shadow_test"],
                "member_pools": [trained], "nonmember_pools": [held_out, "pio_control"], "member_qids": all_qids}
    if side not in SIDES:
        raise ValueError(f"Unknown side {side!r}")
    members, nonmembers = side_pool(side, "member"), side_pool(side, "nonmember")
    spec = {"side": side, "variant": variant, "dev": [side_pool(side, "dev")], "test": [side_pool(side, "test")],
            "member_pools": [members], "nonmember_pools": [nonmembers]}
    if variant == "standard":
        spec.update(train=[{"pool": members, "qids": all_qids}], member_qids=all_qids)
    else:
        primitive = variant[len("xp_"):]
        qid = PRIMITIVE_QID[primitive]
        # Background states (all primitives) keep every primitive trained; members contribute one primitive.
        spec.update(train=[{"pool": f"xp_{side}_background", "qids": all_qids}, {"pool": members, "qids": [qid]}],
                    member_qids=[qid], member_primitive=primitive)
    return spec


def membership_labels(splits, spec):
    labels = {}
    for pool in spec["member_pools"]:
        labels.update({sid: 1 for sid in splits["pools"][pool]})
    for pool in spec["nonmember_pools"]:
        for sid in splits["pools"][pool]:
            if sid in labels:
                raise ValueError(f"{sid} is both member and non-member")
            labels[sid] = 0
    return labels


def restrict_questions(record, qids, split):
    row = copy.deepcopy(record)
    row["split"] = split
    row["questions"] = {q: row["questions"][q] for q in qids}
    for key in ("gold", "gold_label_kind"):
        if isinstance(row.get(key), dict):
            row[key] = {q: row[key][q] for q in qids}
    return row


def build_sft_rows(records, splits, spec):
    rows = {"train": [], "dev": [], "test": []}
    for entry in spec["train"]:
        for sid in splits["pools"][entry["pool"]]:
            rows["train"].append(restrict_questions(records[sid], entry["qids"], "train"))
    for split in ("dev", "test"):
        for pool in spec[split]:
            for sid in splits["pools"][pool]:
                rows[split].append(restrict_questions(records[sid], QUESTION_IDS, split))
    validate_sft_rows(rows, splits, spec)
    return rows


def validate_sft_rows(rows, splits, spec):
    """Membership invariants checked before any training run."""
    where = {}
    for split, split_rows in rows.items():
        for row in split_rows:
            sid = row["state_id"]
            if row["id"] != sid or row["metadata"]["source_group_id"] != sid:
                raise ValueError(f"{sid}: id, state_id and source_group_id must agree")
            if sid in where:
                raise ValueError(f"State {sid} appears in both {where[sid]} and {split}")
            where[sid] = split
    members = {sid for pool in spec["member_pools"] for sid in splits["pools"][pool]}
    nonmembers = {sid for pool in spec["nonmember_pools"] for sid in splits["pools"][pool]}
    not_trained = [sid for sid in members if where.get(sid) != "train"]
    if not_trained:
        raise ValueError(f"{len(not_trained)} member states are missing from the SFT train split")
    leaked = sorted(nonmembers & set(where))
    if leaked:
        raise ValueError(f"{len(leaked)} non-member states appear in SFT data (e.g. {leaked[0]})")
    for row in rows["train"]:
        expected = spec["member_qids"] if row["state_id"] in members else list(QUESTION_IDS)
        if list(row["questions"]) != list(expected):
            raise ValueError(f"{row['state_id']}: trained questions {list(row['questions'])} != {expected}")
    for split in ("dev", "test"):
        for row in rows[split]:
            if list(row["questions"]) != list(QUESTION_IDS):
                raise ValueError(f"{row['state_id']}: {split} rows must keep all questions together")


def write_sft_dir(rows, out_dir):
    """Write train/dev/test JSONL in the layout upstream read_training_records expects."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split in SFT_SPLITS:
        path = out_dir / f"{split}.jsonl"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.write_text(jsonl_text(rows[split]), encoding="utf-8")
        hashes[str(path)] = file_sha256(path)
    upstream_validate_sft_dir(out_dir)
    return hashes


def upstream_validate_sft_dir(out_dir):
    """Run NanoJev's own stdlib schema/split validator (same as ``--validate-only``)."""
    ensure_upstream_on_path()
    from train_pipeline_decisions import data_schema_summary, read_training_records
    records, _ = read_training_records(out_dir)
    return data_schema_summary(records)


def sft_state_ids(sft_dir):
    out = {}
    for split in SFT_SPLITS:
        path = Path(sft_dir) / f"{split}.jsonl"
        out[split] = {row["state_id"]: list(row["questions"]) for row in read_jsonl(path)}
    return out
