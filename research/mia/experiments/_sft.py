"""Train a NanoJev victim on one membership variant with the unmodified upstream SFT entry point.

The SFT files are rebuilt from the dataset into the run directory, membership invariants are
checked before training, and after training the upstream checkpoint's recorded data hashes and
the on-disk SFT files are checked again. D_NM never appears in any file the trainer can read.
"""
import argparse
import json
from pathlib import Path

from mia.config import dump_config, load_config
from mia.data.splits import (VARIANTS, build_sft_rows, load_dataset, membership_spec, sft_state_ids,
                             write_sft_dir)
from mia.nanojev_bridge import run_logged, sft_command
from mia.paths import DEFAULT_ARTIFACTS
from mia.provenance import file_sha256, new_run_dir, utc_stamp, write_json, write_provenance


def verify_trained_membership(run_dir, splits, spec, sft_hashes):
    ckpt_config = json.loads((Path(run_dir) / "checkpoint" / "config.json").read_text(encoding="utf-8"))
    recorded = {Path(k).name: v for k, v in ckpt_config["data_sha256"].items()}
    ours = {Path(k).name: v for k, v in sft_hashes.items()}
    if recorded != ours:
        raise RuntimeError(f"Upstream trainer read different data than written: {recorded} vs {ours}")
    on_disk = sft_state_ids(Path(run_dir) / "sft")
    members = {sid for pool in spec["member_pools"] for sid in splits["pools"][pool]}
    nonmembers = {sid for pool in spec["nonmember_pools"] for sid in splits["pools"][pool]}
    if not members <= set(on_disk["train"]):
        raise RuntimeError("Some members are absent from the trained SFT split")
    if nonmembers & set().union(*map(set, on_disk.values())):
        raise RuntimeError("A non-member reached the SFT files")
    return {"members_in_train": len(members), "nonmembers_in_any_sft_file": 0,
            "train_states": len(on_disk["train"]), "dev_states": len(on_disk["dev"]),
            "test_states": len(on_disk["test"])}


def train_variant(*, data_dir, side, variant, role, sft_config, seed, run_dir, cpu_shim=False, dry_run=False,
                  init_checkpoint=None):
    data_dir = Path(data_dir).resolve()
    records, splits, manifest = load_dataset(data_dir)
    spec = membership_spec(side, variant)
    rows = build_sft_rows(records, splits, spec)
    sft_dir = Path(run_dir) / "sft"
    hashes = write_sft_dir(rows, sft_dir)
    n_train_questions = sum(len(r["questions"]) for r in rows["train"])
    checkpoint_dir = Path(run_dir) / "checkpoint"
    cmd = sft_command(sft_config, sft_dir, checkpoint_dir, seed, n_train_questions, cpu_shim=cpu_shim,
                      init_checkpoint=init_checkpoint)
    write_json(Path(run_dir) / "sft_command.json", {"command": cmd, "n_train_questions": n_train_questions,
                                                    "membership": spec})
    if dry_run:
        print("DRY RUN, not training:\n  " + " ".join(cmd))
        return None
    run_logged(cmd, Path(run_dir) / "train.log")
    verification = verify_trained_membership(run_dir, splits, spec, hashes)
    summary = json.loads((checkpoint_dir / "summary.json").read_text(encoding="utf-8"))
    meta = {
        "schema": "mia-run-meta-v1", "role": role, "side": spec["side"], "variant": variant, "seed": seed,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "weights_sha256": file_sha256(checkpoint_dir / "best.safetensors"),
        "base_model": sft_config["base_model"], "base_revision": sft_config.get("base_revision"),
        "cpu_shim": bool(cpu_shim), "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "data_dir": str(data_dir), "data_splits_sha256": manifest["files"]["splits.json"],
        "data_states_sha256": manifest["files"]["states.jsonl"],
        "sft_dir": str(sft_dir.resolve()), "sft_sha256": hashes, "n_train_questions": n_train_questions,
        "membership": spec, "membership_verification": verification, "command": cmd,
        "train_summary": {k: summary.get(k) for k in ("best_step", "best_dev_target_ce", "training_seconds",
                                                      "max_gpu_allocated_gb")},
        "frozen": True,
    }
    write_json(Path(run_dir) / "run_meta.json", meta)
    return meta


def main_for_role(role):
    """CLI shared by train_shadow.py and train_target.py."""
    parser = argparse.ArgumentParser(description=f"Train the {role} NanoJev victim with upstream SFT")
    parser.add_argument("--config", required=True, help=f"configs/{role}_sft.yaml")
    parser.add_argument("--seed", type=int, required=True, help="Upstream trainer seed")
    parser.add_argument("--data", required=True, help="Dataset directory from generate_mia_data.py")
    parser.add_argument("--variant", default="standard", choices=[v for v in VARIANTS if not v.startswith("pio")],
                        help="standard = all questions of D_M; xp_<primitive> = cross-primitive training")
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument("--run-name")
    parser.add_argument("--cpu-shim", action="store_true", help="Smoke tests only: run upstream SFT on CPU")
    parser.add_argument("--dry-run", action="store_true", help="Write SFT files and the command; do not train")
    args = parser.parse_args()
    config = load_config(args.config)
    cpu_shim = args.cpu_shim or bool(config.get("runtime", {}).get("cpu_shim", False))
    suffix = "_dryrun" if args.dry_run else ""
    name = args.run_name or f"{role}_{args.variant}_seed{args.seed}_{utc_stamp()}{suffix}"
    run_dir = new_run_dir(args.artifacts_root, "models", name)
    dump_config(config, run_dir / "config.yaml")
    write_provenance(run_dir, config=config, config_path=args.config, seed=args.seed,
                     extra={"data_dir": str(Path(args.data).resolve()), "variant": args.variant, "role": role})
    meta = train_variant(data_dir=args.data, side=role, variant=args.variant, role=role, sft_config=config,
                         seed=args.seed, run_dir=run_dir, cpu_shim=cpu_shim, dry_run=args.dry_run)
    print(f"{role} run: {run_dir}" + (f"\ncheckpoint: {meta['checkpoint_dir']}" if meta else ""))
