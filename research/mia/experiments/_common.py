"""Shared plumbing for experiment entry points: CLI, run directories, run metadata, feature cache."""
import argparse
import json
from pathlib import Path

import pandas as pd

from mia.config import dump_config, load_config
from mia.data.splits import load_dataset
from mia.paths import DEFAULT_ARTIFACTS
from mia.provenance import json_sha256, new_run_dir, utc_stamp, write_json, write_provenance

FEATURE_SCHEMA_VERSION = "mia-features-v1"


def base_parser(description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument("--run-name", help="Default: <tag>_seed<seed>_<UTC timestamp>")
    return parser


def start_run(args, kind, tag, config, extra=None):
    """Create a fresh run directory and record config, seed, git state and inputs."""
    name = args.run_name or f"{tag}_seed{args.seed}_{utc_stamp()}"
    run_dir = new_run_dir(args.artifacts_root, kind, name)
    dump_config({**config, "_cli_seed": args.seed}, run_dir / "config.yaml")
    write_provenance(run_dir, config=config, config_path=args.config, seed=args.seed, extra=extra)
    print(f"run directory: {run_dir}", flush=True)
    return run_dir


def load_experiment_config(path):
    return load_config(path)


def resolve_relative(config_path, value):
    """Resolve a path stored in a config relative to that config file."""
    path = Path(value)
    return path if path.is_absolute() else (Path(config_path).resolve().parent / path).resolve()


def load_run_meta(run_dir, expect_role=None, expect_variants=None):
    run_dir = Path(run_dir).resolve()
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    if expect_role and meta["role"] != expect_role:
        raise ValueError(f"{run_dir}: expected a {expect_role} run, got {meta['role']}")
    if expect_variants and meta["variant"] not in expect_variants:
        raise ValueError(f"{run_dir}: expected variant in {expect_variants}, got {meta['variant']}")
    if not Path(meta["checkpoint_dir"]).is_dir():
        raise FileNotFoundError(f"Checkpoint missing for {run_dir}")
    meta["run_dir"] = str(run_dir)
    return meta


def load_data_checked(data_dir, *metas):
    records, splits, manifest = load_dataset(data_dir)
    for meta in metas:
        if meta["data_splits_sha256"] != manifest["files"]["splits.json"]:
            raise ValueError(f"Run {meta['run_dir']} was trained on a different dataset than {data_dir}")
    return records, splits, manifest


def pool_records(records, splits, pools):
    ids = sorted({sid for pool in pools for sid in splits["pools"][pool]})
    return [records[sid] for sid in ids]


def run_summary(meta):
    keys = ("run_dir", "role", "variant", "seed", "checkpoint_dir", "weights_sha256", "cpu_shim")
    return {k: meta.get(k) for k in keys}


def save_table(frame, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_dataset_ids(run_dir, splits, pools):
    write_json(Path(run_dir) / "dataset_ids.json", {"splits_seed": splits["seed"],
                                                    "pools": {p: splits["pools"][p] for p in sorted(set(pools))}})


class FeatureStore:
    """Extract features from frozen checkpoints, cached by (weights, states, settings)."""

    def __init__(self, artifacts_root, device="auto"):
        self.cache_dir = Path(artifacts_root).expanduser().resolve() / "feature_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._scorers = {}

    def _scorer(self, meta, extraction):
        from mia.nanojev_bridge import DecisionScorer
        key = (meta["checkpoint_dir"], extraction.get("precision", "fp32"))
        if key not in self._scorers:
            self._scorers.clear()  # keep at most one model in memory
            self._scorers[key] = DecisionScorer(
                meta["checkpoint_dir"], device=self.device, precision=extraction.get("precision", "fp32"),
                microbatch_questions=extraction.get("microbatch_questions", 32),
                max_microbatch_tokens=extraction.get("max_microbatch_tokens", 65536))
        return self._scorers[key]

    def _cached(self, key_obj, names, compute):
        key = json_sha256(key_obj)
        paths = [self.cache_dir / f"{key}.{name}.csv.gz" for name in names]
        if all(p.exists() for p in paths):
            return [pd.read_csv(p, dtype={"state_id": str, "qid": str}) for p in paths]
        frames = compute()
        for frame, path in zip(frames, paths):
            frame.to_csv(path, index=False)
        (self.cache_dir / f"{key}.key.json").write_text(json.dumps(key_obj, indent=2, default=str) + "\n")
        return frames

    def _base_key(self, kind, meta, records, extraction):
        return {"schema": FEATURE_SCHEMA_VERSION, "kind": kind, "weights_sha256": meta["weights_sha256"],
                "states_sha256": json_sha256([[r["id"], r["state"]] for r in records]),
                "precision": extraction.get("precision", "fp32"), "eps": extraction.get("eps", 1e-6)}

    def confidence(self, meta, records, extraction):
        from mia.attacks.extraction import extract_confidence

        def compute():
            return [extract_confidence(self._scorer(meta, extraction), records, eps=extraction.get("eps", 1e-6))]
        return self._cached(self._base_key("confidence", meta, records, extraction), ["confidence"], compute)[0]

    def counterfactual(self, meta, records, extraction, counterfactual):
        from mia.attacks.extraction import extract_counterfactual
        pairs = [tuple(p) for p in counterfactual["pairs"]]
        key = {**self._base_key("counterfactual", meta, records, extraction), "pairs": pairs,
               "n_replacements": counterfactual["n_replacements"],
               "replacement_seed": counterfactual["replacement_seed"]}

        def compute():
            return list(extract_counterfactual(self._scorer(meta, extraction), records, pairs=pairs,
                                               n_replacements=counterfactual["n_replacements"],
                                               seed=counterfactual["replacement_seed"],
                                               eps=extraction.get("eps", 1e-6),
                                               chunk=counterfactual.get("chunk_states", 32)))
        wide, long = self._cached(key, ["rift_wide", "rift_long"], compute)
        return wide, long

    def noise_floor(self, meta, records, extraction):
        from mia.attacks.extraction import numerical_noise_floor
        return numerical_noise_floor(self._scorer(meta, extraction), records, eps=extraction.get("eps", 1e-6))


def merged_features(store, meta, records, extraction, counterfactual=None):
    """Confidence features, plus RIFT/first-order features when ``counterfactual`` is given."""
    conf = store.confidence(meta, records, extraction)
    if counterfactual is None:
        return conf, None
    wide, long = store.counterfactual(meta, records, extraction, counterfactual)
    merged = conf.merge(wide, on=["state_id", "qid", "primitive"], how="inner", validate="one_to_one")
    if len(merged) != len(conf):
        raise RuntimeError("Confidence and counterfactual features do not cover the same candidates")
    return merged, long
