#!/usr/bin/env python3
"""Paired IN/OUT validation of the RIFT hypothesis (causal sanity check, not a realistic attack).

For each training seed s, two models start from the same theta_0 with identical hyperparameters
and the same number of training states:
  theta_IN(s)  = SFT(pio_base + pio_canary)
  theta_OUT(s) = SFT(pio_base + pio_twin)
Each twin x' matches its canary x in facts, template, language, q, A and y (only nuisance differs).
So every canary has an IN model (theta_IN) and an OUT model (theta_OUT), and every twin the reverse.
pio_control states are trained by neither model; their IN-minus-OUT differences measure how much
two independently trained models differ anyway (the noise floor).

Reported per primitive, pooled over seeds:
  confidence_IN - confidence_OUT and I_IN,ij - I_OUT,ij for trained states vs. control states;
  the same restricted to candidates whose confidence barely moved (|delta logit p(y)| < tol),
  which is the central question: does membership change interactions when confidence does not?

  python research/mia/experiments/run_paired_in_out.py --config research/mia/configs/paired_in_out.yaml \
      --seed 0 --data <data_dir> --train-seeds 101 102 103
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

from mia.attacks.first_order import FIRST_ORDER_FEATURES  # noqa: E402
from mia.attacks.rift import RIFT_FEATURES  # noqa: E402
from mia.config import dump_config, load_config  # noqa: E402
from mia.data.schema import PRIMITIVES  # noqa: E402
from mia.experiments._common import (FeatureStore, base_parser, load_data_checked, load_experiment_config,  # noqa: E402
                                     load_run_meta, merged_features, pool_records, resolve_relative, run_summary,
                                     save_dataset_ids, save_table, start_run)
from mia.experiments._sft import train_variant  # noqa: E402
from mia.provenance import write_json  # noqa: E402

CONFIDENCE_METRICS = ("p_y", "logit_y", "p_max", "h", "loss")
METRICS = CONFIDENCE_METRICS + tuple(RIFT_FEATURES) + tuple(FIRST_ORDER_FEATURES)
POOLS = ("pio_canary", "pio_twin", "pio_control")


def train_pairs(args, data_dir, run_dir, sft_config, cpu_shim):
    pairs = {}
    for seed in args.train_seeds:
        pairs[seed] = {}
        for variant in ("pio_in", "pio_out"):
            model_dir = run_dir / "models" / f"{variant}_seed{seed}"
            model_dir.mkdir(parents=True)
            dump_config(sft_config, model_dir / "config.yaml")
            pairs[seed][variant] = train_variant(data_dir=data_dir, side="shadow", variant=variant, role="pio",
                                                 sft_config=sft_config, seed=seed, run_dir=model_dir,
                                                 cpu_shim=cpu_shim)
            pairs[seed][variant]["run_dir"] = str(model_dir)
    return pairs


def reuse_pairs(run_dirs):
    pairs = {}
    for run in run_dirs:
        meta = load_run_meta(run, "pio", ["pio_in", "pio_out"])
        pairs.setdefault(meta["seed"], {})[meta["variant"]] = meta
    incomplete = [s for s, p in pairs.items() if set(p) != {"pio_in", "pio_out"}]
    if incomplete:
        raise ValueError(f"Seeds without both pio_in and pio_out runs: {incomplete}")
    return pairs


def delta_rows(in_table, out_table, groups, seed):
    """IN-minus-OUT per candidate. For twins theta_OUT is their IN model, so tables are swapped."""
    keys = ["state_id", "qid", "primitive"]
    model_a = in_table.set_index(keys)   # theta_IN(seed)
    model_b = out_table.set_index(keys)  # theta_OUT(seed)
    rows = []
    for key in model_a.index:
        group = groups[key[0]]
        trained_in, trained_out = (model_a, model_b) if group in ("canary", "control") else (model_b, model_a)
        row = {"state_id": key[0], "qid": key[1], "primitive": key[2], "group": group, "train_seed": seed}
        for m in METRICS:
            a, b = trained_in.at[key, m], trained_out.at[key, m]
            row[f"{m}_in"], row[f"{m}_out"], row[f"delta_{m}"] = a, b, a - b
        rows.append(row)
    return rows


def mean_ci(frame, column, n_boot, rng):
    by_state = frame.groupby("state_id")[column].apply(lambda s: s.dropna().to_numpy())
    arrays = [a for a in by_state if len(a)]
    if not arrays:
        return None
    reps = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(arrays), len(arrays))
        reps.append(np.concatenate([arrays[i] for i in pick]).mean())
    return [float(np.quantile(reps, .025)), float(np.quantile(reps, .975))]


def signed_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    out = {"n": int(len(values))}
    if len(values) == 0:
        return out
    out.update(mean=float(values.mean()), median=float(np.median(values)), mean_abs=float(np.abs(values).mean()))
    try:
        out["p_wilcoxon_two_sided"] = float(sps.wilcoxon(values).pvalue)
    except ValueError:
        out["p_wilcoxon_two_sided"] = None
    return out


def compare_groups(frame, metrics, n_boot, rng):
    trained = frame[frame["group"].isin(["canary", "twin"])]
    control = frame[frame["group"] == "control"]
    out = {}
    for m in metrics:
        column = f"delta_{m}"
        entry = {"trained": signed_summary(trained[column]), "control": signed_summary(control[column])}
        entry["trained"]["mean_ci95_state_bootstrap"] = mean_ci(trained, column, n_boot, rng)
        a, b = np.abs(trained[column].dropna()), np.abs(control[column].dropna())
        if len(a) and len(b):
            entry["p_abs_delta_trained_gt_control"] = float(sps.mannwhitneyu(a, b, alternative="greater").pvalue)
        out[m] = entry
    return out


def analyze(deltas, cfg, seed):
    rng = np.random.default_rng(seed)
    tol = cfg["logit_tolerance"]
    results = {}
    for primitive in (*PRIMITIVES, "all"):
        frame = deltas if primitive == "all" else deltas[deltas["primitive"] == primitive]
        stable = frame[np.abs(frame["delta_logit_y"]) < tol]
        block = {"all_candidates": compare_groups(frame, METRICS, cfg["n_bootstrap"], rng),
                 "confidence_stable": {"logit_tolerance": tol,
                                       "n_trained": int(stable["group"].isin(["canary", "twin"]).sum()),
                                       "n_control": int((stable["group"] == "control").sum())}}
        if min(block["confidence_stable"]["n_trained"], block["confidence_stable"]["n_control"]) >= cfg["min_subset"]:
            block["confidence_stable"]["tests"] = compare_groups(stable, RIFT_FEATURES, cfg["n_bootstrap"], rng)
        results[primitive] = block
    return results


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-seeds", nargs="+", type=int, help="Train one IN/OUT pair per seed")
    parser.add_argument("--reuse-runs", nargs="+", help="Existing pio_in/pio_out run dirs instead of training")
    parser.add_argument("--sft-config", help="Override the config's sft_config")
    parser.add_argument("--cpu-shim", action="store_true", help="Smoke tests only")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if bool(args.train_seeds) == bool(args.reuse_runs):
        parser.error("Pass exactly one of --train-seeds or --reuse-runs")
    cfg = load_experiment_config(args.config)
    sft_path = Path(args.sft_config).resolve() if args.sft_config else resolve_relative(args.config, cfg["sft_config"])
    sft_config = load_config(sft_path)
    rift_cfg = load_config(resolve_relative(args.config, cfg["rift_config"]))
    run_dir = start_run(args, "paired_in_out", "paired_in_out", cfg,
                        extra={"data_dir": str(Path(args.data).resolve()), "sft_config": str(sft_path),
                               "train_seeds": args.train_seeds, "reuse_runs": args.reuse_runs})
    cpu_shim = args.cpu_shim or bool(sft_config.get("runtime", {}).get("cpu_shim", False))
    pairs = (train_pairs(args, Path(args.data).resolve(), run_dir, sft_config, cpu_shim) if args.train_seeds
             else reuse_pairs(args.reuse_runs))
    metas = [m for pair in pairs.values() for m in pair.values()]
    records, splits, _ = load_data_checked(args.data, *metas)
    save_dataset_ids(run_dir, splits, ("pio_base",) + POOLS)
    groups = {sid: pool.split("_", 1)[1] for pool in POOLS for sid in splits["pools"][pool]}
    recs = pool_records(records, splits, POOLS)
    store = FeatureStore(args.artifacts_root, args.device)
    rows = []
    for seed, pair in sorted(pairs.items()):
        tables = {}
        for variant, meta in pair.items():
            table, _ = merged_features(store, meta, recs, rift_cfg["extraction"], rift_cfg["counterfactual"])
            save_table(table, run_dir / "features" / f"{variant}_seed{seed}.csv.gz")
            tables[variant] = table
        rows.extend(delta_rows(tables["pio_in"], tables["pio_out"], groups, seed))
    deltas = pd.DataFrame(rows)
    save_table(deltas, run_dir / "paired_deltas.csv.gz")
    results = {"pairs": {str(s): {v: run_summary(m) for v, m in p.items()} for s, p in pairs.items()},
               "n_train_seeds": len(pairs), "analysis": analyze(deltas, cfg["analysis"], args.seed),
               "note": "Causal validation only. Trained = canary/twin under their own IN vs OUT model; "
                       "control = trained by neither (model-difference noise floor)."}
    write_json(run_dir / "results.json", results)
    pooled = results["analysis"]["all"]["all_candidates"]
    print("\nmetric      mean delta (trained)   mean delta (control)   p(|delta| trained > control)")
    for m in ("logit_y", "p_y", *RIFT_FEATURES):
        t, c = pooled[m]["trained"], pooled[m]["control"]
        print(f"{m:10s} {t.get('mean', float('nan')):20.4f} {c.get('mean', float('nan')):22.4f} "
              f"{pooled[m].get('p_abs_delta_trained_gt_control', float('nan')):26.4g}")
    print(f"\nresults: {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
