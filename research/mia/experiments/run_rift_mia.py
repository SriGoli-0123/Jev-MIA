#!/usr/bin/env python3
"""RIFT MIA and the critical comparison against confidence and first-order sensitivity.

Runs only if run_rift_feasibility.py produced a verdict with gate_open_for_rift_mia = true.
(--override-feasibility-gate exists for smoke tests; the override and its reason are recorded
in gate.json and results.json, and such results must not be reported as findings.)

For each primitive, identical target candidates are attacked with simple L2 logistic
regressions trained on the shadow model only, using feature sets such as:
  confidence | first_order (D1..D3) | rift (I12, I13, I23) | confidence+rift | ...
Paired state-level bootstrap tests compare feature sets on the same candidates, and every
set is re-evaluated on overlap subsets where confidence carries little membership signal:
  confidence_matched - 1:1 member/non-member caliper matching on the frozen confidence attack score;
  saturated          - candidates with p_max >= threshold.
Both subsets use target labels and are evaluation-only.

  python research/mia/experiments/run_rift_mia.py --config research/mia/configs/rift.yaml --seed 0 \
      --data <data_dir> --feasibility <rift_feasibility/.../verdict.json> \
      --shadow-run <models/shadow_...> --target-run <models/target_...>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mia.attacks.extraction import attach_labels  # noqa: E402
from mia.attacks.pipeline import attack_pipeline, evaluate_scores, expand_features  # noqa: E402
from mia.data.schema import PRIMITIVES  # noqa: E402
from mia.data.splits import membership_labels  # noqa: E402
from mia.evaluation import plots  # noqa: E402
from mia.evaluation.bootstrap import paired_difference  # noqa: E402
from mia.evaluation.stats import caliper_match  # noqa: E402
from mia.experiments._common import (FeatureStore, base_parser, load_data_checked, load_experiment_config,  # noqa: E402
                                     load_run_meta, merged_features, pool_records, run_summary, save_dataset_ids,
                                     save_table, start_run)
from mia.provenance import write_json  # noqa: E402


def check_gate(verdict, override_reason, cfg, shadow):
    if verdict["counterfactual"] != cfg["counterfactual"]:
        raise ValueError("Counterfactual settings differ from those used in the feasibility run")
    if verdict["data_splits_sha256"] != shadow["data_splits_sha256"]:
        raise ValueError("Feasibility verdict was computed on a different dataset")
    if verdict["gate_open_for_rift_mia"]:
        return {"gate_open": True, "overridden": False, "verdict": verdict["verdict"]}
    if not override_reason:
        print(f"Feasibility verdict: {verdict['verdict']}\n"
              "RIFT MIA is not run because feasibility did not show a reproducible signal (protocol, Milestone 5).")
        raise SystemExit(2)
    return {"gate_open": False, "overridden": True, "override_reason": override_reason, "verdict": verdict["verdict"],
            "warning": "GATE OVERRIDDEN: these results are not evidence for RIFT and must not be reported as findings."}


def subset_metrics(scores, labels, rows, cfg, seed):
    return {name: evaluate_scores(frame, labels, n_boot=cfg["n_bootstrap"], seed=seed, fprs=cfg["fprs"], rows=rows)
            for name, frame in scores.items()}


def comparisons(scores, labels, pairs, cfg, seed, rows=None):
    out = {}
    for a, b in pairs:
        if a not in scores or b not in scores:
            continue
        fa, fb = scores[a], scores[b]
        if rows is not None:
            fa, fb = fa.iloc[rows], fb.iloc[rows]
        if not (fa["state_id"].to_numpy() == fb["state_id"].to_numpy()).all():
            raise RuntimeError("Feature sets were not scored on identical candidates")
        y = fa["state_id"].map(labels).to_numpy(dtype=int)
        out[f"{a} - {b}"] = paired_difference(y, fa["score"], fb["score"], groups=fa["state_id"].to_numpy(),
                                              metric="roc_auc", n_boot=cfg["n_bootstrap"], seed=seed)
    return out


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--feasibility", required=True, help="verdict.json from run_rift_feasibility.py")
    parser.add_argument("--shadow-run", required=True)
    parser.add_argument("--target-run", required=True)
    parser.add_argument("--override-feasibility-gate", metavar="REASON",
                        help="Smoke tests only: run even though feasibility did not open the gate")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    mia_cfg, ev = cfg["mia"], cfg["mia"]["evaluation"]
    shadow = load_run_meta(args.shadow_run, "shadow", ["standard"])
    target = load_run_meta(args.target_run, "target", ["standard"])
    verdict = json.loads(Path(args.feasibility).read_text(encoding="utf-8"))
    gate = check_gate(verdict, args.override_feasibility_gate, cfg, shadow)
    records, splits, _ = load_data_checked(args.data, shadow, target)
    run_dir = start_run(args, "rift_mia", "rift_mia" + ("_GATE_OVERRIDDEN" if gate["overridden"] else ""), cfg,
                        extra={"data_dir": str(Path(args.data).resolve()), "shadow": run_summary(shadow),
                               "target": run_summary(target), "gate": gate,
                               "feasibility_verdict_path": str(Path(args.feasibility).resolve())})
    write_json(run_dir / "gate.json", gate)
    s_spec, t_spec = shadow["membership"], target["membership"]
    s_pools = s_spec["member_pools"] + s_spec["nonmember_pools"]
    t_pools = t_spec["member_pools"] + t_spec["nonmember_pools"]
    save_dataset_ids(run_dir, splits, s_pools + t_pools)

    store = FeatureStore(args.artifacts_root, args.device)
    s_tab, s_long = merged_features(store, shadow, pool_records(records, splits, s_pools), cfg["extraction"],
                                    cfg["counterfactual"])
    t_tab, t_long = merged_features(store, target, pool_records(records, splits, t_pools), cfg["extraction"],
                                    cfg["counterfactual"])
    s_tab, t_tab = s_tab.assign(model_role="shadow"), t_tab.assign(model_role="target")
    s_tab = attach_labels(s_tab, membership_labels(splits, s_spec))
    t_labels = membership_labels(splits, t_spec)
    save_table(s_tab, run_dir / "features" / "shadow_labeled.csv.gz")
    save_table(t_tab, run_dir / "features" / "target_unlabeled.csv.gz")
    save_table(t_long, run_dir / "features" / "target_grids.csv.gz")
    write_json(run_dir / "features" / "target_membership_labels.json", t_labels)

    rng = np.random.default_rng(args.seed)
    results = {"gate": gate, "shadow": run_summary(shadow), "target": run_summary(target), "per_primitive": {}}
    all_scores = []
    plot_dir = run_dir / "plots"
    plot_dir.mkdir()
    plot_errors = []
    for primitive in PRIMITIVES:
        s = s_tab[s_tab["primitive"] == primitive].reset_index(drop=True)
        t = t_tab[t_tab["primitive"] == primitive].reset_index(drop=True)
        units, scores = {}, {}
        for set_name, spec in mia_cfg["feature_sets"].items():
            features = expand_features(spec, primitive)
            attack = mia_cfg["attack"]
            classifier, scored, info = attack_pipeline(s, t, features, C=attack["C"], max_iter=attack["max_iter"],
                                                       calibration_fraction=attack["calibration_fraction"],
                                                       seed=args.seed)
            classifier.save(run_dir / "attackers" / f"{set_name}__{primitive}")
            units[set_name] = {"features": features, "attack": info, "classifier": classifier.describe(),
                               "target_metrics": evaluate_scores(scored, t_labels, n_boot=ev["n_bootstrap"],
                                                                 seed=args.seed, fprs=ev["fprs"], gamma=info["gamma"])}
            scores[set_name] = scored
            all_scores.append(scored.assign(feature_set=set_name, member=scored["state_id"].map(t_labels)))
        y = scores["confidence"]["state_id"].map(t_labels).to_numpy(dtype=int)
        overlap = mia_cfg["overlap"]
        idx, match_info = caliper_match(scores["confidence"]["score"].to_numpy(), y, overlap["caliper_sd"], rng)
        matched = {"match": match_info}
        if match_info["pairs"] >= overlap["min_pairs"]:
            matched["metrics"] = subset_metrics(scores, t_labels, idx, ev, args.seed)
            matched["comparisons"] = comparisons(scores, t_labels, mia_cfg["comparisons"], ev, args.seed, rows=idx)
        saturated_rows = np.flatnonzero(t["p_max"].to_numpy() >= overlap["saturation_p_max"])
        ys = y[saturated_rows]
        saturated = {"threshold_p_max": overlap["saturation_p_max"], "n": int(len(saturated_rows)),
                     "n_members": int(ys.sum()), "n_nonmembers": int((ys == 0).sum())}
        if min(saturated["n_members"], saturated["n_nonmembers"]) >= overlap["min_pairs"]:
            saturated["metrics"] = subset_metrics(scores, t_labels, saturated_rows, ev, args.seed)
            saturated["comparisons"] = comparisons(scores, t_labels, mia_cfg["comparisons"], ev, args.seed,
                                                   rows=saturated_rows)
        results["per_primitive"][primitive] = {
            "feature_sets": units,
            "comparisons_full_target": comparisons(scores, t_labels, mia_cfg["comparisons"], ev, args.seed),
            "overlap_confidence_matched": matched, "overlap_saturated": saturated}
        curves = {name: (frame["state_id"].map(t_labels).to_numpy(), frame["score"].to_numpy())
                  for name, frame in scores.items()}
        plot_errors.append(plots.safe(plots.roc_plot, curves, plot_dir / f"roc_{primitive}.png",
                                      f"target, {primitive}" + (" (GATE OVERRIDDEN)" if gate["overridden"] else "")))
    save_table(pd.concat(all_scores, ignore_index=True), run_dir / "target_scores.csv.gz")
    write_json(run_dir / "results.json", results)
    write_json(run_dir / "plot_errors.json", [e for e in plot_errors if e])
    if gate["overridden"]:
        print("\n" + gate["warning"])
    for primitive, block in results["per_primitive"].items():
        print(f"\n[{primitive}] target ROC-AUC")
        for name, unit in block["feature_sets"].items():
            point = unit["target_metrics"]["point"]
            ci = unit["target_metrics"]["ci95"].get("roc_auc", {})
            print(f"  {name:34s} {point['roc_auc']:.3f} [{ci.get('low', float('nan')):.3f},"
                  f"{ci.get('high', float('nan')):.3f}]")
    print(f"\nresults: {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
