#!/usr/bin/env python3
"""Confidence-based MIA baseline (reproduces the method of arXiv:2610.04985, Sec. IV-A).

1. Query the frozen shadow theta_S on D_M^S and D_NM^S; extract z_S(x) (paper: p(x), h(x)).
2. Fit an L2 logistic-regression attacker g on (z_S(x), M(x)); pick gamma on a shadow
   calibration split; freeze g.
3. Query the frozen target theta_T on D_M^T and D_NM^T; score g(z_T(x)).
4. Only now join target membership labels and report ROC-AUC, PR-AUC, TPR@{0.1,1,5}% FPR
   with state-level bootstrap 95% CIs.

Paper features and exploratory features are separate feature sets. Units: one attacker per
primitive (the paper's per-task setting) and, as an extension, one per state (3 questions).

  python research/mia/experiments/run_confidence_mia.py --config research/mia/configs/confidence_mia.yaml \
      --seed 0 --data <data_dir> --shadow-run <models/shadow_...> --target-run <models/target_...>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mia.attacks.confidence import feature_names  # noqa: E402
from mia.attacks.extraction import attach_labels, state_level_table  # noqa: E402
from mia.attacks.pipeline import attack_pipeline, evaluate_scores  # noqa: E402
from mia.data.schema import PRIMITIVES  # noqa: E402
from mia.data.splits import membership_labels  # noqa: E402
from mia.evaluation import plots  # noqa: E402
from mia.experiments._common import (FeatureStore, base_parser, load_data_checked, load_experiment_config,  # noqa: E402
                                     load_run_meta, merged_features, pool_records, run_summary, save_dataset_ids,
                                     save_table, start_run)
from mia.provenance import write_json  # noqa: E402


def accuracy_report(shadow_table, target_table, target_labels):
    """Task accuracy of each victim on its members and non-members (labels used for reporting only)."""
    report = {}
    target = target_table.assign(member=target_table["state_id"].map(target_labels))
    for name, table in (("shadow", shadow_table), ("target", target)):
        for primitive in PRIMITIVES:
            sub = table[table["primitive"] == primitive]
            report[f"{name}/{primitive}"] = {
                "accuracy_members": float(sub.loc[sub["member"] == 1, "correct"].mean()),
                "accuracy_nonmembers": float(sub.loc[sub["member"] == 0, "correct"].mean()),
                "mean_p_y_members": float(sub.loc[sub["member"] == 1, "p_y"].mean()),
                "mean_p_y_nonmembers": float(sub.loc[sub["member"] == 0, "p_y"].mean())}
    return report


def run_unit(name, shadow, target, features, target_labels, cfg, seed, run_dir, unit, primitive, feature_set):
    attack, evaluation = cfg["attack"], cfg["evaluation"]
    classifier, scored, info = attack_pipeline(shadow, target, features, C=attack["C"], max_iter=attack["max_iter"],
                                               calibration_fraction=attack["calibration_fraction"], seed=seed)
    classifier.save(run_dir / "attackers" / name.replace("/", "__"))
    metrics = evaluate_scores(scored, target_labels, n_boot=evaluation["n_bootstrap"], seed=seed,
                              fprs=evaluation["fprs"], gamma=info["gamma"])
    scored = scored.assign(unit=unit, feature_set=feature_set, primitive=primitive,
                           member=scored["state_id"].map(target_labels))
    return {"unit": unit, "primitive": primitive, "feature_set": feature_set, "features": features,
            "attack": info, "classifier": classifier.describe(), "target_metrics": metrics}, scored


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--shadow-run", required=True)
    parser.add_argument("--target-run", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    shadow = load_run_meta(args.shadow_run, "shadow", ["standard"])
    target = load_run_meta(args.target_run, "target", ["standard"])
    records, splits, _ = load_data_checked(args.data, shadow, target)
    run_dir = start_run(args, "confidence_mia", "confidence_mia", cfg,
                        extra={"data_dir": str(Path(args.data).resolve()), "shadow": run_summary(shadow),
                               "target": run_summary(target)})
    s_spec, t_spec = shadow["membership"], target["membership"]
    s_pools = s_spec["member_pools"] + s_spec["nonmember_pools"]
    t_pools = t_spec["member_pools"] + t_spec["nonmember_pools"]
    save_dataset_ids(run_dir, splits, s_pools + t_pools)

    store = FeatureStore(args.artifacts_root, args.device)
    s_tab, _ = merged_features(store, shadow, pool_records(records, splits, s_pools), cfg["extraction"])
    t_tab, _ = merged_features(store, target, pool_records(records, splits, t_pools), cfg["extraction"])
    s_tab["model_role"], t_tab["model_role"] = "shadow", "target"
    s_labels, t_labels = membership_labels(splits, s_spec), membership_labels(splits, t_spec)
    s_tab = attach_labels(s_tab, s_labels)
    save_table(s_tab, run_dir / "features" / "shadow_confidence_labeled.csv.gz")
    save_table(t_tab, run_dir / "features" / "target_confidence_unlabeled.csv.gz")
    write_json(run_dir / "features" / "target_membership_labels.json", t_labels)

    results = {"shadow": run_summary(shadow), "target": run_summary(target),
               "pipeline_checks": {"shadow_membership_verification": shadow["membership_verification"],
                                   "target_membership_verification": target["membership_verification"],
                                   "shadow_rows": int(len(s_tab)), "target_rows": int(len(t_tab))},
               "model_accuracy": accuracy_report(s_tab, t_tab, t_labels), "units": {}}
    all_scores, curves = [], {}
    for feature_set in cfg["feature_sets"]:
        if "primitive" in cfg["units"]:
            for primitive in PRIMITIVES:
                name = f"{feature_set}/{primitive}"
                result, scored = run_unit(name, s_tab[s_tab["primitive"] == primitive],
                                          t_tab[t_tab["primitive"] == primitive],
                                          feature_names(feature_set, primitive), t_labels, cfg, args.seed, run_dir,
                                          "question", primitive, feature_set)
                results["units"][name] = result
                all_scores.append(scored)
                curves[name] = (scored["member"].to_numpy(), scored["score"].to_numpy())
        if "state" in cfg["units"]:
            columns = {p: feature_names(feature_set, p) for p in PRIMITIVES}
            s_state = state_level_table(s_tab, columns).assign(model_role="shadow")
            s_state["member"] = s_state["state_id"].map(s_labels)
            t_state = state_level_table(t_tab, columns).assign(model_role="target")
            features = [f"{p}.{c}" for p, cs in columns.items() for c in cs]
            name = f"{feature_set}/state"
            result, scored = run_unit(name, s_state, t_state, features, t_labels, cfg, args.seed, run_dir,
                                      "state", "all", feature_set)
            results["units"][name] = result
            all_scores.append(scored)
            curves[name] = (scored["member"].to_numpy(), scored["score"].to_numpy())

    save_table(pd.concat(all_scores, ignore_index=True), run_dir / "target_scores.csv.gz")
    write_json(run_dir / "results.json", results)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir()
    errors = [plots.safe(plots.roc_plot, curves, plot_dir / "roc_target.png", "Confidence MIA on target")]
    for primitive in PRIMITIVES:
        feats = feature_names("paper", primitive)
        errors.append(plots.safe(plots.membership_histograms, s_tab[s_tab["primitive"] == primitive], feats,
                                 plot_dir / f"shadow_{primitive}_features.png", f"shadow theta_S, {primitive}"))
    write_json(run_dir / "plot_errors.json", [e for e in errors if e])

    print_table(results["units"])
    print(f"\nresults: {run_dir / 'results.json'}")


def _num(value):
    return np.nan if value is None else value


def print_table(units):
    print(f"\n{'unit':36s} {'ROC-AUC [95% CI]':>22s} {'TPR@1%':>8s} {'TPR@5%':>8s}")
    for name, result in units.items():
        point, ci = result["target_metrics"]["point"], result["target_metrics"]["ci95"].get("roc_auc", {})
        print(f"{name:36s} {point['roc_auc']:.3f} [{_num(ci.get('low')):.3f},{_num(ci.get('high')):.3f}]"
              f"   {_num(point.get('tpr_at_1pct_fpr')):8.3f} {_num(point.get('tpr_at_5pct_fpr')):8.3f}")


if __name__ == "__main__":
    main()
