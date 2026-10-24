#!/usr/bin/env python3
"""Cross-primitive MIA: can membership acquired through one primitive be inferred through another?

Inputs are six runs trained with --variant xp_boolean / xp_choice / xp_score, three shadow and
three target. In an xp_<P_train> model each member state contributed only its P_train question
to SFT (background states contribute all three). For every (P_train, P_query) cell the attacker
is trained on shadow P_query features of the xp_<P_train> shadow model and evaluated on the
xp_<P_train> target model. Membership always refers to the original P_train training tuple.
Only confidence features are used; RIFT is deliberately not mixed in.

  python research/mia/experiments/run_cross_primitive.py --config research/mia/configs/cross_primitive.yaml \
      --seed 0 --data <data_dir> --shadow-runs <xp_boolean> <xp_choice> <xp_score> \
      --target-runs <xp_boolean> <xp_choice> <xp_score>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from mia.attacks.confidence import feature_names  # noqa: E402
from mia.attacks.cross_primitive import cell_kind, transfer_matrix, variant_primitive  # noqa: E402
from mia.attacks.extraction import attach_labels  # noqa: E402
from mia.attacks.pipeline import attack_pipeline, evaluate_scores  # noqa: E402
from mia.data.schema import PRIMITIVES  # noqa: E402
from mia.data.splits import membership_labels  # noqa: E402
from mia.evaluation import plots  # noqa: E402
from mia.experiments._common import (FeatureStore, base_parser, load_data_checked, load_experiment_config,  # noqa: E402
                                     load_run_meta, merged_features, pool_records, run_summary, save_dataset_ids,
                                     save_table, start_run)
from mia.provenance import write_json  # noqa: E402

XP = [f"xp_{p}" for p in PRIMITIVES]


def by_primitive(run_dirs, role):
    metas = {}
    for run in run_dirs:
        meta = load_run_meta(run, role, XP)
        primitive = variant_primitive(meta["variant"])
        if primitive in metas:
            raise ValueError(f"Two {role} runs for training primitive {primitive}")
        metas[primitive] = meta
    if set(metas) != set(PRIMITIVES):
        raise ValueError(f"Need one {role} run per primitive {PRIMITIVES}, got {sorted(metas)}")
    return metas


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--shadow-runs", nargs=3, required=True)
    parser.add_argument("--target-runs", nargs=3, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    shadows, targets = by_primitive(args.shadow_runs, "shadow"), by_primitive(args.target_runs, "target")
    records, splits, _ = load_data_checked(args.data, *shadows.values(), *targets.values())
    run_dir = start_run(args, "cross_primitive", "cross_primitive", cfg,
                        extra={"data_dir": str(Path(args.data).resolve()),
                               "shadow_runs": {p: run_summary(m) for p, m in shadows.items()},
                               "target_runs": {p: run_summary(m) for p, m in targets.items()}})
    store = FeatureStore(args.artifacts_root, args.device)
    attack, ev = cfg["attack"], cfg["evaluation"]
    cells, all_scores = {}, []
    for train_p in PRIMITIVES:
        s_meta, t_meta = shadows[train_p], targets[train_p]
        s_spec, t_spec = s_meta["membership"], t_meta["membership"]
        s_pools = s_spec["member_pools"] + s_spec["nonmember_pools"]
        t_pools = t_spec["member_pools"] + t_spec["nonmember_pools"]
        save_dataset_ids(run_dir / f"train_{train_p}", splits, s_pools + t_pools)
        s_tab, _ = merged_features(store, s_meta, pool_records(records, splits, s_pools), cfg["extraction"])
        t_tab, _ = merged_features(store, t_meta, pool_records(records, splits, t_pools), cfg["extraction"])
        s_tab = attach_labels(s_tab.assign(model_role="shadow"), membership_labels(splits, s_spec))
        t_tab = t_tab.assign(model_role="target")
        t_labels = membership_labels(splits, t_spec)
        save_table(s_tab, run_dir / "features" / f"train_{train_p}_shadow_labeled.csv.gz")
        save_table(t_tab, run_dir / "features" / f"train_{train_p}_target_unlabeled.csv.gz")
        write_json(run_dir / "features" / f"train_{train_p}_target_membership_labels.json", t_labels)
        for query_p in PRIMITIVES:
            features = feature_names(cfg["feature_set"], query_p)
            classifier, scored, info = attack_pipeline(
                s_tab[s_tab["primitive"] == query_p], t_tab[t_tab["primitive"] == query_p], features,
                C=attack["C"], max_iter=attack["max_iter"], calibration_fraction=attack["calibration_fraction"],
                seed=args.seed)
            classifier.save(run_dir / "attackers" / f"train_{train_p}__query_{query_p}")
            metrics = evaluate_scores(scored, t_labels, n_boot=ev["n_bootstrap"], seed=args.seed, fprs=ev["fprs"],
                                      gamma=info["gamma"])
            cells[(train_p, query_p)] = metrics
            all_scores.append(scored.assign(train_primitive=train_p, query_primitive=query_p,
                                            member=scored["state_id"].map(t_labels)))
            write_json(run_dir / "cells" / f"train_{train_p}__query_{query_p}.json",
                       {"train_primitive": train_p, "query_primitive": query_p, "kind": cell_kind(train_p, query_p),
                        "features": features, "attack": info, "classifier": classifier.describe(),
                        "target_metrics": metrics})
    matrices = {metric: transfer_matrix(cells, metric) for metric in ("roc_auc", "pr_auc", "tpr_at_1pct_fpr",
                                                                      "tpr_at_5pct_fpr")}
    ci = {f"{tp}->{qp}": cells[(tp, qp)]["ci95"].get("roc_auc") for tp in PRIMITIVES for qp in PRIMITIVES}
    write_json(run_dir / "transfer_matrix.json", {"rows_train_primitive": list(PRIMITIVES),
                                                  "cols_query_primitive": list(PRIMITIVES),
                                                  "matrices": matrices, "roc_auc_ci95": ci})
    save_table(pd.concat(all_scores, ignore_index=True), run_dir / "target_scores.csv.gz")
    (run_dir / "plots").mkdir()
    error = plots.safe(plots.heatmap, matrices["roc_auc"], list(PRIMITIVES), list(PRIMITIVES),
                       run_dir / "plots" / "transfer_matrix_auc.png", "Membership transfer (target ROC-AUC)")
    write_json(run_dir / "plot_errors.json", [error] if error else [])
    print("\nROC-AUC (rows: training primitive, cols: query primitive)")
    print(" " * 10 + "".join(f"{q:>10s}" for q in PRIMITIVES))
    for train_p, row in zip(PRIMITIVES, matrices["roc_auc"]):
        print(f"{train_p:>10s}" + "".join(f"{v:10.3f}" for v in row))
    print(f"\nresults: {run_dir / 'transfer_matrix.json'}")


if __name__ == "__main__":
    main()
