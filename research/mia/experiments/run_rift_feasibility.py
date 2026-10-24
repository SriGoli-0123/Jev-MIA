#!/usr/bin/env python3
"""RIFT feasibility: do pairwise fact interactions I = [I12, I13, I23] differ between members and
non-members of the shadow model, reproducibly across shadow training seeds?

No attack classifier is trained on the target here. For each shadow seed and primitive:
  marginal family      - per-feature two-sided tests (AUC permutation, KS) and joint tests
                         (energy distance, cross-validated classifier two-sample test) on [I12,I13,I23];
  beyond_confidence    - CV AUC gain from adding I to the paper confidence features (permutation null),
                         and tests of I on a confidence-matched member/non-member subset;
  beyond_first_order   - CV AUC gain from adding I to confidence + single-fact effects D1..D3 (reported).
Holm correction is applied within each family and seed. A signal is "reproducible" when the same
test is significant in at least max(min_seeds, ceil(min_seed_fraction * n_seeds)) seeds with a
consistent direction. Otherwise the verdict is "RIFT hypothesis unsupported in this setting."

  python research/mia/experiments/run_rift_feasibility.py --config research/mia/configs/rift.yaml --seed 0 \
      --data <data_dir> --shadow-runs <models/shadow_seed1> <models/shadow_seed2> <models/shadow_seed3>
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from mia.attacks.confidence import PAPER_FEATURES  # noqa: E402
from mia.attacks.extraction import attach_labels  # noqa: E402
from mia.attacks.first_order import FIRST_ORDER_FEATURES  # noqa: E402
from mia.attacks.rift import RIFT_FEATURES  # noqa: E402
from mia.data.schema import PRIMITIVES  # noqa: E402
from mia.data.splits import membership_labels  # noqa: E402
from mia.evaluation import plots  # noqa: E402
from mia.evaluation.stats import (caliper_match, classifier_two_sample_test, energy_distance_test, holm,  # noqa: E402
                                  incremental_auc_test, univariate_tests)
from mia.experiments._common import (FeatureStore, base_parser, load_data_checked, load_experiment_config,  # noqa: E402
                                     load_run_meta, merged_features, pool_records, run_summary, save_dataset_ids,
                                     save_table, start_run)
from mia.provenance import write_json  # noqa: E402

UNSUPPORTED = "RIFT hypothesis unsupported in this setting."


def oof_confidence_score(frame, columns, y, seed, folds):
    """Out-of-fold logistic score from confidence features, used only to match on confidence."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return cross_val_predict(model, frame[list(columns)].to_numpy(dtype=float), y, cv=cv,
                             method="predict_proba")[:, 1]


def direction(auc):
    return int(np.sign(round(auc - 0.5, 12)))


def analyze_primitive(sub, primitive, fe, seed):
    rng = np.random.default_rng(seed)
    y = sub["member"].to_numpy(dtype=int)
    X = sub[list(RIFT_FEATURES)].to_numpy(dtype=float)
    conf_cols = list(PAPER_FEATURES[primitive])
    out, tests = {"univariate": {}, "first_order_reference": {}}, []

    def add(family, test, feature, p, sign=0, **stats):
        tests.append({"family": family, "primitive": primitive, "test": test, "feature": feature,
                      "p": float(p), "direction": int(sign), **stats})

    for feature in RIFT_FEATURES:
        r = univariate_tests(sub[feature], y, fe["n_permutations"], rng)
        out["univariate"][feature] = r
        add("marginal", "auc_permutation", feature, r["p_auc_permutation"], direction(r["auc"]), auc=r["auc"])
        add("marginal", "ks", feature, r["p_ks"], 0, ks=r["ks_stat"])
    for feature in FIRST_ORDER_FEATURES:
        out["first_order_reference"][feature] = univariate_tests(sub[feature], y, fe["n_permutations"], rng)
    energy = energy_distance_test(X, y, fe["energy_permutations"], rng, fe["energy_max_n"])
    c2st = classifier_two_sample_test(X, y, fe["c2st_permutations"], rng, fe["cv_folds"], seed)
    out.update(joint_energy=energy, joint_c2st=c2st)
    add("marginal", "energy_distance", "I_joint", energy["p_permutation"], 0, stat=energy["energy_distance"])
    add("marginal", "c2st", "I_joint", c2st["p_permutation"], direction(c2st["cv_auc"]), auc=c2st["cv_auc"])

    beyond_conf = incremental_auc_test(sub[conf_cols], X, y, fe["incremental_permutations"], rng, fe["cv_folds"], seed)
    beyond_fo = incremental_auc_test(sub[conf_cols + list(FIRST_ORDER_FEATURES)], X, y,
                                     fe["incremental_permutations"], rng, fe["cv_folds"], seed)
    out.update(beyond_confidence=beyond_conf, beyond_first_order=beyond_fo)
    add("beyond_confidence", "incremental_auc", "I_joint", beyond_conf["p_permutation_one_sided"], 1,
        delta_auc=beyond_conf["delta_auc"])
    add("beyond_first_order", "incremental_auc", "I_joint", beyond_fo["p_permutation_one_sided"], 1,
        delta_auc=beyond_fo["delta_auc"])

    score = oof_confidence_score(sub, conf_cols, y, seed, fe["cv_folds"])
    idx, info = caliper_match(score, y, fe["matched_caliper_sd"], rng)
    matched = {"match": info}
    if info["pairs"] >= fe["min_matched_pairs"]:
        msub, my = sub.iloc[idx], y[idx]
        matched["confidence_auc_on_matched"] = univariate_tests(score[idx], my, fe["n_permutations"], rng)["auc"]
        for feature in RIFT_FEATURES:
            r = univariate_tests(msub[feature], my, fe["n_permutations"], rng)
            matched[feature] = r
            add("beyond_confidence", "matched_auc_permutation", feature, r["p_auc_permutation"], direction(r["auc"]),
                auc=r["auc"])
        mc = classifier_two_sample_test(msub[list(RIFT_FEATURES)].to_numpy(dtype=float), my,
                                        fe["c2st_permutations"], rng, fe["cv_folds"], seed)
        matched["c2st"] = mc
        add("beyond_confidence", "matched_c2st", "I_joint", mc["p_permutation"], direction(mc["cv_auc"]),
            auc=mc["cv_auc"])
    out["confidence_matched"] = matched
    return out, tests


def apply_holm(tests):
    families = defaultdict(list)
    for i, t in enumerate(tests):
        families[t["family"]].append(i)
    for indices in families.values():
        for i, p in zip(indices, holm([tests[i]["p"] for i in indices])):
            tests[i]["p_holm"] = p
    return tests


def verdict(seed_results, fe):
    n = len(seed_results)
    need = max(fe["min_seeds"], math.ceil(fe["min_seed_fraction"] * n))
    by_test = defaultdict(list)
    for result in seed_results:
        for t in result["tests"]:
            by_test[(t["family"], t["primitive"], t["test"], t["feature"])].append(t)
    reproducible = []
    for key, runs in sorted(by_test.items()):
        significant = [t for t in runs if t["p_holm"] < fe["alpha"]]
        signs = {t["direction"] for t in significant if t["direction"] != 0}
        if len(significant) >= need and len(signs) <= 1:
            reproducible.append({"family": key[0], "primitive": key[1], "test": key[2], "feature": key[3],
                                 "significant_seeds": len(significant), "of_seeds": len(runs),
                                 "direction": signs.pop() if signs else 0})
    families = {r["family"] for r in reproducible}
    marginal = "marginal" in families
    beyond = "beyond_confidence" in families
    if n < fe["min_seeds"]:
        text = f"INCONCLUSIVE: {n} shadow seed(s); at least {fe['min_seeds']} are needed to judge reproducibility."
    elif not marginal and not beyond:
        text = UNSUPPORTED
    elif not beyond:
        text = ("RIFT features differ between members and non-members, but no reproducible membership "
                "information beyond confidence was found in this setting.")
    else:
        text = ("Reproducible RIFT membership signal, including information beyond confidence. "
                "Feasibility evidence only; not an attack result.")
    return {"verdict": text, "gate_open_for_rift_mia": bool(n >= fe["min_seeds"] and (marginal or beyond)),
            "marginal_signal_reproducible": marginal, "beyond_confidence_signal_reproducible": beyond,
            "beyond_first_order_signal_reproducible": "beyond_first_order" in families,
            "seeds_required": need, "n_seeds": n, "alpha": fe["alpha"], "reproducible_tests": reproducible}


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--shadow-runs", nargs="+", required=True, help="Shadow runs trained with different seeds")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    fe = cfg["feasibility"]
    metas = [load_run_meta(r, "shadow", ["standard"]) for r in args.shadow_runs]
    if len({m["seed"] for m in metas}) != len(metas):
        raise ValueError("Shadow runs must use distinct training seeds")
    records, splits, _ = load_data_checked(args.data, *metas)
    run_dir = start_run(args, "rift_feasibility", "rift_feasibility", cfg,
                        extra={"data_dir": str(Path(args.data).resolve()), "shadow_runs": [run_summary(m) for m in metas]})
    store = FeatureStore(args.artifacts_root, args.device)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir()
    seed_results, plot_errors = [], []
    for meta in metas:
        spec = meta["membership"]
        pools = spec["member_pools"] + spec["nonmember_pools"]
        save_dataset_ids(run_dir / f"shadow_seed{meta['seed']}", splits, pools)
        recs = pool_records(records, splits, pools)
        table, long = merged_features(store, meta, recs, cfg["extraction"], cfg["counterfactual"])
        table = attach_labels(table.assign(model_role="shadow"), membership_labels(splits, spec))
        save_table(table, run_dir / "features" / f"shadow_seed{meta['seed']}_labeled.csv.gz")
        save_table(long, run_dir / "features" / f"shadow_seed{meta['seed']}_grids.csv.gz")
        noise = store.noise_floor(meta, recs[:fe["noise_floor_states"]], cfg["extraction"])
        typical = float(np.median(np.abs(table[list(RIFT_FEATURES)].to_numpy())))
        per_primitive, tests = {}, []
        for primitive in PRIMITIVES:
            sub = table[table["primitive"] == primitive].reset_index(drop=True)
            per_primitive[primitive], primitive_tests = analyze_primitive(sub, primitive, fe, args.seed)
            tests.extend(primitive_tests)
            plot_errors.append(plots.safe(plots.membership_histograms, sub, list(RIFT_FEATURES),
                                          plot_dir / f"seed{meta['seed']}_{primitive}_interactions.png",
                                          f"shadow seed {meta['seed']}, {primitive}: I by membership"))
        seed_results.append({"shadow_run": run_summary(meta), "numerical_noise_floor_f": noise,
                             "median_abs_interaction": typical,
                             "noise_warning": bool(noise > 0.1 * typical) if typical > 0 else True,
                             "per_primitive": per_primitive, "tests": apply_holm(tests)})
        write_json(run_dir / f"tests_shadow_seed{meta['seed']}.json", seed_results[-1])
    result = {**verdict(seed_results, fe), "counterfactual": cfg["counterfactual"],
              "extraction": cfg["extraction"], "shadow_runs": [run_summary(m) for m in metas],
              "data_splits_sha256": metas[0]["data_splits_sha256"]}
    write_json(run_dir / "verdict.json", result)
    write_json(run_dir / "plot_errors.json", [e for e in plot_errors if e])
    print("\nVERDICT:", result["verdict"])
    for item in result["reproducible_tests"]:
        print("  reproducible:", item)
    print(f"verdict file: {run_dir / 'verdict.json'}")


if __name__ == "__main__":
    main()
