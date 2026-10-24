"""Query a frozen NanoJev model and turn its candidate logits into per-(state, question) features.

Feature tables never contain membership labels. They carry ``state_id``/``qid`` keys only;
labels are joined from splits.json by the experiment script, and only for the side whose
labels the attacker is allowed to know (the shadow side) before the attack is frozen.
"""
import numpy as np
import pandas as pd

from mia.attacks.confidence import DEFAULT_EPS, confidence_row, label_logit
from mia.attacks.first_order import DELTA_FEATURES, single_fact_effects
from mia.attacks.rift import grid_features
from mia.data import schema
from mia.data.counterfactuals import PAIRS, all_grids, fact_key

KEY = ["state_id", "qid"]


def payload(record, qids=None, state_text=None, payload_id=None):
    questions = record["questions"] if qids is None else {q: record["questions"][q] for q in qids}
    return {"id": payload_id or record["id"], "state": state_text or record["state"], "questions": questions}


def gold_y(record, qid):
    return schema.gold_index(record["questions"][qid], record["gold"][qid])


def extract_confidence(scorer, records, qids=None, eps=DEFAULT_EPS, chunk=512):
    rows = []
    for start in range(0, len(records), chunk):
        part = records[start:start + chunk]
        scores = scorer.score([payload(r, qids) for r in part])
        for record in part:
            for qid in (qids or record["questions"]):
                result = scores[(record["id"], qid)]
                rows.append({"state_id": record["id"], "qid": qid, "primitive": schema.QID_PRIMITIVE[qid],
                             **confidence_row(result["type"], result["logits"], gold_y(record, qid), eps)})
    return pd.DataFrame(rows)


def extract_counterfactual(scorer, records, pairs=PAIRS, n_replacements=3, seed=0, qids=None, eps=DEFAULT_EPS,
                           chunk=32):
    """Score every s11/s10/s01/s00 grid; return (wide per-candidate features, long per-grid rows)."""
    wide, long_rows = [], []
    pair_tuples = [tuple(p) for p in pairs]
    for start in range(0, len(records), chunk):
        part = records[start:start + chunk]
        unique, grids_by_state = {}, {}
        for record in part:
            grids = all_grids(record, pair_tuples, n_replacements, seed)
            grids_by_state[record["id"]] = grids
            for item in grids.values():
                for variant in item["grid"].values():
                    vid = f"{record['id']}|" + "|".join(map(str, fact_key(variant["facts"])))
                    unique.setdefault(vid, variant)
        scores = scorer.score([payload(v, qids, state_text=v["state"], payload_id=vid) for vid, v in unique.items()])
        for record in part:
            sid = record["id"]
            for qid in (qids or record["questions"]):
                y = gold_y(record, qid)
                grid_rows = []
                for (pair, r), item in grids_by_state[sid].items():
                    f = {}
                    for code, variant in item["grid"].items():
                        vid = f"{sid}|" + "|".join(map(str, fact_key(variant["facts"])))
                        f[code] = label_logit(scores[(vid, qid)]["logits"], y, eps)
                    g = grid_features(f)
                    rule = {code: v["variant_rule_gold"][qid] for code, v in item["grid"].items()}
                    row = {"state_id": sid, "qid": qid, "primitive": schema.QID_PRIMITIVE[qid],
                           "pair": f"F{pair[0]}F{pair[1]}", "i": pair[0], "j": pair[1], "replicate": r,
                           "fi_prime": str(item["replacement"][0]), "fj_prime": str(item["replacement"][1]),
                           **g,
                           **{f"rule_gold_changes_{code}": int(rule[code] != rule["11"]) for code in ("10", "01", "00")}}
                    grid_rows.append(row)
                long_rows.extend(grid_rows)
                frame = pd.DataFrame(grid_rows)
                features = {"state_id": sid, "qid": qid, "primitive": schema.QID_PRIMITIVE[qid],
                            "f_orig": float(frame["f11"].iloc[0])}
                for pair in pair_tuples:
                    sub = frame[frame["pair"] == f"F{pair[0]}F{pair[1]}"]
                    features[f"I{pair[0]}{pair[1]}"] = float(sub["I"].mean())
                    features[f"I{pair[0]}{pair[1]}_sd"] = float(sub["I"].std(ddof=0))
                    for name in ("d_j_i1", "d_j_i0", "d_i_j1", "d_i_j0"):
                        features[f"{name}_F{pair[0]}F{pair[1]}"] = float(sub[name].mean())
                features.update(single_fact_effects(grid_rows))
                wide.append(features)
    wide_df = pd.DataFrame(wide)
    missing = [c for c in DELTA_FEATURES if c not in wide_df.columns and len(pair_tuples) == 3]
    if missing:
        raise RuntimeError(f"Missing delta features {missing}")
    return wide_df, pd.DataFrame(long_rows)


def numerical_noise_floor(scorer, records, qids=None, eps=DEFAULT_EPS):
    """Max |f| difference between batched and one-at-a-time scoring of the same states."""
    batched = scorer.score([payload(r, qids) for r in records])
    worst = 0.0
    for record in records:
        single = scorer.score([payload(record, qids)])
        for qid in (qids or record["questions"]):
            y = gold_y(record, qid)
            a = label_logit(batched[(record["id"], qid)]["logits"], y, eps)
            b = label_logit(single[(record["id"], qid)]["logits"], y, eps)
            worst = max(worst, abs(a - b))
    return float(worst)


def attach_labels(table, labels):
    """Join membership labels; every row must have a label (no silent drops)."""
    member = table["state_id"].map(labels)
    if member.isna().any():
        raise ValueError(f"{int(member.isna().sum())} rows have no membership label")
    out = table.copy()
    out["member"] = member.astype(int)
    return out


def state_level_table(table, feature_columns_by_primitive):
    """One row per state: primitive-prefixed features of its three questions side by side."""
    parts = []
    for primitive, columns in feature_columns_by_primitive.items():
        sub = table[table["primitive"] == primitive].set_index("state_id")[list(columns)]
        parts.append(sub.add_prefix(f"{primitive}."))
    merged = pd.concat(parts, axis=1, join="inner").reset_index()
    if merged.isna().any().any():
        raise ValueError("State-level table has missing primitive features")
    return merged


def finite_or_raise(frame, columns):
    values = frame[list(columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite attack features in {columns}")
    return values
