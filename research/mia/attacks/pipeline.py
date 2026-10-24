"""Shadow-to-target attack pipeline shared by every MIA experiment.

attack_pipeline() is the only place an attacker is trained. Its inputs are a labeled
shadow table and an *unlabeled* target table; target membership labels are not an
argument and a labeled target table is rejected. Labels enter only afterwards, in
evaluate_scores(), which never touches the classifier.
"""
import numpy as np
import pandas as pd

from mia.attacks.classifier import AttackClassifier, TargetLabelLeak, stratified_state_split
from mia.attacks.confidence import FEATURE_SETS
from mia.evaluation.bootstrap import bootstrap_summary


def expand_features(spec, primitive):
    """Feature-set spec -> column names. 'confidence' / 'confidence_extended' expand per primitive."""
    columns = []
    for item in spec:
        if item == "confidence":
            columns.extend(FEATURE_SETS["paper"][primitive])
        elif item == "confidence_extended":
            columns.extend(FEATURE_SETS["extended"][primitive])
        else:
            columns.append(item)
    return list(dict.fromkeys(columns))


def key_columns(table):
    return [c for c in ("state_id", "qid", "primitive") if c in table.columns]


def attack_pipeline(shadow_table, target_table, features, *, C=1.0, max_iter=2000, calibration_fraction=0.2,
                    seed=0):
    """Fit g on shadow rows, pick gamma on shadow calibration rows, freeze g, score target rows."""
    if "member" in target_table.columns:
        raise TargetLabelLeak("Target rows must be unlabeled when the attack is trained and applied")
    if set(target_table.get("model_role", pd.Series(["target"])).unique()) != {"target"}:
        raise ValueError("Target table must come from the target model")
    train, calibration = stratified_state_split(shadow_table, calibration_fraction, seed)
    classifier = AttackClassifier(features, C=C, max_iter=max_iter).fit(train)
    gamma, calibration_bacc = classifier.calibrate_threshold(calibration)
    classifier.freeze()
    calibration_scores = classifier.score(calibration)
    scored = target_table[key_columns(target_table)].copy()
    scored["score"] = classifier.score(target_table)
    info = {"gamma": gamma, "shadow_calibration_balanced_accuracy": calibration_bacc,
            "shadow_train_rows": int(len(train)), "shadow_calibration_rows": int(len(calibration)),
            "shadow_calibration_auc": _auc(calibration["member"].to_numpy(), calibration_scores)}
    return classifier, scored, info


def _auc(y, scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None


def evaluate_scores(scored, labels, *, n_boot=1000, seed=0, fprs=(0.001, 0.01, 0.05), gamma=None, rows=None):
    """Join target labels to frozen-attack scores and compute metrics with state-level bootstrap CIs."""
    frame = scored if rows is None else scored.iloc[rows]
    y = frame["state_id"].map(labels)
    if y.isna().any():
        raise ValueError("Scored rows without a target membership label")
    return bootstrap_summary(y.to_numpy(dtype=int), frame["score"].to_numpy(dtype=np.float64),
                             groups=frame["state_id"].to_numpy(), n_boot=n_boot, seed=seed, fprs=tuple(fprs),
                             gamma=gamma)
