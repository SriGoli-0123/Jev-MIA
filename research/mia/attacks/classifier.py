"""L2-regularized logistic-regression membership attacker g(z) = sigmoid(w^T z + b).

Features are standardized on the shadow attack-training rows only. The decision threshold
gamma is chosen on a held-out shadow calibration split to maximize balanced accuracy, as in
arXiv:2610.04985. Fitting refuses any row that did not come from the shadow model, so the
attacker cannot see target membership labels.
"""
import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mia.attacks.extraction import finite_or_raise


class TargetLabelLeak(RuntimeError):
    pass


class AttackClassifier:
    def __init__(self, features, C=1.0, max_iter=2000):
        self.features = list(features)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.scaler = None
        self.model = None
        self.threshold = None
        self.frozen = False

    def fit(self, table):
        if self.frozen:
            raise RuntimeError("Attack classifier is frozen")
        assert_shadow_only(table)
        X = finite_or_raise(table, self.features)
        y = table["member"].to_numpy(dtype=int)
        if len(set(y)) != 2:
            raise ValueError("Shadow attack data must contain members and non-members")
        self.scaler = StandardScaler().fit(X)
        # L2 is the default penalty in every scikit-learn version; passing it explicitly is deprecated in 1.8.
        self.model = LogisticRegression(C=self.C, solver="lbfgs", max_iter=self.max_iter)
        self.model.fit(self.scaler.transform(X), y)
        return self

    def score(self, table):
        X = finite_or_raise(table, self.features)
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def calibrate_threshold(self, table):
        """gamma maximizing balanced accuracy on shadow calibration rows."""
        assert_shadow_only(table)
        scores, y = self.score(table), table["member"].to_numpy(dtype=int)
        best, best_gamma = -1.0, 0.5
        for gamma in np.unique(scores):
            pred = scores >= gamma
            tpr = pred[y == 1].mean()
            tnr = (~pred[y == 0]).mean()
            if (tpr + tnr) / 2 > best:
                best, best_gamma = (tpr + tnr) / 2, float(gamma)
        self.threshold = best_gamma
        return best_gamma, best

    def freeze(self):
        self.frozen = True
        return self

    def describe(self):
        return {"features": self.features, "C": self.C, "penalty": "l2",
                "coef_standardized": self.model.coef_[0].tolist(), "intercept": float(self.model.intercept_[0]),
                "scaler_mean": self.scaler.mean_.tolist(), "scaler_scale": self.scaler.scale_.tolist(),
                "threshold_gamma": self.threshold, "frozen": self.frozen}

    def save(self, path_stem):
        path_stem = Path(path_stem)
        path_stem.parent.mkdir(parents=True, exist_ok=True)
        if Path(f"{path_stem}.joblib").exists():
            raise FileExistsError(f"Refusing to overwrite {path_stem}.joblib")
        joblib.dump({"scaler": self.scaler, "model": self.model, "features": self.features,
                     "threshold": self.threshold}, f"{path_stem}.joblib")
        with open(f"{path_stem}.json", "x", encoding="utf-8") as handle:
            json.dump(self.describe(), handle, indent=2)


def assert_shadow_only(table):
    roles = set(table["model_role"].unique()) if "model_role" in table else {None}
    if roles != {"shadow"}:
        raise TargetLabelLeak(f"Attack fitting/calibration accepts shadow-model rows only, got roles {roles}")


def stratified_state_split(table, fraction, seed):
    """Split shadow rows into attack-train and calibration by state, stratified by membership."""
    rng = random.Random(seed)
    states = table.drop_duplicates("state_id")[["state_id", "member"]]
    calib = set()
    for label in (0, 1):
        ids = sorted(states.loc[states["member"] == label, "state_id"])
        rng.shuffle(ids)
        calib.update(ids[:max(1, round(fraction * len(ids)))])
    mask = table["state_id"].isin(calib)
    return table[~mask].copy(), table[mask].copy()
