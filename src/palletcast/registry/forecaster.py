"""The deployable forecaster: one or more trained members whose site forecasts are summed.

Stored in MLflow as a pyfunc model so S6 can load any registered version the same way.
"""

from __future__ import annotations

import json

import mlflow.pyfunc
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from palletcast.models.common import ROOT

SITES = ["S01", "S02", "S03", "S04"]
NOT_FEATURES = [
    "site_id",
    "customer_id",
    "segment",
    "origin",
    "target_date",
    "target",
    "at_capacity",
    "period",
]
KEYS = ["site_id", "origin", "target_date", "h"]


def feature_matrix(rows, segments):
    """Numeric features + one-hot site (+ segment for customer rows)."""
    X = rows.drop(columns=[c for c in NOT_FEATURES if c in rows.columns])
    for s in SITES:
        X[f"site_{s}"] = (rows["site_id"] == s).astype(float)
    if "segment" in rows.columns:
        for g in segments:
            X[f"seg_{g}"] = (rows["segment"] == g).astype(float)
    return X.astype(float)


def recent_level(rows):
    level = rows["mean_28"].where(rows["mean_28"] > 0, rows["sw_mean8"])
    return level.fillna(1).clip(lower=1)


def make_algo(algo, params, seed=0):
    if algo == "lightgbm":
        return LGBMRegressor(**params, random_state=seed, verbose=-1)
    if algo == "catboost":
        return CatBoostRegressor(**params, random_seed=seed, verbose=0)
    raise ValueError(algo)


def load_params(name):
    return json.loads((ROOT / "reports" / "s3" / "best_params" / f"{name}.json").read_text())


class Forecaster(mlflow.pyfunc.PythonModel):
    def __init__(self, members):
        self.members = members  # list of dicts from configs/registry.yaml
        self.fitted = []
        self.segments = []
        self.columns = {}

    def fit(self, site_df, cust_df, cutoff):
        """Train every member on labels with target_date < cutoff."""
        self.segments = sorted(cust_df["segment"].unique())
        self.fitted = []
        for i, m in enumerate(self.members):
            table = cust_df if m["grain"] == "customer" else site_df
            train = table[(table["target_date"] < cutoff) & table["target"].notna()]
            X = feature_matrix(train, self.segments)
            y = train["target"] / recent_level(train) if m["ratio"] else train["target"]
            model = make_algo(m["algo"], load_params(m["params"]))
            model.fit(X, y)
            self.columns[i] = list(X.columns)
            self.fitted.append(model)
        return self

    def predict_sites(self, site_df, cust_df):
        """Site-level forecasts for the given feature rows: KEYS + yhat."""
        total = None
        for i, (m, model) in enumerate(zip(self.members, self.fitted, strict=True)):
            table = cust_df if m["grain"] == "customer" else site_df
            X = feature_matrix(table, self.segments)[self.columns[i]]
            yhat = model.predict(X)
            if m["ratio"]:
                yhat = yhat * recent_level(table).to_numpy()
            p = table[KEYS].assign(yhat=np.clip(yhat, 0, None) * m["weight"])
            p = p.groupby(KEYS, as_index=False)["yhat"].sum()
            total = (
                p
                if total is None
                else total.merge(p, on=KEYS, suffixes=("", "_m")).assign(
                    yhat=lambda d: d["yhat"] + d.pop("yhat_m")
                )
            )
        return total

    def predict(self, context, model_input, params=None):
        """pyfunc entry point: model_input = {"site": site rows, "customer": customer rows}."""
        return self.predict_sites(model_input["site"], model_input["customer"])
