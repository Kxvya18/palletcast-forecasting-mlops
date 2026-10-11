"""Machine-learning models on the S2 site features: Ridge, Random Forest, LightGBM,
XGBoost, CatBoost.

For each model: try `tuning_trials` random settings on the tuning fold (Q4 2024), keep the
best, then train it on each 2025 fold with every seed. Predictions are made for every
daily origin (the leaderboard uses the Mon/Thu subset, like all other models).

    uv run python -m palletcast.models.ml --models lightgbm xgboost
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from palletcast.models import common

NOT_FEATURES = ["site_id", "origin", "target_date", "target", "at_capacity", "period"]


def make_model(name, params, seed):
    if name == "ridge":
        # linear models cannot handle NaN: fill with the median and add "was missing" flags
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(**params)
        )
    if name == "random_forest":
        return RandomForestRegressor(**params, random_state=seed, n_jobs=-1)
    if name == "lightgbm":
        return LGBMRegressor(**params, random_state=seed, verbose=-1)
    if name == "xgboost":
        return XGBRegressor(**params, random_state=seed)
    if name == "catboost":
        return CatBoostRegressor(**params, random_seed=seed, verbose=0)
    raise ValueError(name)


def random_params(name, rng):
    """One random setting from each model's search space."""
    pick = rng.choice
    if name == "ridge":
        return {"alpha": float(10 ** rng.uniform(-2, 3))}
    if name == "random_forest":
        return {
            "n_estimators": int(pick([200, 400])),
            "max_depth": pick([None, 12, 20]),
            "min_samples_leaf": int(pick([1, 5, 20])),
            "max_features": float(pick([0.3, 0.6, 1.0])),
        }
    if name == "lightgbm":
        return {
            "n_estimators": int(pick([300, 600, 1000])),
            "learning_rate": float(pick([0.02, 0.05, 0.1])),
            "num_leaves": int(pick([15, 31, 63])),
            "min_child_samples": int(pick([10, 30, 100])),
            "subsample": float(pick([0.7, 1.0])),
            "subsample_freq": 1,
            "colsample_bytree": float(pick([0.6, 0.8, 1.0])),
        }
    if name == "xgboost":
        return {
            "n_estimators": int(pick([300, 600, 1000])),
            "learning_rate": float(pick([0.02, 0.05, 0.1])),
            "max_depth": int(pick([4, 6, 8])),
            "min_child_weight": float(pick([1, 5, 20])),
            "subsample": float(pick([0.7, 1.0])),
            "colsample_bytree": float(pick([0.6, 0.8, 1.0])),
        }
    if name == "catboost":
        return {
            "iterations": int(pick([500, 1000])),
            "learning_rate": float(pick([0.03, 0.06, 0.1])),
            "depth": int(pick([4, 6, 8])),
            "l2_leaf_reg": float(pick([1, 3, 10])),
        }
    raise ValueError(name)


def feature_matrix(rows):
    """Numeric features plus one-hot site columns (works the same for every model)."""
    X = rows.drop(columns=NOT_FEATURES)
    for s in ["S01", "S02", "S03", "S04"]:
        X[f"site_{s}"] = (rows["site_id"] == s).astype(float)
    return X.astype(float)


def fit_predict(name, params, seed, site, fold, cfg):
    train = site[(site["target_date"] < common.train_cutoff(fold, cfg)) & site["target"].notna()]
    test = common.eval_rows(site, fold, cfg, every_day=True)
    model = make_model(name, params, seed)
    model.fit(feature_matrix(train), train["target"])
    yhat = np.clip(model.predict(feature_matrix(test)), 0, None)
    return test[["site_id", "origin", "target_date", "h"]].assign(
        fold=fold["name"], model=name, seed=seed, yhat=yhat
    )


def tune(name, site, cfg):
    fold, rng = cfg["tune_fold"], np.random.default_rng(42)
    truth = common.eval_rows(site, fold, cfg)
    best = None
    for trial in range(cfg["tuning_trials"]):
        params = random_params(name, rng)
        t = time.time()
        p = fit_predict(name, params, 0, site, fold, cfg)
        m = common.score(p, truth, cfg)
        common.log_run(
            name,
            fold["name"],
            0,
            {**params, "trial": trial},
            m,
            "ml",
            time.time() - t,
            experiment="s3_tuning",
        )
        print(f"  {name} trial {trial}: WAPE {m['wape']:.4f}")
        if best is None or m["wape"] < best[1]:
            best = (params, m["wape"], p)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", nargs="+", default=["ridge", "random_forest", "lightgbm", "xgboost", "catboost"]
    )
    a = ap.parse_args()
    cfg, _ = common.load_config()
    site = common.site_table()
    for name in a.models:
        params, tune_wape, tune_pred = tune(name, site, cfg)
        print(f"{name}: best tuning WAPE {tune_wape:.4f} with {params}")
        best_dir = common.ROOT / "reports" / "s3" / "best_params"
        best_dir.mkdir(parents=True, exist_ok=True)
        (best_dir / f"{name}.json").write_text(json.dumps(params, indent=1))
        preds = [tune_pred]
        seeds = [0] if name == "ridge" else cfg["seeds"]  # ridge has no randomness
        for fold in cfg["test_folds"]:
            for seed in seeds:
                t = time.time()
                p = fit_predict(name, params, seed, site, fold, cfg)
                m = common.score(p, common.eval_rows(site, fold, cfg), cfg)
                common.log_run(name, fold["name"], seed, params, m, "ml", time.time() - t)
                print(f"{name:14s} {fold['name']} seed {seed}: WAPE {m['wape']:.4f}")
                preds.append(p)
        common.save_predictions(pd.concat(preds), name)


if __name__ == "__main__":
    main()
