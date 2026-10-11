"""S3 experiments, run with LightGBM (statistically tied with the best single models and the
fastest to train). Every variant uses the tuned settings, 3 seeds and the same 2025 scoring.

  E1 grain       site-level model vs customer-level model summed to sites
  E2 outages     masked (NaN) lags vs lags filled with the 8-week same-weekday median
  E3 leakage     honest as-of features vs features built from final data
  E4 objective   squared error vs Tweedie (made for spiky, non-negative counts)
  E5 retraining  retrain every quarter vs every month
  E6 target      predict pallets directly vs predict target / recent level (trees cannot
                 extrapolate beyond training values; a ratio carries over to any level)

    uv run python -m palletcast.models.experiments --only E1 E3
"""

import argparse
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from palletcast.features import build as fb
from palletcast.models import common
from palletcast.models.ml import NOT_FEATURES

OUT = common.ROOT / "reports" / "s3" / "experiments"
SITES = ["S01", "S02", "S03", "S04"]
LAGS = ["sw_lag1", "sw_lag2", "sw_lag3", "sw_lag4", "last_1"]


def site_X(rows):
    X = rows.drop(columns=NOT_FEATURES)
    for s in SITES:
        X[f"site_{s}"] = (rows["site_id"] == s).astype(float)
    return X.astype(float)


def customer_X(rows, segments):
    X = rows.drop(columns=NOT_FEATURES + ["customer_id", "segment"], errors="ignore")
    for s in SITES:
        X[f"site_{s}"] = (rows["site_id"] == s).astype(float)
    for g in segments:
        X[f"seg_{g}"] = (rows["segment"] == g).astype(float)
    return X.astype(float)


def recent_level(rows):
    """The series' recent daily level: 28-day mean of working days (never 0)."""
    level = rows["mean_28"].where(rows["mean_28"] > 0, rows["sw_mean8"])
    return level.fillna(1).clip(lower=1)


def train_and_predict(train, test, make_X, params, seed, ratio=False):
    model = LGBMRegressor(**params, random_state=seed, verbose=-1)
    if ratio:
        model.fit(make_X(train), train["target"] / recent_level(train))
        yhat = model.predict(make_X(test)) * recent_level(test)
    else:
        model.fit(make_X(train), train["target"])
        yhat = model.predict(make_X(test))
    return np.clip(yhat, 0, None)


def to_site_pred(test, yhat, fold, variant, seed):
    """Prediction table at site level (customer rows are summed per site/origin/h)."""
    p = test[["site_id", "origin", "target_date", "h"]].assign(yhat=yhat)
    p = p.groupby(["site_id", "origin", "target_date", "h"], as_index=False)["yhat"].sum()
    return p.assign(fold=fold, model=variant, seed=seed)


def run_variant(
    variant, folds, train_tab, test_tab, make_X, params, cfg, site, seeds, cutoff=None, ratio=False
):
    """Train per fold on train_tab, predict test_tab, score at site level."""
    preds, rows = [], []
    for fold in folds:
        cut = cutoff(fold) if cutoff else common.train_cutoff(fold, cfg)
        train = train_tab[(train_tab["target_date"] < cut) & train_tab["target"].notna()]
        test = test_tab[
            (test_tab["origin"] >= fold["start"])
            & (test_tab["origin"] <= fold["end"])
            & (test_tab["target_date"] <= cfg["last_target_date"])
            & test_tab["origin"].dt.dayofweek.isin(cfg["eval_origin_weekdays"])
        ]
        for seed in seeds:
            yhat = train_and_predict(train, test, make_X, params, seed, ratio)
            p = to_site_pred(test, yhat, fold["name"], variant, seed)
            m = common.score(p, common.eval_rows(site, fold, cfg), cfg)
            rows.append({"variant": variant, "fold": fold["name"], "seed": seed, **m})
            preds.append(p)
    return pd.concat(preds), pd.DataFrame(rows)


def summary(scores):
    s = scores.groupby("variant").agg(
        wape=("wape", "mean"), bias=("bias", "mean"), wape_peak=("wape_peak", "mean")
    )
    s["wape_seed_sd"] = scores.groupby(["variant", "seed"])["wape"].mean().groupby("variant").std()
    return s


def lgbm_params(name="lightgbm"):
    return json.loads((common.ROOT / "reports" / "s3" / "best_params" / f"{name}.json").read_text())


# ---------------------------------------------------------------- the experiments


def e1_grain(cfg, site, seeds):
    cust = pd.read_parquet(common.ROOT / "data" / "features" / "customer.parquet")
    segments = sorted(cust["segment"].unique())
    make_c = lambda r: customer_X(r, segments)  # noqa: E731
    # the customer model gets its own tuning, same budget as every model
    from palletcast.models.ml import random_params

    rng, best = np.random.default_rng(42), None
    for _ in range(cfg["tuning_trials"]):
        p = random_params("lightgbm", rng)
        _, s = run_variant("customer", [cfg["tune_fold"]], cust, cust, make_c, p, cfg, site, [0])
        if best is None or s["wape"].mean() < best[1]:
            best = (p, s["wape"].mean())
    print("customer-level best tuning WAPE", round(best[1], 4), best[0], flush=True)
    a = run_variant(
        "site_level", cfg["test_folds"], site, site, site_X, lgbm_params(), cfg, site, seeds
    )
    b = run_variant(
        "customer_bottom_up", cfg["test_folds"], cust, cust, make_c, best[0], cfg, site, seeds
    )
    return pd.concat([a[1], b[1]]), {"customer_params": best[0]}


def e2_outages(cfg, site, seeds):
    filled = site.copy()
    for c in LAGS:
        filled[c] = filled[c].fillna(filled["sw_median8"])
    a = run_variant(
        "masked", cfg["test_folds"], site, site, site_X, lgbm_params(), cfg, site, seeds
    )
    b = run_variant(
        "imputed", cfg["test_folds"], filled, filled, site_X, lgbm_params(), cfg, site, seeds
    )
    return pd.concat([a[1], b[1]]), {}


def e3_leakage(cfg, site, seeds):
    """Final-data features: as if every record arrived instantly and was never corrected."""
    _, feats_cfg = common.load_config()
    inp = fb.load_inputs(common.ROOT / "data" / "validated")
    v = inp["versions"].sort_values(["shipment_id", "ingested_at"])
    v = v.drop_duplicates("shipment_id", keep="last").copy()
    v["ingested_at"] = v["ship_date"]
    naive_site, _ = fb.build(
        {**inp, "versions": v}, feats_cfg, origins=pd.date_range("2023-03-01", "2025-12-31")
    )
    naive_site = naive_site.drop(columns=["target"]).merge(
        site[["site_id", "origin", "h", "target"]], on=["site_id", "origin", "h"]
    )
    naive_site = naive_site[site.columns]
    honest = run_variant(
        "honest_train_honest_live",
        cfg["test_folds"],
        site,
        site,
        site_X,
        lgbm_params(),
        cfg,
        site,
        seeds,
    )
    real = run_variant(
        "naive_train_honest_live",
        cfg["test_folds"],
        naive_site,
        site,
        site_X,
        lgbm_params(),
        cfg,
        site,
        seeds,
    )
    illusion = run_variant(
        "naive_train_naive_test",
        cfg["test_folds"],
        naive_site,
        naive_site,
        site_X,
        lgbm_params(),
        cfg,
        site,
        seeds,
    )
    return pd.concat([honest[1], real[1], illusion[1]]), {}


def e4_objective(cfg, site, seeds):
    p = lgbm_params()
    a = run_variant("squared_error", cfg["test_folds"], site, site, site_X, p, cfg, site, seeds)
    b = run_variant(
        "tweedie",
        cfg["test_folds"],
        site,
        site,
        site_X,
        {**p, "objective": "tweedie", "tweedie_variance_power": 1.2},
        cfg,
        site,
        seeds,
    )
    return pd.concat([a[1], b[1]]), {}


def e5_retraining(cfg, site, seeds):
    """Monthly: split each 2025 quarter into months and retrain at each month start."""
    months = []
    for start in pd.date_range("2025-01-01", "2025-12-01", freq="MS"):
        end = start + pd.offsets.MonthEnd(0)
        months.append(
            {"name": f"2025-{start.month:02d}", "start": str(start.date()), "end": str(end.date())}
        )
    quarterly = run_variant(
        "quarterly", cfg["test_folds"], site, site, site_X, lgbm_params(), cfg, site, seeds
    )
    monthly_preds, _ = run_variant(
        "monthly", months, site, site, site_X, lgbm_params(), cfg, site, seeds
    )
    # score monthly predictions on the same quarterly folds so the numbers are comparable
    rows = []
    for fold in cfg["test_folds"]:
        p = monthly_preds[
            (monthly_preds["origin"] >= fold["start"]) & (monthly_preds["origin"] <= fold["end"])
        ]
        for seed, g in p.groupby("seed"):
            rows.append(
                {
                    "variant": "monthly",
                    "fold": fold["name"],
                    "seed": seed,
                    **common.score(g, common.eval_rows(site, fold, cfg), cfg),
                }
            )
    return pd.concat([quarterly[1], pd.DataFrame(rows)]), {}


def e6_target(cfg, site, seeds):
    p = lgbm_params()
    a = run_variant("pallets", cfg["test_folds"], site, site, site_X, p, cfg, site, seeds)
    b = run_variant(
        "ratio_to_level", cfg["test_folds"], site, site, site_X, p, cfg, site, seeds, ratio=True
    )
    return pd.concat([a[1], b[1]]), {}


def save_customer_model(cfg, site, seeds):
    """The E1 winner as a leaderboard entry: customer-level LightGBM summed to sites."""
    cust = pd.read_parquet(common.ROOT / "data" / "features" / "customer.parquet")
    segments = sorted(cust["segment"].unique())
    params = json.loads((OUT / "E1_summary.json").read_text())["customer_params"]
    (common.ROOT / "reports" / "s3" / "best_params" / "lightgbm_customer.json").write_text(
        json.dumps(params, indent=1)
    )
    tune, _ = run_variant(
        "lightgbm_customer",
        [cfg["tune_fold"]],
        cust,
        cust,
        lambda r: customer_X(r, segments),
        params,
        cfg,
        site,
        [0],
    )
    test, scores = run_variant(
        "lightgbm_customer",
        cfg["test_folds"],
        cust,
        cust,
        lambda r: customer_X(r, segments),
        params,
        cfg,
        site,
        seeds,
    )
    common.save_predictions(pd.concat([tune, test]), "lightgbm_customer")
    return scores, {}


EXPERIMENTS = {
    "E1": e1_grain,
    "E2": e2_outages,
    "E3": e3_leakage,
    "E4": e4_objective,
    "E5": e5_retraining,
    "E6": e6_target,
    "save_customer_model": save_customer_model,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=list(EXPERIMENTS))
    a = ap.parse_args()
    cfg, _ = common.load_config()
    site = common.site_table()
    OUT.mkdir(parents=True, exist_ok=True)
    for key in a.only:
        scores, extra = EXPERIMENTS[key](cfg, site, cfg["seeds"])
        scores.to_csv(OUT / f"{key}_scores.csv", index=False)
        s = summary(scores)
        (OUT / f"{key}_summary.json").write_text(
            json.dumps({"summary": s.round(5).reset_index().to_dict("records"), **extra}, indent=1)
        )
        print(f"\n{key}:\n{s.round(4).to_string()}", flush=True)


if __name__ == "__main__":
    main()
