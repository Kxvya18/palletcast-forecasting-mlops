"""Shared pieces for the S3 backtest: data, folds, scoring, saving and MLflow logging.

Every model, whatever its family, produces the same prediction table:
    site_id, origin, target_date, h, fold, model, seed, yhat
and is scored the same way, so the leaderboard is a fair comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from palletcast.features import build as fb

DAY = pd.Timedelta(days=1)
ROOT = Path(__file__).resolve().parents[3]
PRED_DIR = ROOT / "reports" / "s3" / "preds"


def load_config():
    train = yaml.safe_load(open(ROOT / "configs" / "train.yaml"))
    feats = yaml.safe_load(open(ROOT / "configs" / "features.yaml"))
    return train, feats


def all_folds(cfg):
    return [cfg["tune_fold"]] + cfg["test_folds"]


def site_table():
    return pd.read_parquet(ROOT / "data" / "features" / "site.parquet")


def eval_rows(site, fold, cfg, every_day=False):
    """The forecasts a fold is scored on: origins inside the fold (Mon/Thu unless
    every_day), targets inside the history period, and a valid target."""
    r = site[
        (site["origin"] >= fold["start"])
        & (site["origin"] <= fold["end"])
        & (site["target_date"] <= cfg["last_target_date"])
        & site["target"].notna()
    ]
    if not every_day:
        r = r[r["origin"].dt.dayofweek.isin(cfg["eval_origin_weekdays"])]
    return r


def train_cutoff(fold, cfg):
    """Training labels must already have been final when the fold starts."""
    return pd.Timestamp(fold["start"]) - cfg["label_settle_days"] * DAY


# ---------------------------------------------------------------- scoring


def wape(y, yhat):
    return float(np.abs(y - yhat).sum() / y.sum())


def bias(y, yhat):
    """Positive = over-forecast, negative = under-forecast (share of volume)."""
    return float((yhat - y).sum() / y.sum())


def score(pred, truth, cfg):
    """pred: prediction table for one model/fold/seed. truth: rows with `target`."""
    m = truth[["site_id", "origin", "h", "target"]].merge(pred, on=["site_id", "origin", "h"])
    y, p = m["target"].to_numpy(), m["yhat"].to_numpy()
    peak_cut = m.groupby("site_id")["target"].transform(lambda t: t.quantile(cfg["peak_quantile"]))
    peak = m["target"] >= peak_cut
    out = {"wape": wape(y, p), "bias": bias(y, p), "wape_peak": wape(y[peak], p[peak]), "n": len(m)}
    for h, g in m.groupby("h"):
        out[f"wape_h{h:02d}"] = wape(g["target"].to_numpy(), g["yhat"].to_numpy())
    return out


# ---------------------------------------------------------------- saving / tracking


def save_predictions(pred, model):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(PRED_DIR / f"{model}.parquet", index=False)


def log_run(model, fold, seed, params, metrics, family, seconds, experiment="s3_backtest"):
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{model}-{fold}-s{seed}"):
        mlflow.set_tags({"model": model, "family": family, "fold": fold, "seed": seed})
        mlflow.log_params(
            {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in params.items()}
        )
        mlflow.log_metrics({**metrics, "seconds": seconds})


# ---------------------------------------------------------------- series for classical / deep


class SiteHistory:
    """Daily site volumes as known at any origin, ready for models that need a gap-free
    series. Outage / flagged days are filled with the median of the same weekday over the
    previous 8 weeks; holiday closures stay 0 (the site really was closed)."""

    def __init__(self, feats_cfg):
        self.cfg = feats_cfg
        self.inp = fb.load_inputs(ROOT / "data" / "validated")
        v = self.inp["versions"]
        self.start = v["ship_date"].min()
        self.days = pd.date_range(self.start, v["ship_date"].max())
        self.sites = pd.Index(self.inp["sites"]["site_id"], name="site_id")
        self.final, self.known = fb.daily_tables(v, "site_id", self.sites, self.days, feats_cfg)
        self.working, self.holiday_closed = fb.site_calendar(
            self.inp["sites"], self.inp["holidays"], self.days
        )
        low = self.inp["flags"][self.inp["flags"]["type"] == "volume_low"]
        self.low_flags = pd.DataFrame(False, index=self.sites, columns=self.days)
        for s, d in zip(low["site_id"], low["date"], strict=True):
            self.low_flags.loc[s, d] = True

    def as_of(self, origin):
        """Series x day table from the first day up to origin-1 as known at origin 06:00."""
        n_days = (origin - self.start).days
        hist = fb.history_as_of(self.final, self.known, origin, n_days)
        hide = fb.unusable_days(
            hist, self.working, self.holiday_closed, self.low_flags, origin, self.cfg
        )
        hide = hide & ~self.holiday_closed.reindex(columns=hist.columns, fill_value=False)
        hist = hist.mask(hide)
        return fill_gaps(hist)

    def long(self, origin, settled_days=0):
        """Long format (unique_id, ds, y) as used by statsforecast / neuralforecast."""
        t = self.as_of(origin)
        if settled_days:
            t = t.iloc[:, :-settled_days]
        out = t.stack().rename("y").reset_index()
        out.columns = ["unique_id", "ds", "y"]
        return out


def fill_gaps(hist):
    """Fill NaN with the median of the same weekday over the previous 8 weeks (else 0)."""
    filled = hist.copy()
    for day in hist.columns[hist.isna().any(axis=0)]:
        prev = [day - 7 * k * DAY for k in range(1, 9)]
        prev = [d for d in prev if d in hist.columns]
        med = hist[prev].median(axis=1) if prev else 0
        filled[day] = hist[day].fillna(med).fillna(0)
    return filled


def to_prediction(fc, origin, fold, model, seed, value_col):
    """Turn a model's forecast for days origin+1..origin+14 into the prediction table."""
    fc = fc.rename(columns={"unique_id": "site_id", "ds": "target_date", value_col: "yhat"})
    fc["h"] = (fc["target_date"] - origin).dt.days
    fc = fc[(fc["h"] >= 1) & (fc["h"] <= 14)]
    return fc.assign(
        origin=origin, fold=fold, model=model, seed=seed, yhat=fc["yhat"].clip(lower=0)
    )[["site_id", "origin", "target_date", "h", "fold", "model", "seed", "yhat"]]
