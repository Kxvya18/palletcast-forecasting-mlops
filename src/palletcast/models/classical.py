"""Classical forecasting models, one per site, refitted at every forecast morning on the
site's history as it was known that morning.

    AutoARIMA, AutoETS, AutoTheta, MSTL  (statsforecast)
    Prophet                              (Meta), with the site country's public holidays

    uv run python -m palletcast.models.classical
"""

import argparse
import logging
import time

import pandas as pd
from prophet import Prophet
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA, AutoETS, AutoTheta

from palletcast.models import common

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

STATS_MODELS = {
    "auto_arima": lambda: AutoARIMA(season_length=7),
    "auto_ets": lambda: AutoETS(season_length=7),
    "theta": lambda: AutoTheta(season_length=7),
    "mstl": lambda: MSTL(season_length=[7, 365]),  # weekly + yearly cycles
}


ARIMA_HISTORY_DAYS = 365  # ARIMA with a weekly season only needs recent data; full history
# made it ~80 s per forecast morning instead of a few seconds


def stats_forecast(history, names):
    """Forecast 15 days after the last known day (origin-1): origin .. origin+14."""
    if names == ["auto_arima"]:
        history = history[history["ds"] > history["ds"].max() - ARIMA_HISTORY_DAYS * common.DAY]
    sf = StatsForecast(
        models=[STATS_MODELS[n]() for n in names], freq="D", n_jobs=1
    )  # 4 series: a process pool costs more than it saves
    fc = sf.forecast(df=history, h=15)
    rename = {type(STATS_MODELS[n]()).__name__: n for n in names}
    return fc.rename(columns=rename)


def prophet_forecast(history, countries):
    out = []
    for site, g in history.groupby("unique_id"):
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
        m.add_country_holidays(country_name=countries[site])
        m.fit(g[["ds", "y"]])
        future = pd.DataFrame({"ds": pd.date_range(g["ds"].max() + common.DAY, periods=15)})
        fc = m.predict(future)[["ds", "yhat"]].assign(unique_id=site)
        out.append(fc.rename(columns={"yhat": "prophet"}))
    return pd.concat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[*STATS_MODELS, "prophet"])
    ap.add_argument("--limit-origins", type=int, default=None, help="for a quick timing test")
    a = ap.parse_args()
    cfg, feats_cfg = common.load_config()
    site = common.site_table()
    hist = common.SiteHistory(feats_cfg)
    countries = hist.inp["sites"].set_index("site_id")["country"].to_dict()
    stats_names = [n for n in a.models if n in STATS_MODELS]

    preds = {n: [] for n in a.models}
    for fold in common.all_folds(cfg):
        truth = common.eval_rows(site, fold, cfg)
        origins = sorted(truth["origin"].unique())[: a.limit_origins]
        seconds = {n: 0.0 for n in a.models}
        for origin in origins:
            origin = pd.Timestamp(origin)
            history = hist.long(origin)
            for n in stats_names:
                t = time.time()
                fc = stats_forecast(history, [n])
                preds[n].append(common.to_prediction(fc, origin, fold["name"], n, 0, n))
                seconds[n] += time.time() - t
            if "prophet" in a.models:
                t = time.time()
                fc = prophet_forecast(history, countries)
                preds["prophet"].append(
                    common.to_prediction(fc, origin, fold["name"], "prophet", 0, "prophet")
                )
                seconds["prophet"] += time.time() - t
        for n in a.models:
            p = pd.concat([x for x in preds[n] if x["fold"].iloc[0] == fold["name"]])
            m = common.score(p, truth, cfg)
            common.log_run(n, fold["name"], 0, {}, m, "classical", seconds[n])
            print(f"{n:11s} {fold['name']:12s} WAPE {m['wape']:.4f}  ({seconds[n]:.0f}s)")
    if a.limit_origins is None:
        for n in a.models:
            common.save_predictions(pd.concat(preds[n]), n)


if __name__ == "__main__":
    main()
