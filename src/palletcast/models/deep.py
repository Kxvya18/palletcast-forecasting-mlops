"""Deep-learning forecasting models (neuralforecast, CPU): N-HiTS, N-BEATS, TFT, LSTM.

One network per model is trained on the 4 site series at the start of each fold (only
labels that were already final), then used at every forecast morning in the fold with the
site history as it was known that morning. Models that accept known-future inputs get two
calendar signals: `closed` (site not shipping that day) and `pre_holiday` (a holiday in
the next 3 days).

    uv run python -m palletcast.models.deep --models nhits tft
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import LSTM, NBEATS, NHITS, TFT

from palletcast.models import common

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

H = 15  # last known day is origin-1, so 15 steps reach origin+14
EXOG = ["closed", "pre_holiday"]


def make_model(name, p, seed):
    common_args = dict(
        h=H,
        input_size=p["input_size"],
        learning_rate=p["learning_rate"],
        max_steps=p["max_steps"],
        loss=MAE(),
        scaler_type="robust",
        random_seed=seed,
        enable_progress_bar=False,
        logger=False,
        enable_model_summary=False,
        accelerator="cpu",
    )
    if name == "nhits":
        return NHITS(**common_args, futr_exog_list=EXOG, mlp_units=[[p["hidden"]] * 2] * 3)
    if name == "nbeats":  # N-BEATS takes no extra inputs
        return NBEATS(**common_args, mlp_units=[[p["hidden"]] * 2] * 3)
    if name == "tft":
        return TFT(**common_args, futr_exog_list=EXOG, hidden_size=p["hidden"])
    if name == "lstm":
        return LSTM(
            **common_args,
            futr_exog_list=EXOG,
            encoder_hidden_size=p["hidden"],
            decoder_hidden_size=p["hidden"],
        )
    raise ValueError(name)


def random_params(rng):
    return {
        "input_size": int(rng.choice([28, 56, 112])),
        "learning_rate": float(rng.choice([5e-4, 1e-3, 3e-3])),
        "max_steps": int(rng.choice([300, 600, 1000])),
        "hidden": int(rng.choice([32, 64, 128])),
    }


def calendar(hist):
    """Known-future inputs for every site and day: closed, pre_holiday."""
    closed = ~hist.working
    hol = hist.holiday_closed
    pre = pd.DataFrame(False, index=hol.index, columns=hol.columns)
    for k in (1, 2, 3):
        pre |= hol.shift(-k, axis=1, fill_value=False)
    out = pd.concat([closed.stack().rename("closed"), pre.stack().rename("pre_holiday")], axis=1)
    out = out.astype(float).reset_index()
    out.columns = ["unique_id", "ds", *EXOG]
    return out


def fit(name, p, seed, hist, cal, fold, cfg):
    start = pd.Timestamp(fold["start"])
    train = hist.long(start, settled_days=cfg["label_settle_days"]).merge(
        cal, on=["unique_id", "ds"]
    )
    nf = NeuralForecast(models=[make_model(name, p, seed)], freq="D")
    nf.fit(df=train)
    return nf


def predict_fold(nf, name, seed, hist, cal, fold, origins):
    model_col = type(nf.models[0]).__name__
    preds = []
    for origin in origins:
        origin = pd.Timestamp(origin)
        df = hist.long(origin).merge(cal, on=["unique_id", "ds"])
        future = cal[(cal["ds"] >= origin) & (cal["ds"] < origin + H * common.DAY)]
        fc = nf.predict(df=df, futr_df=future).reset_index()
        preds.append(common.to_prediction(fc, origin, fold["name"], name, seed, model_col))
    return pd.concat(preds)


def run_one(name, p, seed, hist, cal, site, fold, cfg):
    truth = common.eval_rows(site, fold, cfg)
    t = time.time()
    nf = fit(name, p, seed, hist, cal, fold, cfg)
    pred = predict_fold(nf, name, seed, hist, cal, fold, sorted(truth["origin"].unique()))
    return pred, common.score(pred, truth, cfg), time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["nhits", "nbeats", "tft", "lstm"])
    # TFT trains ~25x slower on CPU; it may need a smaller budget (reported in the leaderboard)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    a = ap.parse_args()
    cfg, feats_cfg = common.load_config()
    trials = a.trials or cfg["tuning_trials"]
    seeds = a.seeds if a.seeds is not None else cfg["seeds"]
    site = common.site_table()
    hist = common.SiteHistory(feats_cfg)
    cal = calendar(hist)

    for name in a.models:
        rng, best = np.random.default_rng(42), None
        for trial in range(trials):
            p = random_params(rng)
            pred, m, sec = run_one(name, p, 0, hist, cal, site, cfg["tune_fold"], cfg)
            common.log_run(
                name,
                cfg["tune_fold"]["name"],
                0,
                {**p, "trial": trial},
                m,
                "deep",
                sec,
                experiment="s3_tuning",
            )
            print(f"  {name} trial {trial}: WAPE {m['wape']:.4f} ({sec:.0f}s) {p}", flush=True)
            if best is None or m["wape"] < best[1]:
                best = (p, m["wape"], pred)
        p, _, tune_pred = best
        print(f"{name}: best tuning WAPE {best[1]:.4f} with {p}", flush=True)
        preds = [tune_pred]
        for fold in cfg["test_folds"]:
            for seed in seeds:
                pred, m, sec = run_one(name, p, seed, hist, cal, site, fold, cfg)
                common.log_run(name, fold["name"], seed, p, m, "deep", sec)
                print(
                    f"{name:6s} {fold['name']} seed {seed}: WAPE {m['wape']:.4f} ({sec:.0f}s)",
                    flush=True,
                )
                preds.append(pred)
        common.save_predictions(pd.concat(preds), name)


if __name__ == "__main__":
    main()
