"""Score every model's saved predictions the same way and build the leaderboard.

* test score  = average over the four 2025 quarters (and over seeds)
* seed spread = standard deviation of the 2025 score across seeds
* ensemble    = average of the 3 best models on the TUNING quarter (never chosen on test)
* check       = for models predicted every day, does the Mon/Thu subset rank them the same?

  uv run python -m palletcast.models.leaderboard
"""

import json

import pandas as pd

from palletcast.models import common

FAMILY = {
    "seasonal_naive": "baseline",
    "sw_median8": "baseline",
    "auto_arima": "classical",
    "auto_ets": "classical",
    "theta": "classical",
    "mstl": "classical",
    "prophet": "classical",
    "ridge": "ml",
    "random_forest": "ml",
    "lightgbm": "ml",
    "xgboost": "ml",
    "catboost": "ml",
    "lightgbm_customer": "ml (customer level)",
    "nhits": "deep",
    "nbeats": "deep",
    "tft": "deep",
    "lstm": "deep",
}


def load_predictions():
    # single models only; the ensemble file is this script's own output
    return {
        p.stem: pd.read_parquet(p)
        for p in sorted(common.PRED_DIR.glob("*.parquet"))
        if not p.stem.startswith("ensemble")
    }


def scores_per_fold(pred, site, cfg, every_day=False):
    rows = []
    for fold in common.all_folds(cfg):
        truth = common.eval_rows(site, fold, cfg, every_day=every_day)
        for seed, p in pred[pred["fold"] == fold["name"]].groupby("seed"):
            rows.append({"fold": fold["name"], "seed": seed, **common.score(p, truth, cfg)})
    return pd.DataFrame(rows)


def seed_average(pred):
    keys = ["site_id", "origin", "target_date", "h", "fold"]
    return pred.groupby(keys, as_index=False)["yhat"].mean().assign(seed=0)


def summarise(name, s, cfg):
    test = s[s["fold"] != cfg["tune_fold"]["name"]]
    per_seed = test.groupby("seed")["wape"].mean()
    h = [c for c in test.columns if c.startswith("wape_h")]
    return {
        "model": name,
        "family": FAMILY.get(name, "ensemble"),
        "wape_2025": test["wape"].mean(),
        "wape_seed_sd": per_seed.std() if len(per_seed) > 1 else 0,
        "bias_2025": test["bias"].mean(),
        "wape_peak_2025": test["wape_peak"].mean(),
        "wape_tune": s.loc[s["fold"] == cfg["tune_fold"]["name"], "wape"].mean(),
        "wape_h1": test["wape_h01"].mean(),
        "wape_h7": test["wape_h07"].mean(),
        "wape_h14": test["wape_h14"].mean(),
        **{f"q_{f}": w for f, w in test.groupby("fold")["wape"].mean().items()},
        "_by_h": test[h].mean().to_dict(),
    }


def main():
    cfg, _ = common.load_config()
    site = common.site_table()
    preds = load_predictions()

    rows = [summarise(n, scores_per_fold(p, site, cfg), cfg) for n, p in preds.items()]
    board = pd.DataFrame(rows)

    # ensemble: 3 best single models on the tuning quarter, seed-averaged, equal weights
    top3 = board.sort_values("wape_tune")["model"].head(3).tolist()
    members = [seed_average(preds[m]) for m in top3]
    ens = members[0].copy()
    for m in members[1:]:
        ens = ens.merge(
            m[["site_id", "origin", "h", "fold", "yhat"]],
            on=["site_id", "origin", "h", "fold"],
            suffixes=("", "_x"),
        )
        ens["yhat"] = ens["yhat"] + ens.pop("yhat_x")
    ens["yhat"] = ens["yhat"] / len(members)
    ens["model"] = "ensemble_top3"
    common.save_predictions(ens, "ensemble_top3")
    board = pd.concat(
        [board, pd.DataFrame([summarise("ensemble_top3", scores_per_fold(ens, site, cfg), cfg)])],
        ignore_index=True,
    )

    board = board.sort_values("wape_2025").reset_index(drop=True)
    by_h = pd.DataFrame({r["model"]: r["_by_h"] for r in board.to_dict("records")}).T
    board = board.drop(columns="_by_h")

    # sanity: Mon/Thu subset vs every day, for models that predicted every day
    check = []
    for n, p in preds.items():
        if p.groupby("fold")["origin"].nunique().min() > 60:
            daily = scores_per_fold(p, site, cfg, every_day=True)
            check.append(
                {
                    "model": n,
                    "wape_monthu": summarise(n, scores_per_fold(p, site, cfg), cfg)["wape_2025"],
                    "wape_daily": summarise(n, daily, cfg)["wape_2025"],
                }
            )
    check = pd.DataFrame(check)

    out = common.ROOT / "reports" / "s3"
    board.to_csv(out / "leaderboard.csv", index=False)
    by_h.to_csv(out / "wape_by_horizon.csv")
    check.to_csv(out / "subset_check.csv", index=False)
    (out / "ensemble.json").write_text(json.dumps({"members": top3}, indent=1))
    pd.set_option("display.width", 200)
    print(board.round(4).to_string(index=False))
    print("\nensemble members (best on tuning quarter):", top3)
    if len(check):
        print("\nMon/Thu vs daily scoring:\n", check.round(4).to_string(index=False))
        rank_ok = (check["wape_monthu"].rank() == check["wape_daily"].rank()).all()
        print("same ranking:", bool(rank_ok))


if __name__ == "__main__":
    main()
