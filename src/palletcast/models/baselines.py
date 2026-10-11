"""B1 seasonal naive and B2 8-week same-weekday median, read straight from the features."""

import time

from palletcast.models import common

BASELINES = {"seasonal_naive": "sw_lag1", "sw_median8": "sw_median8"}


def main():
    cfg, _ = common.load_config()
    site = common.site_table()
    for model, col in BASELINES.items():
        preds = []
        for fold in common.all_folds(cfg):
            t = time.time()
            rows = common.eval_rows(site, fold, cfg, every_day=True)
            # an outage can blank the lag; fall back to the 8-week median, then the 28-day mean
            yhat = rows[col].fillna(rows["sw_median8"]).fillna(rows["mean_28"]).fillna(0)
            p = rows[["site_id", "origin", "target_date", "h"]].assign(
                fold=fold["name"], model=model, seed=0, yhat=yhat.to_numpy()
            )
            preds.append(p)
            m = common.score(p, common.eval_rows(site, fold, cfg), cfg)
            common.log_run(model, fold["name"], 0, {"feature": col}, m, "baseline", time.time() - t)
            print(f"{model:15s} {fold['name']:12s} WAPE {m['wape']:.4f}")
        common.save_predictions(__import__("pandas").concat(preds), model)


if __name__ == "__main__":
    main()
