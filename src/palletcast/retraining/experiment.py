"""S8: which retraining policy works best on April-September 2026?

  frozen     keep the production model of 1 April (v7) all along
  monthly    retrain on the 1st of every month
  triggered  every Monday look at last week's forecast error per site; if any site breaks
             the weekly alert limits, retrain that day

Every retrain is checked by the S4 gate rule before it replaces the current model. Models
train only on labels that were final on the retrain day (21-day settle rule), so a policy
can only learn from a change once 3 weeks of it have been settled.
Nothing is registered: this is an experiment; results go to reports/s8/.

    uv run python -m palletcast.retraining.experiment
"""

import json

import pandas as pd
import yaml

from palletcast.models.common import ROOT, bias, wape
from palletcast.registry import registry
from palletcast.registry.forecaster import Forecaster

CFG = yaml.safe_load(open(ROOT / "configs" / "monitoring.yaml"))["retraining"]
OUT = ROOT / "reports" / "s8"
DAY = pd.Timedelta(days=1)


def retrain(site, cust, run_date, current):
    """Train a candidate for run_date and apply the S4 gate against the current model."""
    h0, h1 = registry.holdout_window(run_date)
    candidate_eval = Forecaster(registry.CFG["models"]["production_candidate"]).fit(site, cust, h0)
    d = registry.decide(
        registry.score_on(candidate_eval, site, cust, h0, h1),
        registry.score_on(current, site, cust, h0, h1),
        registry.CFG["gate"],
    )
    if not d["promote"]:
        return current, False
    cutoff = run_date - registry.CFG["label_settle_days"] * DAY
    return Forecaster(registry.CFG["models"]["production_candidate"]).fit(site, cust, cutoff), True


def predict_period(model, site, cust, start, end):
    """Forecasts made on origins in [start, end) by `model`, with the actual outcome."""
    s = site[(site["origin"] >= start) & (site["origin"] < end) & site["target"].notna()]
    c = cust[(cust["origin"] >= start) & (cust["origin"] < end)]
    p = model.predict_sites(s, c)
    return s[["site_id", "origin", "target_date", "h", "target"]].merge(
        p, on=["site_id", "origin", "target_date", "h"]
    )


def last_week_alert(preds, monday):
    """Did any site break the weekly limits on targets of the previous 7 days?"""
    w = preds[(preds["target_date"] >= monday - 7 * DAY) & (preds["target_date"] < monday)]
    lim = CFG["weekly_alert"]
    for site_id, g in w.groupby("site_id"):
        e, b = wape(g["target"], g["yhat"]), bias(g["target"], g["yhat"])
        if e > lim["max_wape"] or abs(b) > lim["max_abs_bias"]:
            return f"{site_id} wape {e:.3f} bias {b:+.3f}"
    return None


def run_policy(policy, start_model, site, cust):
    start, end = pd.Timestamp(CFG["start"]), pd.Timestamp(CFG["end"])
    model, preds, log = start_model, [], []
    last_retrain = start
    for day in pd.date_range(start, end, freq="W-MON"):
        week_end = min(day + 7 * DAY, end + DAY)
        if day > start:
            done = pd.concat(preds) if preds else pd.DataFrame()
            reason = None
            if policy == "monthly" and (day - 7 * DAY).month != day.month:
                reason = "first Monday of the month"
            if (
                policy == "triggered"
                and (day - last_retrain).days >= CFG["min_days_between_retrains"]
            ):
                reason = last_week_alert(done, day) if len(done) else None
            if reason:
                model, promoted = retrain(site, cust, day, model)
                last_retrain = day
                log.append(
                    {
                        "policy": policy,
                        "date": str(day.date()),
                        "reason": reason,
                        "promoted": promoted,
                    }
                )
                print(
                    f"  {policy}: retrain {day.date()} ({reason}) promoted={promoted}", flush=True
                )
        preds.append(predict_period(model, site, cust, day if day > start else start, week_end))
    p = pd.concat(preds)
    p = p[p["target_date"] <= end]
    return p.assign(policy=policy), log


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    site, cust = registry.load_features()
    start_model, version = registry.load_alias("production")
    print(f"start model: v{version}")
    all_preds, all_logs = [], []
    for policy in ["frozen", "monthly", "triggered"]:
        p, log = run_policy(policy, start_model, site, cust)
        all_preds.append(p)
        all_logs += log
    p = pd.concat(all_preds)
    p["month"] = p["target_date"].dt.to_period("M").astype(str)
    p.to_parquet(OUT / "predictions.parquet", index=False)

    def summarise(g):
        return pd.Series(
            {"wape": wape(g["target"], g["yhat"]), "bias": bias(g["target"], g["yhat"])}
        )

    overall = p.groupby("policy").apply(summarise)
    overall["retrains"] = pd.Series(
        {k: sum(1 for x in all_logs if x["policy"] == k) for k in overall.index}
    )
    by_site = p.groupby(["policy", "site_id"]).apply(summarise)["wape"].unstack()
    s04 = p[(p["site_id"] == "S04") & p["month"].isin(["2026-06", "2026-07", "2026-08"])]
    s04 = s04.groupby(["policy", "month"]).apply(summarise)["bias"].unstack()

    overall.to_csv(OUT / "policies.csv")
    by_site.to_csv(OUT / "policies_by_site.csv")
    s04.to_csv(OUT / "s04_bias_after_contract_loss.csv")
    (OUT / "retrain_log.json").write_text(json.dumps(all_logs, indent=1))
    print("\noverall (Apr-Sep 2026):\n", overall.round(4))
    print("\nWAPE by site:\n", by_site.round(4))
    print("\nS04 bias after the hidden contract loss:\n", s04.round(3))


if __name__ == "__main__":
    main()
