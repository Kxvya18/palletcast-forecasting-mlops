"""S7: watch the deployed model on the live months of 2026.

For each site and month it reports:
  * live error  - WAPE and bias of the frozen production model against what really shipped
  * drift       - Evidently data-drift test of key columns vs the SAME month one year earlier
                  (so normal seasonality is not reported as drift)
  * alerts      - when error, bias or drift crosses the thresholds in configs/monitoring.yaml
It also scores the challengers registered in S4 on 2026, as pre-registered in S3.

    uv run python -m palletcast.monitoring.monitor
"""

import json

import mlflow
import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset

from palletcast.models.common import ROOT, bias, wape
from palletcast.registry.registry import load_alias, load_features

CFG = yaml.safe_load(open(ROOT / "configs" / "monitoring.yaml"))
OUT = ROOT / "reports" / "s7"


def live_predictions(model, site, cust, start, end):
    """Every forecast whose target falls in [start, end], joined with what really shipped."""
    s = site[(site["target_date"] >= start) & (site["target_date"] <= end)]
    c = cust[(cust["target_date"] >= start) & (cust["target_date"] <= end)]
    p = model.predict_sites(s, c)
    return (
        s[["site_id", "origin", "target_date", "h", "target"]]
        .merge(p, on=["site_id", "origin", "target_date", "h"])
        .dropna(subset=["target"])
    )


def error_table(pred, by):
    rows = []
    for key, g in pred.groupby(by):
        rows.append(
            {
                **dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True)),
                "wape": wape(g["target"].to_numpy(), g["yhat"].to_numpy()),
                "bias": bias(g["target"].to_numpy(), g["yhat"].to_numpy()),
                "n": len(g),
            }
        )
    return pd.DataFrame(rows)


def drift_for(site, site_id, month):
    """Evidently drift: this site-month vs the same site-month one year earlier (h = 1 rows)."""
    cols = CFG["drift_columns"]
    rows = site[(site["site_id"] == site_id) & (site["h"] == 1)]
    p = pd.Period(month, "M")
    cur = rows[rows["target_date"].dt.to_period("M") == p][cols]
    ref = rows[rows["target_date"].dt.to_period("M") == p - 12][cols]
    snap = Report([DataDriftPreset()]).run(current_data=cur, reference_data=ref)
    drifted = [
        column for column, is_drift in (value_drift(m) for m in snap.dict()["metrics"]) if is_drift
    ]
    return snap, len(drifted) / len(cols), drifted


def value_drift(metric):
    """(column, drifted?) from one Evidently ValueDrift result, else (None, False).

    Name looks like 'ValueDrift(column=x,method=K-S p_value,threshold=0.05)': p-value
    methods drift when the value is BELOW the threshold, distance methods when ABOVE."""
    name = metric.get("metric_id") or metric.get("metric_name") or ""
    if not name.startswith("ValueDrift"):
        return None, False
    parts = dict(p.split("=", 1) for p in name[len("ValueDrift(") : -1].split(","))
    threshold, value = float(parts["threshold"]), float(metric["value"])
    drifted = value < threshold if "p_value" in parts["method"] else value > threshold
    return parts["column"], drifted


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    site, cust = load_features()
    model, version = load_alias(CFG["frozen_model_alias"])
    months = CFG["live_months"]
    start = pd.Period(months[0], "M").start_time
    end = pd.Period(months[-1], "M").end_time.normalize()

    # 1. live error of the frozen model
    pred = live_predictions(model, site, cust, start, end)
    pred["month"] = pred["target_date"].dt.to_period("M").astype(str)
    err = error_table(pred, ["site_id", "month"])
    weekly = error_table(
        pred.assign(week=pred["target_date"].dt.to_period("W").dt.start_time), ["site_id", "week"]
    )

    # 2. drift vs the same month last year, one Evidently report per site-month
    drift_rows = []
    for s in sorted(site["site_id"].unique()):
        for m in months:
            snap, share, cols = drift_for(site, s, m)
            snap.save_html(str(OUT / f"drift_{s}_{m}.html"))
            drift_rows.append(
                {"site_id": s, "month": m, "drift_share": share, "drifted": ",".join(cols)}
            )
    table = err.merge(pd.DataFrame(drift_rows), on=["site_id", "month"])

    # 3. alerts
    a = CFG["alerts"]
    table["alert_error"] = table["wape"] > a["max_wape"]
    table["alert_bias"] = table["bias"].abs() > a["max_abs_bias"]
    table["alert_drift"] = table["drift_share"] > a["max_drift_share"]
    table["alert"] = table[["alert_error", "alert_bias", "alert_drift"]].any(axis=1)

    table.to_csv(OUT / "monthly_monitoring.csv", index=False)
    weekly.to_csv(OUT / "weekly_error.csv", index=False)
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "frozen_model_version": version,
                "alerts": table[table["alert"]][
                    ["site_id", "month", "wape", "bias", "drift_share", "drifted"]
                ].to_dict("records"),
            },
            indent=1,
            default=str,
        )
    )
    pd.set_option("display.width", 200)
    print(f"frozen model: v{version}")
    print(table.drop(columns=["n", "drifted"]).round(3).to_string(index=False))

    # 4. challengers vs production, all trained for the same date (pre-registered in S3)
    rows = []
    c_start = pd.Timestamp(CFG["challengers_from"])
    for alias in ["challenger-ensemble", "challenger-ratio"]:
        m, v = load_alias(alias)
        p = live_predictions(m, site, cust, c_start, end)
        rows.append(
            {
                "model": f"{alias} (v{v})",
                "wape": wape(p["target"], p["yhat"]),
                "bias": bias(p["target"], p["yhat"]),
            }
        )
    from mlflow import MlflowClient

    v2 = MlflowClient().get_model_version("outbound_forecaster", "2")
    m2 = mlflow.pyfunc.load_model(f"models:/outbound_forecaster/{v2.version}").unwrap_python_model()
    p = live_predictions(m2, site, cust, c_start, end)
    rows.append(
        {
            "model": "production of 2026-01-01 (v2)",
            "wape": wape(p["target"], p["yhat"]),
            "bias": bias(p["target"], p["yhat"]),
        }
    )
    ch = pd.DataFrame(rows)
    ch.to_csv(OUT / "challengers_2026.csv", index=False)
    print(
        "\nchallengers on 2026-01..09 (all trained for 2026-01-01):\n",
        ch.round(4).to_string(index=False),
    )


if __name__ == "__main__":
    main()
