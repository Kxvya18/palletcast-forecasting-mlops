"""Experiment: how far ahead of the in-store start does outbound volume rise?

Uses only promotions that start inside a fixed early estimation window, so the result
cannot peek at the periods S3 will test on. For each promotion, customer volume on day
start+k is divided by that customer's median volume on the same weekday 5-8 weeks before
the start (holiday-free baseline). We report the median ratio and its bootstrap 90% CI.

    uv run python -m palletcast.features.promo_lead --out reports/features
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OFFSETS = range(-21, 22)


def event_study(ship: pd.DataFrame, promo: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    cd = ship.groupby(["customer_id", "ship_date"])["pallets"].sum()
    rows = []
    sel = promo[(promo["start_date"] >= start) & (promo["start_date"] <= end)]
    for p in sel.itertuples():
        for k in OFFSETS:
            d = p.start_date + pd.Timedelta(days=k)
            if d.dayofweek >= 5:  # weekdays only: weekends are a different volume regime
                continue
            ref = [cd.get((p.customer_id, d - pd.Timedelta(weeks=w)), 0) for w in range(5, 9)]
            base = float(np.median(ref))
            if base <= 0:
                continue
            rows.append(
                {
                    "promo_id": p.promo_id,
                    "type": p.uplift_type,
                    "k": k,
                    "ratio": cd.get((p.customer_id, d), 0) / base,
                }
            )
    return pd.DataFrame(rows)


def summarise(es: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for k, g in es.groupby("k"):
        r = g["ratio"].to_numpy()
        boots = np.median(rng.choice(r, (n_boot, len(r))), axis=1)
        out.append(
            {
                "k": k,
                "n": len(r),
                "median": np.median(r),
                "ci_lo": np.quantile(boots, 0.05),
                "ci_hi": np.quantile(boots, 0.95),
            }
        )
    return pd.DataFrame(out).set_index("k")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validated", default="data/validated")
    ap.add_argument("--start", default="2023-03-01")
    ap.add_argument("--end", default="2024-08-31")
    ap.add_argument("--min-uplift", type=float, default=1.10)
    ap.add_argument("--out", default="reports/features")
    a = ap.parse_args(argv)
    v = Path(a.validated)
    ship = pd.read_parquet(v / "shipments.parquet")
    promo = pd.read_parquet(v / "master" / "promo_calendar.parquet")
    es = event_study(ship, promo, a.start, a.end)
    s = summarise(es)
    # the lead window = offsets whose whole 90% CI is above the uplift threshold
    lifted = s.index[s["ci_lo"] >= a.min_uplift].tolist()
    res = {
        "estimation_window": [a.start, a.end],
        "n_promos": int(es["promo_id"].nunique()),
        "min_uplift": a.min_uplift,
        "lifted_offsets": lifted,
        "window": [min(lifted), max(lifted)] if lifted else None,
        "by_offset": s.round(3).reset_index().to_dict("records"),
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "promo_lead.json").write_text(json.dumps(res, indent=1))
    print(s.round(3).to_string())
    print(f"lifted offsets (CI low >= {a.min_uplift}):", lifted)


if __name__ == "__main__":
    main()
