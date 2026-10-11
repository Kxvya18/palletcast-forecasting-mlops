"""Calibrate the site-day volume check on simulated seeds that are NOT the project seed.

Positives are planted partial outages (from each seed's fault manifest); every other scored
site-day is a negative. Thresholds are chosen on calibration seeds and reported on held-out
test seeds, so the numbers in docs/s1_data.md are out-of-sample.

    uv run python -m palletcast.data.calibrate_volume_check --out reports/calibration
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import itertools
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from palletcast.data import make_dataset, validate
from palletcast.data.simulate import load_config

CAL_OFFSETS = [1000, 2000, 3000, 4000, 5000]
TEST_OFFSETS = [6000, 7000, 8000, 9000, 10000]


def score_seed(base: dict, offset: int, work: Path) -> pd.DataFrame:
    cfg = copy.deepcopy(base)
    cfg["seed"] = base["seed"] + offset
    d = work / f"seed_{offset}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    with contextlib.redirect_stdout(io.StringIO()):
        make_dataset.main(["--config", str(d / "cfg.yaml"), "--out", str(d)])
        validate.run(d / "raw", d / "validated", d / "reports", cfg)
    ship = pd.read_parquet(d / "validated" / "shipments.parquet")
    master = validate.load_master(d / "raw")
    daily = ship.groupby(["site_id", "ship_date"])["pallets"].sum()
    start = ship["ship_date"].min()
    sc = validate.volume_scores(
        daily,
        master,
        start,
        start,
        ship["ship_date"].max(),
        cfg["validation"]["outlier_window_weeks"],
    )
    man = json.loads((d / "sim_truth" / "fault_manifest.json").read_text())
    pos = {(o["site_id"], pd.Timestamp(o["date"])) for o in man["partial_outage"]}
    sc["label"] = [(s, t) in pos for s, t in zip(sc["site_id"], sc["date"], strict=True)]
    sc["seed"] = offset
    sc["weekday"] = sc["date"].dt.dayofweek < 5
    return sc


def evaluate(sc: pd.DataFrame, vc: dict) -> dict:
    f = validate.volume_flags(sc, vc)
    pos, neg = sc["label"], ~sc["label"]
    site_years = sc.groupby(["seed", "site_id"])["date"].agg(lambda x: (x.max() - x.min()).days)
    site_years = site_years.sum() / 365.25
    return {
        "recall": float((f & pos).sum() / max(pos.sum(), 1)),
        "recall_weekday": float(
            (f & pos & sc["weekday"]).sum() / max((pos & sc["weekday"]).sum(), 1)
        ),
        "recall_weekend": float(
            (f & pos & ~sc["weekday"]).sum() / max((pos & ~sc["weekday"]).sum(), 1)
        ),
        "n_pos": int(pos.sum()),
        "false_low_per_site_year": float((f & neg & (sc["ratio"] < 1)).sum() / site_years),
        "high_per_site_year": float((f & neg & (sc["ratio"] > 1)).sum() / site_years),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out", default="reports/calibration")
    ap.add_argument("--max-false-per-site-year", type=float, default=0.5)
    a = ap.parse_args(argv)
    base = load_config(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        cal = pd.concat([score_seed(base, o, Path(tmp)) for o in CAL_OFFSETS], ignore_index=True)
        test = pd.concat([score_seed(base, o, Path(tmp)) for o in TEST_OFFSETS], ignore_index=True)

    # Low side: maximise weekday recall s.t. false low alarms <= budget, then overall recall,
    # then prefer the loosest (fewest-alarm) setting.
    rows = []
    for lo, gap in itertools.product(
        np.round(np.arange(0.40, 0.76, 0.05), 2), np.round(np.arange(0.05, 0.41, 0.05), 2)
    ):
        vc = {
            **base["validation"],
            "outlier_low_ratio": lo,
            "outlier_min_gap_share": gap,
            "outlier_high_ratio": np.inf,
            "outlier_high_ratio_near_holiday": np.inf,
        }
        rows.append({"low_ratio": lo, "min_gap_share": gap, **evaluate(cal, vc)})
    grid = pd.DataFrame(rows)
    ok = grid[grid["false_low_per_site_year"] <= a.max_false_per_site_year]
    best = ok.sort_values(
        ["recall_weekday", "recall", "false_low_per_site_year"], ascending=[False, False, True]
    ).iloc[0]

    # High side has no planted positives: pick the smallest thresholds that keep high
    # alarms within the same budget on calibration seeds.
    hrows = []
    for hi, hih in itertools.product([2.0, 2.5, 3.0, 3.5, 4.0], [4.0, 6.0, 8.0, 10.0]):
        vc = {
            **base["validation"],
            "outlier_low_ratio": 0.0,
            "outlier_min_gap_share": float(best["min_gap_share"]),
            "outlier_high_ratio": hi,
            "outlier_high_ratio_near_holiday": hih,
        }
        hrows.append({"high_ratio": hi, "high_ratio_near_holiday": hih, **evaluate(cal, vc)})
    hgrid = pd.DataFrame(hrows)
    hok = hgrid[hgrid["high_per_site_year"] <= a.max_false_per_site_year]
    hbest = hok.sort_values(["high_ratio", "high_ratio_near_holiday"]).iloc[0]

    chosen = {
        "outlier_low_ratio": float(best["low_ratio"]),
        "outlier_min_gap_share": float(best["min_gap_share"]),
        "outlier_high_ratio": float(hbest["high_ratio"]),
        "outlier_high_ratio_near_holiday": float(hbest["high_ratio_near_holiday"]),
    }
    vc = {**base["validation"], **chosen}
    result = {
        "budget_false_alarms_per_site_year": a.max_false_per_site_year,
        "calibration_seeds": CAL_OFFSETS,
        "test_seeds": TEST_OFFSETS,
        "chosen": chosen,
        "calibration": evaluate(cal, vc),
        "test": evaluate(test, vc),
        "current_config_on_test": evaluate(test, base["validation"]),
    }
    grid.to_csv(out / "low_grid.csv", index=False)
    hgrid.to_csv(out / "high_grid.csv", index=False)
    (out / "volume_check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
