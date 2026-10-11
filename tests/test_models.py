"""S3 backtest building blocks, on small hand-made tables (fast)."""

import numpy as np
import pandas as pd
import pytest

from palletcast.models import common

CFG = {
    "tune_fold": {"name": "tune", "start": "2024-10-01", "end": "2024-12-31"},
    "test_folds": [{"name": "q1", "start": "2025-01-01", "end": "2025-03-31"}],
    "last_target_date": "2025-12-31",
    "label_settle_days": 21,
    "eval_origin_weekdays": [0, 3],
    "peak_quantile": 0.9,
}


def test_wape_and_bias():
    y, p = np.array([100.0, 200.0]), np.array([110.0, 170.0])
    assert common.wape(y, p) == pytest.approx(40 / 300)
    assert common.bias(y, p) == pytest.approx(-20 / 300)  # negative = under-forecast


def table(origins, h=(1, 2)):
    rows = [
        {
            "site_id": "S01",
            "origin": o,
            "h": k,
            "target_date": o + pd.Timedelta(days=k),
            "target": 100.0,
        }
        for o in origins
        for k in h
    ]
    return pd.DataFrame(rows)


def test_eval_rows_respects_fold_weekdays_and_history_end():
    origins = pd.date_range("2025-03-25", "2025-04-02")
    t = table(origins)
    t.loc[0, "target"] = np.nan
    r = common.eval_rows(t, CFG["test_folds"][0], CFG)
    assert r["origin"].between("2025-01-01", "2025-03-31").all()
    assert set(r["origin"].dt.dayofweek) <= {0, 3}
    assert r["target"].notna().all()
    late = table(pd.DatetimeIndex(["2025-12-29"]), h=(1, 5))  # a Monday; h=5 lands in 2026
    r = common.eval_rows(late, {"name": "q4", "start": "2025-10-01", "end": "2025-12-31"}, CFG)
    assert (r["target_date"] <= "2025-12-31").all() and len(r) == 1


def test_training_labels_were_final_at_fold_start():
    cut = common.train_cutoff(CFG["test_folds"][0], CFG)
    assert cut == pd.Timestamp("2024-12-11")  # 21 days before 2025-01-01


def test_fill_gaps_uses_same_weekday_median():
    days = pd.date_range("2025-01-01", periods=63)
    hist = pd.DataFrame([np.arange(63, dtype=float)], index=["S01"], columns=days)
    hist.iloc[0, 62] = np.nan
    filled = common.fill_gaps(hist)
    same_weekday = [62 - 7 * k for k in range(1, 9)]
    assert filled.iloc[0, 62] == np.median(same_weekday)


def test_to_prediction_keeps_horizons_1_to_14_and_clips_negatives():
    origin = pd.Timestamp("2025-02-03")
    fc = pd.DataFrame(
        {"unique_id": "S01", "ds": pd.date_range(origin, periods=15), "m": [-5.0] + [10.0] * 14}
    )
    p = common.to_prediction(fc, origin, "q1", "m", 0, "m")
    assert list(p["h"]) == list(range(1, 15))
    assert (p["yhat"] >= 0).all()


def test_score_counts_peak_days_per_site():
    t = table(pd.date_range("2025-01-06", periods=20, freq="7D"), h=(1,))
    t["target"] = np.arange(20, dtype=float) + 1
    pred = t[["site_id", "origin", "h"]].assign(yhat=t["target"].to_numpy())
    m = common.score(pred, t, CFG)
    assert m["wape"] == 0 and m["wape_peak"] == 0 and m["n"] == 20
