"""S2 feature pipeline: point-in-time correctness (no leakage), structure, consistency."""

import numpy as np
import pandas as pd
import pytest
import yaml

from palletcast.features import build as fb
from palletcast.features.asof import known_by_lag

from .conftest import ROOT

FCFG = yaml.safe_load(open(ROOT / "configs" / "features.yaml"))
# Origins chosen to hit hard cases: right after outages, around Christmas, the live period,
# a known onboarding and a known churn.
ORIGINS = pd.to_datetime(
    [
        "2023-06-14",
        "2023-12-22",
        "2024-05-01",
        "2024-09-30",
        "2025-01-03",
        "2025-05-20",
        "2025-12-29",
        "2026-03-02",
        "2026-06-15",
        "2026-09-10",
    ]
)
# plus random origins, so the check does not depend on hand-picked dates
_rng = np.random.default_rng(7)
ORIGINS = ORIGINS.append(
    pd.DatetimeIndex(
        pd.Timestamp("2023-03-08") + pd.to_timedelta(_rng.integers(0, 1300, 6), unit="D")
    )
)
FEATURES_ONLY = ["target", "at_capacity"]  # labels come from the final data by design


@pytest.fixture(scope="module")
def inputs(pipeline):
    return fb.load_inputs(pipeline["dir"] / "validated")


@pytest.fixture(scope="module")
def full(inputs):
    return fb.build(inputs, FCFG)


def truncate(inp: dict, origin: pd.Timestamp) -> dict:
    """Exactly what existed at origin 06:00: records that had arrived, promos announced,
    validation flags already raised."""
    cut = origin + pd.Timedelta(hours=FCFG["forecast_cutoff_hour"])
    out = dict(inp)
    out["versions"] = inp["versions"][inp["versions"]["ingested_at"] < cut]
    out["promo"] = inp["promo"][inp["promo"]["announce_date"] <= origin]
    lag = 1 + FCFG["flag_known_after_days"]
    out["flags"] = inp["flags"][inp["flags"]["date"] <= origin - pd.Timedelta(days=lag)]
    return out


def feature_cols(df):
    return [c for c in df.columns if c not in FEATURES_ONLY]


def rows_at(df, origin, keys):
    return df[df["origin"] == origin].sort_values(keys + ["h"]).reset_index(drop=True)


@pytest.mark.parametrize("origin", ORIGINS, ids=lambda o: str(o.date()))
def test_features_equal_point_in_time_rebuild(inputs, full, origin):
    """Features from the full dataset == features rebuilt from only what existed at the origin.
    If any feature used later-arriving information, these would differ."""
    site_t, cust_t = fb.build(
        truncate(inputs, origin),
        FCFG,
        origins=pd.DatetimeIndex([origin]),
        end=inputs["versions"]["ship_date"].max(),
    )
    for got, want, keys in (
        (site_t, full[0], ["site_id"]),
        (cust_t, full[1], ["site_id", "customer_id"]),
    ):
        a, b = rows_at(want, origin, keys), rows_at(got, origin, keys)
        cols = [c for c in feature_cols(a) if c != "period"]
        assert len(a) == len(b) > 0
        pd.testing.assert_frame_equal(a[cols], b[cols], check_dtype=False)


def test_changing_the_future_changes_no_feature(inputs, full):
    """Corrupt everything that arrives after the origin: features at the origin must not move."""
    origin = pd.Timestamp("2025-05-20")
    cut = origin + pd.Timedelta(hours=FCFG["forecast_cutoff_hour"])
    bad = dict(inputs)
    v = inputs["versions"].copy()
    late = v["ingested_at"] >= cut
    v.loc[late, "pallets"] = v.loc[late, "pallets"] * 7
    bad["versions"] = v
    p = inputs["promo"].copy()
    p.loc[p["announce_date"] > origin, "uplift_type"] = "price_cut"
    p.loc[p["announce_date"] > origin, "start_date"] = origin + pd.Timedelta(days=3)
    bad["promo"] = p
    site_b, cust_b = fb.build(
        bad, FCFG, origins=pd.DatetimeIndex([origin]), end=inputs["versions"]["ship_date"].max()
    )
    for got, want, keys in (
        (site_b, full[0], ["site_id"]),
        (cust_b, full[1], ["site_id", "customer_id"]),
    ):
        a, b = rows_at(want, origin, keys), rows_at(got, origin, keys)
        cols = feature_cols(a)
        pd.testing.assert_frame_equal(a[cols], b[cols], check_dtype=False)
        assert not np.allclose(a["target"].fillna(0), b["target"].fillna(0))  # labels did move


def test_asof_window_covers_all_late_information(inputs):
    """After asof_window_days every day's total is final, so using final values beyond the
    window cannot leak (the window assumption the fast path relies on)."""
    w = FCFG["asof_window_days"]
    for keys in (["site_id"], ["site_id", "customer_id"]):
        known, final = known_by_lag(inputs["versions"], keys, FCFG["forecast_cutoff_hour"], w)
        assert (known[w] == final).all()


def test_one_row_per_series_origin_horizon(full):
    site, cust = full
    assert not site.duplicated(["site_id", "origin", "h"]).any()
    assert not cust.duplicated(["customer_id", "origin", "h"]).any()
    assert set(site["h"]) == set(range(1, 15))
    assert ((site["target_date"] - site["origin"]).dt.days == site["h"]).all()


def test_rows_only_on_expected_operating_days(full, inputs):
    site, cust = full
    hol = inputs["holidays"].merge(inputs["sites"], on="country")
    closed = site.merge(hol, left_on=["site_id", "target_date"], right_on=["site_id", "date"])
    assert closed.empty
    s_keys = set(zip(site["site_id"], site["target_date"], strict=True))
    assert all(k in s_keys for k in zip(cust["site_id"], cust["target_date"], strict=True))


def test_customer_targets_sum_to_site_target(full):
    site, cust = full
    g = cust.groupby(["site_id", "origin", "h"])["target"].sum(min_count=1)
    m = (
        site.set_index(["site_id", "origin", "h"])["target"]
        .to_frame("site")
        .join(g.rename("cust"))
        .dropna()
    )
    assert len(m) > 0.95 * len(site)
    assert (m["site"] == m["cust"]).all()


def test_outage_days_are_masked_not_zero(full, pipeline):
    """A planted full outage must appear as NaN in the lag features, never as 0."""
    site = full[0]
    first = pd.Timestamp(FCFG["first_origin"]) + pd.Timedelta(days=14)
    o = next(x for x in pipeline["manifest"]["outage"] if pd.Timestamp(x["date"]) >= first)
    day = pd.Timestamp(o["date"])
    # for h in 1..6 the most recent same-weekday lag (sw_lag1) of target origin+h is day
    # origin+h-7; choosing origin = day+7-h makes that the outage day
    vals = []
    for h in range(1, 7):
        r = site[
            (site["site_id"] == o["site_id"])
            & (site["origin"] == day + pd.Timedelta(days=7 - h))
            & (site["h"] == h)
        ]
        vals += r["sw_lag1"].tolist()
    assert len(vals) > 0 and np.isnan(vals).all()
    t = site[(site["site_id"] == o["site_id"]) & (site["target_date"] == day)]
    assert t["target"].isna().all()  # and it is never used as a training label


def test_unannounced_customers_and_promos_absent(full, inputs):
    site, cust = full
    c = inputs["customers"]
    late = c[c["onboard_date"].notna()].iloc[-1]  # the live-period onboarding
    rows = cust[cust["customer_id"] == late["customer_id"]]
    notice = pd.Timedelta(days=FCFG["master_data_notice_days"])
    assert rows["origin"].min() >= late["onboard_date"] - notice
    assert rows["target_date"].min() >= late["onboard_date"]
