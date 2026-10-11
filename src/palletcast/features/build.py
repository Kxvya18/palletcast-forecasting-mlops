"""Build the feature tables for the 14-day outbound forecast.

One row answers one question: "standing at origin day O (06:00), how many pallets will
this series ship on target day T = O + h?"  (h = 1..14)

Two tables, as agreed for S3's comparison:
  site.parquet      one series per site
  customer.parquet  one series per customer (summed to sites later)

The one rule that matters most: a feature may only use what was known at the origin.
  * daily volumes are taken "as of" the origin (late rows and corrections come later)
  * outage days are hidden (NaN), not treated as zero demand
  * promotions count only once announced; new/leaving customers only within notice
Rows exist only for target days the site is expected to work.

    uv run python -m palletcast.features.build
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from palletcast.data.simulate import WEEKDAYS
from palletcast.features.asof import known_by_lag

DAY = pd.Timedelta(days=1)
HISTORY_DAYS = 70  # how far back any feature looks (8 same-weekday weeks + horizon)
PROMO_TYPES = ["display", "multibuy", "price_cut"]  # weakest to strongest uplift


def load_inputs(validated: Path) -> dict:
    m = validated / "master"
    return {
        "versions": pd.read_parquet(validated / "shipment_versions.parquet"),
        "flags": pd.read_parquet(validated / "data_flags.parquet"),
        "sites": pd.read_parquet(m / "sites.parquet"),
        "customers": pd.read_parquet(m / "customers.parquet"),
        "promo": pd.read_parquet(m / "promo_calendar.parquet"),
        "holidays": pd.read_parquet(m / "holidays.parquet"),
    }


# --------------------------------------------------------------------------------------
# Step 1: daily volumes, as they were known on each forecast morning
# --------------------------------------------------------------------------------------


def daily_tables(versions, key, series, days, cfg):
    """Returns (final, known).

    final: table series x day with the final daily pallets.
    known: {k: table series x day} with the pallets known k mornings after that day.
    """
    known_long, final_long = known_by_lag(
        versions, [key], cfg["forecast_cutoff_hour"], cfg["asof_window_days"]
    )

    def to_table(values: pd.Series) -> pd.DataFrame:
        t = values.unstack("ship_date")
        return t.reindex(index=series, columns=days).fillna(0)

    final = to_table(final_long)
    known = {k: to_table(known_long[k]) for k in known_long.columns}
    return final, known


def history_as_of(final, known, origin, n_days=HISTORY_DAYS):
    """The last n_days days before the origin, as they looked at origin 06:00.

    Older days are final; the last `asof_window_days` days still had rows missing."""
    days = pd.date_range(origin - n_days * DAY, origin - DAY)
    hist = final.reindex(columns=days)  # days before the data starts become NaN
    for k, table in known.items():
        day = origin - k * DAY
        if day in table.columns:
            hist[day] = table[day]
    return hist


# --------------------------------------------------------------------------------------
# Step 2: hide days that do not show real demand
# --------------------------------------------------------------------------------------


def site_calendar(sites, holidays, days):
    """Two tables site x day: `working` (expected to ship) and `holiday_closed`."""
    working = pd.DataFrame(False, index=sites["site_id"], columns=days)
    holiday_closed = working.copy()
    for s in sites.itertuples():
        works_that_weekday = days.dayofweek.isin(
            [WEEKDAYS.index(d) for d in s.operating_days.split(",")]
        )
        is_holiday = days.isin(holidays.loc[holidays["country"] == s.country, "date"])
        working.loc[s.site_id] = works_that_weekday & ~is_holiday
        holiday_closed.loc[s.site_id] = works_that_weekday & is_holiday
    return working, holiday_closed


def unusable_days(site_hist, working, holiday_closed, low_volume_flags, origin, cfg):
    """Site x day table: True where the day's volume must not be used as a feature.

    * holiday closures - volume was shifted to other days, so not a normal day
    * a working day with nothing known - an outage, not zero demand
    * a day validation flagged as partial feed loss, once that flag existed
    """
    days = site_hist.columns
    outage = working.reindex(columns=days, fill_value=False) & (site_hist == 0)
    flag_age = (origin - days).days >= 1 + cfg["flag_known_after_days"]
    flagged = low_volume_flags.reindex(columns=days, fill_value=False) & flag_age
    return holiday_closed.reindex(columns=days, fill_value=False) | outage | flagged


# --------------------------------------------------------------------------------------
# Step 3: volume features for one origin and horizon
# --------------------------------------------------------------------------------------


def nan_mean(a):
    """Row means ignoring NaN; a row with no usable values gives NaN (without a warning)."""
    n = (~np.isnan(a)).sum(axis=1)
    return np.where(n > 0, np.nansum(a, axis=1) / np.maximum(n, 1), np.nan)


def volume_features(hist, origin, h):
    """Lag and rolling features for target day origin + h, one row per series.

    `hist` columns are the HISTORY_DAYS days before the origin, oldest first."""
    target = origin + h * DAY
    # same weekday as the target, most recent first; the first one must already be known
    first_week = math.ceil((h + 1) / 7)
    sw = hist[[target - 7 * (first_week + m) * DAY for m in range(8)]].to_numpy()
    v = hist.to_numpy()
    previous_28 = nan_mean(v[:, -56:-28])
    mean_28 = nan_mean(v[:, -28:])

    with np.errstate(all="ignore"):
        trend_28 = np.where(previous_28 > 0, mean_28 / previous_28, np.nan)
        median8 = np.array(
            [np.median(r[~np.isnan(r)]) if (~np.isnan(r)).any() else np.nan for r in sw]
        )
    return pd.DataFrame(
        {
            "sw_lag1": sw[:, 0],
            "sw_lag2": sw[:, 1],
            "sw_lag3": sw[:, 2],
            "sw_lag4": sw[:, 3],
            "sw_mean4": nan_mean(sw[:, :4]),
            "sw_mean8": nan_mean(sw),
            "sw_median8": median8,
            "last_1": v[:, -1],
            "mean_7": nan_mean(v[:, -7:]),
            "mean_28": mean_28,
            "trend_28": trend_28,
        },
        index=hist.index,
    )


def volume_rows(inp, cfg, origins, end, days):
    """Steps 1-3 for every origin. Returns site rows, customer rows and the final tables."""
    sites, cust = inp["sites"], inp["customers"]
    site_ids = pd.Index(sites["site_id"], name="site_id")
    cust_ids = pd.Index(cust["customer_id"], name="customer_id")
    site_of = cust.set_index("customer_id")["site_id"]

    site_final, site_known = daily_tables(inp["versions"], "site_id", site_ids, days, cfg)
    cust_final, cust_known = daily_tables(inp["versions"], "customer_id", cust_ids, days, cfg)

    working, holiday_closed = site_calendar(sites, inp["holidays"], days)
    low = inp["flags"][inp["flags"]["type"] == "volume_low"]
    low_volume_flags = pd.DataFrame(False, index=site_ids, columns=days)
    for s, d in zip(low["site_id"], low["date"], strict=True):
        if d in low_volume_flags.columns:
            low_volume_flags.loc[s, d] = True

    site_rows, cust_rows = [], []
    for origin in origins:
        site_hist = history_as_of(site_final, site_known, origin)
        cust_hist = history_as_of(cust_final, cust_known, origin)

        hide = unusable_days(site_hist, working, holiday_closed, low_volume_flags, origin, cfg)
        site_hist = site_hist.mask(hide)
        cust_hist = cust_hist.mask(hide.loc[site_of[cust_ids]].set_axis(cust_ids))
        closed = holiday_closed.reindex(columns=hide.columns, fill_value=False)
        n_masked_28 = (hide & ~closed).iloc[:, -28:].sum(axis=1)

        # each customer's share of its site's volume over the last 28 days
        cust_28 = cust_hist.iloc[:, -28:].sum(axis=1)
        site_28 = site_hist.iloc[:, -28:].sum(axis=1)
        share_28 = (cust_28 / site_28[site_of[cust_ids]].to_numpy()).fillna(0)

        for h in range(cfg["horizons"][0], cfg["horizons"][1] + 1):
            target = origin + h * DAY
            if target > end:
                break
            s = volume_features(site_hist, origin, h)
            s["n_masked_28"] = n_masked_28
            site_rows.append(s.assign(origin=origin, target_date=target, h=h).reset_index())

            c = volume_features(cust_hist, origin, h)
            c["site_id"] = site_of[cust_ids].to_numpy()
            for col in ("sw_mean4", "mean_28", "trend_28"):  # site context for customers
                c[f"site_{col}"] = s.loc[c["site_id"], col].to_numpy()
            c["share_28"] = share_28
            c["n_masked_28"] = n_masked_28[c["site_id"]].to_numpy()
            cust_rows.append(c.assign(origin=origin, target_date=target, h=h).reset_index())

    site_df = pd.concat(site_rows, ignore_index=True)
    cust_df = pd.concat(cust_rows, ignore_index=True)

    # keep only target days the site is expected to work
    w = working.stack()
    working_days = set(w[w].index)
    site_df = site_df[
        [k in working_days for k in zip(site_df["site_id"], site_df["target_date"], strict=True)]
    ]
    cust_df = cust_df[
        [k in working_days for k in zip(cust_df["site_id"], cust_df["target_date"], strict=True)]
    ]
    return site_df, cust_df, site_final, cust_final


# --------------------------------------------------------------------------------------
# Step 4: things known in advance - customers, promotions, holidays, calendar
# --------------------------------------------------------------------------------------


def add_customer_lifecycle(df, customers, cfg):
    """Drop customers the business did not know about yet, or knew had left; add
    days since onboarding and days until a known churn."""
    notice = cfg["master_data_notice_days"] * DAY
    c = customers.set_index("customer_id")
    onboard = df["customer_id"].map(c["onboard_date"])
    churn = df["customer_id"].map(c["churn_date"])
    df["segment"] = df["customer_id"].map(c["segment"])

    known = onboard.isna() | (df["origin"] >= onboard - notice)
    started = onboard.isna() | (df["target_date"] >= onboard)
    churn_known = churn.notna() & (df["origin"] >= churn - notice)
    gone = churn_known & (df["target_date"] >= churn)
    keep = known & started & ~gone

    df["days_since_onboard"] = (df["target_date"] - onboard).dt.days
    df["days_to_churn"] = (churn - df["target_date"]).dt.days.where(churn_known)
    return df[keep].copy()


def add_promo_features(df, promo, cfg):
    """Promotion features per customer row. offset = in-store start - target day.

    Only promotions announced by the origin are visible."""
    lift_lo, lift_hi = cfg["promo_lift_offsets"]
    post_lo, post_hi = cfg["promo_post_offsets"]
    df["promo_in_lift"] = 0.0
    df["promo_in_post"] = 0.0
    df["promo_lift_type"] = 0.0
    df["promo_next_start"] = np.nan
    for t in PROMO_TYPES:
        df[f"promo_lift_{t}"] = 0.0

    for customer, rows in df.groupby("customer_id").groups.items():
        p = promo[promo["customer_id"] == customer]
        if p.empty:
            continue
        # one row per forecast row, one column per promotion of this customer
        origin = df.loc[rows, "origin"].to_numpy()[:, None]
        target = df.loc[rows, "target_date"].to_numpy()[:, None]
        offset = (p["start_date"].to_numpy()[None, :] - target) / np.timedelta64(1, "D")
        announced = p["announce_date"].to_numpy()[None, :] <= origin
        in_lift = announced & (offset >= lift_lo) & (offset <= lift_hi)
        in_post = announced & (offset >= post_lo) & (offset <= post_hi)
        strength = p["uplift_type"].map({t: i + 1 for i, t in enumerate(PROMO_TYPES)})

        df.loc[rows, "promo_in_lift"] = in_lift.sum(axis=1)
        df.loc[rows, "promo_in_post"] = in_post.sum(axis=1)
        df.loc[rows, "promo_lift_type"] = (in_lift * strength.to_numpy()).max(axis=1)
        for t in PROMO_TYPES:
            of_type = (p["uplift_type"] == t).to_numpy()
            df.loc[rows, f"promo_lift_{t}"] = (in_lift & of_type).any(axis=1).astype(float)
        upcoming = np.where(announced & (offset >= lift_lo) & (offset <= 21), offset, np.inf)
        nearest = upcoming.min(axis=1)
        df.loc[rows, "promo_next_start"] = np.where(np.isfinite(nearest), nearest, np.nan)
    return df


def add_site_aggregates(site_df, cust_df):
    """Site rows get customer information summed up: how many customers are active,
    how many are new, and the volume share of customers in a promotion window."""
    c = cust_df.assign(
        new_90=cust_df["days_since_onboard"].between(0, 89),
        post_share=cust_df["share_28"] * (cust_df["promo_in_post"] > 0),
    )
    for t in PROMO_TYPES:
        c[f"promo_lift_share_{t}"] = c["share_28"] * c[f"promo_lift_{t}"]
    agg = c.groupby(["site_id", "origin", "h"]).agg(
        n_active_customers=("customer_id", "size"),
        n_new_customers_90=("new_90", "sum"),
        promo_post_share=("post_share", "sum"),
        promo_lift_share_display=("promo_lift_share_display", "sum"),
        promo_lift_share_multibuy=("promo_lift_share_multibuy", "sum"),
        promo_lift_share_price_cut=("promo_lift_share_price_cut", "sum"),
    )
    return site_df.join(agg, on=["site_id", "origin", "h"])


def add_holiday_distance(df, sites, holidays, cap=15):
    """Days until the next and since the last public holiday in the site's country."""
    country = df["site_id"].map(sites.set_index("site_id")["country"])
    df["days_to_holiday"] = float(cap)
    df["days_since_holiday"] = float(cap)
    for c, rows in df.groupby(country).groups.items():
        hol = np.sort(holidays.loc[holidays["country"] == c, "date"].to_numpy())
        t = df.loc[rows, "target_date"].to_numpy()
        nxt = np.searchsorted(hol, t, side="left")
        prv = np.searchsorted(hol, t, side="right") - 1
        to_next = (hol[np.minimum(nxt, len(hol) - 1)] - t) / np.timedelta64(1, "D")
        since = (t - hol[np.maximum(prv, 0)]) / np.timedelta64(1, "D")
        df.loc[rows, "days_to_holiday"] = np.where(nxt < len(hol), np.minimum(to_next, cap), cap)
        df.loc[rows, "days_since_holiday"] = np.where(prv >= 0, np.minimum(since, cap), cap)
    return df


def add_target(df, final, key, flags, sites=None):
    """Target = final shipped pallets on the target day. Days validation flagged as
    outage / partial loss get NaN: they are not real demand and must not be learned."""
    df = df.join(final.stack().rename("target"), on=[key, "target_date"])
    bad = set(zip(flags["site_id"], flags["date"], strict=True))
    is_bad = [k in bad for k in zip(df["site_id"], df["target_date"], strict=True)]
    df.loc[is_bad, "target"] = np.nan
    if sites is not None:
        df["capacity"] = df["site_id"].map(sites.set_index("site_id")["capacity_pallets_per_day"])
        df["at_capacity"] = df["target"] >= df["capacity"]
    return df


def add_calendar(df, cfg):
    t = df["target_date"]
    df["dow"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["doy"] = t.dt.dayofyear
    df["week"] = t.dt.isocalendar().week.astype(int).to_numpy()
    df["period"] = np.where(t <= pd.Timestamp(cfg["history_end"]), "history", "live")
    return df


# --------------------------------------------------------------------------------------


def build(inp, cfg, origins=None, end=None):
    end = end if end is not None else inp["versions"]["ship_date"].max()
    start = inp["versions"]["ship_date"].min()
    days = pd.date_range(start, end, freq="D")
    if origins is None:
        origins = pd.date_range(cfg["first_origin"], end - DAY, freq="D")

    site_df, cust_df, site_final, cust_final = volume_rows(inp, cfg, origins, end, days)

    cust_df = add_customer_lifecycle(cust_df, inp["customers"], cfg)
    cust_df = add_promo_features(cust_df, inp["promo"], cfg)
    site_df = add_site_aggregates(site_df, cust_df)

    site_df = add_holiday_distance(site_df, inp["sites"], inp["holidays"])
    site_df = add_target(site_df, site_final, "site_id", inp["flags"], inp["sites"])
    cust_df = add_target(cust_df, cust_final, "customer_id", inp["flags"])
    # customers share their site's holiday distances and capacity flag
    site_info = site_df.drop_duplicates(["site_id", "target_date"]).set_index(
        ["site_id", "target_date"]
    )[["days_to_holiday", "days_since_holiday", "at_capacity"]]
    cust_df = cust_df.join(site_info, on=["site_id", "target_date"])

    site_df = add_calendar(site_df, cfg).sort_values(["site_id", "origin", "h"])
    cust_df = add_calendar(cust_df, cfg).sort_values(["customer_id", "origin", "h"])
    return site_df.reset_index(drop=True), cust_df.reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/features.yaml")
    ap.add_argument("--validated", default="data/validated")
    ap.add_argument("--out", default="data/features")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config))
    site_df, cust_df = build(load_inputs(Path(a.validated)), cfg)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    site_df.to_parquet(out / "site.parquet", index=False)
    cust_df.to_parquet(out / "customer.parquet", index=False)
    meta = {
        "site_rows": len(site_df),
        "customer_rows": len(cust_df),
        "site_columns": list(site_df.columns),
        "customer_columns": list(cust_df.columns),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"site rows {len(site_df):,}  customer rows {len(cust_df):,}")


if __name__ == "__main__":
    main()
