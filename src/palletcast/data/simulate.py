"""Simulate outbound shipments (ground truth, before any data-quality faults).

The generative model for customer c on day t is

    mu[c, t] = base_c * growth_c(t) * weekly_site(t) * annual_seg(t) * promo_c(t)
               * lifecycle_c(t) * level_shift_c(t) * lull(t) * exp(ar_c(t))

with public holidays redistributed to neighbouring working days, and
pallets[c, t] ~ NegBin(mean=mu, dispersion=r). Daily pallets are split into truck
shipments. Every assumption is set in configs/data.yaml and documented in docs/s1_data.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import holidays as holidays_lib
import numpy as np
import pandas as pd
import yaml

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class SimOutput:
    sites: pd.DataFrame
    customers: pd.DataFrame
    promo_calendar: pd.DataFrame  # announced promos only (what the business can see)
    promos_all: pd.DataFrame  # includes unannounced promos (truth only)
    holidays: pd.DataFrame
    shipments: pd.DataFrame  # clean, true shipments
    daily_truth: pd.DataFrame  # site x customer x day: mu and true pallets


def annual_profile(bumps: list[list[float]], amp_mult: float = 1.0) -> np.ndarray:
    """Factor for day-of-year 1..366 (index 0 unused), normalised to mean 1 over 1..365."""
    doy = np.arange(367, dtype=float)
    f = np.ones_like(doy)
    for center, width, amp in bumps:
        d = np.abs(doy - center)
        d = np.minimum(d, 365.25 - d)  # circular distance
        f += amp * amp_mult * np.exp(-0.5 * (d / width) ** 2)
    f = np.clip(f, 0.05, None)
    return f / f[1:366].mean()


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def build_holidays(countries: list[str], years: range) -> pd.DataFrame:
    rows = []
    for c in countries:
        for d, name in sorted(holidays_lib.country_holidays(c, years=list(years)).items()):
            rows.append({"country": c, "date": pd.Timestamp(d), "name": name})
    return pd.DataFrame(rows)


def build_customers(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    cc = cfg["customers"]
    segs = list(cc["segment_weights"])
    w = np.array([cc["segment_weights"][s] for s in segs], dtype=float)
    start = pd.Timestamp(cfg["start_date"])
    rows = []
    for si, site in enumerate(cfg["sites"], start=1):
        n = site["n_customers"]
        base = np.exp(
            np.log(cc["base_median_pallets_per_day"]) + cc["base_sigma"] * rng.standard_normal(n)
        )
        base = np.clip(base, *cc["base_clip"])
        seg = rng.choice(segs, size=n, p=w / w.sum())
        order = np.argsort(-base)  # rank 0 = largest customer at the site
        for rank, k in enumerate(order):
            rows.append(
                {
                    "customer_id": f"C{si:02d}{rank + 1:02d}",
                    "site_id": site["site_id"],
                    "segment": seg[k],
                    "temp_zone": "frozen" if rng.random() < cc["frozen_share"] else "chilled",
                    "onboard_date": start,
                    "churn_date": pd.NaT,
                    "sim_base": float(base[k]),
                    "sim_rank": rank,
                    "sim_ramp_on": 0,
                    "sim_ramp_off": 0,
                }
            )
    df = pd.DataFrame(rows)
    ev = cfg["events"]
    for e in ev.get("onboard", []):
        si = [s["site_id"] for s in cfg["sites"]].index(e["site_id"]) + 1
        k = int((df["site_id"] == e["site_id"]).sum()) + 1
        df.loc[len(df)] = {
            "customer_id": f"C{si:02d}{k:02d}",
            "site_id": e["site_id"],
            "segment": e["segment"],
            "temp_zone": "frozen",
            "onboard_date": pd.Timestamp(e["date"]),
            "churn_date": pd.NaT,
            "sim_base": float(e["base"]),
            "sim_rank": k - 1,
            "sim_ramp_on": e["ramp_days"],
            "sim_ramp_off": 0,
        }
    for e in ev.get("churn", []):
        m = (df["site_id"] == e["site_id"]) & (df["sim_rank"] == e["customer_rank"])
        df.loc[m, "churn_date"] = pd.Timestamp(e["date"])
        df.loc[m, "sim_ramp_off"] = e["ramp_days"]
    return df


def build_promos(cfg: dict, customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    pc = cfg["promotions"]
    start, end = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])
    lo, hi = start - pd.Timedelta(days=30), end + pd.Timedelta(days=90)
    span = (hi - lo).days
    types = list(pc["uplift"])
    rows = []
    for c in customers.itertuples():
        n = rng.poisson(pc["per_customer_per_year"] * span / 365.25)
        for _ in range(n):
            s = lo + pd.Timedelta(days=int(rng.integers(0, span)))
            dur = int(rng.integers(pc["duration_days"][0], pc["duration_days"][1] + 1))
            t = types[rng.integers(len(types))]
            if pd.notna(c.onboard_date) and s < c.onboard_date:
                continue
            if pd.notna(c.churn_date) and s > c.churn_date:
                continue
            rows.append(
                {
                    "promo_id": None,
                    "customer_id": c.customer_id,
                    "uplift_type": t,
                    "start_date": s,
                    "end_date": s + pd.Timedelta(days=dur - 1),
                    "announce_date": s
                    - pd.Timedelta(days=int(rng.integers(*pc["announce_lead_days"]))),
                    "sim_uplift": float(rng.uniform(*pc["uplift"][t])),
                    "sim_ship_lead": int(
                        rng.integers(pc["ship_lead_days"][0], pc["ship_lead_days"][1] + 1)
                    ),
                    "sim_announced": bool(rng.random() >= pc["unannounced_share"]),
                }
            )
    df = pd.DataFrame(rows).sort_values(["start_date", "customer_id"]).reset_index(drop=True)
    df["promo_id"] = [f"P{i:05d}" for i in range(1, len(df) + 1)]
    return df


def _site_weekly_matrix(cfg: dict, site: dict, dates: pd.DatetimeIndex) -> np.ndarray:
    """Weekly factor per date, honouring weekly_change events."""
    weekly = np.array(site["weekly"], dtype=float)
    f = weekly[dates.dayofweek]
    for e in cfg["events"].get("weekly_change", []):
        if e["site_id"] == site["site_id"]:
            m = (dates >= pd.Timestamp(e["date"])) & (dates.dayofweek == e["weekday"])
            f = np.where(m, e["new_factor"], f)
    return f


def _redistribute_holidays(
    mu: np.ndarray, closed: np.ndarray, working: np.ndarray, cal: dict
) -> np.ndarray:
    mu = mu.copy()
    T = len(mu)
    for t in np.where(closed)[0]:
        v = mu[t]
        mu[t] = 0.0
        prev = [i for i in range(t - 1, max(t - 15, -1), -1) if working[i]]
        nxt = [i for i in range(t + 1, min(t + 15, T)) if working[i]]
        for share, w, idx in (
            (cal["holiday_shift_prev"], cal["holiday_prev_weights"], prev),
            (cal["holiday_shift_next"], cal["holiday_next_weights"], nxt),
        ):
            idx = idx[: len(w)]
            if idx:
                ww = np.array(w[: len(idx)]) / sum(w[: len(idx)])
                for i, wi in zip(idx, ww, strict=True):
                    mu[i] += share * v * wi
    return mu


def _apply_capacity(demand: np.ndarray, working: np.ndarray, cap: int) -> np.ndarray:
    """Ship at most `cap` pallets per site-day; overflow is backlog for the next working day.

    Each customer is cut back proportionally (largest-remainder rounding)."""
    n, T = demand.shape
    shipped = np.zeros_like(demand)
    carry = np.zeros(n, dtype=np.int64)
    for t in range(T):
        want = demand[:, t] + carry
        if not working[t]:
            carry = want
            continue
        tot = want.sum()
        if tot <= cap:
            shipped[:, t], carry = want, np.zeros(n, dtype=np.int64)
            continue
        exact = want * cap / tot
        s = np.floor(exact).astype(np.int64)
        s[np.argsort(-(exact - s))[: cap - s.sum()]] += 1
        shipped[:, t], carry = s, want - s
    return shipped


def simulate(cfg: dict) -> SimOutput:
    root = np.random.SeedSequence(cfg["seed"])
    s_cust, s_promo, s_year, s_days, s_ship = (np.random.default_rng(s) for s in root.spawn(5))

    start, end = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])
    dates = pd.date_range(start, end, freq="D")
    T = len(dates)
    years_since = ((dates - start).days / 365.25).to_numpy()
    doy = dates.dayofyear.to_numpy()

    sites = pd.DataFrame(
        [
            {
                "site_id": s["site_id"],
                "country": s["country"],
                "capacity_pallets_per_day": s["capacity_pallets_per_day"],
                "operating_days": ",".join(
                    d for d, w in zip(WEEKDAYS, s["weekly"], strict=True) if w > 0
                ),
            }
            for s in cfg["sites"]
        ]
    )
    # weekly_change events can add operating days
    for e in cfg["events"].get("weekly_change", []):
        m = sites["site_id"] == e["site_id"]
        days = sites.loc[m, "operating_days"].iloc[0].split(",")
        if e["new_factor"] > 0 and WEEKDAYS[e["weekday"]] not in days:
            days.append(WEEKDAYS[e["weekday"]])
            days = [d for d in WEEKDAYS if d in days]
            sites.loc[m, "operating_days"] = ",".join(days)

    hol = build_holidays(sites["country"].unique().tolist(), range(start.year, end.year + 2))
    customers = build_customers(cfg, s_cust)
    promos = build_promos(cfg, customers, s_promo)

    cc = cfg["customers"]
    years = range(start.year, end.year + 1)
    ice_amp = {y: max(0.3, 1 + cfg["ice_cream_year_sd"] * s_year.standard_normal()) for y in years}
    year_arr = dates.year.to_numpy()

    lull = cfg["calendar"]["year_end_lull"]
    md = dates.strftime("%m-%d")
    lull_f = np.where((md >= lull["start_md"]) & (md <= lull["end_md"]), lull["factor"], 1.0)

    burn = 120
    truth_rows = []
    for site in cfg["sites"]:
        sid = site["site_id"]
        weekly_f = _site_weekly_matrix(cfg, site, dates)
        country = site["country"]
        hol_dates = set(hol.loc[hol["country"] == country, "date"])
        is_hol = dates.isin(list(hol_dates))
        working = (weekly_f > 0) & ~is_hol
        site_ids, site_mu, site_demand = [], [], []

        for c in customers[customers["site_id"] == sid].itertuples():
            rng = s_days
            g = rng.normal(cc["growth_mean"], cc["growth_sd"])
            amp_mult = rng.uniform(1 - cfg["segment_amp_jitter"], 1 + cfg["segment_amp_jitter"])
            prof = annual_profile(cfg["segments"][c.segment], amp_mult)[doy]
            if c.segment == "ice_cream":
                ya = np.array([ice_amp[y] for y in year_arr])
                prof = 1 + (prof - 1) * ya

            # promotions: uplift during the shipping window, dip afterwards
            promo_f = np.ones(T)
            for p in promos[promos["customer_id"] == c.customer_id].itertuples():
                dur = (p.end_date - p.start_date).days + 1
                ws = p.start_date - pd.Timedelta(days=p.sim_ship_lead)
                in_win = (dates >= ws) & (dates < ws + pd.Timedelta(days=dur))
                post = (dates >= ws + pd.Timedelta(days=dur)) & (
                    dates < ws + pd.Timedelta(days=dur + dur // 2)
                )
                promo_f = np.where(in_win, promo_f * p.sim_uplift, promo_f)
                promo_f = np.where(post, promo_f * cfg["promotions"]["post_dip"], promo_f)

            # lifecycle: onboarding ramp-up, churn ramp-down
            life = np.ones(T)
            if c.sim_ramp_on > 0:
                life = _smoothstep((dates - c.onboard_date).days.to_numpy() / c.sim_ramp_on)
            if pd.notna(c.churn_date):
                left = (c.churn_date - dates).days.to_numpy()
                life = life * _smoothstep(left / max(c.sim_ramp_off, 1))

            level = np.ones(T)
            for e in cfg["events"].get("unannounced_level_shift", []):
                if e["site_id"] == sid and e["customer_rank"] == c.sim_rank:
                    level = np.where(dates >= pd.Timestamp(e["date"]), e["factor"], level)

            # AR(1) log-demand shocks, mean-corrected so E[exp(ar)] = 1
            phi, sig = cc["ar_phi"], cc["ar_sigma"]
            eps = rng.standard_normal(T + burn) * sig
            ar = np.zeros(T + burn)
            for t in range(1, T + burn):
                ar[t] = phi * ar[t - 1] + eps[t]
            ar = ar[burn:] - 0.5 * sig**2 / (1 - phi**2)

            mu = (
                c.sim_base
                * (1 + g) ** years_since
                * weekly_f
                * prof
                * promo_f
                * life
                * level
                * lull_f
                * np.exp(ar)
            )

            # holidays: site closed; volume is pre-built before and caught up after
            mu = _redistribute_holidays(mu, is_hol & (weekly_f > 0), working, cfg["calendar"])
            mu[weekly_f == 0] = 0.0

            r = cc["negbin_dispersion"]
            demand = np.where(mu > 0, rng.negative_binomial(r, r / (r + np.maximum(mu, 1e-9))), 0)
            site_ids.append(c.customer_id)
            site_mu.append(mu)
            site_demand.append(demand.astype(np.int64))

        shipped = _apply_capacity(np.array(site_demand), working, site["capacity_pallets_per_day"])
        for k, cid in enumerate(site_ids):
            truth_rows.append(
                pd.DataFrame(
                    {
                        "site_id": sid,
                        "customer_id": cid,
                        "date": dates,
                        "mu": site_mu[k],
                        "demand_pallets": site_demand[k],
                        "pallets_true": shipped[k],
                    }
                )
            )

    daily = pd.concat(truth_rows, ignore_index=True)
    shipments = _split_into_shipments(cfg, daily, customers, s_ship)

    cust_out = customers[
        ["customer_id", "site_id", "segment", "temp_zone", "onboard_date", "churn_date"]
    ].copy()
    cust_out.loc[cust_out["onboard_date"] == start, "onboard_date"] = pd.NaT  # pre-existing
    promo_cal = promos.loc[
        promos["sim_announced"],
        ["promo_id", "customer_id", "uplift_type", "start_date", "end_date", "announce_date"],
    ]
    return SimOutput(
        sites, cust_out, promo_cal.reset_index(drop=True), promos, hol, shipments, daily
    )


def _split_into_shipments(
    cfg: dict, daily: pd.DataFrame, customers: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    cap, full = cfg["truck_capacity_pallets"], cfg["full_truck_share"]
    zone = customers.set_index("customer_id")["temp_zone"]
    nz = daily[daily["pallets_true"] > 0]
    sites, custs, days, sizes = [], [], [], []
    for s, c, d, p in zip(
        nz["site_id"], nz["customer_id"], nz["date"], nz["pallets_true"], strict=True
    ):
        while p > 0:
            k = cap if rng.random() < full else int(rng.integers(6, cap))
            k = min(k, p)
            sites.append(s)
            custs.append(c)
            days.append(d)
            sizes.append(k)
            p -= k
    n = len(sizes)
    ship_date = pd.DatetimeIndex(days)
    hours = rng.uniform(5, 22, n)
    dispatched = ship_date + pd.to_timedelta(hours, unit="h")
    lat_h = rng.exponential(2.0, n)
    late = rng.random(n) < cfg["ingestion"]["late_share"]
    lo, hi = cfg["ingestion"]["late_days"]
    lat_h = lat_h + np.where(late, rng.integers(lo, hi + 1, n) * 24.0, 0.0)
    ingested = dispatched + pd.to_timedelta(lat_h, unit="h")
    olo, ohi = cfg["order_lead_days"]
    order = (
        ship_date
        - pd.to_timedelta(rng.integers(olo, ohi + 1, n), unit="D")
        + pd.to_timedelta(rng.uniform(6, 20, n), unit="h")
    )

    df = pd.DataFrame(
        {
            "shipment_id": [f"SHP{i:09d}" for i in rng.permutation(n) + 1],
            "site_id": sites,
            "customer_id": custs,
            "ship_date": ship_date,
            "pallets": np.array(sizes, dtype=np.int64),
            "temp_zone": zone.loc[custs].to_numpy(),
            "order_created_at": order.round("s"),
            "ingested_at": ingested.round("s"),
        }
    )
    return df.sort_values(["ingested_at", "shipment_id"]).reset_index(drop=True)
