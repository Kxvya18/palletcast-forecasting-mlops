"""Inject realistic data-quality faults into clean shipments and record exactly what was done.

The manifest is the answer key for validation tests: every injected fault must be caught
and nothing else may be flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_faults(
    cfg: dict, shipments: pd.DataFrame, sites: pd.DataFrame, customers: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, dict]:
    fc = cfg["faults"]
    rng = np.random.default_rng(seed)
    df = shipments.copy()
    manifest: dict = {}

    # 1. Feed outages: whole site-days that never reach the feed.
    start, end = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])
    years = (end - start).days / 365.25
    outages = []
    for sid in sites["site_id"]:
        ship_days = np.sort(df.loc[df["site_id"] == sid, "ship_date"].unique())
        for _ in range(rng.poisson(fc["outages_per_site_year"] * years)):
            i = int(rng.integers(0, len(ship_days) - 3))
            k = int(rng.integers(fc["outage_days"][0], fc["outage_days"][1] + 1))
            for d in ship_days[i : i + k]:
                outages.append({"site_id": sid, "date": pd.Timestamp(d).strftime("%Y-%m-%d")})
    outages = [dict(t) for t in {tuple(o.items()) for o in outages}]
    out_key = set((o["site_id"], o["date"]) for o in outages)
    lost = df.apply(
        lambda r: (r["site_id"], r["ship_date"].strftime("%Y-%m-%d")) in out_key, axis=1
    )
    manifest["outage"] = sorted(outages, key=lambda o: (o["site_id"], o["date"]))
    manifest["outage_lost_pallets"] = int(df.loc[lost, "pallets"].sum())
    df = df.loc[~lost].reset_index(drop=True)

    # 1b. Partial outages: only a share of a site-day's rows reach the feed.
    partial, drop_idx = [], []
    lo, hi = fc["partial_keep_share"]
    for sid in sites["site_id"]:
        sd = df.loc[df["site_id"] == sid, "ship_date"]
        days = np.sort(sd.unique())
        for _ in range(rng.poisson(fc["partial_outages_per_site_year"] * years)):
            d = pd.Timestamp(days[rng.integers(len(days))])
            if any(p["site_id"] == sid and p["date"] == d.strftime("%Y-%m-%d") for p in partial):
                continue
            keep = float(rng.uniform(lo, hi))
            rows = sd.index[sd == d].to_numpy()
            n_drop = int(round(len(rows) * (1 - keep)))
            drop_idx.extend(rng.choice(rows, n_drop, replace=False).tolist())
            partial.append({"site_id": sid, "date": d.strftime("%Y-%m-%d"), "keep": round(keep, 3)})
    manifest["partial_outage"] = sorted(partial, key=lambda o: (o["site_id"], o["date"]))
    manifest["partial_outage_lost_pallets"] = int(df.loc[drop_idx, "pallets"].sum())
    df = df.drop(index=drop_idx).reset_index(drop=True)

    # 2. Disjoint row samples for row-level faults.
    n = len(df)
    kinds = [
        "correction",
        "non_positive_pallets",
        "unknown_customer",
        "site_customer_mismatch",
        "order_after_ship",
        "null_key",
        "exact_duplicate",
    ]
    rates = {k: fc[f"{k}_rate"] for k in kinds}
    counts = {k: max(1, int(round(rates[k] * n))) for k in kinds}
    pool = rng.permutation(n)
    idx, o = {}, 0
    for k in kinds:
        idx[k] = np.sort(pool[o : o + counts[k]])
        o += counts[k]

    # corrections: original row gets wrong pallets; the true row arrives later
    corr = df.loc[idx["correction"]].copy()
    delta = rng.integers(1, 6, len(corr)) * rng.choice([-1, 1], len(corr))
    wrong = corr["pallets"].to_numpy() + delta
    # keep the wrong value plausible (1..truck capacity) so it is only a correction fault
    flip = (wrong < 1) | (wrong > cfg["truck_capacity_pallets"])
    wrong = np.where(flip, corr["pallets"].to_numpy() - delta, wrong)
    df.loc[idx["correction"], "pallets"] = wrong
    corr["ingested_at"] = corr["ingested_at"] + pd.to_timedelta(
        rng.integers(1, 8, len(corr)), unit="D"
    )
    manifest["correction"] = corr["shipment_id"].tolist()

    i = idx["non_positive_pallets"]
    df.loc[i, "pallets"] = np.where(rng.random(len(i)) < 0.5, 0, -df.loc[i, "pallets"])
    manifest["non_positive_pallets"] = df.loc[i, "shipment_id"].tolist()

    i = idx["unknown_customer"]
    df.loc[i, "customer_id"] = [f"C9{rng.integers(100, 999)}" for _ in i]
    manifest["unknown_customer"] = df.loc[i, "shipment_id"].tolist()

    i = idx["site_customer_mismatch"]
    for j in i:
        own = df.at[j, "site_id"]
        other = customers.loc[customers["site_id"] != own, "customer_id"].to_numpy()
        df.at[j, "customer_id"] = other[rng.integers(len(other))]
    manifest["site_customer_mismatch"] = df.loc[i, "shipment_id"].tolist()

    i = idx["order_after_ship"]
    df.loc[i, "order_created_at"] = (
        df.loc[i, "ship_date"]
        + pd.to_timedelta(rng.integers(1, 4, len(i)), unit="D")
        + pd.Timedelta(hours=9)
    )
    manifest["order_after_ship"] = df.loc[i, "shipment_id"].tolist()

    i = idx["null_key"]
    col = np.where(rng.random(len(i)) < 0.5, "site_id", "customer_id")
    for j, c in zip(i, col, strict=True):
        df.at[j, c] = None
    manifest["null_key"] = df.loc[i, "shipment_id"].tolist()

    dups = df.loc[idx["exact_duplicate"]].copy()
    manifest["exact_duplicate"] = dups["shipment_id"].tolist()

    df = pd.concat([df, corr, dups], ignore_index=True)
    df = df.sort_values(["ingested_at", "shipment_id"], kind="stable").reset_index(drop=True)

    # 3. Snapshot time: rows ingested after end_date have not arrived yet.
    snap = end + pd.Timedelta(days=1)
    not_yet = df["ingested_at"] >= snap
    manifest["not_yet_arrived"] = df.loc[not_yet, "shipment_id"].tolist()
    df = df.loc[~not_yet].reset_index(drop=True)
    manifest["counts"] = {k: len(v) for k, v in manifest.items() if isinstance(v, list)}
    return df, manifest


def split_batches(raw: pd.DataFrame, history_cutoff: str) -> dict[str, pd.DataFrame]:
    """History snapshot + monthly live batches, split by ingestion time like a real feed."""
    cut = pd.Timestamp(history_cutoff)
    out = {"history": raw.loc[raw["ingested_at"] < cut].reset_index(drop=True)}
    live = raw.loc[raw["ingested_at"] >= cut]
    for period, g in live.groupby(live["ingested_at"].dt.to_period("M")):
        out[f"live-{period}"] = g.reset_index(drop=True)
    return out
