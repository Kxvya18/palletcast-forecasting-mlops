"""Point-in-time ("as-of") reconstruction of daily volumes.

A record version is visible from its ``ingested_at`` until a newer version of the same
shipment arrives. A forecast made on origin day O sees everything that arrived before
O + cutoff_hour. For each (key, ship_date) we compute the volume known at origins
ship_date + k for k = 1..max_lag, and the final volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)


def version_events(versions: pd.DataFrame) -> pd.DataFrame:
    """+pallets when a version arrives, -pallets when it is superseded."""
    v = versions.sort_values(["shipment_id", "ingested_at"], kind="stable")
    nxt = v.groupby("shipment_id")["ingested_at"].shift(-1)
    add = v.assign(t=v["ingested_at"], delta=v["pallets"].astype(np.int64))
    rem = v.loc[nxt.notna()].assign(
        t=nxt[nxt.notna()], delta=-v.loc[nxt.notna(), "pallets"].astype(np.int64)
    )
    cols = ["site_id", "customer_id", "ship_date", "t", "delta"]
    return pd.concat([add[cols], rem[cols]], ignore_index=True)


def first_visible_lag(t: pd.Series, ship_date: pd.Series, cutoff_hour: int) -> pd.Series:
    """Smallest k >= 1 such that an event at time t is visible to the origin ship_date + k."""
    rel = (t - ship_date - pd.Timedelta(hours=cutoff_hour)) / DAY
    return np.maximum(np.floor(rel).astype(np.int64) + 1, 1)


def known_by_lag(
    versions: pd.DataFrame, keys: list[str], cutoff_hour: int, max_lag: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (known, final).

    known: index = keys + [ship_date], columns 1..max_lag, value = pallets known at origin
           ship_date + k.
    final: same index, total pallets after all versions have arrived.
    """
    ev = version_events(versions)
    ev["k"] = first_visible_lag(ev["t"], ev["ship_date"], cutoff_hour)
    idx = keys + ["ship_date"]
    final = ev.groupby(idx)["delta"].sum()
    win = ev[ev["k"] <= max_lag]
    wide = win.pivot_table(index=idx, columns="k", values="delta", aggfunc="sum", fill_value=0)
    wide = wide.reindex(index=final.index, columns=range(1, max_lag + 1), fill_value=0)
    return wide.cumsum(axis=1), final
