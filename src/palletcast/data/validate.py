"""Validate raw shipment batches in arrival order and build the cumulative validated dataset.

Policy (see docs/s1_data.md):
  * Structural contract breaks (missing/extra columns, invalid master data) -> HARD FAIL.
  * Row-level rule breaks -> row QUARANTINED with a reason; batch continues.
  * Quarantine rate above `max_quarantine_rate` -> HARD FAIL (the feed itself is broken).
  * Exact duplicates are dropped; for repeated shipment_ids the latest ingested row wins
    (corrections). Same id + same ingested_at + different content -> both quarantined.
  * Batch-level anomalies (missing operating days, outliers, late share, staleness) -> WARN.

    uv run python -m palletcast.data.validate --raw data/raw --out data/validated
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

from palletcast.data import schemas
from palletcast.data.simulate import WEEKDAYS, load_config


class ValidationFailed(Exception):
    pass


@dataclass
class BatchResult:
    batch: str
    rows_in: int
    clean: pd.DataFrame
    quarantine: pd.DataFrame
    exact_duplicates_dropped: int
    status: str = "pass"
    errors: list[str] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    superseded_by_correction: int = 0


def frame_hash(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df.reset_index(drop=True), index=True).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def load_master(raw: Path) -> dict[str, pd.DataFrame]:
    m = {
        n: pd.read_parquet(raw / "master" / f"{n}.parquet")
        for n in ("sites", "customers", "promo_calendar", "holidays")
    }
    try:
        schemas.SITES.validate(m["sites"], lazy=True)
        schemas.CUSTOMERS.validate(m["customers"], lazy=True)
        schemas.PROMO_CALENDAR.validate(m["promo_calendar"], lazy=True)
        schemas.HOLIDAYS.validate(m["holidays"], lazy=True)
    except pa.errors.SchemaErrors as e:
        raise ValidationFailed(f"master data invalid:\n{e.failure_cases}") from e
    bad = ~m["customers"]["site_id"].isin(m["sites"]["site_id"])
    if bad.any():
        raise ValidationFailed(f"customers reference unknown sites: {m['customers'][bad]}")
    bad = ~m["promo_calendar"]["customer_id"].isin(m["customers"]["customer_id"])
    if bad.any():
        raise ValidationFailed(f"promos reference unknown customers: {int(bad.sum())} rows")
    return m


def check_rows(batch: pd.DataFrame, name: str, master: dict, cfg: dict) -> BatchResult:
    vc = cfg["validation"]
    res = BatchResult(
        batch=name,
        rows_in=len(batch),
        clean=batch.iloc[:0],
        quarantine=batch.iloc[:0],
        exact_duplicates_dropped=0,
    )
    missing = set(schemas.SHIPMENT_COLUMNS) - set(batch.columns)
    extra = set(batch.columns) - set(schemas.SHIPMENT_COLUMNS)
    if missing or extra:
        res.status = "fail"
        res.errors.append(
            f"column contract broken: missing={sorted(missing)} extra={sorted(extra)}"
        )
        return res
    if len(batch) == 0:
        res.status = "fail"
        res.errors.append("empty batch")
        return res

    # 1. Type coercion: unparseable values quarantine their row instead of failing the column.
    reasons: dict[int, list[str]] = {}
    batch = batch.copy()
    for col, kind in schemas.TYPED_COLUMNS.items():
        raw_col = batch[col]
        conv = (
            pd.to_numeric(raw_col, errors="coerce")
            if kind == "int"
            else pd.to_datetime(raw_col, errors="coerce")
        )
        if kind == "int":
            conv = conv.where(conv.isna() | (conv == conv.round()))  # 12.5 pallets is not valid
        for i in batch.index[raw_col.notna() & conv.isna()]:
            reasons.setdefault(int(i), []).append(f"{col}:unparseable")
        batch[col] = conv
    unparsed = sorted(reasons)
    rest = batch.drop(index=unparsed)
    for col, kind in schemas.TYPED_COLUMNS.items():
        rest[col] = (
            rest[col].astype("Int64") if kind == "int" else rest[col].astype("datetime64[ns]")
        )

    # 2. Row-level rules.
    schema = schemas.shipment_schema(
        master["sites"], master["customers"], cfg["truck_capacity_pallets"]
    )
    try:
        schema.validate(rest, lazy=True)
    except pa.errors.SchemaErrors as e:
        fc = e.failure_cases
        structural = fc[fc["index"].isna()]
        if len(structural):
            res.status = "fail"
            res.errors.append(
                f"structural failures: {structural[['column', 'check']].to_dict('records')}"
            )
            return res
        for ctx, col, chk, i in fc[["schema_context", "column", "check", "index"]].itertuples(
            index=False
        ):
            name = str(chk).split("(")[0]
            tag = name if ctx == "DataFrameSchema" else f"{col}:{name}"
            reasons.setdefault(int(i), []).append(tag)

    bad_idx = sorted(reasons)
    q = batch.loc[bad_idx].copy()
    q["reason"] = [";".join(sorted(set(reasons[i]))) for i in bad_idx]
    ok = batch.drop(index=bad_idx)
    ok["pallets"] = ok["pallets"].astype("int64")
    ok = schema.validate(ok)  # output contract: must pass now

    dup = ok.duplicated(keep="first")
    res.exact_duplicates_dropped = int(dup.sum())
    ok = ok.loc[~dup]

    key = ok.duplicated(["shipment_id", "ingested_at"], keep=False)
    if key.any():
        c = ok.loc[key].copy()
        c["reason"] = "conflicting_duplicate"
        q = pd.concat([q, c])
        ok = ok.loc[~key]

    q["batch"] = name
    res.clean, res.quarantine = ok.reset_index(drop=True), q.reset_index(drop=True)
    rate = len(q) / len(batch)
    if rate > vc["max_quarantine_rate"]:
        res.status = "fail"
        res.errors.append(f"quarantine rate {rate:.4f} > {vc['max_quarantine_rate']}")
    return res


def merge_latest(cum: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Union with latest-ingested-wins per shipment_id. Returns (merged, n_superseded)."""
    allr = pd.concat([cum, new], ignore_index=True)
    allr = allr.drop_duplicates()  # identical re-sends across batches
    allr = allr.sort_values(["shipment_id", "ingested_at"], kind="stable")
    keep = ~allr.duplicated("shipment_id", keep="last")
    return allr.loc[keep].reset_index(drop=True), int((~keep).sum())


def expected_days(master: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Site x date grid of days the site is expected to ship (operating weekday, not a holiday)."""
    dates = pd.date_range(start, end, freq="D")
    rows = []
    hol = master["holidays"]
    for s in master["sites"].itertuples():
        ops = {WEEKDAYS.index(d) for d in s.operating_days.split(",")}
        h = set(hol.loc[hol["country"] == s.country, "date"])
        for d in dates:
            if d.dayofweek in ops and d not in h:
                rows.append((s.site_id, d))
    return pd.DataFrame(rows, columns=["site_id", "date"])


def near_holiday(master: dict) -> set[tuple[str, pd.Timestamp]]:
    """(site, date) within 4 days before / 3 days after a holiday: pre-build and catch-up."""
    out = set()
    for s in master["sites"].itertuples():
        for d in master["holidays"].loc[master["holidays"]["country"] == s.country, "date"]:
            for k in range(-4, 4):
                out.add((s.site_id, d + pd.Timedelta(days=k)))
    return out


def volume_scores(
    daily: pd.Series,
    master: dict,
    start: pd.Timestamp,
    w0: pd.Timestamp,
    w1: pd.Timestamp,
    n_weeks: int,
) -> pd.DataFrame:
    """Score each expected site-day in [w0, w1] against its own recent history.

    ratio     = pallets / median of the same weekday over the previous `n_weeks`
                (holiday-shifted days excluded from the reference)
    gap_share = |pallets - that median| / the site's typical Tue-Thu day over the same
                window, i.e. how big the deviation is in units of a normal working day
    """
    nh = near_holiday(master)
    full = expected_days(master, start, w1)
    full["pallets"] = [
        daily.get((s, d), np.nan) for s, d in zip(full["site_id"], full["date"], strict=True)
    ]
    full["dow"] = full["date"].dt.dayofweek
    full["near_holiday"] = [
        (s, d) in nh for s, d in zip(full["site_id"], full["date"], strict=True)
    ]
    out = []
    for s, gs in full.groupby("site_id"):
        mid = gs[gs["dow"].between(1, 3) & ~gs["near_holiday"]].set_index("date")["pallets"]
        scale = mid.rolling(f"{7 * n_weeks}D", closed="left", min_periods=6).median()
        for _dow, g in gs.groupby("dow"):
            g = g.sort_values("date")
            y = g["pallets"].to_numpy(dtype=float)
            ref_y = np.where(g["near_holiday"], np.nan, y)  # holiday shifts are no baseline
            for i in np.where((g["date"] >= w0).to_numpy() & ~np.isnan(y))[0]:
                ref = ref_y[max(0, i - n_weeks) : i]
                ref = ref[~np.isnan(ref)]
                d = g["date"].iloc[i]
                sc = scale.asof(d) if len(scale) and d > scale.index[0] else np.nan
                if len(ref) < 4 or not sc or np.isnan(sc):
                    continue
                med = float(np.median(ref))
                out.append(
                    (
                        s,
                        d,
                        y[i],
                        med,
                        y[i] / med,
                        abs(y[i] - med) / sc,
                        bool(g["near_holiday"].iloc[i]),
                    )
                )
    return pd.DataFrame(
        out,
        columns=["site_id", "date", "pallets", "ref_median", "ratio", "gap_share", "near_holiday"],
    )


def volume_flags(sc: pd.DataFrame, vc: dict) -> pd.Series:
    """A day alarms only if the ratio is out of range AND the gap is material."""
    hi = np.where(
        sc["near_holiday"], vc["outlier_high_ratio_near_holiday"], vc["outlier_high_ratio"]
    )
    off = (sc["ratio"] < vc["outlier_low_ratio"]) | (sc["ratio"] > hi)
    return off & (sc["gap_share"] >= vc["outlier_min_gap_share"])


def batch_warnings(
    cum: pd.DataFrame,
    res: BatchResult,
    window: tuple[pd.Timestamp, pd.Timestamp],
    master: dict,
    cfg: dict,
) -> list[dict]:
    vc = cfg["validation"]
    w0, w1 = window
    warns: list[dict] = []

    daily = cum.groupby(["site_id", "ship_date"])["pallets"].sum()
    exp = expected_days(master, w0, w1)
    have = set(daily.index)
    miss = exp[[(s, d) not in have for s, d in zip(exp["site_id"], exp["date"], strict=True)]]
    # the last `staleness_days` of the window are judged by the staleness check instead
    miss = miss[miss["date"] <= w1 - pd.Timedelta(days=vc["staleness_days"])]
    for s, d in zip(miss["site_id"], miss["date"], strict=True):
        warns.append(
            {"type": "missing_operating_day", "site_id": s, "date": d.strftime("%Y-%m-%d")}
        )

    sc = volume_scores(daily, master, cum["ship_date"].min(), w0, w1, vc["outlier_window_weeks"])
    for r in sc[volume_flags(sc, vc)].itertuples():
        warns.append(
            {
                "type": "volume_low" if r.ratio < 1 else "volume_high",
                "site_id": r.site_id,
                "date": r.date.strftime("%Y-%m-%d"),
                "pallets": int(r.pallets),
                "ref_median": int(r.ref_median),
                "ratio": round(float(r.ratio), 3),
                "gap_share": round(float(r.gap_share), 3),
                "near_holiday": bool(r.near_holiday),
            }
        )

    b = res.clean
    if len(b):
        lag = (b["ingested_at"].dt.normalize() - b["ship_date"]).dt.days
        late = float((lag > 2).mean())
        if late > vc["max_late_share"]:
            warns.append({"type": "late_arrivals", "share": round(late, 4)})
    for s in master["sites"]["site_id"]:
        last = cum.loc[cum["site_id"] == s, "ship_date"].max()
        e = exp[(exp["site_id"] == s) & (exp["date"] > last)]
        if len(e) > vc["staleness_days"]:
            warns.append(
                {
                    "type": "stale_site",
                    "site_id": s,
                    "last_ship_date": str(last.date()),
                    "missing_operating_days": len(e),
                }
            )
    return warns


def batch_window(name: str, cum_start: pd.Timestamp, cutoff: pd.Timestamp):
    if name == "history":
        return cum_start, cutoff - pd.Timedelta(days=1)
    p = pd.Period(name.removeprefix("live-"), "M")
    return p.start_time, p.end_time.normalize()


def run(raw: Path, out: Path, reports: Path, cfg: dict, upto: str | None = None) -> dict:
    master = load_master(raw)
    files = sorted(
        (raw / "shipments").glob("*.parquet"), key=lambda p: "" if p.stem == "history" else p.stem
    )
    if upto:
        files = [f for f in files if f.stem == "history" or f.stem <= f"live-{upto}"]
    cutoff = pd.Timestamp(cfg["history_cutoff"])

    if out.exists():
        shutil.rmtree(out)
    (out / "master").mkdir(parents=True)
    if reports.exists():
        shutil.rmtree(reports)
    reports.mkdir(parents=True)

    cum = pd.DataFrame(columns=schemas.SHIPMENT_COLUMNS)
    quarantine, versions, flags = [], [], []
    summary = {"batches": [], "status": "pass"}
    for f in files:
        res = check_rows(pd.read_parquet(f), f.stem, master, cfg)
        if res.status != "fail":
            cum, res.superseded_by_correction = merge_latest(
                cum if len(cum) else res.clean.iloc[:0], res.clean
            )
            res.warnings = batch_warnings(
                cum, res, batch_window(f.stem, cum["ship_date"].min(), cutoff), master, cfg
            )
            res.status = "warn" if res.warnings else "pass"
            quarantine.append(res.quarantine)
            versions.append(res.clean)
            flags += [
                {"site_id": w["site_id"], "date": w["date"], "type": w["type"], "batch": res.batch}
                for w in res.warnings
                if w["type"] in ("missing_operating_day", "volume_low")
            ]
        rep = {
            "batch": res.batch,
            "status": res.status,
            "rows_in": res.rows_in,
            "rows_clean": len(res.clean),
            "exact_duplicates_dropped": res.exact_duplicates_dropped,
            "superseded_by_correction": res.superseded_by_correction,
            "quarantined": len(res.quarantine),
            "quarantine_rate": round(len(res.quarantine) / max(res.rows_in, 1), 5),
            "quarantine_reasons": (
                res.quarantine["reason"].value_counts().to_dict() if len(res.quarantine) else {}
            ),
            "errors": res.errors,
            "warnings_by_type": pd.Series([w["type"] for w in res.warnings])
            .value_counts()
            .to_dict()
            if res.warnings
            else {},
            "warnings": res.warnings,
        }
        (reports / f"{res.batch}.json").write_text(json.dumps(rep, indent=1, default=str))
        summary["batches"].append(
            {k: rep[k] for k in ("batch", "status", "rows_in", "quarantined", "warnings_by_type")}
        )
        print(
            f"{res.batch:>14}: {res.status:4}  in={res.rows_in:>7,}  q={len(res.quarantine):>4}  "
            f"dup={res.exact_duplicates_dropped:>3}  corr={res.superseded_by_correction:>3}  "
            f"warn={rep['warnings_by_type']}"
        )
        if res.status == "fail":
            summary["status"] = "fail"
            summary["failed_batch"] = res.batch
            break

    cum = cum.sort_values(["ship_date", "site_id", "customer_id", "shipment_id"]).reset_index(
        drop=True
    )
    cum.to_parquet(out / "shipments.parquet", index=False)
    # Every valid record version with its arrival time, so later stages can rebuild exactly
    # what was known at any moment (point-in-time). Re-sends across batches collapse.
    if versions:
        v = pd.concat(versions, ignore_index=True).drop_duplicates()
        v = v.sort_values(["ingested_at", "shipment_id"]).reset_index(drop=True)
        v.to_parquet(out / "shipment_versions.parquet", index=False)
    fl = pd.DataFrame(flags, columns=["site_id", "date", "type", "batch"])
    fl["date"] = pd.to_datetime(fl["date"])
    fl.to_parquet(out / "data_flags.parquet", index=False)
    q = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame()
    q.to_parquet(out / "quarantine.parquet", index=False)
    for n, m in master.items():
        m.to_parquet(out / "master" / f"{n}.parquet", index=False)
    summary.update(
        {
            "rows_validated": len(cum),
            "rows_quarantined": len(q),
            "validated_hash": frame_hash(cum),
            "last_ship_date": str(cum["ship_date"].max().date()) if len(cum) else None,
        }
    )
    (reports / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/validated")
    ap.add_argument("--reports", default="reports/validation")
    ap.add_argument("--upto", default=None, help="last live month to include, e.g. 2026-03")
    a = ap.parse_args(argv)
    try:
        s = run(Path(a.raw), Path(a.out), Path(a.reports), load_config(a.config), a.upto)
    except ValidationFailed as e:
        raise SystemExit(f"HARD FAIL: {e}") from e
    if s["status"] == "fail":
        raise SystemExit(f"HARD FAIL in batch {s['failed_batch']}; see {a.reports}")


if __name__ == "__main__":
    main()
