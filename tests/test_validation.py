"""Validation must catch every planted fault (recall) and flag nothing else (precision)."""

import pandas as pd
import pytest

from palletcast.data import validate

ROW_FAULT_REASON = {
    "non_positive_pallets": "pallets:greater_than",
    "unknown_customer": "customer_id:isin",
    "site_customer_mismatch": "site_customer_pair_known",
    "order_after_ship": "order_before_ship",
    "null_key": "not_nullable",
}


def all_warnings(pipeline, kind):
    return [
        w for r in pipeline["reports"].values() for w in r.get("warnings", []) if w["type"] == kind
    ]


def test_pipeline_does_not_hard_fail(pipeline):
    assert pipeline["summary"]["status"] == "pass"


@pytest.mark.parametrize("fault", list(ROW_FAULT_REASON))
def test_row_faults_quarantined_exactly(pipeline, fault):
    q = pipeline["quarantine"]
    caught = set(
        q.loc[q["reason"].str.contains(ROW_FAULT_REASON[fault], regex=False), "shipment_id"]
    )
    planted = set(pipeline["manifest"][fault]) - set(pipeline["manifest"]["not_yet_arrived"])
    assert planted == caught


def test_quarantine_contains_only_planted_rows(pipeline):
    m = pipeline["manifest"]
    planted = set().union(*(m[f] for f in ROW_FAULT_REASON)) - set(m["not_yet_arrived"])
    assert set(pipeline["quarantine"]["shipment_id"]) == planted


def test_exact_duplicates_dropped(pipeline):
    dropped = sum(
        r.get("exact_duplicates_dropped", 0) for r in pipeline["reports"].values() if "batch" in r
    )
    assert dropped == len(pipeline["manifest"]["exact_duplicate"])
    assert pipeline["validated"]["shipment_id"].is_unique


def test_corrections_resolve_to_truth(pipeline):
    m = pipeline["manifest"]
    arrived = set(m["correction"]) - set(m["not_yet_arrived"])
    v = pipeline["validated"].set_index("shipment_id")["pallets"]
    t = pipeline["truth"].set_index("shipment_id")["pallets"]
    ids = sorted(arrived & set(v.index))
    assert len(ids) > 0.9 * len(m["correction"])
    assert (v.loc[ids] == t.loc[ids]).all()


def test_outages_reported_as_missing_days(pipeline):
    planted = {(o["site_id"], o["date"]) for o in pipeline["manifest"]["outage"]}
    warned = {(w["site_id"], w["date"]) for w in all_warnings(pipeline, "missing_operating_day")}
    assert warned == planted


def test_partial_outages_low_volume_precision_and_recall(pipeline):
    """No false low-volume alarms; most partial outages caught. The detector is calibrated
    out-of-sample (reports/calibration/volume_check.json); its known blind spot is a partial
    loss on an unusually busy day, which looks like a normal day (docs/s1_data.md, section 5)."""
    planted = {(o["site_id"], o["date"]) for o in pipeline["manifest"]["partial_outage"]}
    warned = {(w["site_id"], w["date"]) for w in all_warnings(pipeline, "volume_low")}
    assert warned <= planted
    assert len(warned & planted) / len(planted) >= 0.8


def test_volume_check_flags_weekday_feed_loss(pipeline, cfg):
    """Deterministic: drop 70% of a normal Wednesday at every site; each must alarm."""
    master = validate.load_master(pipeline["dir"] / "raw")
    daily = pipeline["validated"].groupby(["site_id", "ship_date"])["pallets"].sum().astype(float)
    day = pd.Timestamp("2025-03-12")  # a Wednesday with no holidays nearby
    for s in master["sites"]["site_id"]:
        daily.loc[(s, day)] *= 0.3
    sc = validate.volume_scores(
        daily,
        master,
        daily.index.get_level_values(1).min(),
        day,
        day,
        cfg["validation"]["outlier_window_weeks"],
    )
    assert len(sc) == 4 and validate.volume_flags(sc, cfg["validation"]).all()


def test_high_volume_alarm_rate_is_low(pipeline):
    site_days = pipeline["validated"].groupby(["site_id", "ship_date"]).ngroups
    assert len(all_warnings(pipeline, "volume_high")) / site_days < 0.001


def test_end_to_end_reconciliation_with_truth(pipeline):
    """Every true shipment is either validated with its true value, or explained by a fault."""
    m, truth, raw, val = (pipeline[k] for k in ("manifest", "truth", "raw", "validated"))
    t = truth.set_index("shipment_id")
    assert set(val["shipment_id"]) <= set(t.index)

    stale_corr = set(m["correction"]) & set(m["not_yet_arrived"])
    v = val.set_index("shipment_id")
    ok = v.index.difference(sorted(stale_corr))
    assert (v.loc[ok, "pallets"] == t.loc[ok, "pallets"]).all()

    missing = set(t.index) - set(v.index)
    never_arrived = set(t.index) - set(raw["shipment_id"].dropna())
    quarantined = set(pipeline["quarantine"]["shipment_id"])
    assert missing == never_arrived | quarantined

    lost_days = {(o["site_id"], o["date"]) for o in m["outage"] + m["partial_outage"]}
    na = t.loc[sorted(never_arrived)]
    explained = [
        (s, d.strftime("%Y-%m-%d")) in lost_days or i in set(m["not_yet_arrived"])
        for i, s, d in zip(na.index, na["site_id"], na["ship_date"], strict=True)
    ]
    assert all(explained)


# ---- contract / hard-fail behaviour on hand-made batches ----


@pytest.fixture(scope="module")
def master(pipeline):
    return validate.load_master(pipeline["dir"] / "raw")


@pytest.fixture(scope="module")
def good_batch(pipeline):
    return pd.read_parquet(pipeline["dir"] / "raw" / "shipments" / "live-2026-03.parquet")


def test_missing_column_hard_fails(good_batch, master, cfg):
    r = validate.check_rows(good_batch.drop(columns="pallets"), "x", master, cfg)
    assert r.status == "fail" and "contract" in r.errors[0]


def test_extra_column_hard_fails(good_batch, master, cfg):
    r = validate.check_rows(good_batch.assign(foo=1), "x", master, cfg)
    assert r.status == "fail"


def test_empty_batch_hard_fails(good_batch, master, cfg):
    assert validate.check_rows(good_batch.iloc[:0], "x", master, cfg).status == "fail"


def test_quarantine_rate_threshold_hard_fails(good_batch, master, cfg):
    b = good_batch.copy()
    n = int(len(b) * 0.05)
    b.loc[b.index[:n], "pallets"] = 0
    r = validate.check_rows(b, "x", master, cfg)
    assert r.status == "fail" and "quarantine rate" in r.errors[0]


def test_unparseable_value_is_quarantined_not_fatal(good_batch, master, cfg):
    b = good_batch.copy()
    b["pallets"] = b["pallets"].astype(object)
    b.loc[b.index[0], "pallets"] = "twelve"
    r = validate.check_rows(b, "x", master, cfg)
    assert r.status != "fail"
    assert b.loc[b.index[0], "shipment_id"] in set(r.quarantine["shipment_id"])


def test_more_pallets_than_a_truck_is_quarantined(good_batch, master, cfg):
    b = good_batch.copy()
    b.loc[b.index[0], "pallets"] = 330  # cases recorded as pallets
    r = validate.check_rows(b, "x", master, cfg)
    q = r.quarantine.set_index("shipment_id")["reason"]
    assert q[b.loc[b.index[0], "shipment_id"]] == "pallets:less_than_or_equal_to"


def test_conflicting_duplicate_quarantined(good_batch, master, cfg):
    b = good_batch.copy()
    clash = b.iloc[[0]].copy()
    clash["pallets"] = clash["pallets"].where(clash["pallets"] < 2, clash["pallets"] - 1) + (
        clash["pallets"] < 2
    )
    r = validate.check_rows(pd.concat([b, clash], ignore_index=True), "x", master, cfg)
    sid = clash["shipment_id"].iloc[0]
    assert (
        r.quarantine.loc[r.quarantine["shipment_id"] == sid, "reason"] == "conflicting_duplicate"
    ).sum() == 2
    assert sid not in set(r.clean["shipment_id"])


def test_invalid_master_data_hard_fails(pipeline, tmp_path):
    import shutil

    raw = tmp_path / "raw"
    shutil.copytree(pipeline["dir"] / "raw", raw)
    c = pd.read_parquet(raw / "master" / "customers.parquet")
    pd.concat([c, c.iloc[[0]]]).to_parquet(raw / "master" / "customers.parquet", index=False)
    with pytest.raises(validate.ValidationFailed):
        validate.load_master(raw)
