"""Generator properties: reproducibility and the structure later stages rely on."""

import copy

import numpy as np
import pandas as pd

from palletcast.data.simulate import _apply_capacity, annual_profile, simulate
from palletcast.data.validate import frame_hash


def test_same_seed_same_data(cfg):
    a, b = simulate(cfg), simulate(cfg)
    assert frame_hash(a.shipments) == frame_hash(b.shipments)
    assert frame_hash(a.daily_truth) == frame_hash(b.daily_truth)


def test_different_seed_different_data(cfg):
    c2 = copy.deepcopy(cfg)
    c2["seed"] += 1
    assert frame_hash(simulate(cfg).shipments) != frame_hash(simulate(c2).shipments)


def test_annual_profile_has_mean_one():
    for bumps in ([[165, 45, 1.2]], [[300, 40, 0.15], [200, 30, -0.15]]):
        assert abs(annual_profile(bumps)[1:366].mean() - 1) < 1e-9


def test_capacity_never_exceeded_and_pallets_conserved():
    rng = np.random.default_rng(0)
    demand = rng.integers(0, 400, size=(6, 200))
    working = rng.random(200) > 0.2
    shipped = _apply_capacity(demand, working, cap=1000)
    assert (shipped.sum(axis=0) <= 1000).all()
    assert (shipped[:, ~working] == 0).all()
    # whatever is not shipped by the end is the final backlog, never lost or invented
    assert shipped.sum() <= demand.sum()


def test_truth_respects_capacity_and_closures(pipeline):
    d = pd.read_parquet(pipeline["dir"] / "sim_truth" / "daily_truth.parquet")
    sites = pd.read_parquet(pipeline["dir"] / "raw" / "master" / "sites.parquet")
    hol = pd.read_parquet(pipeline["dir"] / "raw" / "master" / "holidays.parquet")
    tot = d.groupby(["site_id", "date"])["pallets_true"].sum().reset_index()
    tot = tot.merge(sites, on="site_id")
    assert (tot["pallets_true"] <= tot["capacity_pallets_per_day"]).all()
    closed = tot.merge(hol, left_on=["country", "date"], right_on=["country", "date"])
    assert (closed["pallets_true"] == 0).all()


def test_shipments_sum_to_daily_truth(pipeline):
    t = pipeline["truth"].groupby(["site_id", "customer_id", "ship_date"])["pallets"].sum()
    d = pd.read_parquet(pipeline["dir"] / "sim_truth" / "daily_truth.parquet")
    d = d[d["pallets_true"] > 0].set_index(["site_id", "customer_id", "date"])["pallets_true"]
    assert (t.sort_index().to_numpy() == d.sort_index().to_numpy()).all()
    assert (pipeline["truth"]["pallets"] <= 33).all()
