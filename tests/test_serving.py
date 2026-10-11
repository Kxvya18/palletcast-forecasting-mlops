"""S6: the API must give exactly the same forecast as the offline pipeline.

Needs the local data and model registry (skipped where they are not available, e.g. CI)."""

import numpy as np
import pandas as pd
import pytest

from palletcast.models.common import ROOT

pytestmark = pytest.mark.skipif(
    not (ROOT / "mlflow.db").exists() or not (ROOT / "data" / "features" / "site.parquet").exists(),
    reason="needs local MLflow registry and DVC data",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from palletcast.serving.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def offline():
    from palletcast.serving.app import state

    site = pd.read_parquet(ROOT / "data" / "features" / "site.parquet")
    cust = pd.read_parquet(ROOT / "data" / "features" / "customer.parquet")
    return state, site, cust


@pytest.mark.parametrize("origin", ["2026-01-15", "2026-04-06", "2026-07-23", "2026-09-16"])
def test_api_equals_offline_forecast(client, offline, origin):
    state, site, cust = offline
    o = pd.Timestamp(origin)
    want = state["model"].predict_sites(site[site["origin"] == o], cust[cust["origin"] == o])
    got = client.get("/forecast", params={"origin": origin}).json()["forecast"]
    got = pd.DataFrame(got)
    m = want.assign(date=want["target_date"].dt.strftime("%Y-%m-%d")).merge(
        got, on=["site_id", "date", "h"]
    )
    assert len(m) == len(want) == len(got) > 0
    np.testing.assert_allclose(m["pallets"], m["yhat"].round(1), atol=0.051)


def test_latest_forecast_covers_14_days_per_working_day(client):
    r = client.get("/forecast", params={"site": "S02"}).json()
    hs = [f["h"] for f in r["forecast"]]
    assert set(hs) <= set(range(1, 15)) and len(hs) >= 10  # S02 works 7 days; holidays excluded


def test_future_origin_is_refused(client):
    assert client.get("/forecast", params={"origin": "2030-01-01"}).status_code == 400


def test_health_reports_model_and_data(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["data_until"] == "2026-09-30"
