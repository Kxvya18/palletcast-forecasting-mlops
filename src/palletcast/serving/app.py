"""S6: forecast API.

    GET  /health                      which model and data the service is using
    GET  /forecast?site=S02&origin=   14-day forecast made at origin 06:00 (default: the
                                      morning after the latest data)
    POST /reload                      load the model that currently has the production label

Features are built with exactly the same S2 code used for training, from the data as it
was known at the origin, so live forecasts cannot drift from what was tested.

    uv run uvicorn palletcast.serving.app:app --port 8000
"""

import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException

from palletcast.features import build
from palletcast.models.common import ROOT

MODEL_URI = os.environ.get("MODEL_URI", "models:/outbound_forecaster@production")
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data" / "validated"))
DAY = pd.Timedelta(days=1)

state = {}


def load_model():
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    )
    model = mlflow.pyfunc.load_model(MODEL_URI)
    state["model"] = model.unwrap_python_model()
    state["model_info"] = {"uri": MODEL_URI, "run_id": model.metadata.run_id}


@asynccontextmanager
async def lifespan(app):
    load_model()
    state["inputs"] = build.load_inputs(DATA_DIR)
    state["cfg"] = yaml.safe_load(open(ROOT / "configs" / "features.yaml"))
    state["data_until"] = state["inputs"]["versions"]["ship_date"].max()
    yield


app = FastAPI(title="PalletCast outbound forecast", lifespan=lifespan)


def forecast_sites(origin: pd.Timestamp) -> pd.DataFrame:
    """Build the origin's features (as known at 06:00) and predict all sites."""
    site_df, cust_df = build.build(
        state["inputs"], state["cfg"], origins=pd.DatetimeIndex([origin]), end=origin + 14 * DAY
    )
    return state["model"].predict_sites(site_df, cust_df)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": state["model_info"],
        "data_until": str(state["data_until"].date()),
    }


@app.get("/forecast")
def forecast(site: str | None = None, origin: date | None = None):
    latest = state["data_until"] + DAY
    origin = pd.Timestamp(origin) if origin else latest
    if origin > latest:
        raise HTTPException(
            400, f"no data yet for origin {origin.date()}; latest is {latest.date()}"
        )
    if origin < pd.Timestamp(state["cfg"]["first_origin"]):
        raise HTTPException(400, "origin before the first possible forecast date")
    p = forecast_sites(origin)
    if site:
        if site not in set(p["site_id"]):
            raise HTTPException(404, f"unknown site {site}")
        p = p[p["site_id"] == site]
    rows = [
        {
            "site_id": r.site_id,
            "date": str(r.target_date.date()),
            "h": int(r.h),
            "pallets": round(float(r.yhat), 1),
        }
        for r in p.sort_values(["site_id", "h"]).itertuples()
    ]
    return {"origin": str(origin.date()), "model": state["model_info"], "forecast": rows}


@app.post("/reload")
def reload():
    load_model()
    return {"reloaded": state["model_info"]}
