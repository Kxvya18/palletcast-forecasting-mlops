import json
from pathlib import Path

import pandas as pd
import pytest

from palletcast.data import make_dataset, validate
from palletcast.data.simulate import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "data.yaml"


@pytest.fixture(scope="session")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory, cfg):
    """Full-size generate + validate run, shared by all tests."""
    d = tmp_path_factory.mktemp("data")
    make_dataset.main(["--config", str(CONFIG), "--out", str(d)])
    summary = validate.run(d / "raw", d / "validated", d / "reports", cfg)
    reports = {p.stem: json.loads(p.read_text()) for p in (d / "reports").glob("*.json")}
    return {
        "dir": d,
        "summary": summary,
        "reports": reports,
        "manifest": json.loads((d / "sim_truth" / "fault_manifest.json").read_text()),
        "truth": pd.read_parquet(d / "sim_truth" / "shipments_clean.parquet"),
        "raw": pd.concat(
            [pd.read_parquet(p) for p in (d / "raw" / "shipments").glob("*.parquet")],
            ignore_index=True,
        ),
        "validated": pd.read_parquet(d / "validated" / "shipments.parquet"),
        "quarantine": pd.read_parquet(d / "validated" / "quarantine.parquet"),
    }
