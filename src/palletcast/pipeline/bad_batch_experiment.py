"""S5 experiment: a corrupted monthly batch must stop the pipeline before any training.

Copies the raw data to a scratch folder, corrupts the April 2026 batch (5% of pallet counts
set to 0, above the 1% quarantine limit from S1) and runs the monthly flow on it.
Expected: the validate step fails, later steps never run, production is unchanged.

    uv run python -m palletcast.pipeline.bad_batch_experiment
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
from mlflow import MlflowClient

from palletcast.models.common import ROOT
from palletcast.pipeline.flow import monthly_run
from palletcast.registry import registry


def production_version():
    registry.setup_mlflow()
    return (
        MlflowClient().get_model_version_by_alias(registry.CFG["model_name"], "production").version
    )


def main():
    before = production_version()
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        shutil.copytree(ROOT / "data" / "raw", raw)
        batch = raw / "shipments" / "live-2026-04.parquet"
        b = pd.read_parquet(batch)
        bad_rows = b.sample(frac=0.05, random_state=0).index
        b.loc[bad_rows, "pallets"] = 0
        b.to_parquet(batch, index=False)

        state = monthly_run("2026-04", raw_dir=raw, return_state=True)
    after = production_version()
    print(f"flow state: {state.type.value} | production before v{before}, after v{after}")
    assert state.is_failed() and before == after
    print("OK: corrupted batch stopped the pipeline; production model unchanged")


if __name__ == "__main__":
    main()
