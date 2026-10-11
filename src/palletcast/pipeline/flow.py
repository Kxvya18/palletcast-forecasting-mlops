"""S5: the monthly pipeline as a Prefect flow.

    new monthly batch -> validate (S1) -> build features (S2) -> gate & register (S4)

Each month's run works in its own folder runs/<run_date>/ so runs never overwrite each
other or the DVC-tracked data. If validation hard-fails, the flow stops there and no model
is trained on bad data.

    uv run python -m palletcast.pipeline.flow --months 2026-01 2026-02 2026-03
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from prefect import flow, get_run_logger, task

from palletcast.data import validate
from palletcast.data.simulate import load_config
from palletcast.features import build
from palletcast.models.common import ROOT
from palletcast.registry import registry


@task
def check_new_batch(raw_dir: Path, month: str) -> None:
    batch = raw_dir / "shipments" / f"live-{month}.parquet"
    if not batch.exists():
        raise FileNotFoundError(f"no data batch for {month}: {batch}")
    get_run_logger().info(f"new batch found: {batch.name}")


@task
def validate_data(raw_dir: Path, run_dir: Path, month: str) -> dict:
    summary = validate.run(
        raw_dir,
        run_dir / "validated",
        run_dir / "validation_reports",
        load_config(ROOT / "configs" / "data.yaml"),
        upto=month,
    )
    if summary["status"] == "fail":
        raise RuntimeError(
            f"validation HARD FAIL in batch {summary['failed_batch']}; stopping before any training"
        )
    get_run_logger().info(f"validated rows: {summary['rows_validated']:,}")
    return summary


@task(retries=1)
def build_features(run_dir: Path) -> None:
    build.main(["--validated", str(run_dir / "validated"), "--out", str(run_dir / "features")])


@task(retries=1)  # one retry: the MLflow database can be briefly locked
def gate_and_register(run_dir: Path, run_date: pd.Timestamp) -> dict:
    site, cust = registry.load_features(run_dir / "features")
    decision = registry.run_monthly(site, cust, run_date)
    get_run_logger().info(f"gate decision: promote={decision['promote']} ({decision['reason']})")
    return decision


@flow(name="monthly-forecast-pipeline")
def monthly_run(month: str, raw_dir: Path = ROOT / "data" / "raw"):
    """Process the data batch for `month` (e.g. "2026-01") on the 1st of the next month."""
    run_date = (pd.Period(month, "M") + 1).start_time
    run_dir = ROOT / "runs" / str(run_date.date())
    run_dir.mkdir(parents=True, exist_ok=True)

    check_new_batch(raw_dir, month)
    summary = validate_data(raw_dir, run_dir, month)
    build_features(run_dir)
    decision = gate_and_register(run_dir, run_date)

    result = {
        "month": month,
        "run_date": str(run_date.date()),
        "rows_validated": summary["rows_validated"],
        "decision": decision,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(result, indent=1, default=str))
    return result


@flow(name="replay-months")
def replay(months: list[str]):
    """Run the monthly pipeline for several months in order (a replay of the past)."""
    return [monthly_run(m) for m in months]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", required=True)
    replay(ap.parse_args().months)
