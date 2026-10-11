"""Generate the simulated raw feed: master data, faulty shipment batches, and the ground truth.

uv run python -m palletcast.data.make_dataset --config configs/data.yaml --out data
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from palletcast.data.faults import inject_faults, split_batches
from palletcast.data.simulate import load_config, simulate


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out", default="data")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    sim = simulate(cfg)
    raw, manifest = inject_faults(cfg, sim.shipments, sim.sites, sim.customers, cfg["seed"] + 1)

    out = Path(a.out)
    raw_dir, truth_dir = out / "raw", out / "sim_truth"
    for d in (raw_dir, truth_dir):
        if d.exists():
            shutil.rmtree(d)
    (raw_dir / "master").mkdir(parents=True)
    (raw_dir / "shipments").mkdir(parents=True)
    truth_dir.mkdir(parents=True)

    sim.sites.to_parquet(raw_dir / "master" / "sites.parquet", index=False)
    sim.customers.to_parquet(raw_dir / "master" / "customers.parquet", index=False)
    sim.promo_calendar.to_parquet(raw_dir / "master" / "promo_calendar.parquet", index=False)
    sim.holidays.to_parquet(raw_dir / "master" / "holidays.parquet", index=False)
    for name, b in split_batches(raw, cfg["history_cutoff"]).items():
        b.to_parquet(raw_dir / "shipments" / f"{name}.parquet", index=False)

    # Ground truth is for evaluating the pipeline only; nothing downstream may read it.
    sim.shipments.to_parquet(truth_dir / "shipments_clean.parquet", index=False)
    sim.daily_truth.to_parquet(truth_dir / "daily_truth.parquet", index=False)
    sim.promos_all.to_parquet(truth_dir / "promos_all.parquet", index=False)
    (truth_dir / "fault_manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"clean shipments: {len(sim.shipments):,}  raw rows: {len(raw):,}")
    print("faults:", json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
