"""S4 experiment: does the gate accept a good candidate and reject a broken one?

The broken candidate is trained on shuffled answers, as if that month's data were corrupt.
Nothing is registered here; the decisions are saved to reports/s4/decisions/.

    uv run python -m palletcast.registry.gate_experiment --run-date 2026-01-01
"""

import argparse

import numpy as np
import pandas as pd

from palletcast.registry import registry
from palletcast.registry.forecaster import Forecaster


def broken_candidate(site, cust, cutoff):
    """Same model type, but the training answers are shuffled (= garbage data)."""
    rng = np.random.default_rng(0)
    bad = cust.copy()
    bad["target"] = rng.permutation(bad["target"].to_numpy())
    return Forecaster(registry.CFG["models"]["production_candidate"]).fit(site, bad, cutoff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date", default="2026-01-01")
    run_date = pd.Timestamp(ap.parse_args().run_date)
    site, cust = registry.load_features()
    h0, _ = registry.holdout_window(run_date)

    good = registry.gate(site, cust, run_date)
    registry.save_decision(good, f"experiment_{run_date.date()}_good")

    bad = registry.gate(site, cust, run_date, candidate_eval=broken_candidate(site, cust, h0))
    registry.save_decision(bad, f"experiment_{run_date.date()}_broken")

    for name, d in (("good candidate", good), ("broken candidate", bad)):
        print(
            f"{name:17s} WAPE {d['candidate']['wape']:.3f}  bias {d['candidate']['bias']:+.3f}"
            f"  | production WAPE {d['production']['wape']:.3f}  -> "
            f"{'PROMOTE' if d['promote'] else 'REJECT'} ({d['reason']})"
        )


if __name__ == "__main__":
    main()
