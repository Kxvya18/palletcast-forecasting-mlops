"""Train a model, register it in MLflow with its lineage, and run the promotion gate.

Run date R = the morning a (monthly) training run happens.
  * training labels:  target_date < R - label_settle_days        (final at R)
  * gate holdout:     target_date in [H0, H1), H1 = R - settle, H0 = H1 - holdout_days
The gate trains the candidate a second time without the holdout (labels < H0), scores it
and the current production model on the holdout, and promotes the full candidate only if
it is not worse by more than the allowed margins.

    uv run python -m palletcast.registry.registry bootstrap --run-date 2025-12-01
    uv run python -m palletcast.registry.registry gate --run-date 2026-01-01
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient

from palletcast.data.validate import frame_hash
from palletcast.models import common
from palletcast.registry.forecaster import Forecaster

DAY = pd.Timedelta(days=1)
CFG = yaml.safe_load(open(common.ROOT / "configs" / "registry.yaml"))
DECISIONS = common.ROOT / "reports" / "s4" / "decisions"


def setup_mlflow():
    mlflow.set_tracking_uri(f"sqlite:///{common.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("s4_registry")
    return MlflowClient()


def git_commit():
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=common.ROOT
    )
    return out.stdout.strip() or "unknown"


def load_features(features_dir=None):
    d = features_dir or common.ROOT / "data" / "features"
    return pd.read_parquet(d / "site.parquet"), pd.read_parquet(d / "customer.parquet")


def holdout_window(run_date):
    h1 = run_date - CFG["label_settle_days"] * DAY
    return h1 - CFG["holdout_days"] * DAY, h1


def score_on(model, site, cust, start, end):
    """Score a model on all forecasts whose target falls in [start, end)."""
    s = site[(site["target_date"] >= start) & (site["target_date"] < end) & site["target"].notna()]
    c = cust[(cust["target_date"] >= start) & (cust["target_date"] < end)]
    pred = model.predict_sites(s, c)
    m = common.score(pred, s, {"peak_quantile": 0.9})
    return {k: m[k] for k in ("wape", "bias", "wape_peak", "n")}


def train_and_register(key, site, cust, run_date, alias=None, extra_tags=None):
    """Train model `key` on everything settled at run_date and register a new version."""
    client = setup_mlflow()
    cutoff = run_date - CFG["label_settle_days"] * DAY
    t = time.time()
    model = Forecaster(CFG["models"][key]).fit(site, cust, cutoff)
    with mlflow.start_run(run_name=f"{key}-{run_date.date()}") as run:
        tags = {
            "model_key": key,
            "run_date": str(run_date.date()),
            "train_cutoff": str(cutoff.date()),
            "git_commit": git_commit(),
            "features_hash_site": frame_hash(site),
            "features_hash_customer": frame_hash(cust),
            **(extra_tags or {}),
        }
        mlflow.set_tags(tags)
        mlflow.log_params({"members": json.dumps(CFG["models"][key])})
        mlflow.log_metric("train_seconds", time.time() - t)
        info = mlflow.pyfunc.log_model(
            name="model", python_model=model, registered_model_name=CFG["model_name"]
        )
    version = str(info.registered_model_version)
    for k, v in tags.items():
        client.set_model_version_tag(CFG["model_name"], version, k, v)
    if alias:
        client.set_registered_model_alias(CFG["model_name"], alias, version)
    print(f"registered {CFG['model_name']} v{version} ({key}) alias={alias} run={run.info.run_id}")
    return version, model


def load_alias(alias):
    setup_mlflow()
    try:
        m = mlflow.pyfunc.load_model(f"models:/{CFG['model_name']}@{alias}")
    except Exception:  # noqa: BLE001 - alias not set yet
        return None, None
    version = MlflowClient().get_model_version_by_alias(CFG["model_name"], alias).version
    return m.unwrap_python_model(), version


def gate(site, cust, run_date, key="production_candidate", candidate_eval=None):
    """Decide whether a fresh `key` model replaces production. Returns the decision dict.

    candidate_eval lets the S4 experiment inject a deliberately broken candidate."""
    h0, h1 = holdout_window(run_date)
    prod, prod_version = load_alias("production")
    if candidate_eval is None:
        candidate_eval = Forecaster(CFG["models"][key]).fit(site, cust, h0)
    cand = score_on(candidate_eval, site, cust, h0, h1)
    decision = {
        "run_date": str(run_date.date()),
        "holdout": [str(h0.date()), str(h1.date())],
        "candidate": cand,
        "production_version": prod_version,
    }
    if prod is None:
        decision.update(promote=True, reason="no production model yet")
    else:
        p = score_on(prod, site, cust, h0, h1)
        decision.update(production=p, **decide(cand, p, CFG["gate"]))
    return decision


def decide(cand, prod, g):
    """The gate rule: not worse than production by more than the allowed margins."""
    wape_ok = cand["wape"] <= prod["wape"] + g["max_wape_increase"]
    bias_ok = cand["bias"] >= prod["bias"] - g["max_bias_worsening"]
    return {
        "wape_ok": bool(wape_ok),
        "bias_ok": bool(bias_ok),
        "promote": bool(wape_ok and bias_ok),
        "reason": "passes gate" if wape_ok and bias_ok else "candidate worse than production",
    }


def save_decision(decision, name=None):
    DECISIONS.mkdir(parents=True, exist_ok=True)
    path = DECISIONS / f"{name or decision['run_date']}.json"
    path.write_text(json.dumps(decision, indent=1))
    print(json.dumps({k: decision[k] for k in ("run_date", "promote", "reason")}))
    return path


def run_monthly(site, cust, run_date):
    """One monthly cycle: gate a fresh candidate; if it passes, register it as production."""
    decision = gate(site, cust, run_date)
    if decision["promote"]:
        version, _ = train_and_register(
            "production_candidate",
            site,
            cust,
            run_date,
            alias="production",
            extra_tags={"gate": "promoted"},
        )
        decision["new_version"] = version
    save_decision(decision)
    return decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["bootstrap", "gate", "challengers"])
    ap.add_argument("--run-date", required=True)
    a = ap.parse_args()
    run_date = pd.Timestamp(a.run_date)
    site, cust = load_features()
    if a.action == "bootstrap":  # first production model, no gate to pass
        train_and_register(
            "production_candidate",
            site,
            cust,
            run_date,
            alias="production",
            extra_tags={"gate": "bootstrap"},
        )
    elif a.action == "challengers":
        train_and_register("challenger_ensemble", site, cust, run_date, alias="challenger-ensemble")
        train_and_register("challenger_ratio", site, cust, run_date, alias="challenger-ratio")
    else:
        run_monthly(site, cust, run_date)


if __name__ == "__main__":
    main()
