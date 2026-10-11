# S4–S7 · Registry, orchestration, serving, monitoring

## S4 · Model registry and promotion gate (MLflow)

**What:** every deployable model is saved in MLflow as `outbound_forecaster` with a version
number and **lineage tags**: git commit, a fingerprint of the exact feature data, run date and
training cutoff. Aliases mark what's in use: `production`, `challenger-ensemble`,
`challenger-ratio`. Rolling back = moving the `production` alias to an older version.

**Model wrapper** (`registry/forecaster.py`): a model is a list of members
(level: site/customer, algorithm, ratio target on/off, weight). The production model and both
S3 challengers are the same class, so S6 loads any of them the same way.

**Promotion gate** (`registry/registry.py`, `configs/registry.yaml`): at each monthly run the
candidate is retrained *without* the latest 4 settled weeks and scored there against the
current production model. It's promoted unless:
- its WAPE is more than **0.3 points worse**, or
- its bias is more than **1 point more negative** (deeper under-forecasting).

It's a "not worse" rule, not "must be better": S3/E5 showed fresher models forecast better,
so healthy monthly updates shouldn't be blocked.

**Experiment** (1 Jan 2026):

| Candidate | Holdout WAPE | Bias | Decision |
|---|---|---|---|
| normal | 13.8% | +0.5% | promote (production was 13.8%) |
| trained on shuffled answers (corrupt data) | **38.3%** | −12.3% | **reject** |

**Honest limitation:** with monthly runs and a 4-week holdout, the "candidate without the
holdout" is almost the same model as last month's production. In steady operation the gate is
therefore a **safety check against broken training or code changes**, not evidence that the new
model is better. Bad *data* is stopped earlier, by validation (S5 experiment). Whether
retraining helps is measured on live data (S7/S8).

Registry history: v1 bootstrap (1 Dec 2025), v2 (1 Jan 2026), v3/v4 challengers (1 Jan 2026),
v5–v7 from the S5 replay (1 Feb, 1 Mar, 1 Apr 2026).

## S5 · Orchestration (Prefect)

**What:** one flow per month (`pipeline/flow.py`):
`check batch → validate (S1) → build features (S2) → gate & register (S4) → summary`.
Each run works in its own folder `runs/<run date>/`, so it never touches the DVC-tracked data.
Prefect logs every task, retries the two steps that can fail transiently, and shows the
history in its dashboard (`prefect server start`, port 4200).

**Replay of January–March 2026:**

| Run date | Data month | Candidate WAPE | Production WAPE | Decision |
|---|---|---|---|---|
| 2026-02-01 | January | 10.1% | 10.4% | promoted → v5 |
| 2026-03-01 | February | 11.0% | 11.0% | promoted → v6 |
| 2026-04-01 | March | 11.3% | 11.4% | promoted → v7 |

**Corrupted-batch experiment** (`pipeline/bad_batch_experiment.py`): April's batch with 5% of
pallet counts set to 0 → validation HARD FAIL (limit 1%) → the flow stops after 2 tasks, no
training, **production stays v7**. It shows red in the dashboard.

Scheduling is a one-line Prefect deployment (`monthly_run.serve(cron="0 7 1 * *")`); runs were
started by hand here to replay the past.

## S6 · Serving (FastAPI + Docker)

**What:** `serving/app.py`:
- `GET /forecast?site=S02&origin=2026-07-23`: 14-day forecast made at 06:00 on that morning
  (default: the morning after the latest data)
- `GET /health`: model and data in use
- `POST /reload`: pick up a newly promoted production model without restarting

Features are built **with the same S2 code**, from data as known that morning.

**Key test** (`tests/test_serving.py`): on four mornings across 2026, the API's forecast equals
the offline pipeline's forecast for every site and horizon, to within rounding to 0.1 pallets.
So there's no training–serving skew in the code path.

**Docker:** `Dockerfile` installs only the **core** dependency group (no training stack). It
bakes in one exported production model (`serving.export_model` → `serving_model/`), and the
validated data is mounted at run time:

```bash
uv run python -m palletcast.serving.export_model
docker build -t palletcast-forecast:v7 .
docker run -p 8000:8000 -v "$PWD/data/validated:/app/data/validated:ro" palletcast-forecast:v7
curl "http://127.0.0.1:8000/forecast?site=S04"
```

## S7 · Monitoring (live error + Evidently drift)

**Setup:** production v7 (deployed 1 Apr 2026) is **frozen** for April–September 2026, to see
what happens without retraining. Per site and month (`monitoring/monitor.py`):
- **live error** (WAPE, bias) against what really shipped;
- **drift**: Evidently tests of 9 key columns vs the **same month one year earlier**, so
  normal seasonality isn't reported as drift. HTML reports in `reports/s7/`.

Alerts: WAPE > 16% (~1.5× the S3 test error), |bias| > 7%, or > 50% of columns drifted.

**Did it catch the planted 2026 events?** (weekly bias of the frozen model)

| Event | First week | Then |
|---|---|---|
| S02 new ice-cream customer + Saturday shift (Mar/Apr) | **−21%** (under-forecast) | shrinks to ~0 by mid-May as recent-history features catch up |
| S04 hidden contract loss (1 Jun, not in master data) | **+18%** (over-forecast) | +13–21% in most weeks through mid-July: **needs retraining (S8)** |

Both are caught **in the first week** by error/bias monitoring.

**Drift was much less useful:** frequent alerts with no forecast problem (e.g. S01 in July and
September, driven by year-on-year growth), and it **missed** S04's contract loss (3 of 9 columns
drifted). Conclusion: alert on **error and bias**; use drift reports to *investigate*.

**Pre-registered challenger test** (S3 → scored on Jan–Sep 2026, all trained for 1 Jan 2026):

| Model | WAPE 2026 | Bias |
|---|---|---|
| **customer-level LightGBM (production choice)** | **10.9%** | −1.1% |
| ratio-target challenger | 12.0% | −2.3% |
| ensemble challenger | 13.0% | −6.2% |

The model chosen in S3 for simplicity also wins on genuinely unseen data. The ensemble, which
the tuning-quarter rule slightly preferred, is last. One period, but it supports the choice.

## Limitations
- The registry lives in a local SQLite file (`mlflow.db`) and is not shared. A team would run
  an MLflow server with shared storage.
- The Docker image serves one fixed model version; promotion means rebuilding the image
  (or running without Docker and calling `/reload`).
- Alert thresholds are judgement calls tied to S3's error level, not tuned.
