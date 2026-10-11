# S1 · Data: ingestion, validation, versioning

**Business problem (whole project):** forecast each site's outbound pallets per day for the
next 14 days, so a automated cold store can plan dock slots, trucks and shifts.

**S1's job:** deliver a raw feed that behaves like a real WMS export, and a validated,
versioned dataset that downstream stages can trust, plus a precise record of what was
lost or flagged.

> **All data is simulated.** No real company data is used. Every number below is an assumption
> chosen to be plausible, not calibrated to a real site. The parameters live in
> `configs/data.yaml`.

---

## 1. Why simulate

Real data isn't available. A simulator also gives us something real data can't: **ground
truth** (`data/sim_truth/`). Every validation rule, and later every drift detector, can then
be scored with precision and recall instead of judged by eye. The truth folder is for
auditing only. No pipeline stage may read it.

## 2. Generative model

For customer *c* on day *t*:

```
mu[c,t] = base_c · (1+g_c)^years · weekly_site(t) · annual_segment(doy) · promo_c(t)
          · lifecycle_c(t) · level_shift_c(t) · year_end_lull(t) · exp(AR1_c(t))
demand[c,t] ~ NegBin(mean = mu, dispersion r = 20)
```

then site-level physics:

1. **Holidays:** the site is closed. 60% of the lost volume is pre-built over the 3 previous
   working days (weights .5/.3/.2), 30% is caught up over the next 2 (.6/.4), and 10% is lost.
2. **Capacity:** each site has a hard daily outbound limit (crane and dock throughput). Demand
   above it becomes backlog for the next working day, cut back proportionally per customer.
   Without this, the first draft put up to 2.8× normal volume on 24 December, which no
   automated warehouse could physically ship.
3. **Shipments:** daily pallets are split into trucks, about 75% full loads (33 pallets, EU
   standard) and the rest partial. Each shipment gets an order timestamp 1–5 days earlier and
   an ingestion timestamp (hours later; 3% arrive 2–10 days late).

| Component | Assumption | Why |
|---|---|---|
| Sites | 4 (NL, GB, DE, ES), 9–11 customers each | multi-country → different holiday calendars |
| Weekly | Mon catch-up ≈1.1×, Sat 0.2–0.4×, Sun closed (GB 0.1×) | DE Sunday HGV ban; typical DC rhythm |
| Annual | segment-specific bumps (ice cream summer, bakery/seafood/ready meals Christmas) | outbound peaks *ahead of* consumption |
| Ice-cream amplitude | varies by year (sd 15%) | weather; deliberately **unobserved** |
| Promotions | ~6/customer/year, 7–21 days, uplift by type (price cut 1.5–2×, multibuy 1.3–1.6×, display 1.15–1.35×) | |
| Promo timing | outbound uplift starts 5–10 days **before** in-store start; 0.85× dip afterwards | retailer DCs stocked ahead |
| Promo calendar | announced 4–8 weeks ahead; 10% never announced | gives S2 a real leakage trap and missing information |
| Noise | AR(1) on log-demand (φ = .6, σ = .1) + negative binomial | persistence + overdispersed counts |
| Growth | ~5%/yr per customer | |

### Planted business events (for S7/S8)

| When | Site | Event | Visible in master data? |
|---|---|---|---|
| 2024-05 | S01 | new frozen-veg customer, 8-week ramp | yes (`onboard_date`) |
| 2025-02 | S03 | new ready-meals customer | yes |
| 2025-06 | S04 | customer churns, 4-week ramp-down | yes (`churn_date`) |
| 2026-03 | S02 | large new ice-cream customer (live period) | yes |
| 2026-04 | S02 | Saturday shift added (Sat 0.40× → 0.85×) | only via `operating_days` |
| 2026-06 | S04 | largest customer loses a contract (−35%) | **no**; must be detected from data |

## 3. The raw feed

- `data/raw/master/`: sites, customers, promo_calendar (announced only), holidays.
- `data/raw/shipments/`: `history.parquet` (everything **ingested** before 2026-01-01) and
  `live-YYYY-MM.parquet` monthly batches, partitioned by **ingestion** time like a real
  feed. Late rows and corrections therefore land in later batches than their ship date.
- Snapshot time is 2026-09-30 end of day. Rows ingested after that "have not arrived yet".

### Planted data-quality faults

| Fault | Rate | What it mimics |
|---|---|---|
| Exact duplicate rows | 0.2% | re-sent messages |
| Corrections | 0.3% | wrong pallet count first; corrected row arrives 1–7 days later |
| Non-positive pallets | 0.05% | keying errors, reversals |
| Unknown customer ID | 0.05% | master-data lag, typos |
| Customer at wrong site | 0.03% | mis-mapped account |
| Order after ship date | 0.03% | clock / timezone errors |
| Null key | 0.02% | broken export |
| Full outage | ~0.75/site/yr, 1–3 days | interface down, data never arrives |
| Partial outage | ~0.75/site/yr, 10–45% of rows arrive | interface partially down |

Every fault is recorded in `data/sim_truth/fault_manifest.json`, which the tests use as the
answer key.

## 4. Validation policy

**Change from the original plan.** The plan said "hard fail on bad rows". Production feeds
are never perfectly clean, so failing the whole batch for 0.1% bad rows would stop
forecasting every day. The policy is:

| Level | Condition | Action |
|---|---|---|
| Structural | missing/extra columns, empty batch, invalid master data | **HARD FAIL**, nothing merged |
| Row | unparseable value, null key, pallets ≤ 0 or > 33 (one truck), unknown customer, wrong site–customer pair, order after ship, ingested before ship | **QUARANTINE** row with reason |
| Batch | quarantine rate > 1% | **HARD FAIL** (the feed itself is broken) |
| Duplicates | exact duplicate | drop |
| | same `shipment_id`, later `ingested_at` | latest wins (correction) |
| | same `shipment_id` + same `ingested_at`, different content | quarantine both |
| Batch health | expected operating day with no rows | **WARN** `missing_operating_day` |
| | site-day volume abnormal vs its own history (§5) | **WARN** `volume_low` / `volume_high` |
| | > 6% of rows arrive > 2 days late | **WARN** `late_arrivals` |
| | > 2 expected operating days with no data at batch end | **WARN** `stale_site` |

Two implementation details matter:
- **Type coercion is a separate step before pandera.** pandera's built-in coercion
  fails the *whole column* if one value can't be parsed, which turned a single `"twelve"` into
  a batch hard-fail. The validator parses each typed column itself and quarantines only the
  unparseable rows.
- Real business changes (onboarding, churn, the S04 contract loss) are **not** data errors,
  and validation must not flag them. Detecting them is S7's job.

## 5. Calibrating the volume check

The volume check exists to catch **data** problems (a partial feed, a unit error), not real
demand swings such as promotions, peaks or capacity backlog.

- **First attempt:** a robust z-score against the same weekday over the previous 8 weeks.
  It raised 184 warnings on history, mostly holiday pre-builds and promotions. That's
  alarm fatigue.
- **Second attempt:** a ratio rule with thresholds read off the project seed's fault-free
  distribution. It was perfect on that seed but **overfit**: on an unseen seed it raised
  about 50 false alarms, almost all on low-volume Saturdays, where one truck more or less
  is a 3–5× swing.
- **Final rule:** alarm only if the ratio is out of range **and** the absolute gap is
  material, measured in units of the site's typical Tue–Thu day. Thresholds were chosen on 5
  calibration seeds and are reported on 5 held-out seeds
  (`src/palletcast/data/calibrate_volume_check.py`, results in `reports/calibration/`).

Budget: at most 0.5 false alarms per site per year on each side. Positives are planted
partial outages; every other scored site-day is a negative.

| Chosen | Value |
|---|---|
| low ratio | < 0.50× same-weekday median |
| high ratio | > 3.0× (> 10× within 4 days before / 3 after a holiday) |
| materiality | gap ≥ 5% of the site's typical Tue–Thu day |

| Seeds | Partial outages | Recall | Weekday / weekend | False low alarms / site-yr | High alarms / site-yr |
|---|---|---|---|---|---|
| Calibration (5) | 53 | 0.87 | 0.88 / 0.85 | 0.48 | 0.45 |
| **Held-out test (5)** | 55 | **1.00** | 1.00 / 1.00 | **0.08** | **0.22** |

The calibration-to-test difference is sampling noise on about 50 positives. Pooled recall,
about **0.94**, is the honest figure. The low grid (`reports/calibration/low_grid.csv`) shows
the trade-off: catching every partial outage would cost up to 9 false alarms per site-year.

**Blind spot:** a partial loss on an unusually busy day. On the project seed, S04 on
2025-08-04 lost 62% of a promo-peak day (1,540 → 602 pallets). The result was 0.54× its
reference, which looks like a normal quiet Monday. A site-day volume check cannot
separate these cases.

## 6. Versioning (DVC)

`dvc.yaml` defines two stages: `generate` → `validate`. Raw, truth and validated data are
DVC-tracked, and the validation reports are git-tracked (small JSON). The default remote is
a local folder configured in `.dvc/config.local` (untracked, machine-specific). Moving to
S3/GCS is a one-line `dvc remote add`. Re-running `dvc repro --force` from scratch
reproduces **byte-identical** outputs: `dvc.lock` is unchanged.

```bash
uv sync
uv run dvc repro          # generate + validate
uv run pytest -q          # full-size run checked against the fault manifest
```

## 7. Results on the project seed

| | Value |
|---|---|
| True shipments (simulated) | 251,146 |
| Raw rows delivered (10 batches) | 250,419 |
| Validated shipments | 248,729 |
| Quarantined | 449, every one a planted fault: 125 unknown customer, 124 non-positive pallets, 75 wrong site, 75 order after ship, 50 null key (the 450th planted bad row had not arrived by snapshot) |
| Exact duplicates dropped | 498 / 498 planted |
| Corrections applied | 748 planted; validated value = truth for every correction that arrived |
| Full outages | 33 / 33 site-days reported as `missing_operating_day`, no false reports |
| Partial outages | 8 / 9 reported as `volume_low`, 0 false (miss explained in §5) |
| `volume_high` | 1: S02 2025-11-15, a genuine Saturday at 3.1× (two overlapping promotions at the largest customer pushed Thu/Fri to capacity, and the backlog shipped Saturday) |
| Hard fails | 0 |
| Volume lost vs truth | 63,891 pallets (0.96%): 50,782 from outages, the rest quarantined rows and unarrived corrections |
| Validated-data hash | `9938c205acecbcfb` (`reports/validation/summary.json`) |

All of the above is asserted in `tests/` (28 tests, full-size data, no mocks). That includes
an end-to-end reconciliation: every true shipment is either in the validated set with its
true value, or explained by a planted fault.

## 8. What S2 must handle (from the EDA notebook)

1. **Outage days are missing data, not zero demand.** They are flagged by validation and
   must be masked or imputed before computing lags.
2. **Weekly cycle dominates** (ACF ≈ .75 at every multiple of 7). Lag-7/14/28 and
   same-weekday rolling statistics; seasonal-naive is the baseline to beat in S3.
3. **Holiday pre-build and catch-up** need days-to and days-since holiday features, not
   just an is-holiday flag.
4. **Promotions lead outbound by about 10 days.** Promo features must be aligned to the
   shipping window, and only promos with `announce_date` ≤ forecast origin may be used
   (leakage).
5. **Seasonality is segment-specific.** Site totals mix segments, so customer-mix features
   or customer-level modelling with aggregation may help.
6. **Overdispersed counts.** Use scale-free metrics (MASE/WAPE) and consider a
   Poisson/Tweedie objective.
7. **Capacity clipping:** observed volume is *shipped*, not *demand*. Peak days are censored
   at capacity. Planning needs demand, so S3 should note where the target is censored.

## 9. Known limitations

- Parameters are plausible guesses, not fitted to a real site. Conclusions are about the
  pipeline's behaviour, not about any real operator's actual demand.
- Partial loss on low-volume days (Saturdays, Sundays) is mostly undetectable at site-day
  granularity; see the weekend recall in §5. Customer-level checks could help but would
  flag real promo swings. Not pursued in S1.
- A whole-day replay with **new** shipment IDs (a doubled feed) is only detectable if it
  exceeds the high-ratio threshold. No row-level rule can catch it.
- Late rows for the last days before the snapshot are, by construction, still missing.
  The most recent days are always incomplete, and S2/S6 must account for this
  (e.g. don't treat yesterday as final).
