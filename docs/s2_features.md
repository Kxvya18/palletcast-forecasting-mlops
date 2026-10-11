# S2 · Feature pipeline

**Job of S2:** turn validated shipments into a table where every row is one forecasting
question, and every column is a clue the warehouse genuinely had when it asked it.

> One row = "standing at origin day **O** at 06:00, how many pallets will this series ship
> on target day **T = O + h**?", for h = 1…14.

Decisions agreed with the user:

| Decision | Choice | Why |
|---|---|---|
| Grain | **both** site-level and customer-level tables | Promotions, onboarding and churn are customer events. S3 measures which grain forecasts sites better. |
| Forecast frequency | **rolling, every day** | how dock planning works; more training rows |
| Multi-day strategy | **direct**: one model, `h` is a feature | no compounding errors, one model to run |
| Outage days | **masked (NaN)** | they're missing data, not zero demand. S3 tests imputation as an alternative. |
| Target | **shipped pallets** | what we observe. Capped at capacity on peak days; flagged with `at_capacity`. |

---

## 1. Point-in-time correctness: the rule behind everything

**Data leakage** = a feature contains information that didn't exist at the origin. The model
then looks great in backtests and fails live. S1 planted three leakage traps, and S2 handles
each one:

| Trap | Handling |
|---|---|
| Late rows (3% arrive 2–10 days late) | volumes rebuilt **as of 06:00 on the origin** from `ingested_at` |
| Corrections (wrong value first, fixed later) | the as-of view shows the *wrong* value until the fix arrives |
| Promotions announced late | a promotion counts only if `announce_date ≤ origin` |
| New / leaving customers | known only 28 days before onboarding / churn (`master_data_notice_days`) |
| Validation flags | a partial-outage flag for day *d* exists from origin *d + 2* |

To support this, S1's validator now also saves **every valid version of every record with
its arrival time** (`shipment_versions.parquet`) and the outage flags (`data_flags.parquet`).

**How the as-of view works** (`features/asof.py`): a record version adds its pallets when it
arrives and removes them when a correction replaces it. Summing those additions and removals
up to the origin's 06:00 cutoff gives exactly what the database showed that morning.

**Proof there is no leakage** (`tests/test_features.py`):
1. For 16 origins (10 hand-picked hard cases + 6 random), rebuild the features from a copy of
   the data where **everything that arrived after the origin is deleted**. Every feature must
   be identical to the full build.
2. Multiply every pallet count that arrived after the origin by 7 and rewrite every
   later-announced promotion. No feature at the origin may change; only the labels do.

## 2. Experiment: how complete is recent data?

At 06:00, how much of a day's final volume is already in the database?

| Mornings after shipping | Median day | Worst 1% | Days already exact |
|---|---|---|---|
| 1 (yesterday) | 97.2% | 86% | 21% |
| 3 | 97.7% | 87% | 27% |
| 7 | 100% | 91% | 50% |
| 14 | 100% | 100% | 99.98% |
| 21 | 100% | 100% | 100% |

**Decision:** use recent days, but always as they looked that morning. Training and live
forecasting then see the same incomplete numbers. Building features from final data instead
would make yesterday look ~3% higher in training than it ever looks live (notebook §5:
median as-of/final = 0.972, 5th percentile 0.912). That's **training–serving skew**. S3
will measure its cost.

The 21-day window (`asof_window_days`) is verified: after 21 days every day is final
(asserted in tests).

## 3. Experiment: promotion lead window

Estimated **only on promotions starting 2023-03 … 2024-08** (321 promotions), so it can't
peek at the periods S3 will test on. For each promotion: customer volume around the in-store
start ÷ the same weekday 5–8 weeks earlier, with a bootstrap 90% confidence interval.

- Outbound is clearly lifted (whole CI ≥ 1.1×) from **8 days before to 7 days after** the
  in-store start, peaking ~1.45× around 2–4 days before.
- A post-promotion dip (~0.9×) appears 14–16 days after the start.

These set `promo_lift_offsets: [-7, 8]` and `promo_post_offsets: [-16, -8]` (written as
*start − target day*). The windows come from data, not from the simulator's settings.

## 4. Feature dictionary

All "volume" features use the as-of view and are NaN on masked days.
**sw** = same weekday as the target day.

### Both tables
| Feature | Meaning |
|---|---|
| `h` | horizon: days ahead (1–14) |
| `dow`, `month`, `doy`, `week` | calendar of the target day |
| `sw_lag1` … `sw_lag4` | volume on the same weekday, the 1st–4th most recent weeks already known |
| `sw_mean4`, `sw_mean8`, `sw_median8` | mean / median of the last 4 or 8 known same-weekday values |
| `last_1` | yesterday's volume, as known this morning |
| `mean_7`, `mean_28` | average daily volume over the last 7 / 28 days |
| `trend_28` | last 28 days ÷ the 28 days before (growth or decline) |
| `n_masked_28` | how many of the last 28 days were hidden as outages: tells the model when its clues are thin |
| `days_to_holiday`, `days_since_holiday` | distance to the nearest public holiday (capped at 15): captures pre-build and catch-up |
| `at_capacity` | *label-side flag, not a feature*: the target day hit site capacity (shipped < demand) |
| `target` | final shipped pallets on the target day; NaN on outage / partial-outage days |
| `period` | `history` (target ≤ 2025-12-31, for S3) or `live` (2026, for S7/S8) |

### Site table only
| Feature | Meaning |
|---|---|
| `n_active_customers` | customers known and active on the target day |
| `n_new_customers_90` | of those, onboarded within the last 90 days (still ramping up) |
| `promo_lift_share_{display,multibuy,price_cut}` | share of site volume (last 28 days) from customers with that promotion type in its lift window |
| `promo_post_share` | share of site volume from customers in a post-promotion dip |
| `capacity` | site's daily outbound limit |

### Customer table only
| Feature | Meaning |
|---|---|
| `segment` | product type (ice cream, fries, …) |
| `site_sw_mean4`, `site_mean_28`, `site_trend_28` | the site's level and trend, as context |
| `share_28` | this customer's share of site volume, last 28 days |
| `promo_in_lift`, `promo_in_post` | number of announced promotions in the lift / dip window |
| `promo_lift_type` | strongest promotion type in the lift window (0 none, 1 display, 2 multibuy, 3 price cut) |
| `promo_lift_{display,multibuy,price_cut}` | 1 if a promotion of that type is in the lift window |
| `promo_next_start` | days from the target day to the nearest announced promotion start (−7…21) |
| `days_since_onboard`, `days_to_churn` | customer lifecycle, only when known |

## 5. Which rows exist

- Origins: every day from 2023-03-01 (after an 8-week warm-up) to the day before the data
  ends. Targets must fall within the data.
- **Only target days the site is expected to work.** A Sunday at a closed site has a known
  answer (0). Including it would make every accuracy number look better than it is.
- Customer rows only for customers known and active on the target day.
- Result: **63,515 site rows** and **631,797 customer rows**. Customer targets sum
  *exactly* to the site target on every row where both exist (tested).

## 6. Sanity results (notebook `02_s2_features.ipynb`)

- Same-weekday features carry most of the signal (Spearman ρ ≈ 0.85–0.90 at every
  horizon). Yesterday’s volume is only strongly informative when it’s the same weekday as the
  target (h = 6, 13: ρ ≈ 0.88), or one day off (h = 5, 12: ρ ≈ 0.6).
- Seasonal-naive check: the same-weekday median of 8 weeks, used directly as a forecast,
  has a **WAPE of 14–15%**; last week alone 16–18%. That's the bar S3's model must beat.
  (WAPE = total absolute error ÷ total actual volume.)

## 7. How to run

```bash
uv run dvc repro                 # generate → validate → promo_lead → features
uv run pytest -q tests/test_features.py
```

The full feature build takes ~5 minutes. It's written as plain pandas, one origin at a time,
so it's easy to read. A faster array-based version produced identical output but was much
harder to follow, and was replaced on purpose.

## 8. Known limitations

- The 28-day notice for onboarding / churn is an assumption, not data.
- A partial-outage flag is assumed available two days later. A real daily validation run
  would need to match this.
- Promotion windows are estimated on 2023-03 … 2024-08. If behaviour changes, they should be
  re-estimated. That's a candidate for S8 (retraining).
- `at_capacity` marks censored labels, but S2 doesn't try to reconstruct uncapped demand
  (agreed: predict shipped pallets first).
