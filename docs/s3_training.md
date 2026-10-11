# S3 · Training, backtesting, experiment tracking

**Job of S3:** compare model families fairly, pick the model to deploy, and record every
run so the choice can be checked and reproduced.

## 1. Rules of the contest (fixed before any results)

| Rule | Setting | Why |
|---|---|---|
| Untouched data | **2026 is never used** | reserved for S7/S8 as a truly unseen test |
| Test periods | the four quarters of **2025** | each quarter forecast as if we were there |
| Training data | targets ≥ 21 days before the quarter starts | those answers were final when the quarter began (S2 completeness experiment) |
| Tuning | **Q4 2024 only**, 10 random tries per tunable model | settings and choices are never made on test data |
| Seeds | 3 for every model with randomness | a "win" must not be luck |
| Forecasts scored | Monday and Thursday mornings, 14 days ahead, working days only | ~5,800 forecasts/year. Checked: daily scoring gives the **same ranking** (`reports/s3/subset_check.csv`). |
| Metric | **WAPE** = total abs. error ÷ total volume (headline); bias; peak-day WAPE (busiest 10%); WAPE by horizon | |

Every model writes the same prediction table and is scored by the same code
(`models/common.py`), so the comparison is like for like.

## 2. Models

| Family | Models | Input |
|---|---|---|
| Baselines | seasonal naive (same weekday last week), 8-week same-weekday median | S2 features |
| Classical | AutoARIMA, AutoETS, AutoTheta, MSTL (statsforecast), Prophet with country holidays | each site's own as-of history, **refit every forecast morning** |
| ML | Ridge, Random Forest, LightGBM, XGBoost, CatBoost | all S2 site features, retrained each quarter |
| Deep | N-HiTS, N-BEATS, LSTM (neuralforecast, CPU); calendar inputs `closed`, `pre_holiday` | site histories, retrained each quarter |
| Ensemble | mean of the 3 best on the tuning quarter | |

Practical constraints (documented, not hidden):
- **AutoARIMA** uses the last 365 days of history. Full history took ~80 s per forecast
  morning; a weekly-season ARIMA doesn't need more.
- **TFT** was dropped from this round. It took ~27 min per training run on CPU, so an equal
  budget would have taken ~10 h. Given the other deep models' results it's very unlikely to
  win. It can be added later with a GPU.
- Classical and deep models can't take gaps, so outage days are filled with the 8-week
  same-weekday median. Holiday closures stay 0.

## 3. Leaderboard (2025, mean of 4 quarters and seeds)

| Model | Family | WAPE 2025 | ± seeds | Bias | Peak-day WAPE | Tuning WAPE |
|---|---|---|---|---|---|---|
| **LightGBM, customer level → sites** | ML | **10.82%** | 0.02 | −2.2% | **16.5%** | 11.45% |
| Ensemble (LGBM-customer, LGBM, CatBoost) | ensemble | 11.39% | – | −3.2% | 17.7% | **11.14%** |
| CatBoost | ML | 12.37% | 0.06 | −4.0% | 18.6% | 11.58% |
| XGBoost | ML | 12.40% | 0.02 | −3.1% | 18.6% | 11.84% |
| LightGBM (site level) | ML | 12.44% | 0.05 | −3.3% | 18.6% | 11.54% |
| Random Forest | ML | 12.47% | 0.05 | −3.0% | 20.5% | 12.42% |
| Ridge | ML | 12.82% | – | −3.8% | 19.6% | 12.56% |
| N-BEATS | deep | 13.93% | 0.11 | −3.6% | 23.6% | 14.23% |
| Prophet | classical | 14.37% | – | −1.8% | 22.6% | 13.84% |
| N-HiTS | deep | 14.59% | 0.11 | −3.2% | 22.2% | 14.59% |
| **8-week median (baseline)** | baseline | **14.60%** | – | −3.4% | 24.4% | 15.06% |
| AutoETS | classical | 14.61% | – | −5.0% | 25.7% | 15.51% |
| LSTM | deep | 14.77% | 0.05 | −2.8% | 23.2% | 14.10% |
| AutoARIMA | classical | 14.90% | – | −4.4% | 25.3% | 15.25% |
| Theta | classical | 14.97% | – | −5.7% | 26.1% | 15.84% |
| Seasonal naive | baseline | 17.16% | – | −3.1% | 23.4% | 17.74% |
| MSTL | classical | 18.18% | – | −3.4% | 25.4% | 20.88% |

Full table with per-quarter and per-horizon scores: `reports/s3/leaderboard.csv`,
`reports/s3/wape_by_horizon.csv`. Every run is in MLflow (`uv run mlflow ui`).

**Reading it:**
1. Models that use the S2 clues (promotions, holidays, customers) win clearly: **~12.4%
   vs 14.6%** for the best simple rule.
2. The five ML algorithms are nearly tied. The library choice barely matters.
3. Classical and deep models barely beat, or lose to, the 8-week median. With 4 series and
   strong promotion/holiday effects, models that see only a series' own past are at a
   disadvantage. Deep models also need far more series.
4. **The largest gain came from the data level (customer vs site), not the algorithm.**
5. All models under-forecast (§5) and peak days stay hard (~16–19% for the best).

## 4. Experiments (LightGBM; same settings, 3 seeds, same 2025 scoring)

| | Experiment | Result (WAPE 2025) | Decision |
|---|---|---|---|
| E1 | site-level vs **customer-level summed to sites** (customer model tuned with the same budget) | **10.82%** vs 12.44%; tuning quarter 11.45% vs 11.54% | **adopt customer level** |
| E2 | outage days masked vs imputed (8-week median) | 12.44% vs 12.42% | no difference, keep masking |
| E3 | honest as-of features vs features from final data | honest 12.44%, naive-trained used honestly 12.40% (bias −3.7% vs −3.3%), naive both ways 12.39% | see below |
| E4 | squared error vs Tweedie loss | 12.44% vs 12.33%, bias −4.1% | keep squared error |
| E5 | retrain quarterly vs **monthly** | 12.44% vs **11.99%** | input for S8: retrain monthly |
| E6 | predict pallets vs predict ratio to recent level | 12.44% vs 11.77% on 2025, **but 11.54% vs 12.29% on tuning** | **not adopted**; registered challenger for 2026 |

**E3, honestly:** training–serving skew costs very little *here*. The model leans on
same-weekday values ≥ 7 days old, which are already final by the forecast morning. The only
effect is a slightly larger under-forecast. The point-in-time pipeline from S2 is still
needed for correctness (leakage is proven absent by tests), but in this dataset it barely
changes accuracy. We report that rather than overstate it.

## 5. Why every model under-forecasts

Bias of site-level LightGBM by site and quarter shows **S03 under-forecast by 10–18% in
2025 Q1–Q2**, when a new ready-meals customer ramped up and S03 reached record volumes.
**Tree models cannot extrapolate**: a tree's output is an average of past training
answers, so record highs are capped. Capacity-limited days are also under-forecast (−26%,
36 days), as expected for censored labels.

This explains E1: at customer level the new customer is its own series at a normal size,
so nothing is "record-breaking". E6 (ratio target) was designed *after* seeing this on
2025, so it was judged on the tuning quarter, where it lost. It's therefore registered as
a challenger **before** anyone looks at 2026, which has a similar event (S02's large new
customer).

## 6. Choice of model to deploy

By the pre-declared rule (choose on the tuning quarter), the **ensemble** ranks first
(11.14% vs 11.45%). The 2025 test says the customer-level LightGBM alone is better (10.82%
vs 11.39%).

**Decision: deploy customer-level LightGBM alone.** The reason is *not* its test score,
since choosing on test would be self-deception. The reasons are:
- the ensemble's advantage is 0.3 points on a single, noisy quarter;
- the ensemble mixes 3 models at 2 levels, which multiplies the cost of every later stage
  (registry, serving, monitoring, retraining).

The ensemble and the ratio-target model are kept as **challengers** to evaluate on 2026 in
S7/S8.

Deployed configuration: `reports/s3/best_params/lightgbm_customer.json`.

## 7. How to run

```bash
uv run python -m palletcast.models.baselines
uv run python -m palletcast.models.ml            # ~1 h
uv run python -m palletcast.models.classical     # ~2 h (refits every forecast morning)
uv run python -m palletcast.models.deep --models nhits nbeats lstm   # ~1.5 h
uv run python -m palletcast.models.experiments   # E1-E6, ~1.5 h
uv run python -m palletcast.models.leaderboard
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
On macOS without Homebrew, run `scripts/macos_openmp_fix.sh` once after `uv sync`
(LightGBM/XGBoost need the OpenMP library).

The backtest is not a DVC stage: it takes hours and is run deliberately. Its outputs
(leaderboard, predictions, best settings) are saved under `reports/s3/`. The production
training job comes in S4/S5.

## 8. Limitations
- One tuning quarter is a small basis for choices. A longer validation window would be
  more robust, at the cost of less training data.
- Customer-level versions of XGBoost/CatBoost weren't run. Since the algorithms tie at
  site level, little difference is expected.
- TFT not evaluated (CPU cost).
- No prediction intervals yet (a range like "1,800–2,300 pallets"). Planners would value
  them; a candidate for later.
