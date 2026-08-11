# NBA Fantasy Basketball Projections

An end-to-end machine learning pipeline that projects **next-season per-game NBA statistics** for every active player, then converts those projections into fantasy points for a custom scoring league.

The system pulls 24 seasons of NBA data (2000-01 → 2024-25), engineers 269 features per player-season, trains a separate model family for each of 12 target stats, and selects the best model or ensemble per stat against a held-out season.

![Python](https://img.shields.io/badge/python-3.12-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.7-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.0-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Results

### Held-out validation (2023-24 season, n = 400 players)

Each model is trained on 2000-01 → 2022-23 and evaluated on 2023-24, a season it never sees during training or tuning. The naive baseline is *"next season = last season"*, which is a genuinely strong baseline for NBA per-game rate stats.

| Target | Baseline MAE | Model MAE | R² | Improvement | Winning approach |
|--------|-------------:|----------:|----:|------------:|------------------|
| PTS    | 2.326 | **2.062** | 0.849 | 11.3% | Weighted average |
| REB    | 0.823 | **0.759** | 0.818 |  7.8% | Simple average |
| AST    | 0.627 | **0.568** | 0.846 |  9.4% | Weighted average |
| STL    | 0.214 | **0.189** | 0.591 | 11.5% | Simple average |
| BLK    | 0.142 | **0.124** | 0.811 | 12.6% | XGBoost |
| TOV    | 0.316 | **0.292** | 0.787 |  7.7% | Simple average |
| MIN    | 3.967 | **3.556** | 0.736 | 10.4% | Position XGBoost |
| FG3M   | 0.351 | **0.326** | 0.779 |  7.0% | Weighted average |
| FGA    | 1.742 | **1.525** | 0.847 | 12.5% | Simple average |
| FTA    | 0.564 | **0.510** | 0.851 |  9.5% | Weighted average |
| FG_PCT | 0.039 | **0.031** | 0.628 | 21.5% | Weighted average |
| FT_PCT | 0.085 | **0.067** | 0.296 | 21.3% | Simple average |

Two honest caveats: `FT_PCT` is close to unpredictable year-over-year (R² 0.30), and `STL` is noisy (R² 0.59). Both still beat the baseline, but treat their point estimates loosely.

### Live backtest (2025-26 projections vs. actual results, n = 406 players)

Projections generated from 2024-25 data before the season, scored against the real 2025-26 stats via `evaluate_predictions.py`:

| Stat | MAE | Bias | MAPE |
|------|----:|-----:|-----:|
| PTS  | 2.53 | −0.24 | 13.5% |
| REB  | 0.83 | +0.04 | 14.1% |
| AST  | 0.67 | −0.02 | 18.5% |
| STL  | 0.20 | −0.04 | 15.1% |
| BLK  | 0.15 | +0.02 | 23.0% |
| TOV  | 0.32 | −0.02 | 16.8% |
| FGA  | 1.79 | −0.01 | 12.7% |
| FTA  | 0.68 | −0.19 | 22.0% |

Errors on unseen live data land close to validation error, and bias is small and mostly negative — the model is mildly conservative, understating volume for players whose roles expanded.

---

## Quickstart

### Prerequisites
- Python 3.12
- Internet access for stages 1–4 (live NBA API calls)

### Installation

```bash
git clone https://github.com/IsaacPressman/fantasy_basketball_predictor.git
cd fantasy_basketball_predictor

python3.12 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run the pipeline

```bash
# Full pipeline: data collection through final projections
python run_pipeline.py

# Modeling only (skips the API calls — requires data/raw/ to already exist)
python run_pipeline.py --start 5

# A single stage
python run_pipeline.py --start 17 --end 17
```

On Windows, prefix commands so that Unicode console output works on cp1252 terminals:

```
set PYTHONIOENCODING=utf-8 && venv/Scripts/python.exe -X utf8 run_pipeline.py
```

A full run from scratch takes roughly 30–60 minutes, dominated by NBA API rate limits (stages 1–4) and Optuna hyperparameter search (stages 14–16).

### Score projections against reality

```bash
python evaluate_predictions.py
```

Fetches actual results for the projected season and writes MAE / bias / MAPE per stat to `predictions/evaluation/`. Actual stats are cached to `data/raw/` after the first fetch.

---

## Pipeline

The pipeline is 18 stages, orchestrated by `run_pipeline.py`. Each stage reads the previous stage's output from disk, so any stage can be re-run in isolation.

| # | Stage | Module | Output |
|---|-------|--------|--------|
| 1 | Fetch player stats | `data_collection/nba_stats_fetcher.py` | `data/raw/player_stats_*.csv` |
| 2 | Fetch advanced stats | `data_collection/advanced_stats_fetcher.py` | `data/raw/advanced_stats_*.csv` |
| 3 | Fetch team stats | `data_collection/team_stats_fetcher.py` | `data/raw/team_stats_*.csv` |
| 4 | Fetch player bio info | `data_collection/player_info_fetcher.py` | `data/raw/player_info.csv` |
| 5 | Merge datasets | `feature_engineering/merge_datasets.py` | `data/processed/master_dataset.csv` |
| 6 | Create targets | `feature_engineering/create_targets.py` | `NEXT_*` columns |
| 7 | Lag features | `feature_engineering/lag_features.py` | 1/2/3-season lags, trends |
| 8 | Derived features | `feature_engineering/derived_features.py` | rates, efficiency, interactions |
| 9 | Availability features | `feature_engineering/availability_features.py` | workload & injury-risk signals |
| 10 | Validate features | `feature_engineering/validate_features.py` | `data/processed/final_*.csv` |
| 11 | Prepare training data | `models/prepare_training_data.py` | `data/processed/prepared/*.csv` |
| 12 | Baselines | `models/baseline.py` | `models/baseline_results/` |
| 13 | Linear models | `models/linear_model.py` | `models/linear/` |
| 14 | Random Forest | `models/random_forest.py` | `models/random_forest/` |
| 15 | XGBoost | `models/gradient_boosting.py` | `models/xgboost/` |
| 16 | Position-specific XGBoost | `models/position_models.py` | `models/position_xgboost/` |
| 17 | Ensemble & selection | `models/ensemble.py` | `models/ensemble_results/ensemble_results.json` |
| 18 | Generate projections | `models/generate_predictions.py` | `predictions/nba_projections_*.csv` |

```
NBA API → data/raw/ → data/processed/ → data/processed/prepared/ → models/ → predictions/
```

---

## Methodology

### Prediction targets

Twelve per-game statistics, each modeled independently as a `NEXT_*` target (the player's value in the *following* season):

```
PTS  REB  AST  STL  BLK  TOV  MIN  FG3M  FGA  FTA  FG_PCT  FT_PCT
```

### Temporal split

Because the task is forecasting, the split is strictly chronological — no random shuffling, no future data leaking backward.

| Split | Seasons | Rows |
|-------|---------|-----:|
| Train | 2000-01 → 2022-23 (22 seasons) | 6,817 |
| Validation | 2023-24 | 400 |
| Projection input | 2024-25 | 508 |

The validation season is used **only** for model selection and ensemble weighting. Optuna objectives never touch it — hyperparameters are tuned by cross-validation inside the training set, and ensemble weights come from 5-fold CV on the training set (`calculate_weights_from_cv`). This keeps 2023-24 a clean estimate of out-of-sample error.

*(The 2004-05 season is a full lockout and is excluded entirely. 2011-12 (66 games), 2019-20 (63 games), and 2020-21 (72 games) are shortened; `SEASON_MAX_GAMES` normalizes availability features accordingly.)*

### Features (269 total)

| Group | Examples |
|-------|----------|
| Current-season box score | `PTS`, `REB`, `AST`, `FGA`, `FG_PCT`, `MIN`, `GP` |
| Advanced metrics | `ADV_USG_PCT`, `ADV_AST_PCT`, `ADV_TS_PCT`, `ADV_PIE` |
| Team context | `TEAM_PACE`, `TEAM_OFF_RATING`, `TEAM_NET_RATING`, `TEAM_W_PCT` |
| Lags (1–3 seasons) | `PTS_LAG1`, `MIN_LAG2`, `USG_PCT_LAG3` — 71 features |
| Trends & deltas | `PTS_TREND`, `MIN_CHANGE`, `GP_IMPROVING`, `WORKLOAD_TREND` |
| Rate normalization | `PTS_PER36`, `REB_PACE_ADJ`, `USAGE_MINUTES_PRODUCT` |
| Biographical | `AGE`, `HEIGHT_INCHES`, `WEIGHT`, `BMI`, `DRAFT_NUMBER`, position dummies |
| Availability & workload | `GP_CONSISTENCY`, `AVAILABILITY_SCORE`, `INJURY_PRONE_PATTERN`, `LOAD_MANAGEMENT` |
| Interactions | `AGE_USAGE_INTERACTION`, `PACE_USAGE_INTERACTION`, `AGE_MINUTES_RISK` |

The availability block exists because minutes and games played drive fantasy value more than efficiency does — modeling *who stays on the floor* matters as much as modeling *how well they play*.

### Model families

Each of the 12 targets gets its own model from each family, and the best per-target approach wins:

1. **Linear** — Ridge / Lasso / ElasticNet with feature scaling
2. **Random Forest** — Optuna-tuned, 25 trials
3. **XGBoost** — Optuna-tuned, 30 trials
4. **Position XGBoost** — separate Guard / Wing / Big models, 20 trials per group
5. **Stacking** — Ridge meta-learner over the base model predictions
6. **Simple / weighted average** — weights derived from training-set CV performance

`models/ensemble_results/ensemble_results.json` is the pipeline's key artifact: it records `best_approach` and per-model `weights` for every target, and drives model loading in stage 18. In practice the simple and weighted averages win most targets — a normal result when base models are individually strong and make partially uncorrelated errors.

### Uncertainty

Each projected stat ships with `_LOW` / `_HIGH` columns: a 95% interval of ±1.96 × the standard deviation of that model's validation residuals, clipped at zero for counting stats. This is a constant-width band, not a per-player heteroscedastic interval — it communicates typical model error, not individual player risk.

### Fantasy scoring

Projections are converted to fantasy points using a custom league formula, implemented once in `src/constants.py::calculate_fantasy_points()`:

```
PTS×1 + REB×1 + AST×2 + STL×4 + BLK×4 + FGM×2 + FGA×(−1) + TOV×(−2) + FG3M×1 + FT_MISS×(−1)
```

where `FT_MISS = FTA − FTM`. To use a different league's scoring, edit `FANTASY_SCORING` in `src/constants.py` — nothing else needs to change.

---

## Outputs

`predictions/` per run, timestamped:

| File | Contents |
|------|----------|
| `nba_projections_<season>_<timestamp>.csv` | All ~500 players: 12 projected stats, 95% intervals, fantasy points, rank |
| `top_100_projections_<timestamp>.csv` | Top 100 by projected fantasy points |
| `projection_summary_<timestamp>.txt` | Human-readable top-10 board and league-wide averages |
| `evaluation/accuracy_summary_<season>.csv` | Post-season MAE / bias / MAPE per stat |

Model diagnostics live alongside the artifacts: per-target feature importances in `models/{random_forest,xgboost}/feature_importances/`, and metrics in `models/*_results/`.

---

## Project structure

```
fantasy_basketball_predictor/
├── run_pipeline.py               # Stage orchestrator (--start / --end)
├── evaluate_predictions.py       # Backtest projections vs. actual results
├── src/
│   ├── constants.py              # Targets, season order, splits, fantasy scoring
│   ├── config.py                 # Canonical filesystem paths
│   ├── data_collection/          # Stages 1–4  (NBA API)
│   ├── feature_engineering/      # Stages 5–10
│   └── models/                   # Stages 11–18
├── data/
│   ├── raw/                      # NBA API pulls
│   └── processed/
│       └── prepared/             # X_train / X_val / X_pred, targets, metadata
├── models/                       # Serialized models, metrics, feature importances
└── predictions/                  # Projections + post-season evaluation
```

### Two files worth knowing

- **`src/constants.py`** — the single source of truth for target columns, season ordering, games-per-season adjustments, the train/val/predict split, and the fantasy scoring formula. Change the configuration here, not in individual modules.
- **`src/config.py`** — canonical directory paths. Import from here rather than hardcoding path strings.

---

## Known limitations

- **Position coverage.** `player_info.csv` comes from the bulk `leaguedashplayerbiostats` endpoint and only covers players active from 2020-21 onward. Historical players have no `POSITION`, so position is inferred from height for pre-2020 seasons and the Guard/Wing/Big models are thinner there than the overall dataset size suggests.
- **No mid-season roster context.** Projections are made from end-of-season aggregates and don't account for offseason trades, signings, or announced role changes. A player who changes teams in July is projected on last year's team context.
- **Rookies are out of scope.** Every model requires at least one prior NBA season of features, so incoming rookies get no projection.
- **`FT_PCT` and `STL` are weakly predictable.** See the caveat under Results.
- **NBA API rate limits.** Stage 4 deliberately uses 5 bulk calls instead of one `commonplayerinfo` call per player — the per-player approach exhausts DNS/connection limits after roughly 200 requests.

## Data source

All data comes from the public NBA Stats API via the [`nba_api`](https://github.com/swar/nba_api) package. This project is unofficial and not affiliated with or endorsed by the NBA.

## License

MIT
