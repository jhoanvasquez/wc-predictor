# 2026 World Cup Match Predictor

A single-file-per-phase ML pipeline that predicts win / draw / loss probabilities for any international match. Built as a first real ML project — readable code, real data, no black boxes.

```
python predict.py "Colombia" "DR Congo"
```

```
==============================================================
  Colombia  vs  DR Congo
  2026 FIFA World Cup  |  neutral venue
==============================================================
  Elo  :  Colombia 1949  vs  DR Congo 1716  (diff +233)
  H2H  :  no prior meetings in dataset
--------------------------------------------------------------
  Colombia                  win     66.2%
  Draw                              20.2%
  DR Congo                  win     13.6%
--------------------------------------------------------------
  PICK: Colombia  (66.2%)   [CONFIDENT]
==============================================================
```

## Quick start

```bash
pip install -r requirements.txt

python fetch_data.py   # download ~5 MB of match history
python elo.py          # compute Elo ratings for 319 teams
python features.py     # engineer 17 features per match
python train.py        # train XGBoost, evaluate on 2022-2024
python predict.py "Spain" "Morocco"
```

## How it works

There is no magic — most of the signal comes from the features, not the model.

| Component | What it does |
|-----------|-------------|
| **Elo engine** | Assigns every team a rating starting at 1500. Updates after each match — bigger swing for upsets and blowouts. Single strongest predictor. |
| **Feature engineering** | Builds 17 signals per match: Elo difference, win rate over last 5 and 10 games, goal difference form, rest days, head-to-head record, neutral venue flag. |
| **XGBoost classifier** | Gradient-boosted trees trained on ~15,000 matches (2006–2021). Outputs three probabilities that sum to 1. |
| **Temporal validation** | Train on matches before 2022, test on 2022–2024. No future data leaks into training. |

## Results on held-out test set (2022–2024)

| Metric | Model | Baseline |
|--------|-------|----------|
| Accuracy | 58.6% | 47.4% |
| Log-loss | 0.907 | 1.053 |

Baseline = always predict training-set class frequencies.

## Pipeline scripts

| Script | Phase | Output |
|--------|-------|--------|
| `fetch_data.py` | 1 — Data | `data_cache/results_clean.csv` |
| `elo.py` | 2 — Elo engine | `data_cache/elo_ratings.csv`, `matches_with_elo.csv` |
| `features.py` | 3 — Features | `data_cache/features.csv` |
| `train.py` | 4 — Model | `data_cache/model.pkl` |
| `predict.py` | 5 — CLI | stdout |

## What the model can and cannot do

**Can do**
- Rank teams by recent form and historical quality
- Give calibrated probability estimates (not just a pick)
- Beat a naive "always pick the higher-rated team" baseline

**Cannot see**
- Injuries or suspensions on match day
- Starting lineups and rotation decisions
- Expected goals (xG) — the stat modern models rely on
- Tournament context ("we only need a draw to advance")

Draws are consistently under-predicted — a known limitation of win/draw/loss models on low-scoring sports. On the held-out test set, only 5% of actual draws were correctly predicted.

## Data

Historical results from [martj42/international_results](https://github.com/martj42/international_results). Downloaded automatically on first run.

## Stack

`pandas` · `numpy` · `xgboost` · `scikit-learn` · `matplotlib` · `requests`
