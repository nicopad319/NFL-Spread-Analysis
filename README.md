# NFL Spread Market Efficiency Study

Can publicly available NFL data predict spread outcomes well enough to be profitable? This project tries to answer that question by building and testing two models (logistic regression and a PyTorch neural network) against 15 seasons of NFL data.

The short answer: not really. And that's actually the interesting finding.

Built with Python, scikit-learn, and PyTorch. Data covers the 2010-2024 regular season (3,840+ games).

---

## Key Results

| Metric | Result |
|---|---|
| Statistically significant features | 1 out of 9 tested |
| Only significant feature | Defensive EPA differential (p = 0.020) |
| Walk-forward accuracy (mean, 2014-2024) | ~53.2% |
| Historical home cover rate (baseline) | ~47.6% |
| Backtesting ROI (half-Kelly, 54% threshold) | -5.5% |
| Final bankroll ($1,000 start, 11-year sim) | ~$490 |

The model beats the naive baseline directionally, but 53% accuracy isn't enough to overcome -110 juice (52.38% breakeven). The neural network comparison confirms this isn't a modeling problem; there just isn't enough signal in the public data to exploit profitably.

---

## Project Structure

```
nfl-spread-analysis/
├── explore.py                   # full analysis pipeline
├── nfl_spread_analysis.ipynb    # narrative notebook with methodology and findings
└── README.md
```

---

## How It Works

### Data Sources
- **nfl_data_py** for schedules (results, spreads, weather, rest) and play-by-play EPA data
- **SBR Historical Odds** for opening lines from 2010-2021 (manually compiled, used for line movement analysis)

### Features Tested

| Feature | Description | p-value | Significant? |
|---|---|---|---|
| `diff_def_epa` | Defensive EPA differential (home minus away) | 0.020 | yes |
| `diff_rolling_diff` | Rolling 4-game scoring differential | 0.053 | no |
| `large_favorite` | Home favored by 7+ points | 0.092 | no |
| `temp` | Game-time temperature | 0.085 | no |
| `home_is_favorite` | Binary: is home team favored | 0.565 | no |
| `diff_off_epa` | Offensive EPA differential | 0.636 | no |
| `rest_diff` | Rest days differential | 0.710 | no |
| `wind` | Wind speed | 0.617 | no |
| `dome_team_outdoors` | Dome team playing in cold (under 40F) | 0.898 | no |

Two features were engineered and tested but cut from the final model:
- **QB-adjusted EPA**: EPA restricted to the season starter's plays. Dropped 37% of the sample due to incomplete early-season PBP data, and showed no predictive signal (p = 0.977). The market already adjusts for known QB situations.
- **Opening line movement**: SBR open to close. Validates correctly (r = 0.973 vs. nfl_data_py spread_line), but zero correlation with outcomes (p = 0.858).

### Validation Approach

**Walk-forward validation**: train on all seasons before the test year, test on each season from 2014 to 2024. The only defensible approach for time-series data since it prevents future information from leaking into training.

**Logistic regression with strong regularization** (C = 0.01). This shrinks noise feature coefficients toward zero, so the model doesn't overfit to the 8 features with no real signal.

### Neural Network Comparison (Section 9)

To test whether model complexity helps, a feedforward neural network was trained and compared against logistic regression on the same 2024 holdout set.

Architecture: 9 inputs -> Linear(64) -> Linear(32) -> Linear(16) -> Linear(1), with BatchNorm, ReLU, and Dropout(0.3) after each hidden layer. Trained with Adam (lr=1e-3), BCEWithLogitsLoss, and a learning rate scheduler.

Split: train on 2010-2022, validate on 2023 (for hyperparameter tuning), test on 2024 (reported once).

**Result**: both models perform similarly on the 2024 holdout, which confirms the issue is the data, not the model. Adding more complexity doesn't help when there isn't much signal to learn.

### Backtesting

Bets sized using half-Kelly criterion based on the model's edge over the 52.38% breakeven:

```
edge       = p_model - 0.5238
kelly_frac = edge / 0.9091
bet_size   = (kelly_frac / 2) * bankroll
```

Only games where model confidence exceeds 54% are bet. Both sides are considered (if the model strongly favors away, the away side is bet).

A threshold sensitivity test showed uncalibrated probabilities: win rate fell as confidence threshold rose, which is the opposite of what a well-calibrated model does. This is a known limitation of logistic regression on noisy data.

---

## How to Run

**Requirements:**
```
pip install nfl_data_py pandas numpy scikit-learn scipy openpyxl torch jupyter
```

**Run the script:**
```bash
python explore.py
```

**Or open the notebook:**
```bash
jupyter notebook nfl_spread_analysis.ipynb
```

PBP data downloads on first run (~5 minutes for 15 seasons). Subsequent runs use nfl_data_py's local cache.

Note: SBR Excel files are not included in this repo. Line movement features will be skipped if the `sbr_data/` folder is absent; all other features compute normally.

---

## Walk-Forward Results by Season

| Season | Games | Accuracy | Baseline | vs. Baseline |
|---|---|---|---|---|
| 2014 | 256 | 52.0% | 47.3% | +4.7% |
| 2015 | 256 | 54.3% | 48.4% | +5.9% |
| 2016 | 256 | 51.2% | 49.2% | +2.0% |
| 2017 | 256 | 52.7% | 47.3% | +5.5% |
| 2018 | 256 | 54.3% | 46.9% | +7.4% |
| 2019 | 256 | 53.5% | 48.8% | +4.7% |
| 2020 | 256 | 51.6% | 46.9% | +4.7% |
| 2021 | 272 | 55.1% | 48.5% | +6.6% |
| 2022 | 272 | 51.5% | 46.3% | +5.1% |
| 2023 | 272 | 54.0% | 48.9% | +5.1% |
| 2024 | 272 | 52.2% | 47.4% | +4.8% |
| **Mean** | | **53.2%** | **47.6%** | **+5.1%** |

Beats the naive baseline in all 11 out-of-sample seasons. But the 52.38% juice threshold is the bar that matters for profitability, not the naive baseline.

---

## Final Model Coefficients (trained 2010-2023, tested on 2024)

| Feature | Coefficient |
|---|---|
| diff_def_epa | +0.071 |
| large_favorite | +0.030 |
| wind | +0.005 |
| dome_team_outdoors | +0.002 |
| home_is_favorite | -0.006 |
| rest_diff | -0.006 |
| diff_off_epa | -0.007 |
| diff_rolling_diff | -0.012 |
| temp | -0.033 |

With C=0.01 regularization, `diff_def_epa` ends up doing almost all the work. Its coefficient is 2x the next largest feature. Everything else gets shrunk close to zero, which lines up with the p-value results.

---

## Conclusion

The NFL betting market prices publicly available information quickly. A model built on EPA, scoring trends, weather, rest, and line movement beats a naive baseline directionally but can't generate positive ROI after juice. Consistent with the academic literature on sports betting market efficiency (Dare and McDonald, 1996; Borghesi, 2007).

The most useful finding: defensive EPA differential is the only metric with a statistically significant relationship to spread outcomes, and even that signal is too weak to exploit profitably on its own. The neural network comparison reinforces this; more model complexity doesn't help when the signal ceiling is this low.

---

*Data: nfl_data_py (MIT License). Analysis covers 2010-2024 NFL regular season.*
