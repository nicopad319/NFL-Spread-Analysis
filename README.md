# NFL Spread Market Efficiency Study

**Can publicly available NFL metrics predict spread outcomes profitably?**

This project investigates whether freely available data — team efficiency metrics, weather, rest, and opening line movement — contains information not already priced into NFL betting spreads. It is framed as a **market efficiency study**, not a prediction tool. The null result is the finding.

Built with Python, scikit-learn, and `nfl_data_py`. Analysis covers 2010–2024 regular season (3,840+ games).

---

## Key Findings

| Metric | Result |
|---|---|
| Statistically significant features | 1 of 9 tested |
| Only significant feature | Defensive EPA differential (p = 0.020) |
| Walk-forward accuracy (mean, 2014–2024) | ~53.2% |
| Historical home cover rate (baseline) | ~47.6% |
| Backtesting ROI (half-Kelly, 54% threshold) | −5.5% |
| Final bankroll ($1,000 start, 11-year sim) | ~$490 |

**Bottom line:** The model beats the naive baseline directionally, but 53% accuracy is not enough to overcome -110 juice (52.38% breakeven). The NFL spread market is largely efficient with respect to publicly available quantitative data.

---

## Project Structure

```
nfl-spread-analysis/
├── explore.py                   # Full analysis pipeline
├── nfl_spread_analysis.ipynb    # Narrative notebook with methodology and findings
└── README.md
```

---

## Methodology

### Data Sources
- **nfl_data_py** — schedules (game results, spreads, weather) and play-by-play EPA data
- **SBR Historical Odds** — manually compiled opening lines for 2010–2021 (line movement analysis)

### Features Tested

| Feature | Description | p-value | Significant? |
|---|---|---|---|
| `diff_def_epa` | Defensive EPA differential (home − away) | 0.020 | ✓ |
| `diff_rolling_diff` | Rolling 4-game scoring differential | 0.053 | — |
| `large_favorite` | Home favored by 7+ points | 0.092 | — |
| `temp` | Game-time temperature | 0.085 | — |
| `home_is_favorite` | Binary: home team favored | 0.565 | — |
| `diff_off_epa` | Offensive EPA differential | 0.636 | — |
| `rest_diff` | Rest days differential | 0.710 | — |
| `wind` | Wind speed | 0.617 | — |
| `dome_team_outdoors` | Dome team playing in cold (< 40°F) | 0.898 | — |

Two additional features were engineered and tested but excluded from the final model:
- **QB-adjusted EPA** — EPA restricted to the season starter's plays. Dropped 37% of the sample due to incomplete early-season PBP data; showed no predictive signal (p = 0.977)
- **Opening line movement** — shift from SBR open to close. Validates correctly (r = 0.973 vs. nfl_data_py spread_line), but shows no correlation with outcomes (p = 0.858)

### Validation

**Walk-forward validation** — train on all seasons prior to the test year, test on each season from 2014–2024. This is the only defensible approach for time-series data: it prevents future information from leaking into training and simulates real deployment.

**Logistic regression with strong regularization** (C = 0.01). High regularization shrinks coefficients for noise features toward zero, preventing the model from overfitting the one marginally significant feature.

### Backtesting

Bets are sized using **half-Kelly criterion** based on the model's edge over the 52.38% breakeven:

```
edge       = p_model − 0.5238
kelly_frac = edge / 0.9091
bet_size   = (kelly_frac / 2) × bankroll
```

Only games where model confidence exceeds a threshold (54% default) are bet. Both sides are considered — if the model strongly favors the away team, the away side is bet.

A threshold sensitivity test revealed **uncalibrated probabilities**: win rate fell as confidence threshold rose (opposite of a well-calibrated model), confirming the model's raw scores should not be taken as reliable probability estimates.

---

## How to Run

**Requirements:**
```
pip install nfl_data_py pandas numpy scikit-learn scipy openpyxl jupyter
```

**Run the script:**
```bash
python explore.py
```

**Or open the notebook:**
```bash
jupyter notebook nfl_spread_analysis.ipynb
```

The notebook downloads PBP data on first run (~5 minutes for 15 seasons). Subsequent runs use `nfl_data_py`'s local cache.

> Note: SBR Excel files are not included in this repo. Line movement features will be skipped if the `sbr_data/` directory is absent — all other features compute normally.

---

## Results Detail

### Walk-Forward Accuracy by Season

| Season | Games | Accuracy | Baseline | vs. Baseline |
|---|---|---|---|---|
| 2014 | 256 | 52.0% | 47.3% | +4.7% ▲ |
| 2015 | 256 | 54.3% | 48.4% | +5.9% ▲ |
| 2016 | 256 | 51.2% | 49.2% | +2.0% ▲ |
| 2017 | 256 | 52.7% | 47.3% | +5.5% ▲ |
| 2018 | 256 | 54.3% | 46.9% | +7.4% ▲ |
| 2019 | 256 | 53.5% | 48.8% | +4.7% ▲ |
| 2020 | 256 | 51.6% | 46.9% | +4.7% ▲ |
| 2021 | 272 | 55.1% | 48.5% | +6.6% ▲ |
| 2022 | 272 | 51.5% | 46.3% | +5.1% ▲ |
| 2023 | 272 | 54.0% | 48.9% | +5.1% ▲ |
| 2024 | 272 | 52.2% | 47.4% | +4.8% ▲ |
| **Mean** | | **53.2%** | **47.6%** | **+5.1%** |

The model beats the naive baseline in all 11 out-of-sample test seasons. But beating "always predict the historical cover rate" is a low bar — the 52.38% juice threshold is what matters for profitability.

---

## Conclusion

The NFL betting market prices publicly available information quickly and accurately. A model built on EPA, scoring trends, weather, rest, and line movement beats a naive baseline directionally but cannot generate positive ROI after juice. This is consistent with the academic literature on sports betting market efficiency (Dare & McDonald, 1996; Borghesi, 2007).

The most actionable finding: **defensive efficiency (EPA allowed) is the only metric with a statistically significant relationship to spread outcomes**, and even that signal is too weak to exploit profitably on its own.

---

*Data: nfl_data_py (MIT License). Analysis covers 2010–2024 NFL regular season.*
