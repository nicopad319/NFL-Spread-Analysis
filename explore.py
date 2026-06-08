import nfl_data_py as nfl
import pandas as pd
import numpy as np
import os
import re
import openpyxl
from datetime import date as date_cls
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats

# ── SBR line data ─────────────────────────────────────────────────────────────
# Files named like nfl_21-22.xlsx, nfl_20-21.xlsx, etc.
# Format: two rows per game (V=away, H=home)
#   V row Open/Close = game total (large number ~40-60)
#   H row Open/Close = point spread (small number <30, or 'pk' for pick-em)
#   Exception: when away team is the favorite, spread appears on V row instead
#   Sign convention: positive = that row's team is favored (gives points)
#   Matches nfl_data_py: positive spread_line = home favored

SBR_DIR = r'C:\Users\padil\OneDrive\Documents\nfl-model\sbr_data'

SBR_TEAM_MAP = {
    # Standard names
    'Arizona': 'ARI', 'Atlanta': 'ATL', 'Baltimore': 'BAL', 'Buffalo': 'BUF',
    'Carolina': 'CAR', 'Chicago': 'CHI', 'Cincinnati': 'CIN', 'Cleveland': 'CLE',
    'Dallas': 'DAL', 'Denver': 'DEN', 'Detroit': 'DET', 'GreenBay': 'GB',
    'Houston': 'HOU', 'Indianapolis': 'IND', 'Jacksonville': 'JAX',
    'KansasCity': 'KC', 'LAChargers': 'LAC', 'LARams': 'LA', 'LasVegas': 'LV',
    'Miami': 'MIA', 'Minnesota': 'MIN', 'NewEngland': 'NE', 'NewOrleans': 'NO',
    'NYGiants': 'NYG', 'NYJets': 'NYJ', 'Oakland': 'LV', 'Philadelphia': 'PHI',
    'Pittsburgh': 'PIT', 'SanDiego': 'LAC', 'SanFrancisco': 'SF',
    'Seattle': 'SEA', 'StLouis': 'LA', 'TampaBay': 'TB', 'Tennessee': 'TEN',
    'Washington': 'WAS', 'WashingtonFootball': 'WAS', 'Commanders': 'WAS',
    # Alternate / malformed names seen across seasons
    'St.Louis': 'LA',        # pre-2016 Rams
    'LosAngeles': 'LA',      # 2016 season (only Rams in LA that year)
    'LVRaiders': 'LV',       # Las Vegas Raiders alt name
    'KCChiefs': 'KC',        # Kansas City alt name
    'Kansas': 'KC',          # truncated copy-paste
    'Tampa': 'TB',           # truncated copy-paste
    'BuffaloBills': 'BUF',   # full team name copy-paste artifact
    'NewYork': 'NYG',        # confirmed: 2013 home vs Oakland = Giants at MetLife
    'Washingtom': 'WAS',     # typo (m instead of n)
}

def _parse_spread_val(val):
    """'pk'/'PK' → 0.0, None → NaN, else float."""
    if val is None:
        return np.nan
    if str(val).strip().lower() in ('pk', 'ev', 'pick'):
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

def _is_spread(val):
    """True if val looks like a point spread rather than a game total."""
    if val is None:
        return False
    if str(val).strip().lower() in ('pk', 'ev', 'pick'):
        return True
    try:
        return abs(float(val)) < 30
    except (ValueError, TypeError):
        return False

def _season_year_from_filename(path):
    name = os.path.basename(path)
    m = re.search(r'(\d{2})-\d{2}', name)
    if m:
        return 2000 + int(m.group(1))
    m = re.search(r'(\d{4})', name)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract season year from: {name}")

def _mmdd_to_date(mmdd_val, season_year):
    mmdd = int(mmdd_val)
    month = mmdd // 100
    day   = mmdd % 100
    year  = season_year if month >= 3 else season_year + 1
    return str(date_cls(year, month, day))

def parse_sbr_file(path):
    season_year = _season_year_from_filename(path)
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    all_rows = list(ws.iter_rows(values_only=True))

    # Strip header and blank rows
    data = [r for r in all_rows if r[2] not in (None, 'VH')]

    games = []
    i = 0
    while i < len(data) - 1:
        r1, r2 = data[i], data[i + 1]

        # Skip neutral-site games (Super Bowl, London, etc.)
        if r1[2] == 'N' or r2[2] == 'N':
            i += 1
            continue

        # Identify visitor / home rows
        if   r1[2] == 'V' and r2[2] == 'H': v_row, h_row = r1, r2
        elif r1[2] == 'H' and r2[2] == 'V': h_row, v_row = r1, r2
        else:
            i += 1
            continue

        date_val = v_row[0] or h_row[0]
        if date_val is None:
            i += 2
            continue

        # Columns: Date(0) Rot(1) VH(2) Team(3) 1st(4) 2nd(5) 3rd(6) 4th(7)
        #          Final(8) Open(9) Close(10) ML(11) 2H(12)
        v_open, v_close = v_row[9], v_row[10]
        h_open, h_close = h_row[9], h_row[10]

        v_is_spread = _is_spread(v_open)
        h_is_spread = _is_spread(h_open)

        if h_is_spread and not v_is_spread:
            # Normal case: home row has spread, away row has total
            # Positive = home is favored (home gives points) — matches nfl_data_py convention
            spread_open  =  _parse_spread_val(h_open)
            spread_close =  _parse_spread_val(h_close)
        elif v_is_spread and not h_is_spread:
            # Away team is the favorite — spread is on the visitor row
            # Negate so positive = home is favored in final convention
            spread_open  = -_parse_spread_val(v_open)
            spread_close = -_parse_spread_val(v_close)
        else:
            # Ambiguous row (both or neither look like a spread) — skip
            i += 2
            continue

        # Sanity check: close should look like a spread too
        # Copy-paste errors can put a total in the close column
        if spread_close is not None and abs(spread_close) >= 30:
            spread_close = spread_open  # line didn't move, use open as close

        away_team = SBR_TEAM_MAP.get(str(v_row[3]).strip(), str(v_row[3]).strip())
        home_team = SBR_TEAM_MAP.get(str(h_row[3]).strip(), str(h_row[3]).strip())

        try:
            gameday = _mmdd_to_date(date_val, season_year)
        except Exception:
            i += 2
            continue

        games.append({
            'gameday':     gameday,
            'away_team':   away_team,
            'home_team':   home_team,
            'spread_open': spread_open,
            'spread_close_sbr': spread_close,
        })
        i += 2

    return pd.DataFrame(games)

def load_all_sbr(sbr_dir):
    frames = []
    for fname in sorted(os.listdir(sbr_dir)):
        if fname.endswith('.xlsx') or fname.endswith('.xls'):
            path = os.path.join(sbr_dir, fname)
            try:
                df = parse_sbr_file(path)
                frames.append(df)
                print(f"  Loaded {fname}: {len(df)} games")
            except Exception as e:
                print(f"  SKIP {fname}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

SEASONS = list(range(2010, 2025))
DOME_TEAMS = ['NO', 'ATL', 'IND', 'DET', 'MIN', 'LV', 'DAL', 'ARI']
TEAM_ABBREV_MAP = {
    'JAC': 'JAX', 'LVR': 'LV', 'STL': 'LA', 'SD': 'LAC',
    'OAK': 'LV', 'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE',
    'HST': 'HOU',
}

# ── Schedules ─────────────────────────────────────────────────────────────────

schedules = nfl.import_schedules(SEASONS)
schedules = schedules[schedules['game_type'] == 'REG'].copy()
schedules['home_covered'] = (schedules['result'] > schedules['spread_line']).astype(int)

# Normalize team abbreviations to match PBP data
# OAK→LV, SD→LAC, STL→LA so EPA merges don't silently fail for relocated franchises
schedules['home_team'] = schedules['home_team'].replace(TEAM_ABBREV_MAP)
schedules['away_team'] = schedules['away_team'].replace(TEAM_ABBREV_MAP)

schedules['temp_known'] = schedules['temp'].notna()
schedules['temp'] = schedules['temp'].fillna(70)
schedules['wind'] = schedules['wind'].fillna(0)

schedules['rest_diff'] = schedules['home_rest'] - schedules['away_rest']
schedules['home_is_favorite'] = (schedules['spread_line'] < 0).astype(int)
schedules['large_favorite'] = (schedules['spread_line'] < -7).astype(int)
schedules['dome_team_outdoors'] = (
    schedules['home_team'].isin(DOME_TEAMS) &
    schedules['temp_known'] &
    (schedules['temp'] < 40)
).astype(int)

schedules = schedules.sort_values('gameday').reset_index(drop=True)

# ── Rolling scoring stats (per team per season) ───────────────────────────────

home_games = schedules[['game_id', 'gameday', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
home_games.columns = ['game_id', 'gameday', 'season', 'week', 'team', 'opponent', 'points_scored', 'points_allowed']

away_games = schedules[['game_id', 'gameday', 'season', 'week', 'away_team', 'home_team', 'away_score', 'home_score']].copy()
away_games.columns = ['game_id', 'gameday', 'season', 'week', 'team', 'opponent', 'points_scored', 'points_allowed']

game_log = pd.concat([home_games, away_games]).sort_values('gameday').reset_index(drop=True)

game_log['rolling_scored'] = game_log.groupby(['team', 'season'])['points_scored'].transform(
    lambda x: x.shift(1).rolling(4, min_periods=1).mean()
)
game_log['rolling_allowed'] = game_log.groupby(['team', 'season'])['points_allowed'].transform(
    lambda x: x.shift(1).rolling(4, min_periods=1).mean()
)
game_log['rolling_diff'] = game_log['rolling_scored'] - game_log['rolling_allowed']

team_stats = game_log[['game_id', 'team', 'rolling_diff']].copy()

schedules = schedules.merge(
    team_stats.rename(columns={'team': 'home_team', 'rolling_diff': 'home_rolling_diff'}),
    on=['game_id', 'home_team'], how='left'
)
schedules = schedules.merge(
    team_stats.rename(columns={'team': 'away_team', 'rolling_diff': 'away_rolling_diff'}),
    on=['game_id', 'away_team'], how='left'
)
schedules['diff_rolling_diff'] = schedules['home_rolling_diff'] - schedules['away_rolling_diff']

# ── EPA features ──────────────────────────────────────────────────────────────

pbp = nfl.import_pbp_data(SEASONS)
pbp = pbp[pbp['play_type'].isin(['pass', 'run']) & pbp['epa'].notna()].copy()
pbp['posteam'] = pbp['posteam'].replace(TEAM_ABBREV_MAP)
pbp['defteam'] = pbp['defteam'].replace(TEAM_ABBREV_MAP)

offense_epa = pbp.groupby(['game_id', 'posteam'])['epa'].mean().reset_index()
offense_epa.columns = ['game_id', 'team', 'off_epa_per_play']
defense_epa = pbp.groupby(['game_id', 'defteam'])['epa'].mean().reset_index()
defense_epa.columns = ['game_id', 'team', 'def_epa_per_play']

team_epa = offense_epa.merge(defense_epa, on=['game_id', 'team'])
team_epa = team_epa.merge(
    schedules[['game_id', 'gameday', 'season']].drop_duplicates(), on='game_id'
).sort_values('gameday').reset_index(drop=True)

team_epa['rolling_off_epa'] = team_epa.groupby(['team', 'season'])['off_epa_per_play'].transform(
    lambda x: x.shift(1).rolling(4, min_periods=1).mean()
)
team_epa['rolling_def_epa'] = team_epa.groupby(['team', 'season'])['def_epa_per_play'].transform(
    lambda x: x.shift(1).rolling(4, min_periods=1).mean()
)

schedules = schedules.merge(
    team_epa[['game_id', 'team', 'rolling_off_epa', 'rolling_def_epa']].rename(columns={
        'team': 'home_team', 'rolling_off_epa': 'home_rolling_off_epa', 'rolling_def_epa': 'home_rolling_def_epa'
    }),
    on=['game_id', 'home_team'], how='left'
)
schedules = schedules.merge(
    team_epa[['game_id', 'team', 'rolling_off_epa', 'rolling_def_epa']].rename(columns={
        'team': 'away_team', 'rolling_off_epa': 'away_rolling_off_epa', 'rolling_def_epa': 'away_rolling_def_epa'
    }),
    on=['game_id', 'away_team'], how='left'
)

schedules['diff_off_epa'] = schedules['home_rolling_off_epa'] - schedules['away_rolling_off_epa']
schedules['diff_def_epa'] = schedules['home_rolling_def_epa'] - schedules['away_rolling_def_epa']

# ── QB-adjusted EPA ───────────────────────────────────────────────────────────
# Motivation: team EPA is noisy because backup QB games drag down the rolling
# average even though the market already knows the starter is injured.
# Starter-only EPA isolates the signal from the QB the team actually plans to start.
#
# Method:
#   1. Game-day starter  = QB with most pass attempts in that game
#   2. Season starter    = QB who was game-day starter most often that season
#   3. Starter EPA       = EPA per play restricted to the season starter's throws
#   4. starter_played    = fraction of last 4 games the season starter actually played
#                          (low value = recent injury, market may not have fully adjusted)

pass_plays = pbp[
    (pbp['play_type'] == 'pass') &
    (pbp['passer_player_id'].notna())
].copy()

# Step 1 — game-day starter per team per game
qb_attempts = (pass_plays
    .groupby(['game_id', 'posteam', 'passer_player_id'])
    .size()
    .reset_index(name='attempts'))
qb_attempts = qb_attempts.sort_values('attempts', ascending=False)
game_starters = (qb_attempts
    .groupby(['game_id', 'posteam'])
    .first()
    .reset_index()
    [['game_id', 'posteam', 'passer_player_id']]
    .rename(columns={'posteam': 'team', 'passer_player_id': 'game_starter_id'}))

# Step 2 — season starter per team
# pbp already has a season column — use it directly to avoid duplicate columns
game_starters = game_starters.merge(
    pass_plays[['game_id', 'season']].drop_duplicates(), on='game_id'
)
season_starter_counts = (game_starters
    .groupby(['season', 'team', 'game_starter_id'])
    .size()
    .reset_index(name='games_started')
    .sort_values('games_started', ascending=False))
season_starters = (season_starter_counts
    .groupby(['season', 'team'])
    .first()
    .reset_index()
    [['season', 'team', 'game_starter_id']]
    .rename(columns={'game_starter_id': 'season_starter_id'}))

# Flag each game: did the season starter play?
game_starters = game_starters.merge(season_starters, on=['season', 'team'])
game_starters['starter_played'] = (
    game_starters['game_starter_id'] == game_starters['season_starter_id']
).astype(int)

# Step 3 — EPA per game restricted to season starter's throws
# pass_plays already has season — merge directly without re-joining schedules
starter_plays = pass_plays.merge(
    season_starters.rename(columns={'team': 'posteam'}),
    on=['season', 'posteam']
)
starter_plays = starter_plays[
    starter_plays['passer_player_id'] == starter_plays['season_starter_id']
]

starter_epa = (starter_plays
    .groupby(['game_id', 'posteam'])['epa']
    .mean()
    .reset_index()
    .rename(columns={'posteam': 'team', 'epa': 'starter_off_epa_per_play'}))

# Merge starter_played flag in
starter_epa = starter_epa.merge(
    game_starters[['game_id', 'team', 'starter_played']], on=['game_id', 'team']
)

# Step 4 — rolling (per team per season, no lookahead)
# groupby dropped season — get both gameday and season from schedules
starter_epa = starter_epa.merge(
    schedules[['game_id', 'gameday', 'season']].drop_duplicates(), on='game_id'
).sort_values('gameday').reset_index(drop=True)

starter_epa['rolling_starter_off_epa'] = (
    starter_epa
    .groupby(['team', 'season'])['starter_off_epa_per_play']
    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
)
starter_epa['rolling_starter_played'] = (
    starter_epa
    .groupby(['team', 'season'])['starter_played']
    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
)

# Step 5 — merge into schedules
schedules = schedules.merge(
    starter_epa[['game_id', 'team', 'rolling_starter_off_epa', 'rolling_starter_played']].rename(columns={
        'team': 'home_team',
        'rolling_starter_off_epa': 'home_rolling_starter_off_epa',
        'rolling_starter_played': 'home_rolling_starter_played',
    }),
    on=['game_id', 'home_team'], how='left'
)
schedules = schedules.merge(
    starter_epa[['game_id', 'team', 'rolling_starter_off_epa', 'rolling_starter_played']].rename(columns={
        'team': 'away_team',
        'rolling_starter_off_epa': 'away_rolling_starter_off_epa',
        'rolling_starter_played': 'away_rolling_starter_played',
    }),
    on=['game_id', 'away_team'], how='left'
)

# Positive = home team's starter has been better / more available recently
schedules['diff_starter_off_epa'] = (
    schedules['home_rolling_starter_off_epa'] - schedules['away_rolling_starter_off_epa']
)
schedules['diff_starter_played'] = (
    schedules['home_rolling_starter_played'] - schedules['away_rolling_starter_played']
)

# ── SBR line movement ─────────────────────────────────────────────────────────
print("\nLoading SBR opening line data...")
sbr = load_all_sbr(SBR_DIR)

if not sbr.empty:
    schedules = schedules.merge(
        sbr[['gameday', 'away_team', 'home_team', 'spread_open', 'spread_close_sbr']],
        on=['gameday', 'away_team', 'home_team'],
        how='left'
    )

    # Validate: spread_close_sbr should correlate tightly with nfl_data_py spread_line
    # If signs are flipped, negate both so convention is consistent
    check = schedules[['spread_line', 'spread_close_sbr']].dropna()
    corr = check.corr().iloc[0, 1]
    print(f"SBR close vs nfl_data_py spread_line correlation: {corr:.3f}")
    if corr < 0:
        print("  Signs flipped — negating SBR spreads to match nfl_data_py convention")
        schedules['spread_open']      = -schedules['spread_open']
        schedules['spread_close_sbr'] = -schedules['spread_close_sbr']

    # Positive line_movement = home became bigger favorite open → close
    # This captures sharp money direction: a big move signals informed bettors
    schedules['line_movement'] = schedules['spread_close_sbr'] - schedules['spread_open']
    n_lines = schedules['line_movement'].notna().sum()
    print(f"Games with line movement data: {n_lines} / {len(schedules)}")
else:
    schedules['line_movement'] = np.nan
    print("No SBR files found — line_movement will be excluded")

# ── Model ─────────────────────────────────────────────────────────────────────

# Features grouped by statistical significance from correlation analysis:
#
#   Significant (p<0.05):  diff_def_epa
#   Borderline (p<0.10):   diff_rolling_diff, large_favorite, temp
#   Not significant:       all others
#
# QB features (diff_starter_off_epa, diff_starter_played) were tested but excluded:
#   - drop 37.4% of games due to missing PBP passer data
#   - show no correlation with home_covered (p=0.977, p=0.274)
#   - not worth the sample size cost
_base_features = [
    'diff_def_epa',        # p=0.020 — only confirmed signal
    'diff_rolling_diff',   # p=0.053 — borderline
    'large_favorite',      # p=0.092 — borderline
    'temp',                # p=0.085 — borderline
    'home_is_favorite',    # p=0.565 — not significant, kept for completeness
    'diff_off_epa',        # p=0.636 — not significant, kept for completeness
    'rest_diff',           # p=0.710 — not significant, kept for completeness
    'wind',                # p=0.617 — not significant, kept for completeness
    'dome_team_outdoors',  # p=0.898 — not significant, kept for completeness
]

HAS_LINE_DATA = sbr is not None and not sbr.empty
FEATURES_WITHOUT_LINE = _base_features

model_data = schedules.dropna(subset=FEATURES_WITHOUT_LINE + ['home_covered']).copy()

# ── Walk-forward validation ───────────────────────────────────────────────────
# Train on all prior seasons, test on each season from 2014 onward.
# Returns predicted probabilities alongside predictions for backtesting.

def walk_forward(model_data, features, start=2014, end=2025):
    season_results = []
    all_preds = []

    for test_season in range(start, end):
        train = model_data[model_data['season'] < test_season]
        test  = model_data[model_data['season'] == test_season].copy()
        if len(train) < 100 or len(test) == 0:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[features])
        X_test  = scaler.transform(test[features])

        clf = LogisticRegression(C=0.01, max_iter=1000)
        clf.fit(X_train, train['home_covered'])

        test['pred']      = clf.predict(X_test)
        test['prob_home'] = clf.predict_proba(X_test)[:, 1]

        season_results.append({
            'season':   test_season,
            'games':    len(test),
            'accuracy': accuracy_score(test['home_covered'], test['pred']),
            'baseline': test['home_covered'].mean(),
        })
        all_preds.append(test)

    return pd.DataFrame(season_results), pd.concat(all_preds)

res, predictions = walk_forward(model_data, FEATURES_WITHOUT_LINE)

# ── Backtesting ───────────────────────────────────────────────────────────────
# Simulate betting using model predicted probabilities.
# Only bet when model confidence exceeds threshold (filters out near-50/50 games).
# Bet sizing: half-Kelly based on edge over the -110 juice breakeven (52.38%).
# Payout: risk $110 to win $100 → net payout ratio = 100/110 = 0.9091

JUICE          = 110        # standard spread bet: risk $110
WIN_PAYOUT     = 100        # to win $100
PAYOUT_RATIO   = WIN_PAYOUT / JUICE          # 0.9091
BREAKEVEN      = JUICE / (JUICE + WIN_PAYOUT) # 0.5238

def backtest(predictions, threshold=0.54, starting_bankroll=1000.0):
    bankroll = starting_bankroll
    season_rows = []

    for season, group in predictions.groupby('season'):
        bets, wins, wagered, profit = 0, 0, 0.0, 0.0

        for _, row in group.iterrows():
            prob = row['prob_home']
            # Flip to bet on whichever side the model favors
            bet_home = prob >= threshold
            bet_away = (1 - prob) >= threshold

            if not bet_home and not bet_away:
                continue

            p = prob if bet_home else (1 - prob)
            outcome_correct = (row['home_covered'] == 1) if bet_home else (row['home_covered'] == 0)

            # Half-Kelly bet sizing
            edge      = p - BREAKEVEN
            kelly     = edge / PAYOUT_RATIO
            half_kelly = max(kelly / 2, 0)
            bet_size  = round(half_kelly * bankroll, 2)

            if bet_size <= 0:
                continue

            bets    += 1
            wagered += bet_size

            if outcome_correct:
                wins    += 1
                bankroll += bet_size * PAYOUT_RATIO
                profit   += bet_size * PAYOUT_RATIO
            else:
                bankroll -= bet_size
                profit   -= bet_size

        win_rate = wins / bets if bets > 0 else float('nan')
        roi      = profit / wagered * 100 if wagered > 0 else float('nan')
        season_rows.append({
            'season':    season,
            'bets':      bets,
            'win_rate':  win_rate,
            'wagered':   round(wagered, 0),
            'profit':    round(profit, 0),
            'roi_%':     round(roi, 1),
            'bankroll':  round(bankroll, 0),
        })

    return pd.DataFrame(season_rows), bankroll

# Test multiple thresholds to find optimal confidence cutoff
print('Loading backtests across confidence thresholds...')
for thresh in [0.54, 0.56, 0.58, 0.60]:
    bt, final_bk = backtest(predictions, threshold=thresh)
    total_bets   = bt['bets'].sum()
    total_w      = bt['wagered'].sum()
    total_p      = bt['profit'].sum()
    roi          = total_p / total_w * 100 if total_w > 0 else 0
    avg_win      = (bt['win_rate'] * bt['bets']).sum() / total_bets if total_bets > 0 else 0
    print(f'  threshold={thresh:.0%}  bets={int(total_bets):4d}  '
          f'win%={avg_win:.1%}  ROI={roi:+.1f}%  final=${final_bk:,.0f}')

# Use 0.54 as primary for full breakdown table
bt_results, final_bankroll = backtest(predictions, threshold=0.54)

# ── Final model coefficients & signal strength ────────────────────────────────
train_final = model_data[model_data['season'] < 2024]
test_final  = model_data[model_data['season'] == 2024]

scaler_final = StandardScaler()
model_final  = LogisticRegression(C=0.01, max_iter=1000)
model_final.fit(scaler_final.fit_transform(train_final[FEATURES_WITHOUT_LINE]),
                train_final['home_covered'])

coef_df = pd.DataFrame({
    'feature':     FEATURES_WITHOUT_LINE,
    'coefficient': model_final.coef_[0],
}).sort_values('coefficient', ascending=False)

corr_rows = []
for f in FEATURES_WITHOUT_LINE:
    r, p = stats.pearsonr(model_data[f].fillna(0), model_data['home_covered'])
    corr_rows.append({'feature': f, 'r': round(r, 4), 'p': round(p, 3),
                      'significant': '*' if p < 0.05 else ''})
corr_df = pd.DataFrame(corr_rows).sort_values('p')

# ── Output ────────────────────────────────────────────────────────────────────
SEP = '─' * 60

print(f'\n{SEP}')
print('  NFL SPREAD MODEL  |  2010–2024  |  Logistic Regression')
print(SEP)

print(f'\n  Dataset')
print(f'  {"Total scheduled games:":30s} {len(schedules)}')
print(f'  {"Games used in model:":30s} {len(model_data)}')
print(f'  {"Dropped (missing features):":30s} {len(schedules) - len(model_data)}')
print(f'  {"  — week 1 rolling NaN:":30s} ~241  (no prior-season data)')
print(f'  {"  — QB features (excluded):":30s} 1458  (37% drop, p=0.977, not worth it)')
print(f'  {"  — team abbrev mismatches:":30s} fixed  (OAK/SD/STL → LV/LAC/LA)')

print(f'\n{SEP}')
print('  FEATURE SIGNAL')
print(SEP)
print(f'\n  {"Feature":<25}  {"r":>7}  {"p":>6}  {"sig":>4}')
print(f'  {"-"*25}  {"-"*7}  {"-"*6}  {"-"*4}')
for _, row in corr_df.iterrows():
    print(f'  {row["feature"]:<25}  {row["r"]:>+7.4f}  {row["p"]:>6.3f}  {row["significant"]:>4}')

print(f'\n{SEP}')
print('  WALK-FORWARD ACCURACY  (train on all prior seasons)')
print(SEP)
print(f'\n  {"Season":>7}  {"Games":>6}  {"Accuracy":>9}  {"Baseline":>9}  {"vs Baseline":>12}')
print(f'  {"─"*7}  {"─"*6}  {"─"*9}  {"─"*9}  {"─"*12}')
for _, row in res.iterrows():
    vs = row['accuracy'] - row['baseline']
    marker = ' ▲' if vs > 0 else ' ▼'
    print(f'  {int(row["season"]):>7}  {int(row["games"]):>6}  '
          f'{row["accuracy"]:>9.1%}  {row["baseline"]:>9.1%}  '
          f'{vs:>+11.1%}{marker}')
print(f'  {"─"*7}  {"─"*6}  {"─"*9}  {"─"*9}  {"─"*12}')
print(f'  {"MEAN":>7}  {"":>6}  {res["accuracy"].mean():>9.1%}  '
      f'{res["baseline"].mean():>9.1%}  '
      f'{(res["accuracy"] - res["baseline"]).mean():>+11.1%}')
print(f'  Seasons above baseline: {(res["accuracy"] > res["baseline"]).sum()} / {len(res)}')

print(f'\n{SEP}')
print(f'  BACKTESTING  |  threshold={0.54:.0%}  |  half-Kelly sizing  |  -110 juice')
print(SEP)
print(f'\n  {"Season":>7}  {"Bets":>5}  {"Win%":>6}  {"Wagered":>9}  {"Profit":>8}  {"ROI%":>6}  {"Bankroll":>10}')
print(f'  {"─"*7}  {"─"*5}  {"─"*6}  {"─"*9}  {"─"*8}  {"─"*6}  {"─"*10}')
for _, row in bt_results.iterrows():
    print(f'  {int(row["season"]):>7}  {int(row["bets"]):>5}  '
          f'{row["win_rate"]:>6.1%}  ${row["wagered"]:>8,.0f}  '
          f'{"+" if row["profit"]>=0 else ""}{row["profit"]:>7,.0f}  '
          f'{row["roi_%"]:>+6.1f}%  ${row["bankroll"]:>9,.0f}')
print(f'  {"─"*7}  {"─"*5}  {"─"*6}  {"─"*9}  {"─"*8}  {"─"*6}  {"─"*10}')
total_bets    = bt_results['bets'].sum()
total_wagered = bt_results['wagered'].sum()
total_profit  = bt_results['profit'].sum()
overall_roi   = total_profit / total_wagered * 100 if total_wagered > 0 else 0
print(f'  {"TOTAL":>7}  {int(total_bets):>5}  {"":>6}  '
      f'${total_wagered:>8,.0f}  {"+" if total_profit>=0 else ""}{total_profit:>7,.0f}  '
      f'{overall_roi:>+6.1f}%  ${final_bankroll:>9,.0f}')
print(f'\n  Starting bankroll: $1,000  →  Final: ${final_bankroll:,.0f}  '
      f'({"+" if final_bankroll>=1000 else ""}{final_bankroll-1000:,.0f})')
print(f'  Breakeven win rate at -110 juice: {BREAKEVEN:.1%}')

print(f'\n{SEP}')
print('  FINAL MODEL COEFFICIENTS  (trained 2010-2023, tested on 2024)')
print(SEP)
print(f'\n  2024 accuracy: {accuracy_score(test_final["home_covered"], model_final.predict(scaler_final.transform(test_final[FEATURES_WITHOUT_LINE]))):>6.1%}')
print(f'  2024 baseline: {test_final["home_covered"].mean():>6.1%}')
print(f'\n  {"Feature":<25}  {"Coefficient":>12}')
print(f'  {"-"*25}  {"-"*12}')
for _, row in coef_df.iterrows():
    print(f'  {row["feature"]:<25}  {row["coefficient"]:>+12.6f}')

print(f'\n{SEP}\n')
