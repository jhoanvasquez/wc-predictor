from collections import defaultdict
import numpy as np
import pandas as pd
from pathlib import Path

MATCHES_PATH  = Path("data_cache/matches_with_elo.csv")
FEATURES_PATH = Path("data_cache/features.csv")

DIVIDER = "-" * 52

# These are the columns the model will actually train on (not date/team names).
FEATURE_COLS = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "neutral",
    "home_win_rate_5",
    "home_win_rate_10",
    "home_gd_5",
    "home_goals_scored_5",
    "away_win_rate_5",
    "away_win_rate_10",
    "away_gd_5",
    "away_goals_scored_5",
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "h2h_home_win_rate",
    "h2h_matches",
]


# ── Feature helpers ───────────────────────────────────────────────────────────

def form_stats(history: list, n: int) -> dict:
    """Win rate and goal numbers over the last N matches in a team's history."""
    recent = history[-n:]
    if not recent:
        return {"win_rate": np.nan, "gd": np.nan, "scored": np.nan}
    wins   = sum(1 for m in recent if m["result"] == "W")
    gd     = sum(m["scored"] - m["conceded"] for m in recent)
    scored = sum(m["scored"] for m in recent) / len(recent)
    return {"win_rate": wins / len(recent), "gd": gd, "scored": scored}


# ── Main loop ─────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    team_hist = defaultdict(list)   # team -> [{result, scored, conceded}, ...]
    last_date = {}                  # team -> date of their most recent match
    h2h       = defaultdict(list)   # sorted(team_a, team_b) -> [winner_team | None, ...]

    rows = []

    for _, m in df.iterrows():
        home = m["home_team"]
        away = m["away_team"]
        date = m["date"]
        res  = m["result"]   # "H", "D", "A"  (from home team's perspective)

        # ── Compute features using state BEFORE this match ────────────────
        hf5  = form_stats(team_hist[home], 5)
        hf10 = form_stats(team_hist[home], 10)
        af5  = form_stats(team_hist[away], 5)
        af10 = form_stats(team_hist[away], 10)

        home_rest = (date - last_date[home]).days if home in last_date else np.nan
        away_rest = (date - last_date[away]).days if away in last_date else np.nan

        pair     = tuple(sorted([home, away]))
        prior    = h2h[pair]
        h2h_n    = len(prior)
        # Store the winning team name (not "H"/"A") so the lookup is order-independent.
        h2h_home_wins = sum(1 for winner in prior if winner == home)

        rows.append({
            # identifiers — kept for debugging, not fed to the model
            "date":       date,
            "home_team":  home,
            "away_team":  away,
            "tournament": m["tournament"],

            # target label
            "result": res,

            # Elo
            "elo_diff":  m["elo_diff"],
            "home_elo":  m["home_elo"],
            "away_elo":  m["away_elo"],
            "neutral":   int(m["neutral"]),

            # form — home
            "home_win_rate_5":     hf5["win_rate"],
            "home_win_rate_10":    hf10["win_rate"],
            "home_gd_5":           hf5["gd"],
            "home_goals_scored_5": hf5["scored"],

            # form — away
            "away_win_rate_5":     af5["win_rate"],
            "away_win_rate_10":    af10["win_rate"],
            "away_gd_5":           af5["gd"],
            "away_goals_scored_5": af5["scored"],

            # rest
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_diff":      home_rest - away_rest,  # NaN propagates if either is NaN

            # head-to-head
            "h2h_home_win_rate": h2h_home_wins / h2h_n if h2h_n > 0 else np.nan,
            "h2h_matches":       h2h_n,
        })

        # ── Update state AFTER recording the row ──────────────────────────
        team_hist[home].append({
            "result":   "W" if res == "H" else ("D" if res == "D" else "L"),
            "scored":   m["home_score"],
            "conceded": m["away_score"],
        })
        team_hist[away].append({
            "result":   "W" if res == "A" else ("D" if res == "D" else "L"),
            "scored":   m["away_score"],
            "conceded": m["home_score"],
        })

        last_date[home] = date
        last_date[away] = date

        # Record the winner by name — makes H2H lookup order-independent
        if res == "H":
            winner = home
        elif res == "A":
            winner = away
        else:
            winner = None   # draw
        h2h[pair].append(winner)

    return pd.DataFrame(rows)


# ── Analysis ──────────────────────────────────────────────────────────────────

def show_correlations(df: pd.DataFrame):
    # Encode result as a number so we can compute Pearson correlation.
    # +1 = home win, 0 = draw, -1 = away win.
    outcome = df["result"].map({"H": 1, "D": 0, "A": -1})

    corrs = []
    for col in FEATURE_COLS:
        if col in df.columns and df[col].notna().sum() > 200:
            r = df[col].corr(outcome)
            if not np.isnan(r):
                corrs.append((col, r))

    corrs.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{DIVIDER}")
    print(" FEATURE CORRELATIONS WITH OUTCOME")
    print(f"{DIVIDER}")
    print("  (+) favors home win  |  (-) favors away win")
    print()
    for col, r in corrs:
        bar  = "#" * int(abs(r) * 40)
        sign = "+" if r >= 0 else "-"
        print(f"  {col:<25} {sign}{abs(r):.3f}  {bar}")


def show_missing(df: pd.DataFrame):
    print(f"\n{DIVIDER}")
    print(" MISSING VALUES")
    print(f"{DIVIDER}")
    print("  NaN appears where a team has no history yet (first few matches).")
    print()
    any_missing = False
    for col in FEATURE_COLS:
        n = df[col].isna().sum()
        if n > 0:
            any_missing = True
            pct = n / len(df) * 100
            print(f"  {col:<25} {n:>5,} NaN  ({pct:.1f}%)")
    if not any_missing:
        print("  none")
    print()
    print("  XGBoost handles NaN natively -- no imputation needed.")


def main():
    print(f"\n{DIVIDER}")
    print(" LOADING")
    print(DIVIDER)
    df = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  {len(df):,} matches")

    print(f"\n{DIVIDER}")
    print(" BUILDING FEATURES")
    print(DIVIDER)
    print("  processing ...")
    features = build_features(df)
    print(f"  done: {len(features):,} rows x {len(FEATURE_COLS)} features")

    show_correlations(features)
    show_missing(features)

    features.to_csv(FEATURES_PATH, index=False)

    print(f"\n{DIVIDER}")
    print(" PHASE 3 DONE")
    print(DIVIDER)
    print(f"  features.csv  ->  {len(features):,} rows, {len(FEATURE_COLS)} feature columns")
    print("  Next: python train.py")
    print(DIVIDER)


if __name__ == "__main__":
    main()
