import numpy as np
import pandas as pd
from pathlib import Path

CLEAN_PATH   = Path("data_cache/results_clean.csv")
ELO_PATH     = Path("data_cache/elo_ratings.csv")
MATCHES_PATH = Path("data_cache/matches_with_elo.csv")

INITIAL_ELO = 1500  # every new team starts here
K           = 40    # how aggressively ratings move after each match
HOME_ADV    = 100   # Elo points added to home team's effective rating

DIVIDER = "-" * 52


# ── Core Elo math ────────────────────────────────────────────────────────────

def expected_score(elo_a: float, elo_b: float) -> float:
    """
    Probability team A beats team B based on rating gap.
    A 200-point gap -> ~75% expected win rate for A.
    A 400-point gap -> ~91% expected win rate for A.
    """
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def result_to_score(home_goals: int, away_goals: int) -> float:
    """Win=1.0, Draw=0.5, Loss=0.0 (from home team's perspective)."""
    if home_goals > away_goals:
        return 1.0
    if home_goals == away_goals:
        return 0.5
    return 0.0


def goal_weight(goal_diff: int) -> float:
    """
    Winning 4-0 is more informative than winning 1-0.
    The weight scales the K-factor so blowouts move ratings further.
    """
    if goal_diff <= 1:
        return 1.00
    if goal_diff == 2:
        return 1.50
    return 1.75


# ── Engine ─────────────────────────────────────────────────────────  

def run_elo(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    ratings: dict[str, float] = {}
    rows = []

    for _, match in df.iterrows():
        home      = match["home_team"]
        away      = match["away_team"]
        neutral   = match["neutral"]
        imp       = match["importance"]
        h_goals   = int(match["home_score"])
        a_goals   = int(match["away_score"])

        home_elo = ratings.get(home, INITIAL_ELO)
        away_elo = ratings.get(away, INITIAL_ELO)

        # Home teams get a rating boost when computing expected score.
        # On neutral ground there is no boost — both teams are treated equally.
        home_elo_adj = home_elo + (0 if neutral else HOME_ADV)

        exp_home = expected_score(home_elo_adj, away_elo)
        exp_away = 1 - exp_home

        act_home = result_to_score(h_goals, a_goals)
        act_away = 1 - act_home

        gw   = goal_weight(abs(h_goals - a_goals))
        k_eff = K * imp * gw   # final update magnitude for this match

        # Store pre-match ratings — the model will only ever see
        # what was known BEFORE the match was played.
        rows.append({
            "date":        match["date"],
            "home_team":   home,
            "away_team":   away,
            "home_score":  h_goals,
            "away_score":  a_goals,
            "tournament":  match["tournament"],
            "neutral":     neutral,
            "importance":  imp,
            "home_elo":    home_elo,
            "away_elo":    away_elo,
            "elo_diff":    home_elo - away_elo,   # key feature for the model
            "result":      "H" if act_home == 1 else ("D" if act_home == 0.5 else "A"),
        })

        # Update both teams after the match
        ratings[home] = home_elo + k_eff * (act_home - exp_home)
        ratings[away] = away_elo + k_eff * (act_away - exp_away)

    return ratings, pd.DataFrame(rows)


# ── Validation ───────────────────────────────────────────────────────────────

def calibration_check(matches: pd.DataFrame):
    """
    If Elo is working, teams with a 70% expected win rate should win
    roughly 70% of the time. This table checks that.
    """
    matches = matches.copy()

    # Expected home win probability using pre-match ratings + home advantage
    matches["exp_home_win"] = matches.apply(
        lambda r: expected_score(
            r["home_elo"] + (0 if r["neutral"] else HOME_ADV),
            r["away_elo"]
        ),
        axis=1,
    )

    # actual_score uses 1.0/0.5/0.0 to match what Elo's expected_score represents.
    # Elo expected score = P(win) + 0.5*P(draw) — not just P(win).
    # Comparing against wins-only would always look underpredicted.
    matches["actual_score"] = matches["result"].map({"H": 1.0, "D": 0.5, "A": 0.0})

    bins   = [0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]
    labels = ["< 40%", "40-50%", "50-60%", "60-70%", "70-80%", "> 80%"]
    matches["bucket"] = pd.cut(matches["exp_home_win"], bins=bins, labels=labels, right=False)

    table = (
        matches.groupby("bucket", observed=True)["actual_score"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "actual_score_rate", "count": "matches"})
    )

    print(f"\n{DIVIDER}")
    print(" CALIBRATION CHECK")
    print(f"{DIVIDER}")
    print("  Expected score = P(win) + 0.5*P(draw). Actual score matches that scale.")
    print()
    print(f"  {'expected':>12}  {'actual score':>12}  {'matches':>8}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*8}")
    for bucket, row in table.iterrows():
        print(f"  {str(bucket):>12}  {row['actual_score_rate']:>12.1%}  {int(row['matches']):>8,}")


# ── Output ───────────────────────────────────────────────────────────────────

def print_top_teams(ratings: dict, n: int = 20):
    ranked = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{DIVIDER}")
    print(f" TOP {n} TEAMS (current Elo ratings)")
    print(f"{DIVIDER}")
    for rank, (team, elo) in enumerate(ranked[:n], 1):
        bar = "#" * int((elo - 1400) / 20)
        print(f"  {rank:>2}.  {team:<22} {elo:>6.0f}  {bar}")


def main():
    print(f"\n{DIVIDER}")
    print(" LOADING DATA")
    print(DIVIDER)
    df = pd.read_csv(CLEAN_PATH, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  {len(df):,} matches, sorted chronologically")
    print(f"  {df['date'].min().date()} -> {df['date'].max().date()}")

    print(f"\n{DIVIDER}")
    print(" RUNNING ELO ENGINE")
    print(DIVIDER)
    print("  processing ...")
    ratings, matches = run_elo(df)
    print(f"  done. {len(ratings)} teams rated.")

    print_top_teams(ratings)
    calibration_check(matches)

    # Save
    rating_df = (
        pd.DataFrame(ratings.items(), columns=["team", "elo"])
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    rating_df.to_csv(ELO_PATH, index=False)
    matches.to_csv(MATCHES_PATH, index=False)

    print(f"\n{DIVIDER}")
    print(" PHASE 2 DONE")
    print(DIVIDER)
    print(f"  elo_ratings.csv     -> {len(rating_df)} teams")
    print(f"  matches_with_elo.csv -> {len(matches):,} rows")
    print("  Next: python features.py")
    print(DIVIDER)


if __name__ == "__main__":
    main()
