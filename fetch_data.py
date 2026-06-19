import sys
import requests
import pandas as pd
from pathlib import Path

DATA_URL   = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
DATA_DIR   = Path("data_cache")
RAW_PATH   = DATA_DIR / "results.csv"
CLEAN_PATH = DATA_DIR / "results_clean.csv"

DIVIDER = "-" * 52


def fetch():
    DATA_DIR.mkdir(exist_ok=True)
    if RAW_PATH.exists():
        print(f"cache hit  ->  {RAW_PATH}")
        return
    print("downloading results (~5 MB) ...")
    r = requests.get(DATA_URL, timeout=60)
    r.raise_for_status()
    RAW_PATH.write_bytes(r.content)
    print(f"saved  ->  {RAW_PATH}")


def load() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH, parse_dates=["date"])


def describe(df: pd.DataFrame):
    # Schema
    print(f"\n{DIVIDER}\n SCHEMA\n{DIVIDER}")
    print(df.dtypes.to_string())

    # Overview
    print(f"\n{DIVIDER}\n OVERVIEW\n{DIVIDER}")
    print(f"  rows      : {len(df):,}")
    print(f"  date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  teams     : {df['home_team'].nunique()} unique")

    # Sample row — helps you see what one record looks like
    print(f"\n{DIVIDER}\n SAMPLE ROW\n{DIVIDER}")
    sample = df.sample(1, random_state=42).iloc[0]
    for col in df.columns:
        print(f"  {col:<18}: {sample[col]}")

    # Tournament breakdown — you'll use this to assign match importance weights
    print(f"\n{DIVIDER}\n TOP TOURNAMENT TYPES\n{DIVIDER}")
    for name, count in df["tournament"].value_counts().head(10).items():
        print(f"  {count:>6,}  {name}")

    # Outcome distribution — this is your baseline to beat
    total     = len(df)
    home_wins = (df["home_score"] > df["away_score"]).sum()
    draws     = (df["home_score"] == df["away_score"]).sum()
    away_wins = (df["home_score"] < df["away_score"]).sum()
    print(f"\n{DIVIDER}\n OUTCOME DISTRIBUTION (full history)\n{DIVIDER}")
    print(f"  home win  : {home_wins:>6,}  ({home_wins / total * 100:.1f}%)")
    print(f"  draw      : {draws:>6,}  ({draws  / total * 100:.1f}%)")
    print(f"  away win  : {away_wins:>6,}  ({away_wins / total * 100:.1f}%)")
    print()
    print("  * home advantage is real -- your model needs to account for it.")
    print("  * draws are ~25% of all matches -- harder to predict than they look.")


def assign_importance(df: pd.DataFrame) -> pd.DataFrame:
    # Importance controls how much a match swings Elo ratings.
    # A World Cup final should move ratings more than a friendly.
    # These weights are a first guess — you'll tune them in Phase 2.
    tier = {
        "FIFA World Cup":                 1.00,
        "UEFA Euro":                      0.90,
        "Copa América":                   0.90,
        "Africa Cup of Nations":          0.85,
        "AFC Asian Cup":                  0.85,
        "CONCACAF Gold Cup":              0.85,
        "Oceania Nations Cup":            0.80,
        "FIFA World Cup qualification":   0.75,
        "UEFA Nations League":            0.70,
        "Friendly":                       0.40,
    }
    df["importance"] = df["tournament"].map(tier).fillna(0.60)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 2006 gives Elo enough warm-up time (~20 years of matches) without
    # pulling in data so old it no longer reflects current team quality.
    df = df[df["date"] >= "2006-01-01"].copy()
    df = assign_importance(df)
    df = df.reset_index(drop=True)
    df.to_csv(CLEAN_PATH, index=False)

    print(f"\n{DIVIDER}\n AFTER FILTER\n{DIVIDER}")
    print(f"  kept  : {len(df):,} matches (2006 -> today)")
    print(f"  saved : {CLEAN_PATH}")
    return df


def main():
    fetch()
    df = load()
    describe(df)
    clean(df)

    print(f"\n{DIVIDER}")
    print(" PHASE 1 DONE")
    print(DIVIDER)
    print("  Next: python elo.py")
    print(DIVIDER)


if __name__ == "__main__":
    main()
