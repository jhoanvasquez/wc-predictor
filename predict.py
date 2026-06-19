import sys
import difflib
import numpy as np
import pandas as pd
import joblib
from datetime import date
from pathlib import Path

MODEL_PATH   = Path("data_cache/model.pkl")
META_PATH    = Path("data_cache/model_meta.pkl")
ELO_PATH     = Path("data_cache/elo_ratings.csv")
MATCHES_PATH = Path("data_cache/matches_with_elo.csv")

TODAY = date.today()
THICK = "=" * 62
THIN  = "-" * 62


# ── Team name resolution ──────────────────────────────────────────────────────

def resolve_team(name: str, known: list) -> str:
    """
    Fuzzy-match a user-typed name against the list of known teams.
    Tries exact match, then substring, then difflib similarity.
    """
    # Exact (case-insensitive)
    for t in known:
        if t.lower() == name.lower():
            return t

    # Substring: "Congo" matches "DR Congo", "Iran" matches "IR Iran"
    sub = [t for t in known if name.lower() in t.lower() or t.lower() in name.lower()]
    sub.sort(key=len)   # prefer shorter match ("DR Congo" before "Republic of Congo")

    # Difflib similarity
    fuzzy = difflib.get_close_matches(name, known, n=3, cutoff=0.45)

    # Combine, dedup, preserve rank
    combined = list(dict.fromkeys(sub + fuzzy))

    if not combined:
        print(f"\n  No team found matching '{name}'.")
        print("  Check the spelling or try a longer name.")
        sys.exit(1)

    resolved = combined[0]
    if resolved.lower() != name.lower():
        print(f"  '{name}' -> '{resolved}'", end="")
        others = combined[1:3]
        if others:
            print(f"  (other options: {', '.join(others)})", end="")
        print()

    return resolved


# ── Form and H2H ─────────────────────────────────────────────────────────────

def team_form(matches: pd.DataFrame, team: str) -> dict:
    mask   = (matches["home_team"] == team) | (matches["away_team"] == team)
    played = matches[mask].sort_values("date")

    if played.empty:
        return {k: np.nan for k in
                ["win_rate_5", "win_rate_10", "gd_5", "goals_scored_5", "last_date"]}

    records = []
    for _, m in played.iterrows():
        is_home  = m["home_team"] == team
        scored   = m["home_score"] if is_home else m["away_score"]
        conceded = m["away_score"] if is_home else m["home_score"]
        won      = (m["result"] == "H" and is_home) or (m["result"] == "A" and not is_home)
        records.append({"win": int(won), "scored": scored, "conceded": conceded})

    r5  = records[-5:]
    r10 = records[-10:]

    return {
        "win_rate_5":      sum(r["win"] for r in r5)  / len(r5),
        "win_rate_10":     sum(r["win"] for r in r10) / len(r10),
        "gd_5":           sum(r["scored"] - r["conceded"] for r in r5),
        "goals_scored_5":  sum(r["scored"] for r in r5) / len(r5),
        "last_date":       played["date"].max(),
    }


def h2h_record(matches: pd.DataFrame, team_a: str, team_b: str) -> dict:
    mask = (
        ((matches["home_team"] == team_a) & (matches["away_team"] == team_b)) |
        ((matches["home_team"] == team_b) & (matches["away_team"] == team_a))
    )
    h2h = matches[mask]

    if h2h.empty:
        return {"win_rate": np.nan, "n": 0, "w": 0, "d": 0, "l": 0}

    w = d = 0
    for _, m in h2h.iterrows():
        res        = m["result"]
        a_is_home  = m["home_team"] == team_a
        if res == "D":
            d += 1
        elif (res == "H" and a_is_home) or (res == "A" and not a_is_home):
            w += 1

    l = len(h2h) - w - d
    return {"win_rate": w / len(h2h), "n": len(h2h), "w": w, "d": d, "l": l}


# ── Prediction ────────────────────────────────────────────────────────────────

def confidence_tag(prob: float) -> str:
    if prob >= 0.70: return "LOCK"
    if prob >= 0.60: return "CONFIDENT"
    if prob >= 0.52: return "SLIGHT EDGE"
    return "COIN FLIP"


def predict(team_a: str, team_b: str,
            model, le, feat_cols: list,
            elo_df: pd.DataFrame, matches: pd.DataFrame) -> None:

    elo_idx = elo_df.set_index("team")["elo"]
    elo_a   = elo_idx[team_a]
    elo_b   = elo_idx[team_b]

    fa  = team_form(matches, team_a)
    fb  = team_form(matches, team_b)
    h2h = h2h_record(matches, team_a, team_b)

    def days_since(last_date):
        if pd.isna(last_date):
            return np.nan
        return (TODAY - pd.Timestamp(last_date).date()).days

    rest_a = days_since(fa["last_date"])
    rest_b = days_since(fb["last_date"])
    rest_d = (rest_a - rest_b) if not (np.isnan(rest_a) or np.isnan(rest_b)) else np.nan

    row = {
        "elo_diff":            elo_a - elo_b,
        "home_elo":            elo_a,
        "away_elo":            elo_b,
        "neutral":             1,           # World Cup = neutral venue
        "home_win_rate_5":     fa["win_rate_5"],
        "home_win_rate_10":    fa["win_rate_10"],
        "home_gd_5":           fa["gd_5"],
        "home_goals_scored_5": fa["goals_scored_5"],
        "away_win_rate_5":     fb["win_rate_5"],
        "away_win_rate_10":    fb["win_rate_10"],
        "away_gd_5":           fb["gd_5"],
        "away_goals_scored_5": fb["goals_scored_5"],
        "home_rest_days":      rest_a,
        "away_rest_days":      rest_b,
        "rest_diff":           rest_d,
        "h2h_home_win_rate":   h2h["win_rate"],
        "h2h_matches":         h2h["n"],
    }

    X         = pd.DataFrame([row])[feat_cols]
    probs_raw = model.predict_proba(X)[0]

    # le.classes_ = ["A", "D", "H"]  ->  probs_raw indices 0, 1, 2
    p = {cls: probs_raw[i] for i, cls in enumerate(le.classes_)}
    p_a, p_d, p_b = p["H"], p["D"], p["A"]

    pick      = team_a if p_a >= p_b else team_b
    pick_prob = max(p_a, p_b)
    tag       = confidence_tag(pick_prob)

    # ── Print ─────────────────────────────────────────────────────────────────
    print()
    print(THICK)
    print(f"  {team_a}  vs  {team_b}")
    print(f"  2026 FIFA World Cup  |  neutral venue")
    print(THICK)
    print(f"  Elo  :  {team_a} {elo_a:.0f}  vs  {team_b} {elo_b:.0f}  (diff {elo_a-elo_b:+.0f})")
    if h2h["n"] > 0:
        print(f"  H2H  :  {h2h['w']}W {h2h['d']}D {h2h['l']}L for {team_a} in {h2h['n']} prior meetings")
    else:
        print(f"  H2H  :  no prior meetings in dataset")
    print(THIN)
    print(f"  {team_a:<24}  win    {p_a:>6.1%}")
    print(f"  {'Draw':<24}         {p_d:>6.1%}")
    print(f"  {team_b:<24}  win    {p_b:>6.1%}")
    print(THIN)
    print(f"  PICK: {pick}  ({pick_prob:.1%})   [{tag}]")
    print(THICK)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print()
        print("  Usage: python predict.py \"Team A\" \"Team B\"")
        print("  Example: python predict.py \"Colombia\" \"DR Congo\"")
        sys.exit(1)

    model     = joblib.load(MODEL_PATH)
    meta      = joblib.load(META_PATH)
    le        = meta["label_encoder"]
    feat_cols = meta["feature_cols"]

    elo_df  = pd.read_csv(ELO_PATH)
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])

    known = elo_df["team"].tolist()
    team_a = resolve_team(sys.argv[1], known)
    team_b = resolve_team(sys.argv[2], known)

    if team_a == team_b:
        print("  ERROR: both names resolved to the same team.")
        sys.exit(1)

    predict(team_a, team_b, model, le, feat_cols, elo_df, matches)


if __name__ == "__main__":
    main()
