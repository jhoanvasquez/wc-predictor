import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.preprocessing import LabelEncoder

FEATURES_PATH = Path("data_cache/features.csv")
MODEL_PATH    = Path("data_cache/model.pkl")
META_PATH     = Path("data_cache/model_meta.pkl")

# Everything before this date trains the model.
# 2022-2024 is the held-out set — the model never sees it during training.
TRAIN_CUTOFF = "2022-01-01"
TEST_CUTOFF  = "2025-01-01"

DIVIDER = "-" * 52

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


# ── Baselines ─────────────────────────────────────────────────────────────────

def majority_class_accuracy(y_true: np.ndarray) -> float:
    """Accuracy of always predicting the most common class."""
    counts = np.bincount(y_true)
    return counts.max() / len(y_true)


def prior_logloss(y_train: np.ndarray, y_test: np.ndarray, n_classes: int) -> float:
    """
    Log-loss when always predicting the training-set class frequencies.
    This is a stronger baseline than uniform — it knows the priors.
    """
    priors = np.bincount(y_train, minlength=n_classes) / len(y_train)
    priors = np.clip(priors, 1e-7, 1 - 1e-7)
    y_pred = np.tile(priors, (len(y_test), 1))
    return log_loss(y_test, y_pred)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_metrics(y_test, y_pred_class, y_pred_prob, y_train, le):
    acc      = accuracy_score(y_test, y_pred_class)
    ll       = log_loss(y_test, y_pred_prob)
    base_acc = majority_class_accuracy(y_test)
    base_ll  = prior_logloss(y_train, y_test, n_classes=len(le.classes_))

    print(f"\n{DIVIDER}")
    print(" EVALUATION  (held-out test: 2022-2024)")
    print(DIVIDER)
    print()
    print(f"  {'metric':<14} {'model':>8}  {'baseline':>10}  {'edge':>8}")
    print(f"  {'-'*14} {'-'*8}  {'-'*10}  {'-'*8}")
    print(f"  {'accuracy':<14} {acc:>8.1%}  {base_acc:>10.1%}  {acc-base_acc:>+7.1%}")
    print(f"  {'log-loss':<14} {ll:>8.3f}  {base_ll:>10.3f}  {ll-base_ll:>+7.3f}")
    print()
    print("  log-loss: lower is better.")
    print("  baseline: always predict training-set class frequencies.")


def print_confusion(y_test, y_pred_class, le):
    cm      = confusion_matrix(y_test, y_pred_class)
    classes = le.classes_   # ["A", "D", "H"]

    print(f"\n{DIVIDER}")
    print(" CONFUSION MATRIX  (rows = actual, cols = predicted)")
    print(DIVIDER)
    print()
    header = "  ".join(f"{c:>6}" for c in classes)
    print(f"            {header}    correct")
    print(f"  {'-'*50}")
    for i, cls in enumerate(classes):
        row      = "  ".join(f"{cm[i, j]:>6,}" for j in range(len(classes)))
        total    = cm[i].sum()
        correct  = cm[i, i]
        pct      = correct / total * 100
        label    = {"A": "away win", "D": "draw   ", "H": "home win"}[cls]
        flag     = "  <- hard" if cls == "D" else ""
        print(f"  {label}  {row}    {pct:.0f}%{flag}")
    print()
    print("  Draws are consistently mispredicted -- that's a known weakness")
    print("  of win/draw/loss models on low-scoring sports.")


def print_feature_importance(model):
    pairs = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    max_imp = pairs[0][1]

    print(f"\n{DIVIDER}")
    print(" FEATURE IMPORTANCE  (XGBoost gain)")
    print(DIVIDER)
    print()
    for col, imp in pairs:
        bar = "#" * int(imp / max_imp * 32)
        print(f"  {col:<25} {imp:.4f}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{DIVIDER}")
    print(" LOADING")
    print(DIVIDER)
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])

    train_df = df[df["date"] <  TRAIN_CUTOFF]
    test_df  = df[(df["date"] >= TRAIN_CUTOFF) & (df["date"] < TEST_CUTOFF)]

    print(f"  train : {len(train_df):,} matches  (2006 -> 2021)")
    print(f"  test  : {len(test_df):,}  matches  (2022 -> 2024)")

    # Encode target. Fix alphabetical order so classes are always A=0, D=1, H=2.
    le = LabelEncoder()
    le.fit(["A", "D", "H"])

    y_train = le.transform(train_df["result"])
    y_test  = le.transform(test_df["result"])
    X_train = train_df[FEATURE_COLS]
    X_test  = test_df[FEATURE_COLS]

    print(f"\n{DIVIDER}")
    print(" TRAINING")
    print(DIVIDER)
    print("  XGBClassifier  300 trees, lr=0.05, depth=4 ...")

    model = XGBClassifier(
        objective        = "multi:softprob",
        num_class        = 3,
        n_estimators     = 300,
        learning_rate    = 0.05,
        max_depth        = 4,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        eval_metric      = "mlogloss",
        random_state     = 42,
        verbosity        = 0,
    )
    model.fit(X_train, y_train)
    print("  done.")

    y_pred_prob  = model.predict_proba(X_test)
    y_pred_class = model.predict(X_test)

    print_metrics(y_test, y_pred_class, y_pred_prob, y_train, le)
    print_confusion(y_test, y_pred_class, le)
    print_feature_importance(model)

    # Save model + metadata needed by the CLI in Phase 6
    joblib.dump(model, MODEL_PATH)
    joblib.dump({"label_encoder": le, "feature_cols": FEATURE_COLS}, META_PATH)

    print(f"\n{DIVIDER}")
    print(" PHASE 4 DONE")
    print(DIVIDER)
    print(f"  model.pkl       -> trained XGBClassifier")
    print(f"  model_meta.pkl  -> label encoder + feature list")
    print("  Next: python predict.py \"Spain\" \"Morocco\"")
    print(DIVIDER)


if __name__ == "__main__":
    main()
