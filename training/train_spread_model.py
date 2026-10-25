"""
Spread model v1 - train, compare with baselines, pick a cost-based alert threshold.
Team C2 - Project 3 (unofficial IndiGo social listening prototype; synthetic data)

Task: 60 minutes after a post, predict whether it reaches 200+ reshares in 24 h.
Crisis alert = negative sentiment AND predicted spread.
Every alert threshold (model AND baselines) is chosen on the VALIDATION set by
minimising expected cost, then reported once on the untouched TEST set.

Run from the repo root:  python training/train_spread_model.py   (after the generator)
Out:  models/spread_model_v1.joblib, models/metrics_v1.json, eval/test_predictions.csv
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
COST_FALSE_ALARM = 500      # Rs: ~20 min of a PR analyst reviewing a non-crisis (assumption)
COST_MISSED_CRISIS = 25000  # Rs: incremental cost of a crisis caught late (assumption)

# Only information available 60 minutes after posting. hidden_*, reshares_24h,
# spread and crisis are NEVER features (that would be leakage).
CATEGORICAL = ["language", "topic", "sentiment", "account_type"]
LOG_NUMERIC = ["followers", "early_reshares_60m", "early_replies_60m"]
PLAIN_NUMERIC = ["urgency_level", "hour_sin", "hour_cos", "verified", "has_media", "sarcasm"]
FEATURES = CATEGORICAL + LOG_NUMERIC + PLAIN_NUMERIC


def add_features(df):
    df = df.copy()
    df["urgency_level"] = df["urgency"].map({"low": 0, "medium": 1, "high": 2})
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("log", Pipeline([("log1p", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]), LOG_NUMERIC),
        ("num", StandardScaler(), PLAIN_NUMERIC),
    ])


def evaluate(alert, crisis):
    tp = int((alert & crisis).sum()); fp = int((alert & ~crisis).sum()); fn = int((~alert & crisis).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    cost = fp * COST_FALSE_ALARM + fn * COST_MISSED_CRISIS
    return {"alerts": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3), "recall": round(recall, 3),
            "cost_per_1000_mentions_rs": round(cost / len(alert) * 1000)}


def best_threshold(score, neg, crisis, candidates):
    """Pick the score threshold with the lowest expected cost on validation data."""
    costs = [evaluate(neg & (score >= t), crisis)["cost_per_1000_mentions_rs"] for t in candidates]
    return candidates[int(np.argmin(costs))]


def main():
    df = add_features(pd.read_csv(ROOT / "data" / "synthetic_mentions_v1.csv"))
    train, rest = train_test_split(df, test_size=0.30, stratify=df["spread"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["spread"], random_state=SEED)
    sets = {name: (d, (d["sentiment"] == "negative").to_numpy(), d["crisis"].astype(bool).to_numpy())
            for name, d in [("val", val), ("test", test)]}

    models = {
        "Logistic regression (v1)": LogisticRegression(class_weight="balanced", max_iter=2000),
        "Gradient boosting (v1b)": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    }
    fitted = {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor()), ("clf", clf)]).fit(train[FEATURES], train["spread"])
        fitted[name] = pipe

    # Each method = a score per post; alert when negative AND score >= threshold
    scorers = {
        "Follower-count rule": lambda d: d["followers"].to_numpy().astype(float),
        "Early-reshare rule": lambda d: d["early_reshares_60m"].to_numpy().astype(float),
        **{name: (lambda d, p=pipe: p.predict_proba(d[FEATURES])[:, 1]) for name, pipe in fitted.items()},
    }
    grids = {
        "Follower-count rule": np.array([1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5]),
        "Early-reshare rule": np.arange(1, 31, dtype=float),
    }

    results = {}
    vd, vneg, vcrisis = sets["val"]
    td, tneg, tcrisis = sets["test"]
    results["Never alert"] = {**evaluate(np.zeros(len(td), bool), tcrisis), "threshold": None, "pr_auc": None}
    for name, scorer in scorers.items():
        vscore, tscore = scorer(vd), scorer(td)
        grid = grids.get(name, np.unique(np.quantile(vscore[vneg], np.linspace(0.5, 0.995, 200))))
        thr = best_threshold(vscore, vneg, vcrisis, grid)
        res = evaluate(tneg & (tscore >= thr), tcrisis)
        ranking = np.where(tneg, tscore, -np.inf)
        ranking = np.where(np.isinf(ranking), np.nanmin(tscore) - 1, ranking)
        res["pr_auc"] = round(float(average_precision_score(tcrisis, ranking)), 3)
        res["threshold"] = round(float(thr), 4)
        results[name] = res

    # Bootstrap 95% intervals for the v1 model's test precision and recall
    rng = np.random.default_rng(SEED)
    lr = fitted["Logistic regression (v1)"]
    alert_lr = tneg & (lr.predict_proba(td[FEATURES])[:, 1] >= results["Logistic regression (v1)"]["threshold"])
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(td), len(td))
        e = evaluate(alert_lr[idx], tcrisis[idx]); boots.append((e["precision"], e["recall"]))
    boots = np.array(boots)
    ci = {"precision_95ci": np.percentile(boots[:, 0], [2.5, 97.5]).round(3).tolist(),
          "recall_95ci": np.percentile(boots[:, 1], [2.5, 97.5]).round(3).tolist()}

    summary = {
        "data": {"rows": len(df), "train": len(train), "val": len(val), "test": len(test),
                 "spread_rate": round(df["spread"].mean(), 3), "crisis_rate": round(df["crisis"].mean(), 3),
                 "test_crises": int(tcrisis.sum())},
        "costs_rs": {"false_alarm": COST_FALSE_ALARM, "missed_crisis": COST_MISSED_CRISIS},
        "results_test": results, "logreg_bootstrap": ci,
    }
    with open(ROOT / "models" / "metrics_v1.json", "w") as f:
        json.dump(summary, f, indent=2)
    joblib.dump(lr, ROOT / "models" / "spread_model_v1.joblib")
    # Test-set scores for the app's threshold slider (Model results tab)
    pd.DataFrame({
        "p_spread": lr.predict_proba(td[FEATURES])[:, 1].round(5),
        "negative": tneg.astype(int), "crisis": tcrisis.astype(int),
        "followers": td["followers"].to_numpy(), "early_reshares_60m": td["early_reshares_60m"].to_numpy(),
    }).to_csv(ROOT / "eval" / "test_predictions.csv", index=False)

    print(json.dumps(summary["data"]), "\n")
    print(pd.DataFrame(results).T[["threshold", "alerts", "tp", "fp", "fn", "precision", "recall",
                                    "pr_auc", "cost_per_1000_mentions_rs"]].to_string())
    print("\nLogReg bootstrap:", ci)


if __name__ == "__main__":
    main()
