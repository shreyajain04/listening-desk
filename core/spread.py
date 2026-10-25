"""Scores how likely a post is to reach 200+ reshares in 24 hours, 60 minutes after posting.
Model: logistic regression v1 from training/train_spread_model.py (synthetic data)."""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from core.config import MODELS

CATEGORICAL = ["language", "topic", "sentiment", "account_type"]
LOG_NUMERIC = ["followers", "early_reshares_60m", "early_replies_60m"]
PLAIN_NUMERIC = ["urgency_level", "hour_sin", "hour_cos", "verified", "has_media", "sarcasm"]
FEATURES = CATEGORICAL + LOG_NUMERIC + PLAIN_NUMERIC

FRIENDLY = {
    "followers": "Follower count", "early_reshares_60m": "Reshares in the first hour",
    "early_replies_60m": "Replies in the first hour", "urgency_level": "Urgency",
    "verified": "Verified account", "has_media": "Has a photo or video", "sarcasm": "Sarcasm",
    "sentiment_negative": "Negative tone", "sentiment_positive": "Positive tone",
    "sentiment_neutral": "Neutral tone", "account_type_influencer": "Influencer account",
    "account_type_journalist": "Journalist account", "account_type_regular": "Regular account",
}


def _friendly(name: str) -> str:
    if name in FRIENDLY:
        return FRIENDLY[name]
    if name.startswith("topic_"):
        return "Topic: " + name[6:].replace("_", " ")
    if name.startswith("language_"):
        return "Language: " + name[9:]
    return name.replace("_", " ").capitalize()


def calibrate(p_balanced, base_rate: float):
    """The model was trained with balanced class weights, which inflates its scores.
    Prior correction turns them back into real chances: true odds = weighted odds * base_rate / (1 - base_rate)."""
    r = base_rate / (1 - base_rate)
    p_balanced = np.clip(p_balanced, 1e-9, 1 - 1e-9)
    return p_balanced * r / (p_balanced * r + (1 - p_balanced))


@lru_cache(maxsize=1)
def load():
    """Return (model, alert threshold as a real chance, metrics, base rate)."""
    model = joblib.load(MODELS / "spread_model_v1.joblib")
    metrics = json.loads((MODELS / "metrics_v1.json").read_text())
    base_rate = metrics["data"]["spread_rate"]
    raw = metrics["results_test"]["Logistic regression (v1)"]["threshold"]
    return model, float(calibrate(raw, base_rate)), metrics, base_rate


def _frame(post: dict, label) -> pd.DataFrame:
    row = {
        "language": label.language if label.language in ("english", "hinglish") else "english",
        "topic": label.topic, "sentiment": label.sentiment,
        "account_type": post.get("account_type", "regular"),
        "followers": float(post.get("followers", 0)),
        "early_reshares_60m": float(post.get("early_reshares_60m", 0)),
        "early_replies_60m": float(post.get("early_replies_60m", 0)),
        "urgency_level": {"low": 0, "medium": 1, "high": 2}[label.urgency],
        "hour_sin": np.sin(2 * np.pi * float(post.get("hour", 12)) / 24),
        "hour_cos": np.cos(2 * np.pi * float(post.get("hour", 12)) / 24),
        "verified": int(post.get("verified", 0)), "has_media": int(post.get("has_media", 0)),
        "sarcasm": int(label.sarcasm),
    }
    return pd.DataFrame([row])[FEATURES]


def score(post: dict, label) -> tuple[float, list[str]]:
    """Return (real chance of 200+ reshares in 24 h, top 3 reasons pushing it up)."""
    model, _, _, base_rate = load()
    X = _frame(post, label)
    p = float(calibrate(model.predict_proba(X)[0, 1], base_rate))
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    names = list(prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL)) + LOG_NUMERIC + PLAIN_NUMERIC
    contrib = prep.transform(X)
    contrib = np.asarray(contrib.todense() if hasattr(contrib, "todense") else contrib)[0] * clf.coef_[0]
    order = np.argsort(-contrib)
    reasons = [_friendly(names[i]) for i in order[:5] if contrib[i] > 0.15][:3]
    return p, reasons
