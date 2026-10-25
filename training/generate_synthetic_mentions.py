"""
Synthetic mention generator v1 - IndiGo social mentions (unofficial educational prototype,
not affiliated with IndiGo; every post and number here is synthetic)
Team C2 - Foundations of AI for Business, Project 3

Why synthetic: no public dataset pairs Indian brand mentions with how far
they spread, and X/Twitter data is costly and restricted. Every rule below is
an ASSUMPTION, written down so the team can defend it, critique it, or change it.

Design principles
- Features the model may use are only those knowable 60 minutes after posting.
- Two hidden drivers are NOT given to the model:
    hidden_appeal       - how emotionally compelling the wording is
    hidden_news_pickup  - a random shock (media or a big account amplifies it)
  so the outcome cannot be perfectly predicted from the features.
- Language (English vs Hinglish) has NO effect on spread, by design. Any
  language gap seen later must therefore come from the LLM classifier.

Run from the repo root:  python training/generate_synthetic_mentions.py
Out:  data/synthetic_mentions_v1.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SEED = 42
N = 10000
SPREAD_THRESHOLD = 200      # "spread" = 200+ reshares within 24 hours
rng = np.random.default_rng(SEED)

TOPICS = ["delay", "cancellation", "baggage", "refund", "staff_behaviour",
          "safety", "pricing", "app_website", "other"]
TOPIC_P = [0.22, 0.10, 0.12, 0.12, 0.09, 0.03, 0.08, 0.10, 0.14]

# P(negative, neutral, positive) given topic
SENT_BY_TOPIC = {
    "delay": (0.80, 0.15, 0.05), "cancellation": (0.85, 0.12, 0.03),
    "baggage": (0.80, 0.15, 0.05), "refund": (0.80, 0.17, 0.03),
    "staff_behaviour": (0.60, 0.10, 0.30), "safety": (0.85, 0.13, 0.02),
    "pricing": (0.55, 0.35, 0.10), "app_website": (0.65, 0.30, 0.05),
    "other": (0.15, 0.45, 0.40),
}
# P(low, medium, high) urgency for NEGATIVE posts, by topic
URG_NEG = {
    "safety": (0.05, 0.15, 0.80), "cancellation": (0.15, 0.45, 0.40),
    "delay": (0.25, 0.50, 0.25), "baggage": (0.25, 0.50, 0.25),
    "refund": (0.35, 0.50, 0.15), "staff_behaviour": (0.30, 0.45, 0.25),
}
URG_NEG_DEFAULT = (0.55, 0.35, 0.10)
URG_NON_NEG = (0.85, 0.13, 0.02)
URG_LEVELS = ["low", "medium", "high"]

ACCOUNT_TYPES = ["regular", "influencer", "journalist"]
ACCOUNT_P = [0.88, 0.08, 0.04]
FOLLOWER_MEDIAN = {"regular": 250, "influencer": 40000, "journalist": 15000}
FOLLOWER_SIGMA = {"regular": 1.3, "influencer": 1.0, "journalist": 1.0}
VERIFIED_P = {"regular": 0.02, "influencer": 0.50, "journalist": 0.60}

# Posting hour: quiet at night, busier 8:00-23:00
HOUR_W = np.array([1, 0.6, 0.4, 0.3, 0.3, 0.5, 1, 2, 3, 3.5, 3.5, 3.5,
                   3.5, 3.5, 3.5, 3.5, 3.5, 4, 4.5, 4.5, 4, 3.5, 3, 2])
HOUR_P = HOUR_W / HOUR_W.sum()


def gamma_poisson(mean, shape=2.0):
    """Over-dispersed counts (negative binomial via a gamma-Poisson mix)."""
    return rng.poisson(mean * rng.gamma(shape, 1.0 / shape, size=np.shape(mean)))


def main():
    df = pd.DataFrame({"post_id": [f"M{i:05d}" for i in range(N)]})
    df["language"] = rng.choice(["english", "hinglish"], N, p=[0.6, 0.4])
    df["topic"] = rng.choice(TOPICS, N, p=TOPIC_P)
    df["sentiment"] = [rng.choice(["negative", "neutral", "positive"], p=SENT_BY_TOPIC[t])
                       for t in df["topic"]]
    neg = (df["sentiment"] == "negative").to_numpy()
    df["sarcasm"] = (neg & (rng.random(N) < 0.10)).astype(int)
    df["urgency"] = [rng.choice(URG_LEVELS, p=URG_NEG.get(t, URG_NEG_DEFAULT) if s == "negative" else URG_NON_NEG)
                     for t, s in zip(df["topic"], df["sentiment"])]
    urg = df["urgency"].map({"low": 0, "medium": 1, "high": 2}).to_numpy()

    df["account_type"] = rng.choice(ACCOUNT_TYPES, N, p=ACCOUNT_P)
    df["followers"] = [max(5, int(rng.lognormal(np.log(FOLLOWER_MEDIAN[a]), FOLLOWER_SIGMA[a])))
                       for a in df["account_type"]]
    df["verified"] = [int(rng.random() < VERIFIED_P[a]) for a in df["account_type"]]
    evidence_topic = df["topic"].isin(["baggage", "staff_behaviour", "safety"]).to_numpy()
    df["has_media"] = (rng.random(N) < 0.25 + 0.15 * (evidence_topic & neg)).astype(int)
    df["hour"] = rng.choice(24, N, p=HOUR_P)
    peak = df["hour"].between(18, 22).to_numpy()

    log_f = np.log1p(df["followers"].to_numpy())
    appeal = rng.normal(0, 1, N)                       # hidden
    df["hidden_appeal"] = appeal.round(3)

    # Early engagement in the first 60 minutes (observable at scoring time)
    early_mean = np.exp(-4.0 + 0.45 * log_f + 0.5 * neg + 0.35 * urg + 0.4 * df["has_media"]
                        + 0.3 * df["verified"] + 0.2 * peak + 0.2 * df["sarcasm"] + 0.6 * appeal)
    df["early_reshares_60m"] = gamma_poisson(early_mean)
    df["early_replies_60m"] = gamma_poisson(0.8 * early_mean * (1 + 0.5 * neg))

    # Growth from hour 1 to hour 24 (NOT observable at scoring time)
    growth_mean = np.exp(-0.2 + 0.9 * np.log1p(df["early_reshares_60m"]) + 0.25 * log_f
                         + 0.4 * neg + 0.25 * urg + 0.8 * appeal + rng.normal(0, 0.6, N))
    pickup_p = np.where(neg & (urg == 2), 0.06, 0.02)
    pickup = (rng.random(N) < pickup_p).astype(int)    # hidden
    df["hidden_news_pickup"] = pickup
    growth_mean = growth_mean * np.where(pickup == 1, np.exp(rng.normal(2.5, 0.5, N)), 1.0)
    df["reshares_24h"] = df["early_reshares_60m"] + gamma_poisson(growth_mean)

    df["spread"] = (df["reshares_24h"] >= SPREAD_THRESHOLD).astype(int)
    df["crisis"] = (df["spread"].astype(bool) & neg).astype(int)

    df.to_csv(ROOT / "data" / "synthetic_mentions_v1.csv", index=False)
    print(f"rows={len(df)}  spread rate={df.spread.mean():.3f}  crisis rate={df.crisis.mean():.3f}")
    print(df.groupby("language")["spread"].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
