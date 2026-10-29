"""Scores the LLM classifier against human labels (the gold set).
Humans label eval/gold_set_template.csv first, without looking at the app's answers."""
import pandas as pd

from core.classifier import classify_batch
from core.config import CLASSIFY_BATCH_SIZE, LANGUAGES, SENTIMENTS, TOPICS, URGENCIES
from core.guards import clean_input

ALLOWED = {"sentiment": SENTIMENTS, "topic": TOPICS, "urgency": URGENCIES, "language": LANGUAGES}
FIELDS = ["sentiment", "topic", "urgency", "sarcasm", "language"]
YES, NO = {"yes", "y", "true", "1"}, {"no", "n", "false", "0"}


def load_gold(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean and check the human labels. Returns (usable rows, list of problems)."""
    problems = []
    missing = [c for c in ["id", "text", *FIELDS] if c not in df.columns]
    if missing:
        return pd.DataFrame(), [f"Missing columns: {', '.join(missing)}. Use the template's column names."]
    df = df.copy()
    for c in ["id", "text", *FIELDS]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    for c in ALLOWED:
        df[c] = df[c].str.lower().str.replace(" ", "_")
    df["sarcasm"] = df["sarcasm"].str.lower()
    keep = []
    for _, r in df.iterrows():
        if not r["text"]:
            continue
        bad = [f"{c}='{r[c]}'" for c in ALLOWED if r[c] not in ALLOWED[c]]
        if r["sarcasm"] not in YES | NO:
            bad.append(f"sarcasm='{r['sarcasm']}'")
        if bad:
            problems.append(f"{r['id']}: " + ", ".join(bad))
        else:
            keep.append(r)
    gold = pd.DataFrame(keep)
    if not gold.empty:
        gold["sarcasm"] = gold["sarcasm"].isin(YES)
    return gold, problems


def run_classifier(gold: pd.DataFrame, use_llm: bool = True, on_call=None) -> dict:
    items = [(r["id"], clean_input(r["text"]).text) for _, r in gold.iterrows()]
    labels = {}
    for i in range(0, len(items), CLASSIFY_BATCH_SIZE):
        labels.update(classify_batch(items[i:i + CLASSIFY_BATCH_SIZE], use_llm=use_llm, on_call=on_call))
    return labels


def score(gold: pd.DataFrame, labels: dict) -> dict:
    rows = []
    for _, r in gold.iterrows():
        lab = labels[r["id"]]
        row = {"id": r["id"], "text": r["text"], "ai_source": lab.source}
        for f in FIELDS:
            row[f"human_{f}"] = r[f]
            row[f"ai_{f}"] = getattr(lab, f)
            row[f"{f}_ok"] = row[f"human_{f}"] == row[f"ai_{f}"]
        rows.append(row)
    res = pd.DataFrame(rows)
    acc = {f: float(res[f"{f}_ok"].mean()) for f in FIELDS}
    slices = {}
    groups = {"English": res["human_language"] == "english",
              "Hinglish or Hindi": res["human_language"].isin(["hinglish", "hindi"]),
              "Sarcastic": res["human_sarcasm"],
              "Not sarcastic": ~res["human_sarcasm"]}
    for name, mask in groups.items():
        if mask.sum():
            slices[name] = {"posts": int(mask.sum()), "sentiment_accuracy": float(res.loc[mask, "sentiment_ok"].mean()),
                            "topic_accuracy": float(res.loc[mask, "topic_ok"].mean())}
    mistakes = res[~res[[f"{f}_ok" for f in FIELDS]].all(axis=1)]
    confusion = pd.crosstab(res["human_sentiment"], res["ai_sentiment"], rownames=["Human"], colnames=["AI"])
    return {"n": len(res), "accuracy": acc, "slices": slices, "mistakes": mistakes, "table": res,
            "confusion": confusion, "llm_used": bool((res["ai_source"].isin(["Groq", "Gemini"])).all())}
