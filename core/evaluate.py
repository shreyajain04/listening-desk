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
    """Clean and check human labels. Sentiment is required; the other four fields are optional, and a field
    is scored only where a person filled it in. Returns (usable rows, list of problems)."""
    problems = []
    missing = [c for c in ["id", "text", "sentiment"] if c not in df.columns]
    if missing:
        return pd.DataFrame(), [f"Missing columns: {', '.join(missing)}. Use the template's column names."]
    df = df.copy()
    for c in ["id", "text", *FIELDS]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).str.strip()
    for c in ALLOWED:
        df[c] = df[c].str.lower().str.replace(" ", "_")
    df["sarcasm"] = df["sarcasm"].str.lower()
    keep = []
    for _, r in df.iterrows():
        if not r["text"]:
            continue
        if not r["sentiment"]:
            problems.append(f"{r['id']}: sentiment is empty")
            continue
        bad = [f"{c}='{r[c]}'" for c in ALLOWED if r[c] and r[c] not in ALLOWED[c]]
        if r["sarcasm"] and r["sarcasm"] not in YES | NO:
            bad.append(f"sarcasm='{r['sarcasm']}'")
        if bad:
            problems.append(f"{r['id']}: " + ", ".join(bad))
        else:
            keep.append(r)
    gold = pd.DataFrame(keep)
    if not gold.empty:
        gold["sarcasm"] = gold["sarcasm"].map(lambda v: True if v in YES else False if v in NO else None)
        gold = gold.astype(object).where(gold != "", None)   # blank = not labelled (no forward-filling)
    return gold, problems


def run_classifier(gold: pd.DataFrame, use_llm: bool = True, on_call=None) -> dict:
    items = [(r["id"], clean_input(r["text"]).text) for _, r in gold.iterrows()]
    labels = {}
    for i in range(0, len(items), CLASSIFY_BATCH_SIZE):
        labels.update(classify_batch(items[i:i + CLASSIFY_BATCH_SIZE], use_llm=use_llm, on_call=on_call))
    return labels


def score(gold: pd.DataFrame, labels: dict) -> dict:
    """Accuracy per field, counted only over posts where a person gave that label."""
    rows = []
    for _, r in gold.iterrows():
        lab = labels[r["id"]]
        row = {"id": r["id"], "text": r["text"], "ai_source": lab.source}
        for f in FIELDS:
            human = r[f]
            row[f"ai_{f}"] = getattr(lab, f)
            missing = human is None or (not isinstance(human, (str, bool)) and pd.isna(human))
            row[f"human_{f}"] = None if missing else human
            row[f"{f}_ok"] = None if missing else bool(human == getattr(lab, f))
        rows.append(row)
    res = pd.DataFrame(rows)
    acc, counted = {}, {}
    for f in FIELDS:
        has = res[f"{f}_ok"].notna()
        if has.any():
            acc[f] = float(res.loc[has, f"{f}_ok"].astype(bool).mean())
            counted[f] = int(has.sum())
    slices = {}
    lang = res["human_language"]
    sarc = res["human_sarcasm"]
    groups = {"English": lang == "english", "Hinglish or Hindi": lang.isin(["hinglish", "hindi"]),
              "Sarcastic": sarc == True, "Not sarcastic": sarc == False}  # noqa: E712
    for name, mask in groups.items():
        mask = mask.fillna(False).astype(bool)
        if mask.sum():
            topic_ok = res.loc[mask, "topic_ok"].dropna()
            slices[name] = {"posts": int(mask.sum()),
                            "sentiment_accuracy": float(res.loc[mask, "sentiment_ok"].astype(bool).mean()),
                            "topic_accuracy": float(topic_ok.astype(bool).mean()) if len(topic_ok) else None}
    wrong = res[[f"{f}_ok" for f in FIELDS]].apply(lambda col: col == False)  # noqa: E712
    mistakes = res[wrong.any(axis=1)]
    confusion = pd.crosstab(res["human_sentiment"], res["ai_sentiment"], rownames=["Human"], colnames=["AI"])
    return {"n": len(res), "accuracy": acc, "counted": counted, "slices": slices, "mistakes": mistakes,
            "table": res, "confusion": confusion, "llm_used": bool(res["ai_source"].isin(["Groq", "Gemini"]).all())}
