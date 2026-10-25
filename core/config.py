"""Shared settings. Change values here, not scattered through the code."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
EVAL = ROOT / "eval"

# Must match the labels the spread model was trained on (training/generate_synthetic_mentions.py)
TOPICS = ["delay", "cancellation", "baggage", "refund", "staff_behaviour",
          "safety", "pricing", "app_website", "other"]
SENTIMENTS = ["positive", "neutral", "negative"]
URGENCIES = ["low", "medium", "high"]
LANGUAGES = ["english", "hinglish", "hindi", "other"]

TOPIC_LABELS = {
    "delay": "Delay", "cancellation": "Cancellation", "baggage": "Baggage", "refund": "Refund",
    "staff_behaviour": "Staff behaviour", "safety": "Safety", "pricing": "Pricing",
    "app_website": "App or website", "other": "Other",
}

# Guardrail limits
MAX_INPUT_CHARS = 1000
MAX_REPLY_CHARS = 280
LLM_CALLS_PER_WINDOW = 25        # per browser session
LLM_WINDOW_SECONDS = 600         # 10 minutes
CLASSIFY_BATCH_SIZE = 5          # mentions per classifier call (keeps tokens per minute low)

# Default model IDs. Override them in Streamlit secrets without touching code.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def secret(name: str, default: str | None = None) -> str | None:
    """Read a value from Streamlit secrets, then environment variables."""
    try:
        import streamlit as st
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(name, default)
