"""Decides where each mention goes. Business rules live here, in one readable place."""

ROUTES = {
    "crisis": "Crisis alert",
    "reply": "Needs a reply",
    "routine": "Routine",
    "check": "Check by hand",
}


def route(label, p_spread: float, threshold: float) -> tuple[str, str]:
    """Return (route key, plain-language reason)."""
    if not label.valid:
        return "check", "The classifier could not label this post after three tries."
    if label.injection_attempt:
        return "check", "Looks like an attempt to manipulate the system. A reviewer decides whether to reply."
    if label.topic == "safety" and label.sentiment != "positive":
        return "crisis", "Safety-related posts always go to the PR lead, whatever their spread score."
    if label.sentiment == "negative" and p_spread >= threshold:
        return "crisis", f"Negative, with a {p_spread:.1%} chance of spreading, above the {threshold:.1%} alert threshold."
    if label.needs_reply or label.sentiment == "negative":
        return "reply", "A complaint or question that deserves a reply."
    return "routine", "No reply needed. Logged for reporting."
