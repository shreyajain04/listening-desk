"""Guardrails that run before and after every LLM call. No LLM needed here."""
import re
from dataclasses import dataclass, field

from core.config import MAX_INPUT_CHARS, MAX_REPLY_CHARS

# Personal data patterns (Indian context)
PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Booking reference: 6 characters mixing letters and digits; IndiGo flight numbers (6E123, 6E2134) are kept
PNR = re.compile(r"\b(?!6E\d{2,4}\b)(?=[A-Z0-9]{6}\b)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b")

INJECTION_CUES = re.compile(
    r"ignore (all |any |the |your )?(previous |prior |above )?(instructions|rules|prompt)"
    r"|system prompt|you are now|act as|disregard (the |your )?(rules|instructions)"
    r"|developer mode|reveal (your )?(prompt|instructions)",
    re.IGNORECASE,
)

# Promises a public reply must never make (tone guide TG-6). Phrasings, not single words,
# so "check the status of your refund request" passes but "we will refund you" does not.
PROMISE_PATTERNS = {
    "a rupee amount": re.compile(r"(₹\s?[\d,]+|\brs\.?\s?\d[\d,]*|\binr\s?\d[\d,]*)", re.IGNORECASE),
    "a refund": re.compile(r"\b(will|we'll|shall|going to)\s+(\w+\s+){0,3}refund|\bfull refund|\brefund (you|of|in full)"
                           r"|\bget (a|your) refund|\brefund\s+(mil|de|kar)\w*\s*(jayega|denge|diya)", re.IGNORECASE),
    "compensation": re.compile(r"\b(will|we'll|shall)\s+(\w+\s+){0,3}compensat|\bcompensation (of|will|is)", re.IGNORECASE),
    "a voucher": re.compile(r"\bvoucher", re.IGNORECASE),
    "a guarantee": re.compile(r"\bguarantee", re.IGNORECASE),
    "a deadline": re.compile(r"\bwithin \d+\s?(minutes?|hours?|days?|hrs?)\b|\bby (today|tonight|tomorrow)\b", re.IGNORECASE),
    "something free": re.compile(r"\bfree (ticket|flight|upgrade|meal)", re.IGNORECASE),
}


@dataclass
class CleanInput:
    text: str                       # what the LLM sees
    redacted: list[str] = field(default_factory=list)
    truncated: bool = False
    injection_suspected: bool = False


def clean_input(raw: str) -> CleanInput:
    text = (raw or "").replace("\x00", "").strip()
    truncated = len(text) > MAX_INPUT_CHARS
    text = text[:MAX_INPUT_CHARS]
    redacted = []
    for label, pattern, token in [("phone number", PHONE, "[PHONE]"),
                                  ("email", EMAIL, "[EMAIL]"),
                                  ("booking code", PNR, "[BOOKING]")]:
        if pattern.search(text):
            redacted.append(label)
            text = pattern.sub(token, text)
    return CleanInput(text=text, redacted=redacted, truncated=truncated,
                      injection_suspected=bool(INJECTION_CUES.search(text)))


def check_draft(reply: str, cited: list[str], retrieved: dict[str, str], needs_human: bool) -> list[str]:
    """Return problems with a draft reply. An empty list means it may be approved."""
    issues = []
    reply = (reply or "").strip()
    if not reply:
        return ["The draft is empty."]
    if len(reply) > MAX_REPLY_CHARS:
        issues.append(f"Too long for X: {len(reply)} of {MAX_REPLY_CHARS} characters.")
    if not cited:
        issues.append("No playbook section is cited.")
    unknown = [c for c in cited if c not in retrieved]
    if unknown:
        issues.append(f"Cites sections that were not retrieved: {', '.join(unknown)}.")
    for label, pattern in PROMISE_PATTERNS.items():
        match = pattern.search(reply)
        if match:
            issues.append(f"Promises {label} (\"{match.group(0).strip()}\"), which public replies must not do (TG-6).")
    if PHONE.search(reply) or EMAIL.search(reply) or PNR.search(reply):
        issues.append("Contains personal data (phone, email or booking code).")
    if needs_human:
        issues.append("The drafter says the playbook does not cover this case.")
    return issues
