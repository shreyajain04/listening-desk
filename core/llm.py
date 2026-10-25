"""One function, chat_json, hides which provider answers.
Order: Groq (free plan) -> Gemini (free tier) -> raise LLMError so callers fall back to offline rules.
Plain HTTP via `requests`, so no provider SDK versions can break the app."""
import json
import re

import requests

from core.config import DEFAULT_GEMINI_MODEL, DEFAULT_GROQ_MODEL, secret

TIMEOUT = 30


class LLMError(Exception):
    pass


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    return json.loads(text[start:end + 1])


def _groq(system: str, user: str, max_tokens: int) -> str:
    key = secret("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set")
    body = {
        "model": secret("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    r = requests.post(url, json=body, headers=headers, timeout=TIMEOUT)
    if r.status_code == 400 and "response_format" in r.text:   # model without JSON mode
        body.pop("response_format")
        r = requests.post(url, json=body, headers=headers, timeout=TIMEOUT)
    if r.status_code != 200:
        raise LLMError(f"Groq HTTP {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"]


def _gemini(system: str, user: str, max_tokens: int) -> str:
    key = secret("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set")
    model = secret("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens,
                             "responseMimeType": "application/json",
                             "thinkingConfig": {"thinkingBudget": 0}},
    }
    r = requests.post(url, json=body, headers=headers, timeout=TIMEOUT)
    if r.status_code == 400 and "thinking" in r.text.lower():   # model without thinking control
        body["generationConfig"].pop("thinkingConfig")
        r = requests.post(url, json=body, headers=headers, timeout=TIMEOUT)
    if r.status_code != 200:
        raise LLMError(f"Gemini HTTP {r.status_code}: {r.text[:200]}")
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


PROVIDERS = [("Groq", _groq), ("Gemini", _gemini)]


def chat_json(system: str, user: str, max_tokens: int = 900) -> tuple[dict, str]:
    """Return (parsed JSON, provider name). Tries each provider in order."""
    errors = []
    for name, call in PROVIDERS:
        try:
            return _parse_json(call(system, user, max_tokens)), name
        except (LLMError, requests.RequestException, ValueError, KeyError, IndexError) as exc:
            errors.append(f"{name}: {exc}")
    raise LLMError(" | ".join(errors) or "no provider configured")


def provider_status() -> list[tuple[str, str]]:
    """Ping each provider with a tiny request. Used on the About tab."""
    out = []
    for name, call in PROVIDERS:
        try:
            _parse_json(call('Reply with JSON only.', 'Return {"ok": true}', 20))
            out.append((name, "Working"))
        except Exception as exc:
            out.append((name, f"Not working: {str(exc)[:160]}"))
    return out
