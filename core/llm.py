"""One function, chat_json, hides which provider answers.
Order: Groq (free plan) -> Gemini (free tier) -> raise LLMError so callers fall back to offline rules.
Plain HTTP via `requests`, so no provider SDK versions can break the app."""
import json
import re
import time

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


REASONING_HEADROOM = 1500   # thinking models spend tokens before the answer; leave room for both


def _post_with_fallbacks(url: str, headers: dict, body: dict, optional: list[tuple[list[str], str]]) -> requests.Response:
    """POST, and if the API rejects an optional setting (HTTP 400 naming it), drop it and try again."""
    r = requests.post(url, json=body, headers=headers, timeout=TIMEOUT)
    for path, word in optional:
        if r.status_code != 400 or word not in r.text.lower():
            continue
        target = body
        for key in path[:-1]:
            target = target.get(key, {})
        if path[-1] in target:
            target.pop(path[-1])
            r = requests.post(url, json=body, headers=headers, timeout=TIMEOUT)
    return r


def _groq(system: str, user: str, max_tokens: int) -> str:
    key = secret("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set")
    model = secret("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens + REASONING_HEADROOM,
        "response_format": {"type": "json_object"},
    }
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"     # short thinking: faster and uses less of the free quota
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    optional = [(["response_format"], "response_format"), (["reasoning_effort"], "reasoning")]
    r = _post_with_fallbacks(url, headers, body, optional)
    if r.status_code == 429:                  # per-minute limit: wait briefly once, then let Gemini take over
        wait = float(r.headers.get("retry-after", "0") or 0)
        if 0 < wait <= 12:
            time.sleep(wait)
            r = _post_with_fallbacks(url, headers, body, optional)
    if r.status_code != 200:
        raise LLMError(f"Groq HTTP {r.status_code}: {r.text[:200]}")
    content = r.json()["choices"][0]["message"].get("content") or ""
    if not content.strip():
        raise LLMError("Groq returned an empty answer")
    return content


def _gemini(system: str, user: str, max_tokens: int) -> str:
    key = secret("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY is not set")
    model = secret("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    thinking = {"thinkingLevel": "low"} if model.startswith("gemini-3") else {"thinkingBudget": 0}
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens + REASONING_HEADROOM,
                             "responseMimeType": "application/json", "thinkingConfig": thinking},
    }
    r = _post_with_fallbacks(url, headers, body, [(["generationConfig", "thinkingConfig"], "thinking")])
    if r.status_code != 200:
        raise LLMError(f"Gemini HTTP {r.status_code}: {r.text[:200]}")
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


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
