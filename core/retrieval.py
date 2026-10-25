"""Finds the playbook sections relevant to a mention.
v1: TF-IDF keyword search (scikit-learn, no extra install).
Planned upgrade: fastembed embeddings. Keep the same `retrieve()` signature so nothing else changes."""
import re
from dataclasses import dataclass
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.config import DATA

HEADING = re.compile(r"^## \[(?P<id>[A-Z]{2}-\d+)\] (?P<title>.+)$", re.MULTILINE)
ALWAYS_INCLUDE = "TG-1"      # voice basics apply to every reply
# The section written for each topic is always offered to the drafter, then keyword search adds the rest
TOPIC_SECTION = {"safety": "PB-3", "baggage": "PB-4", "refund": "PB-5", "delay": "PB-6",
                 "staff_behaviour": "PB-7", "pricing": "PB-9", "app_website": "PB-10", "cancellation": "PB-2"}


@dataclass
class Section:
    id: str
    title: str
    body: str
    source: str


@lru_cache(maxsize=1)
def load_sections() -> list[Section]:
    sections = []
    for path in sorted((DATA / "playbook").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        matches = list(HEADING.finditer(text))
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append(Section(m["id"], m["title"].strip(), text[m.end():end].strip(), path.stem))
    return sections


@lru_cache(maxsize=1)
def _index():
    sections = load_sections()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vec.fit_transform([f"{s.title} {s.title} {s.body}" for s in sections])
    return vec, matrix


def retrieve(query: str, k: int = 3, topic: str | None = None) -> list[Section]:
    """Voice basics + the topic's own section + keyword matches, k sections besides TG-1."""
    sections = load_sections()
    vec, matrix = _index()
    scores = cosine_similarity(vec.transform([query]), matrix)[0]
    picked = [s for s in sections if s.id == ALWAYS_INCLUDE]
    mapped = next((s for s in sections if s.id == TOPIC_SECTION.get(topic or "")), None)
    if mapped:
        picked.append(mapped)
    for i in scores.argsort()[::-1]:
        if len(picked) >= k + 1:
            break
        if sections[i] not in picked:
            picked.append(sections[i])
    return picked


def section_by_id(section_id: str) -> Section | None:
    return next((s for s in load_sections() if s.id == section_id), None)
