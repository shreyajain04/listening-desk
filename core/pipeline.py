"""Runs one or many mentions through the system. Streamlit-free, so it can be tested on its own."""
from dataclasses import dataclass

from core.classifier import Classification, classify_batch
from core.config import CLASSIFY_BATCH_SIZE
from core.guards import CleanInput, clean_input
from core.router import route
from core.spread import load, score


@dataclass
class Analysis:
    post: dict
    clean: CleanInput
    label: Classification
    p_spread: float
    reasons: list[str]
    route: str
    route_reason: str

    @property
    def id(self) -> str:
        return self.post["post_id"]


def analyze(posts: list[dict], use_llm: bool = True, on_call=None) -> list[Analysis]:
    cleaned = {p["post_id"]: clean_input(p["text"]) for p in posts}
    labels: dict[str, Classification] = {}
    ids = [p["post_id"] for p in posts]
    for i in range(0, len(ids), CLASSIFY_BATCH_SIZE):
        chunk = [(pid, cleaned[pid].text) for pid in ids[i:i + CLASSIFY_BATCH_SIZE]]
        labels.update(classify_batch(chunk, use_llm=use_llm, on_call=on_call))
    _, threshold, _, _ = load()
    out = []
    for p in posts:
        lab = labels[p["post_id"]]
        if cleaned[p["post_id"]].injection_suspected:
            lab.injection_attempt = True
        prob, reasons = score(p, lab)
        r, why = route(lab, prob, threshold)
        out.append(Analysis(p, cleaned[p["post_id"]], lab, prob, reasons, r, why))
    return out
