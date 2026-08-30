"""Offline evaluation of the retrieval and generation stages.

Running a fixed set of (question, expected key phrase) pairs lets us see
whether the retriever surfaces the right material and whether the model
actually uses it. This is intentionally simple: loud, clear numbers beat
a complicated benchmark no one can reproduce.

Metrics:

* Retrieval hit rate @ k - the fraction of questions where a chunk
  containing the expected key phrase is among the top-k results.
* Answer coverage - the fraction of model answers that mention the
  expected key phrase (optional; requires running the generator).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .retriever import Retriever

AnswerFn = Callable[[str], str]


@dataclass
class EvalResult:
    """Results for the whole question set, plus per-question detail."""

    retrieval_hit_rate: float
    answer_coverage: float | None
    details: list[dict]

    def to_dict(self) -> dict:
        return {
            "retrieval_hit_rate": self.retrieval_hit_rate,
            "answer_coverage": self.answer_coverage,
            "questions": len(self.details),
            "details": self.details,
        }


def load_questions(path: str | Path) -> list[dict]:
    """Read question rows: ``[{"question": "...", "key_phrase": "..."}]``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        {"question": row["question"], "key_phrase": row["key_phrase"]} for row in raw
    ]


def evaluate(
    retriever: Retriever,
    rows: list[dict],
    answer_fn: AnswerFn | None = None,
    top_k: int | None = None,
) -> EvalResult:
    k = top_k or retriever.top_k
    details: list[dict] = []

    for row in rows:
        key = row["key_phrase"].lower()
        hits = retriever.retrieve(row["question"], top_k=k)
        hit_rate_ok = any(key in hit.text.lower() for hit in hits)

        coverage_ok: bool | None = None
        if answer_fn is not None:
            answer = answer_fn(row["question"])
            coverage_ok = key in answer.lower()

        details.append(
            {
                "question": row["question"],
                "key_phrase": row["key_phrase"],
                "retrieved": hit_rate_ok,
                "answer_covered": coverage_ok,
                "top_contexts": [hit.metadata.get("source") for hit in hits],
            }
        )

    hit_rate = sum(d["retrieved"] for d in details) / len(details) if details else 0.0
    coverage = None
    if answer_fn is not None:
        answered = [d for d in details if d["answer_covered"] is not None]
        coverage = (
            sum(1 for d in answered if d["answer_covered"]) / len(answered)
            if answered
            else 0.0
        )

    return EvalResult(
        retrieval_hit_rate=hit_rate, answer_coverage=coverage, details=details
    )
