"""Ground answer generation on retrieved context.

Provide the model with only the retrieved excerpts and ask it to answer
from those. Because the model is never shown the whole corpus, it cannot
hallucinate from unrelated content: every claim is tied to a numbered
source the user can check.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from openai import OpenAI

from .store import SearchHit

SYSTEM_PROMPT = (
    "You are a careful research assistant that answers questions using "
    "only the numbered source excerpts below.\n"
    "Rules:\n"
    "- Base every claim on the excerpts; never use outside knowledge.\n"
    "- If the excerpts do not contain the answer, say \"I don't know\" "
    "and stop.\n"
    "- Cite the excerpt number after each claim, e.g. [1] or [2].\n"
    "- Answer in a few concise sentences."
)


@dataclass
class Answer:
    """A generation result plus the sources it was built from."""

    text: str
    sources: list[dict]


def _render_context(hits: list[SearchHit]) -> str:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        location = hit.metadata.get("source", "unknown")
        page = hit.metadata.get("page")
        label = f"{location} (page {page})" if page is not None else location
        blocks.append(f"[{number}] Source: {label}\n{hit.text}")
    return "\n\n".join(blocks)


class Generator(ABC):
    """Produces an answer from a question and its retrieved context."""

    @abstractmethod
    def generate(self, question: str, hits: list[SearchHit]) -> Answer: ...


class ChatGenerator(Generator):
    """Answers via a chat-completion model."""

    def __init__(self, model: str, api_key: str | None = None, client: OpenAI | None = None):
        self.model = model
        self._client = client or OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, question: str, hits: list[SearchHit]) -> Answer:
        user_message = (
            f"{_render_context(hits)}\n\n"
            f"Question: {question}"
        )
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        answer_text = response.choices[0].message.content or ""
        return Answer(text=answer_text, sources=[hit.metadata for hit in hits])