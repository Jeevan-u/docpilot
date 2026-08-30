"""Token-aware text chunking.

Documents are split into overlapping segments that each fit inside a token
budget. Keeping an overlap between neighbours means a concept that straddles
the boundary between two chunks is still recoverable when we search.

Splitting strategy, applied in order of decreasing granularity:

1. Break the text into paragraphs (blank-line separated).
2. Break each paragraph into sentences.
3. Fill chunks greedily with sentences until the token budget is reached.
4. Seed the next chunk with the tail of the previous one (the overlap).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

from .loader import Document

_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """A piece of a document plus enough context to trace it back."""

    text: str
    metadata: dict = field(default_factory=dict)
    token_count: int = 0


class TokenChunker:
    """Split documents into overlapping, token-bounded chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        if chunk_size <= 0 or overlap < 0:
            raise ValueError("chunk_size must be positive and overlap non-negative")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._encoding = tiktoken.get_encoding("o200k_base")

    def chunk_document(self, document: Document) -> list[Chunk]:
        atoms = [atom for atom in self._to_atoms(document.text) if atom]
        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_tokens = 0

        for atom in atoms:
            token_count = self._token_count(atom)

            if token_count > self.chunk_size:
                if buffer:
                    chunks.append(self._build(buffer, document, len(chunks)))
                    buffer, buffer_tokens = [], 0
                for piece in self._hard_split(atom):
                    chunks.append(self._build([piece], document, len(chunks)))
                continue

            if buffer and buffer_tokens + token_count > self.chunk_size:
                chunks.append(self._build(buffer, document, len(chunks)))
                buffer = self._overlap_tail(buffer)
                buffer_tokens = sum(self._token_count(text) for text in buffer)

            buffer.append(atom)
            buffer_tokens += token_count

        if buffer:
            chunks.append(self._build(buffer, document, len(chunks)))

        return chunks

    def _build(self, parts: list[str], document: Document, index: int) -> Chunk:
        text = " ".join(parts).strip()
        metadata = {
            **document.metadata,
            "source": document.source,
            "page": document.page,
            "chunk_index": index,
        }
        return Chunk(text=text, metadata=metadata, token_count=self._token_count(text))

    def _overlap_tail(self, buffer: list[str]) -> list[str]:
        """Return the trailing part of a buffer sized to the overlap budget.

        Whole sentences are preferred so boundaries land cleanly. Only when
        a single sentence is itself larger than the whole budget do we trim
        it to avoid carrying practically all of the previous chunk.
        """
        selected: list[str] = []
        used = 0
        for text in reversed(buffer):
            room = self.overlap - used
            if room <= 0:
                break
            count = self._token_count(text)
            if count <= room:
                selected.append(text)
                used += count
            elif not selected:
                selected.append(self._truncate(text, room))
                used = self.overlap
            else:
                break
        return list(reversed(selected))

    def _hard_split(self, text: str) -> list[str]:
        """Split an over-long passage into overlapping fixed-size pieces."""
        ids = self._encoding.encode(text)
        step = max(self.chunk_size - self.overlap, 1)
        pieces = [
            self._encoding.decode(ids[start : start + self.chunk_size])
            for start in range(0, len(ids), step)
        ]
        return [piece for piece in pieces if piece.strip()]

    def _truncate(self, text: str, max_tokens: int) -> str:
        ids = self._encoding.encode(text)
        return self._encoding.decode(ids[:max_tokens])

    def _token_count(self, text: str) -> int:
        return len(self._encoding.encode(text))

    @staticmethod
    def _to_atoms(text: str) -> list[str]:
        paragraphs = _PARAGRAPH_BOUNDARY.split(text)
        atoms: list[str] = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            atoms.extend(sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(paragraph))
        return atoms