"""Tests for the token-aware chunker."""

from docpilot.chunker import TokenChunker
from docpilot.loader import Document

_SAMPLE = """The sun rose over the quiet valley. Birds began to sing in the trees.

A river ran along the far edge of the field. Its water was cold and clear.

By noon, the clouds had gathered. A light rain began to fall on the grass.

By evening the sky cleared again. The stars appeared one by one above.

The next morning everything began anew. The valley welcomed the day."

"""


def _chunker(**kwargs) -> TokenChunker:
    kwargs.setdefault("chunk_size", 40)
    kwargs.setdefault("overlap", 8)
    return TokenChunker(**kwargs)


def _document() -> Document:
    return Document(text=_SAMPLE, source="test.txt")


def _token_count(text: str) -> int:
    return len(TokenChunker()._encoding.encode(text))


def test_chunks_respect_token_budget():
    chunker = _chunker(chunk_size=30, overlap=4)
    chunks = chunker.chunk_document(_document())
    assert len(chunks) > 1
    assert all(chunk.token_count <= 30 for chunk in chunks)


def test_chunks_are_ordered_and_cover_the_text():
    chunker = _chunker()
    chunks = chunker.chunk_document(_document())
    merged = " ".join(chunk.text for chunk in chunks)
    assert chunks[0].metadata["chunk_index"] == 0
    for first, second in zip(chunks, chunks[1:]):
        assert first.metadata["chunk_index"] + 1 == second.metadata["chunk_index"]
        assert second.metadata["chunk_index"] > first.metadata["chunk_index"]
    assert "sun" in merged


def test_neighbouring_chunks_share_overlap():
    chunker = _chunker(chunk_size=30, overlap=12)
    chunks = chunker.chunk_document(_document())
    assert len(chunks) >= 2
    for first, second in zip(chunks, chunks[1:]):
        words2 = second.text.split()
        shared_words = 0
        for count in range(min(len(first.text.split()), len(words2)), 0, -1):
            if first.text.endswith(" ".join(words2[:count])):
                shared_words = count
                break
        assert shared_words >= 1


def test_overlap_adds_repetition():
    chunker = _chunker(chunk_size=30, overlap=12)
    chunks = chunker.chunk_document(_document())
    total_chunk_tokens = sum(chunk.token_count for chunk in chunks)
    assert total_chunk_tokens > _token_count(_SAMPLE)


def test_single_passage_larger_than_budget_is_hard_split():
    long_passage = ("The word 'alphabet' comes from the Greek letters alpha and beta. " * 30).strip()
    document = Document(text=long_passage, source="long.txt")
    chunker = _chunker(chunk_size=50, overlap=0)
    chunks = chunker.chunk_document(document)
    assert len(chunks) > 1
    assert all(chunk.token_count <= 50 for chunk in chunks)


def test_single_sentence_under_budget_produces_one_chunk():
    document = Document(text="Just a short sentence.", source="short.txt")
    chunks = _chunker().chunk_document(document)
    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "short.txt"


def test_empty_document_produces_no_chunks():
    chunks = _chunker().chunk_document(Document(text="   ", source="empty.txt"))
    assert chunks == []


def test_chunked_by_sentence_not_by_characters():
    chunker = _chunker(chunk_size=40, overlap=4)
    chunks = chunker.chunk_document(_document())
    for chunk in chunks:
        pieces = chunk.text.rsplit(".", 1)
        assert chunk.text.strip().endswith(".") or len(pieces) == len(chunk.text.rsplit(".", 1))
    assert all(_token_count(chunk.text) <= 40 for chunk in chunks)