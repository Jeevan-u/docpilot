"""Load text content from common document formats.

Supported formats: plain text (.txt), Markdown (.md, .mdx) and PDF (.pdf).
Directories are walked recursively, so you can point the loader at a whole
folder of notes and it will pick up every supported file inside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".mdx", ".pdf"}


@dataclass
class Document:
    """A single unit of source material: text plus provenance."""

    text: str
    source: str
    page: int | None = None
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """Reads supported files and yields :class:`Document` objects."""

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)
        if path.is_dir():
            documents: list[Document] = []
            for file in sorted(path.rglob("*")):
                if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                    documents.extend(self._load_file(file))
            return documents
        if path.is_file():
            return self._load_file(path)
        raise FileNotFoundError(f"No such file or directory: {path}")

    def _load_file(self, path: Path) -> list[Document]:
        if path.suffix.lower() == ".pdf":
            return self._load_pdf(path)
        return [
            Document(
                text=path.read_text(encoding="utf-8", errors="replace"),
                source=str(path),
            )
        ]

    def _load_pdf(self, path: Path) -> list[Document]:
        documents: list[Document] = []
        try:
            reader = PdfReader(str(path))
        except Exception as exc:  # corrupt or encrypted
            raise ValueError(f"Could not read PDF {path}: {exc}") from exc
        for page_number, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                documents.append(
                    Document(text=text, source=str(path), page=page_number)
                )
        return documents
