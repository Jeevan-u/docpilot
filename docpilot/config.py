"""Central configuration for the application.

Settings live in a single :class:`Config` object so every other module
can read them from one place. Values are overridable through environment
variables, which keeps secrets (like the API key) out of the repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_path(name: str, fallback: str) -> Path:
    return Path(os.getenv(name, fallback)).expanduser()


@dataclass
class Config:
    """Runtime settings shared across the whole application."""

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    index_dir: Path = field(
        default_factory=lambda: _env_path("INDEX_DIR", "data/index")
    )
    api_key: str = os.getenv("OPENAI_API_KEY", "")

    def require_api_key(self) -> str:
        """Return the API key or raise a helpful error."""
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env "
                "and add your key, or export OPENAI_API_KEY."
            )
        return self.api_key