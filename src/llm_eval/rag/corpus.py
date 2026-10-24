"""Load a folder of Markdown documents and split them into paragraph chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage. ``doc_id`` is the file stem, e.g. ``refunds``."""

    chunk_id: str
    doc_id: str
    text: str


def chunk_document(doc_id: str, text: str, paragraphs_per_chunk: int = 1) -> list[Chunk]:
    """Split on blank lines, drop headings, and group paragraphs into chunks."""
    paragraphs = [
        " ".join(p.split())
        for p in text.split("\n\n")
        if p.strip() and not p.lstrip().startswith("#")
    ]
    chunks = []
    for start in range(0, len(paragraphs), paragraphs_per_chunk):
        body = " ".join(paragraphs[start : start + paragraphs_per_chunk])
        chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, body))
    return chunks


def load_corpus(directory: str | Path, paragraphs_per_chunk: int = 1) -> list[Chunk]:
    """Load every ``*.md`` file in ``directory`` (except README.md) as chunks."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")
    chunks: list[Chunk] = []
    for path in sorted(root.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        chunks.extend(
            chunk_document(path.stem, path.read_text(encoding="utf-8"), paragraphs_per_chunk)
        )
    if not chunks:
        raise ValueError(f"No documents found in {root}")
    return chunks
