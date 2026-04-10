from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        cleaned = text.strip()
        if not cleaned:
            return []

        # Keep punctuation with the sentence by splitting on whitespace/newlines
        # that follow common sentence terminators.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])(?:\s+|\n+)", cleaned) if s.strip()]

        if not sentences:
            return []

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        separators = self.separators if self.separators else [""]
        chunks = self._split(text, separators)
        return [c for c in (chunk.strip() for chunk in chunks) if c]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        current_text = current_text or ""
        if not current_text.strip():
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            remaining_separators = [""]

        sep = remaining_separators[0]
        rest = remaining_separators[1:]

        # Base-case fallback: no separator => fixed-size slicing
        if sep == "":
            out: list[str] = []
            for start in range(0, len(current_text), self.chunk_size):
                piece = current_text[start : start + self.chunk_size]
                if piece:
                    out.append(piece)
            return out

        parts = current_text.split(sep)

        out: list[str] = []
        buffer = ""
        for part in parts:
            if not part:
                continue

            candidate = part if not buffer else f"{buffer}{sep}{part}"
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue

            if buffer:
                out.append(buffer)
                buffer = ""

            if len(part) <= self.chunk_size:
                buffer = part
            else:
                # Part still too large; try next separators.
                out.extend(self._split(part, rest))

        if buffer:
            out.append(buffer)

        return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    denom = math.sqrt(_dot(vec_a, vec_a)) * math.sqrt(_dot(vec_b, vec_b))
    if denom == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / denom


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed = FixedSizeChunker(chunk_size=chunk_size, overlap=max(0, chunk_size // 10)).chunk(text)
        by_sent = SentenceChunker(max_sentences_per_chunk=3).chunk(text)
        rec = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        def stats(chunks: list[str]) -> dict:
            count = len(chunks)
            avg_length = (sum(len(c) for c in chunks) / count) if count else 0.0
            return {"count": count, "avg_length": avg_length, "chunks": chunks}

        return {
            "fixed_size": stats(fixed),
            "by_sentences": stats(by_sent),
            "recursive": stats(rec),
        }
