from __future__ import annotations

import json
import os
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore


@dataclass(frozen=True)
class BenchmarkQuery:
    query: str
    gold_answer: str
    metadata_filter: dict | None = None


class MarkdownHeaderChunker:
    """
    Custom chunking strategy for Markdown-like docs.

    Idea:
        - Split by headings (#, ##, ###...) to preserve section-level coherence.
        - If a section is still too large, fall back to RecursiveChunker.
    """

    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = chunk_size
        self._fallback = RecursiveChunker(chunk_size=chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        lines = text.splitlines()
        sections: list[list[str]] = []
        current: list[str] = []

        def is_heading(line: str) -> bool:
            stripped = line.lstrip()
            return stripped.startswith("#") and len(stripped) > 1 and stripped[1] in {" ", "#"}

        for line in lines:
            if is_heading(line) and current:
                sections.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            sections.append(current)

        chunks: list[str] = []
        for sec_lines in sections:
            sec = "\n".join(sec_lines).strip()
            if not sec:
                continue
            if len(sec) <= self.chunk_size:
                chunks.append(sec)
            else:
                chunks.extend(self._fallback.chunk(sec))
        return chunks


def _detect_language(text: str) -> str:
    if not text:
        return "unknown"
    lowered = text.lower()
    if any(ch in lowered for ch in ["đ", "ă", "â", "ê", "ô", "ơ", "ư"]):
        return "vi"
    return "en"


def load_documents_from_data_dir(data_dir: Path) -> list[Document]:
    allowed_extensions = {".md", ".txt"}
    docs: list[Document] = []

    for path in sorted(data_dir.glob("*")):
        if path.suffix.lower() not in allowed_extensions or not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                id=path.stem,
                content=content,
                metadata={
                    "source": str(path).replace("\\", "/"),
                    "extension": path.suffix.lower(),
                    "language": _detect_language(content),
                    "category": "sample",
                },
            )
        )
    return docs


def chunk_documents(docs: Iterable[Document], chunker: object) -> list[Document]:
    chunk_fn = getattr(chunker, "chunk", None)
    if not callable(chunk_fn):
        raise TypeError("chunker must have a .chunk(text) -> list[str] method")

    out: list[Document] = []
    for doc in docs:
        chunks = chunk_fn(doc.content)
        for idx, content in enumerate(chunks):
            out.append(
                Document(
                    id=f"{doc.id}__chunk_{idx}",
                    content=content,
                    metadata={
                        **(doc.metadata or {}),
                        "doc_id": doc.id,
                        "chunk_index": idx,
                    },
                )
            )
    return out


def _load_queries(config_path: Path) -> list[BenchmarkQuery]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    queries: list[BenchmarkQuery] = []
    for item in raw.get("queries", []):
        queries.append(
            BenchmarkQuery(
                query=str(item.get("query", "")).strip(),
                gold_answer=str(item.get("gold_answer", "")).strip(),
                metadata_filter=item.get("metadata_filter"),
            )
        )
    return [q for q in queries if q.query]


def _select_embedder() -> Callable[[str], list[float]]:
    # Phase 2 benchmarks should be fast and reproducible.
    # Default to mock embeddings unless the user explicitly opts in via env vars.
    provider = os.getenv("PHASE2_EMBEDDING_PROVIDER") or os.getenv(EMBEDDING_PROVIDER_ENV) or "mock"
    provider = provider.strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    if provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    return _mock_embed


def _md_escape_cell(s: str, limit: int = 140) -> str:
    if s is None:
        return ""
    s = " ".join(str(s).split())
    if len(s) > limit:
        s = s[: limit - 3] + "..."
    return s.replace("|", "\\|")


def run_benchmark(
    *,
    docs: list[Document],
    queries: list[BenchmarkQuery],
    strategies: dict[str, object],
    embedding_fn: Callable[[str], list[float]],
    top_k: int = 3,
) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}

    for strat_name, chunker in strategies.items():
        chunked_docs = chunk_documents(docs, chunker)
        store = EmbeddingStore(collection_name=f"phase2_{strat_name}", embedding_fn=embedding_fn)
        store.add_documents(chunked_docs)

        strat_rows: list[dict] = []
        for q in queries:
            if q.metadata_filter:
                retrieved = store.search_with_filter(q.query, top_k=top_k, metadata_filter=q.metadata_filter)
            else:
                retrieved = store.search(q.query, top_k=top_k)

            top1 = retrieved[0] if retrieved else {"content": "", "score": 0.0, "metadata": {}}
            strat_rows.append(
                {
                    "query": q.query,
                    "gold_answer": q.gold_answer,
                    "filter": q.metadata_filter,
                    "top1_preview": _md_escape_cell(top1.get("content", "")),
                    "top1_score": float(top1.get("score") or 0.0),
                    "top1_source": _md_escape_cell((top1.get("metadata") or {}).get("source", "")),
                    "top1_doc_id": _md_escape_cell((top1.get("metadata") or {}).get("doc_id", "")),
                }
            )

        results[strat_name] = strat_rows

    return results


def render_markdown_tables(results: dict[str, list[dict]]) -> str:
    lines: list[str] = []
    for strat_name, rows in results.items():
        lines.append(f"\n## Strategy: `{strat_name}`\n")
        lines.append("| # | Query | Filter | Top-1 Retrieved Chunk (preview) | Score | Source | doc_id |")
        lines.append("|---:|---|---|---|---:|---|---|")
        for i, r in enumerate(rows, start=1):
            filt = json.dumps(r.get("filter"), ensure_ascii=False) if r.get("filter") else ""
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(i),
                        _md_escape_cell(r.get("query", ""), 80),
                        _md_escape_cell(filt, 50),
                        _md_escape_cell(r.get("top1_preview", ""), 160),
                        f"{r.get('top1_score', 0.0):.3f}",
                        _md_escape_cell(r.get("top1_source", ""), 60),
                        _md_escape_cell(r.get("top1_doc_id", ""), 30),
                    ]
                )
                + " |"
            )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Run Phase 2 retrieval benchmark.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Relative path to data directory containing .md/.txt files (default: data).",
    )
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["all", "fixed_400_40", "sentences_2", "recursive_400", "md_headers_400"],
        help="Run one strategy only or all strategies.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output markdown path (relative to repo root), e.g. report/benchmark_recursive_400.md",
    )
    args = parser.parse_args()

    day_dir = Path(__file__).parent
    data_dir = day_dir / args.data_dir
    config_path = day_dir / "report" / "phase2_queries.json"

    docs = load_documents_from_data_dir(data_dir)
    if not docs:
        print(f"No documents found in {data_dir} (.md/.txt).")
        return 1

    if not config_path.exists():
        print(f"Missing config: {config_path}")
        print("Create it from the provided template and rerun.")
        return 1

    queries = _load_queries(config_path)
    if not queries:
        print("No queries found in config.")
        return 1

    embedding_fn = _select_embedder()

    all_strategies: dict[str, object] = {
        "fixed_400_40": FixedSizeChunker(chunk_size=400, overlap=40),
        "sentences_2": SentenceChunker(max_sentences_per_chunk=2),
        "recursive_400": RecursiveChunker(chunk_size=400),
        "md_headers_400": MarkdownHeaderChunker(chunk_size=400),
    }
    strategies = all_strategies if args.strategy == "all" else {args.strategy: all_strategies[args.strategy]}

    results = run_benchmark(docs=docs, queries=queries, strategies=strategies, embedding_fn=embedding_fn, top_k=3)
    markdown = render_markdown_tables(results)

    if args.output:
        out_path = day_dir / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"Saved markdown result to: {out_path}")
    else:
        # Single write avoids line-by-line interleave artifacts on some Windows terminals.
        sys.stdout.write(markdown)
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

