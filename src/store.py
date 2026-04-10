from __future__ import annotations

import os
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        # Default to in-memory for reliability in classroom environments.
        # Opt-in to ChromaDB by setting USE_CHROMA=1.
        use_chroma = os.getenv("USE_CHROMA", "").strip().lower() in {"1", "true", "yes", "y"}
        if use_chroma:
            try:
                import chromadb  # noqa: F401

                client = chromadb.Client()
                self._collection = client.get_or_create_collection(name=self._collection_name)
                self._use_chroma = True
            except Exception:
                self._use_chroma = False
                self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        record_id = f"{doc.id}:{self._next_index}"
        self._next_index += 1
        metadata = dict(doc.metadata or {})
        metadata.setdefault("doc_id", doc.id)
        return {
            "id": record_id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": metadata,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        q_emb = self._embedding_fn(query)
        scored: list[dict[str, Any]] = []
        for r in records:
            score = _dot(q_emb, r.get("embedding", []))
            scored.append({**r, "score": float(score)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: max(0, top_k)]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        if self._use_chroma and self._collection is not None:
            ids: list[str] = []
            documents: list[str] = []
            embeddings: list[list[float]] = []
            metadatas: list[dict[str, Any]] = []
            for doc in docs:
                rec = self._make_record(doc)
                ids.append(rec["id"])
                documents.append(rec["content"])
                embeddings.append(rec["embedding"])
                metadatas.append(rec["metadata"])
            try:
                self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
                return
            except Exception:
                # fall back to in-memory below
                self._use_chroma = False
                self._collection = None

        for doc in docs:
            self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            q_emb = self._embedding_fn(query)
            try:
                res = self._collection.query(
                    query_embeddings=[q_emb],
                    n_results=max(1, top_k),
                    include=["documents", "metadatas", "distances"],
                )
                out: list[dict[str, Any]] = []
                docs = (res.get("documents") or [[]])[0]
                metadatas = (res.get("metadatas") or [[]])[0]
                distances = (res.get("distances") or [[]])[0]
                for content, metadata, distance in zip(docs, metadatas, distances):
                    # Convert distance (often cosine distance) to a similarity-like score.
                    score = 1.0 - float(distance) if distance is not None else 0.0
                    out.append({"content": content, "metadata": metadata or {}, "score": score})
                out.sort(key=lambda x: x["score"], reverse=True)
                return out[: max(0, top_k)]
            except Exception:
                # fall back to in-memory
                self._use_chroma = False
                self._collection = None

        results = self._search_records(query, self._store, top_k)
        # Ensure the public-facing shape includes content + metadata + score
        return [{"content": r["content"], "metadata": r["metadata"], "score": r["score"]} for r in results]

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            try:
                return int(self._collection.count())
            except Exception:
                pass
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            q_emb = self._embedding_fn(query)
            try:
                res = self._collection.query(
                    query_embeddings=[q_emb],
                    n_results=max(1, top_k),
                    where=metadata_filter,
                    include=["documents", "metadatas", "distances"],
                )
                out: list[dict[str, Any]] = []
                docs = (res.get("documents") or [[]])[0]
                metadatas = (res.get("metadatas") or [[]])[0]
                distances = (res.get("distances") or [[]])[0]
                for content, metadata, distance in zip(docs, metadatas, distances):
                    score = 1.0 - float(distance) if distance is not None else 0.0
                    out.append({"content": content, "metadata": metadata or {}, "score": score})
                out.sort(key=lambda x: x["score"], reverse=True)
                return out[: max(0, top_k)]
            except Exception:
                self._use_chroma = False
                self._collection = None

        filtered = [
            r
            for r in self._store
            if all((r.get("metadata") or {}).get(k) == v for k, v in metadata_filter.items())
        ]
        results = self._search_records(query, filtered, top_k)
        return [{"content": r["content"], "metadata": r["metadata"], "score": r["score"]} for r in results]

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        removed_any = False

        if self._use_chroma and self._collection is not None:
            try:
                # Best-effort delete by metadata filter.
                self._collection.delete(where={"doc_id": doc_id})
                # Chroma's delete doesn't report count reliably across versions.
                return True
            except Exception:
                self._use_chroma = False
                self._collection = None

        before = len(self._store)
        self._store = [r for r in self._store if (r.get("metadata") or {}).get("doc_id") != doc_id]
        removed_any = len(self._store) < before
        return removed_any
