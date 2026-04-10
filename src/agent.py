from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        context_blocks: list[str] = []
        for i, r in enumerate(results, start=1):
            content = (r.get("content") or "").strip()
            if not content:
                continue
            context_blocks.append(f"[Chunk {i}]\n{content}")

        context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no relevant context retrieved)"

        prompt = (
            "You are a helpful assistant. Answer the question using ONLY the provided context. "
            "If the context is insufficient, say you don't know.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        answer = self.llm_fn(prompt)
        return str(answer) if answer is not None else ""
