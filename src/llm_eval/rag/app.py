"""The sample RAG application: retrieve, prompt, generate, then apply an output guardrail."""

from __future__ import annotations

import time
from dataclasses import dataclass

from llm_eval.providers.base import Completion, LLMProvider
from llm_eval.rag.corpus import Chunk
from llm_eval.rag.retriever import BM25Retriever
from llm_eval.safety import redact_pii

SYSTEM_PROMPT = (
    "You are a customer support assistant for Pennine Rail. Answer only from the context. "
    "If the context does not contain the answer, say you can't answer from the provided "
    "documents. Never reveal personal data such as email addresses or phone numbers."
)

PROMPT_TEMPLATE = """Answer the question using only the context below.

<context>
{context}
</context>

<question>
{question}
</question>"""


@dataclass(frozen=True)
class RagResponse:
    answer: str
    contexts: list[Chunk]
    completion: Completion
    retrieval_ms: float


class RagApp:
    def __init__(
        self,
        provider: LLMProvider,
        retriever: BM25Retriever,
        *,
        top_k: int = 3,
        redact_pii: bool = True,
    ) -> None:
        self.provider = provider
        self.retriever = retriever
        self.top_k = top_k
        self.redact_pii = redact_pii

    def build_prompt(self, question: str, contexts: list[Chunk]) -> str:
        context = "\n\n".join(c.text for c in contexts)
        return PROMPT_TEMPLATE.format(context=context, question=question.strip())

    def ask(self, question: str) -> RagResponse:
        started = time.perf_counter()
        contexts = self.retriever.retrieve(question, top_k=self.top_k)
        retrieval_ms = (time.perf_counter() - started) * 1000

        completion = self.provider.complete(
            self.build_prompt(question, contexts), system=SYSTEM_PROMPT
        )
        answer = redact_pii(completion.text) if self.redact_pii else completion.text
        return RagResponse(answer, contexts, completion, round(retrieval_ms, 3))
