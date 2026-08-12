from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import Settings
from search import (
    FALLBACK_RESPONSE,
    GenerationError,
    RetrievalError,
    SemanticSearchService,
    build_context,
    initialize_search,
)


def document(text: str):
    return SimpleNamespace(page_content=text, metadata={})


def make_settings() -> Settings:
    return Settings(
        ai_provider="openai",
        api_key="test-key",
        database_url="postgresql+psycopg://user:password@localhost/db",
        pdf_path=ROOT / "document.pdf",
        collection_name="document",
        embedding_model="text-embedding-3-small",
        llm_model="gpt-5-nano",
    )


class VectorStore:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def similarity_search_with_score(self, question, k):
        self.calls.append((question, k))
        if self.error:
            raise self.error
        return self.results


class LLM:
    def __init__(self, response="grounded answer", error=None):
        self.response = response
        self.error = error
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.response)


class SearchTests(unittest.TestCase):
    def test_context_preserves_order_and_excludes_scores(self):
        context = build_context([(document("first"), 0.123), (document("second"), 9.9)])

        self.assertEqual(context, "first\n\n---\n\nsecond")
        self.assertNotIn("0.123", context)

    def test_answer_uses_k_ten_and_guarded_ordered_context(self):
        store = VectorStore([(document("first fact"), 0.1), (document("second fact"), 0.2)])
        llm = LLM("answer")
        service = SemanticSearchService(store, llm)

        answer = service.answer("  question?  ")

        self.assertEqual(answer, "answer")
        self.assertEqual(store.calls, [("question?", 10)])
        prompt = llm.prompts[0]
        self.assertLess(prompt.index("first fact"), prompt.index("second fact"))
        self.assertIn(FALLBACK_RESPONSE, prompt)
        self.assertIn("Nunca invente ou use conhecimento externo", prompt)

    def test_empty_results_produce_empty_context_and_still_invoke_model(self):
        llm = LLM(FALLBACK_RESPONSE)
        service = SemanticSearchService(VectorStore([]), llm)

        self.assertEqual(service.answer("unknown"), FALLBACK_RESPONSE)
        context_section = llm.prompts[0].split("REGRAS:", 1)[0]
        self.assertEqual(context_section.strip(), "CONTEXTO:")

    def test_response_content_blocks_are_normalized(self):
        llm = LLM([{"type": "text", "text": "part one"}, {"text": " part two"}])
        service = SemanticSearchService(VectorStore([]), llm)

        self.assertEqual(service.answer("question"), "part one part two")

    def test_retrieval_errors_are_safe_and_distinct(self):
        service = SemanticSearchService(
            VectorStore(error=RuntimeError("password=secret")),
            LLM(),
        )

        with self.assertRaises(RetrievalError) as caught:
            service.answer("question")
        self.assertNotIn("secret", str(caught.exception))
        self.assertIn("PostgreSQL", str(caught.exception))

    def test_generation_errors_are_safe_and_distinct(self):
        service = SemanticSearchService(
            VectorStore([]),
            LLM(error=RuntimeError("api-key-secret")),
        )

        with self.assertRaises(GenerationError) as caught:
            service.answer("question")
        self.assertNotIn("secret", str(caught.exception))
        self.assertIn("modelo", str(caught.exception))

    def test_initialization_reuses_shared_settings(self):
        settings = make_settings()
        calls = []

        service = initialize_search(
            settings,
            embeddings_factory=lambda received: calls.append(("embedding", received)) or "e",
            vector_store_factory=lambda received, embeddings: calls.append(
                ("store", received, embeddings)
            )
            or "v",
            chat_model_factory=lambda received: calls.append(("llm", received)) or "l",
        )

        self.assertEqual(service.vector_store, "v")
        self.assertEqual(service.llm, "l")
        self.assertTrue(all(call[1] is settings for call in calls))


if __name__ == "__main__":
    unittest.main()
