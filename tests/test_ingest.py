from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import Settings
from ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    IngestionError,
    ingest_pdf,
    load_pdf,
    persist_chunks,
    split_documents,
)


def document(text: str, **metadata):
    return SimpleNamespace(page_content=text, metadata=dict(metadata))


def make_settings(pdf_path: Path) -> Settings:
    return Settings(
        ai_provider="openai",
        api_key="test-key",
        database_url="postgresql+psycopg://user:password@localhost/db",
        pdf_path=pdf_path,
        collection_name="current_document",
        embedding_model="text-embedding-3-small",
        llm_model="gpt-5-nano",
    )


class IngestionTests(unittest.TestCase):
    def test_missing_pdf_fails_before_loader_is_created(self):
        loader_called = False

        def loader_cls(_path):
            nonlocal loader_called
            loader_called = True

        with self.assertRaisesRegex(IngestionError, "não encontrado"):
            load_pdf(ROOT / "missing.pdf", loader_cls=loader_cls)
        self.assertFalse(loader_called)

    def test_load_pdf_preserves_metadata_and_adds_missing_values(self):
        class Loader:
            def __init__(self, path):
                self.path = path

            def load(self):
                return [document("one", page=7, custom="kept"), document("two")]

        pages = load_pdf(ROOT / "document.pdf", loader_cls=Loader)

        self.assertEqual(pages[0].metadata["page"], 7)
        self.assertEqual(pages[0].metadata["custom"], "kept")
        self.assertEqual(pages[1].metadata["page"], 1)
        self.assertEqual(pages[1].metadata["source"], str(ROOT / "document.pdf"))

    def test_splitter_uses_exact_parameters_and_removes_empty_chunks(self):
        captured = {}

        class Splitter:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def split_documents(self, pages):
                self.pages = pages
                return [document("useful", page=0), document("   ", page=0)]

        chunks = split_documents([document("page")], splitter_cls=Splitter)

        self.assertEqual(captured["chunk_size"], CHUNK_SIZE)
        self.assertEqual(captured["chunk_overlap"], CHUNK_OVERLAP)
        self.assertEqual([chunk.page_content for chunk in chunks], ["useful"])

    def test_empty_text_is_rejected_before_persistence(self):
        class Splitter:
            def __init__(self, **_kwargs):
                pass

            def split_documents(self, _pages):
                return [document("  ")]

        with self.assertRaisesRegex(IngestionError, "texto útil"):
            split_documents([document("page")], splitter_cls=Splitter)

    def test_persistence_replaces_only_configured_collection_each_run(self):
        calls = []

        class Store:
            def add_documents(self, chunks):
                calls[-1]["chunks"] = chunks

        def vector_factory(settings, embeddings, **kwargs):
            calls.append(
                {
                    "collection": settings.collection_name,
                    "embeddings": embeddings,
                    **kwargs,
                }
            )
            return Store()

        settings = make_settings(ROOT / "document.pdf")
        chunks = [document("chunk")]
        for _ in range(2):
            persist_chunks(
                chunks,
                settings,
                embeddings_factory=lambda _settings: "embedding-client",
                vector_store_factory=vector_factory,
            )

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call["replace_collection"] for call in calls))
        self.assertTrue(
            all(call["collection"] == "current_document" for call in calls)
        )

    def test_complete_pipeline_returns_non_empty_chunk_count(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
            settings = make_settings(Path(temp_pdf.name))

            class Loader:
                def __init__(self, _path):
                    pass

                def load(self):
                    return [document("page text")]

            class Splitter:
                def __init__(self, **_kwargs):
                    pass

                def split_documents(self, pages):
                    return [document(pages[0].page_content, **pages[0].metadata)]

            class Store:
                def add_documents(self, chunks):
                    self.chunks = chunks

            count = ingest_pdf(
                settings,
                loader_cls=Loader,
                splitter_cls=Splitter,
                embeddings_factory=lambda _settings: object(),
                vector_store_factory=lambda *_args, **_kwargs: Store(),
            )

        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
