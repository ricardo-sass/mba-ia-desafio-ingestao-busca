from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import ConfigurationError, Settings, load_settings
from providers import create_chat_model, create_embeddings, create_vector_store


class FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def make_settings(provider: str) -> Settings:
    return Settings(
        ai_provider=provider,
        api_key="test-key",
        database_url="postgresql+psycopg://user:password@localhost/db",
        pdf_path=ROOT / "document.pdf",
        collection_name="test_collection",
        embedding_model=(
            "text-embedding-3-small"
            if provider == "openai"
            else "models/embedding-001"
        ),
        llm_model="gpt-5-nano" if provider == "openai" else "gemini-test",
    )


class SettingsTests(unittest.TestCase):
    def test_openai_defaults_and_only_openai_key_is_required(self):
        settings = load_settings({"OPENAI_API_KEY": "openai-test"})

        self.assertEqual(settings.ai_provider, "openai")
        self.assertEqual(settings.embedding_model, "text-embedding-3-small")
        self.assertEqual(settings.llm_model, "gpt-5-nano")
        self.assertEqual(settings.collection_name, "document_pdf")
        self.assertEqual(settings.pdf_path, ROOT / "document.pdf")

    def test_gemini_defaults_and_only_google_key_is_required(self):
        settings = load_settings(
            {"AI_PROVIDER": "GeMiNi", "GOOGLE_API_KEY": "google-test"}
        )

        self.assertEqual(settings.ai_provider, "gemini")
        self.assertEqual(settings.embedding_model, "models/embedding-001")
        self.assertEqual(settings.llm_model, "gemini-2.5-flash-lite")

    def test_unsupported_provider_is_rejected(self):
        with self.assertRaisesRegex(ConfigurationError, "openai, gemini"):
            load_settings({"AI_PROVIDER": "unsupported"})

    def test_selected_provider_key_is_required(self):
        with self.assertRaisesRegex(ConfigurationError, "OPENAI_API_KEY"):
            load_settings({"AI_PROVIDER": "openai"})
        with self.assertRaisesRegex(ConfigurationError, "GOOGLE_API_KEY"):
            load_settings({"AI_PROVIDER": "gemini"})


class ProviderFactoryTests(unittest.TestCase):
    def test_openai_factories_receive_requested_models(self):
        settings = make_settings("openai")

        embeddings = create_embeddings(settings, openai_cls=FakeClient)
        chat = create_chat_model(settings, openai_cls=FakeClient)

        self.assertEqual(embeddings.kwargs["model"], "text-embedding-3-small")
        self.assertEqual(chat.kwargs["model"], "gpt-5-nano")
        self.assertEqual(embeddings.kwargs["api_key"], "test-key")

    def test_gemini_factories_receive_requested_models(self):
        settings = make_settings("gemini")

        embeddings = create_embeddings(settings, gemini_cls=FakeClient)
        chat = create_chat_model(settings, gemini_cls=FakeClient)

        self.assertEqual(embeddings.kwargs["model"], "models/embedding-001")
        self.assertEqual(embeddings.kwargs["google_api_key"], "test-key")
        self.assertEqual(chat.kwargs["temperature"], 0)

    def test_vector_store_uses_shared_settings_and_replace_flag(self):
        settings = make_settings("openai")
        vector_store = create_vector_store(
            settings,
            object(),
            replace_collection=True,
            vector_store_cls=FakeClient,
        )

        self.assertEqual(vector_store.kwargs["collection_name"], "test_collection")
        self.assertEqual(vector_store.kwargs["connection"], settings.database_url)
        self.assertTrue(vector_store.kwargs["pre_delete_collection"])


if __name__ == "__main__":
    unittest.main()
