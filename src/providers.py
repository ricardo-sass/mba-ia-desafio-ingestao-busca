"""Factories for provider clients and the shared PGVector store."""

from __future__ import annotations

from typing import Any, Callable

from config import Settings


def create_embeddings(
    settings: Settings,
    *,
    openai_cls: Callable[..., Any] | None = None,
    gemini_cls: Callable[..., Any] | None = None,
) -> Any:
    """Create the embedding client selected in ``settings``."""

    if settings.ai_provider == "openai":
        if openai_cls is None:
            from langchain_openai import OpenAIEmbeddings

            openai_cls = OpenAIEmbeddings
        return openai_cls(model=settings.embedding_model, api_key=settings.api_key)

    if gemini_cls is None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        gemini_cls = GoogleGenerativeAIEmbeddings
    return gemini_cls(model=settings.embedding_model, google_api_key=settings.api_key)


def create_chat_model(
    settings: Settings,
    *,
    openai_cls: Callable[..., Any] | None = None,
    gemini_cls: Callable[..., Any] | None = None,
) -> Any:
    """Create the text generation client selected in ``settings``."""

    if settings.ai_provider == "openai":
        if openai_cls is None:
            from langchain_openai import ChatOpenAI

            openai_cls = ChatOpenAI
        return openai_cls(model=settings.llm_model, api_key=settings.api_key)

    if gemini_cls is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        gemini_cls = ChatGoogleGenerativeAI
    return gemini_cls(
        model=settings.llm_model,
        google_api_key=settings.api_key,
        temperature=0,
    )


def create_vector_store(
    settings: Settings,
    embeddings: Any,
    *,
    replace_collection: bool = False,
    vector_store_cls: Callable[..., Any] | None = None,
) -> Any:
    """Create a PGVector instance scoped to the configured collection."""

    if vector_store_cls is None:
        from langchain_postgres import PGVector

        vector_store_cls = PGVector

    return vector_store_cls(
        embeddings=embeddings,
        collection_name=settings.collection_name,
        connection=settings.database_url,
        use_jsonb=True,
        pre_delete_collection=replace_collection,
    )
