"""Runtime configuration shared by ingestion and semantic search."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # Allows dependency-free unit tests before installation.
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_PROVIDERS = ("openai", "gemini")


class ConfigurationError(RuntimeError):
    """Raised when required runtime settings are invalid or missing."""


@dataclass(frozen=True, slots=True)
class Settings:
    ai_provider: str
    api_key: str
    database_url: str
    pdf_path: Path
    collection_name: str
    embedding_model: str
    llm_model: str


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Load settings, applying provider-specific defaults and validation.

    ``environ`` exists primarily to make configuration deterministic in tests. When
    omitted, values from a local ``.env`` file are loaded without overriding values
    already present in the process environment.
    """

    if environ is None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        values: Mapping[str, str] = os.environ
    else:
        values = environ

    provider = values.get("AI_PROVIDER", "openai").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        allowed = ", ".join(SUPPORTED_PROVIDERS)
        raise ConfigurationError(
            f"AI_PROVIDER inválido. Use um dos valores: {allowed}."
        )

    if provider == "openai":
        key_name = "OPENAI_API_KEY"
        default_embedding = "text-embedding-3-small"
        default_llm = "gpt-5-nano"
    else:
        key_name = "GOOGLE_API_KEY"
        default_embedding = "models/embedding-001"
        default_llm = "gemini-2.5-flash-lite"

    api_key = values.get(key_name, "").strip()
    if not api_key:
        raise ConfigurationError(
            f"A variável {key_name} é obrigatória para o provedor {provider}."
        )

    database_url = values.get(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/rag",
    ).strip()
    if not database_url:
        raise ConfigurationError("DATABASE_URL não pode estar vazia.")

    collection_name = values.get("COLLECTION_NAME", "document_pdf").strip()
    if not collection_name:
        raise ConfigurationError("COLLECTION_NAME não pode estar vazia.")

    configured_path = Path(values.get("PDF_PATH", "document.pdf").strip())
    if not configured_path.is_absolute():
        configured_path = PROJECT_ROOT / configured_path

    embedding_model = (values.get("EMBEDDING_MODEL") or default_embedding).strip()
    llm_model = (values.get("LLM_MODEL") or default_llm).strip()
    if not embedding_model:
        raise ConfigurationError("EMBEDDING_MODEL não pode estar vazio.")
    if not llm_model:
        raise ConfigurationError("LLM_MODEL não pode estar vazio.")

    return Settings(
        ai_provider=provider,
        api_key=api_key,
        database_url=database_url,
        pdf_path=configured_path.resolve(),
        collection_name=collection_name,
        embedding_model=embedding_model,
        llm_model=llm_model,
    )
