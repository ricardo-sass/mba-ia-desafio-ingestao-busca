"""Semantic retrieval and context-grounded answer generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from config import Settings, load_settings
from providers import create_chat_model, create_embeddings, create_vector_store


FALLBACK_RESPONSE = "Não tenho informações necessárias para responder sua pergunta."

PROMPT_TEMPLATE = """CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


class SearchError(RuntimeError):
    """Base exception for safe errors exposed to the terminal interface."""


class SearchInitializationError(SearchError):
    """Raised when vector or model clients cannot be initialized."""


class RetrievalError(SearchError):
    """Raised when vector retrieval fails."""


class GenerationError(SearchError):
    """Raised when answer generation fails or returns no text."""


def build_context(results: Iterable[tuple[Any, float]]) -> str:
    """Join retrieved page content in result order, excluding scores/metadata."""

    contents = [
        str(document.page_content).strip()
        for document, _score in results
        if str(document.page_content).strip()
    ]
    return "\n\n---\n\n".join(contents)


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(getattr(block, "text", None), str):
                parts.append(block.text)
        text = "".join(parts).strip()
    else:
        text = str(content).strip() if content is not None else ""

    if not text:
        raise GenerationError("O modelo não retornou uma resposta em texto.")
    return text


@dataclass(slots=True)
class SemanticSearchService:
    vector_store: Any
    llm: Any

    def answer(self, question: str) -> str:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("A pergunta não pode estar vazia.")

        try:
            results = self.vector_store.similarity_search_with_score(
                normalized_question,
                k=10,
            )
        except Exception as exc:
            raise RetrievalError(
                "Não foi possível consultar os vetores no PostgreSQL."
            ) from exc

        prompt = PROMPT_TEMPLATE.format(
            contexto=build_context(results),
            pergunta=normalized_question,
        )
        try:
            response = self.llm.invoke(prompt)
        except Exception as exc:
            raise GenerationError(
                "Não foi possível gerar a resposta com o modelo configurado."
            ) from exc
        return _response_text(response)


def initialize_search(
    settings: Settings | None = None,
    *,
    embeddings_factory: Callable[[Settings], Any] = create_embeddings,
    chat_model_factory: Callable[[Settings], Any] = create_chat_model,
    vector_store_factory: Callable[..., Any] = create_vector_store,
) -> SemanticSearchService:
    """Initialize reusable retrieval and generation clients once."""

    active_settings = settings or load_settings()
    try:
        embeddings = embeddings_factory(active_settings)
        vector_store = vector_store_factory(active_settings, embeddings)
        llm = chat_model_factory(active_settings)
    except Exception as exc:
        raise SearchInitializationError(
            "Não foi possível inicializar o banco vetorial ou o provedor de IA."
        ) from exc
    return SemanticSearchService(vector_store=vector_store, llm=llm)


def search_prompt(
    question: str | None = None,
    *,
    service: SemanticSearchService | None = None,
) -> SemanticSearchService | str:
    """Compatibility helper: initialize the service or answer one question."""

    active_service = service or initialize_search()
    if question is None:
        return active_service
    return active_service.answer(question)
