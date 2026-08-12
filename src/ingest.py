"""Load, split, embed, and persist the configured PDF."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from config import ConfigurationError, Settings, load_settings
from providers import create_embeddings, create_vector_store


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


class IngestionError(RuntimeError):
    """Raised when the PDF ingestion pipeline cannot finish safely."""


def load_pdf(
    pdf_path: Path,
    *,
    loader_cls: Callable[[str], Any] | None = None,
) -> list[Any]:
    """Load all readable PDF pages and retain their source/page metadata."""

    if not pdf_path.is_file():
        raise IngestionError(f"Arquivo PDF não encontrado: {pdf_path}")

    if loader_cls is None:
        from langchain_community.document_loaders import PyPDFLoader

        loader_cls = PyPDFLoader

    try:
        pages = list(loader_cls(str(pdf_path)).load())
    except Exception as exc:
        raise IngestionError(
            "Não foi possível ler o PDF. Verifique se o arquivo é válido e não está protegido."
        ) from exc

    if not pages:
        raise IngestionError("O PDF não contém páginas legíveis para ingestão.")

    for page_number, page in enumerate(pages):
        page.metadata.setdefault("source", str(pdf_path))
        page.metadata.setdefault("page", page_number)
    return pages


def split_documents(
    documents: Iterable[Any],
    *,
    splitter_cls: Callable[..., Any] | None = None,
) -> list[Any]:
    """Split pages with the parameters required by the challenge."""

    if splitter_cls is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter_cls = RecursiveCharacterTextSplitter

    splitter = splitter_cls(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(list(documents))
    non_empty_chunks = [
        chunk for chunk in chunks if str(chunk.page_content).strip()
    ]
    if not non_empty_chunks:
        raise IngestionError("O PDF não contém texto útil para vetorização.")
    return non_empty_chunks


def persist_chunks(
    chunks: Sequence[Any],
    settings: Settings,
    *,
    embeddings_factory: Callable[[Settings], Any] = create_embeddings,
    vector_store_factory: Callable[..., Any] = create_vector_store,
) -> None:
    """Replace only the configured collection, then store all current chunks."""

    try:
        embeddings = embeddings_factory(settings)
    except Exception as exc:
        raise IngestionError(
            "Não foi possível inicializar o modelo de embeddings configurado."
        ) from exc

    try:
        vector_store = vector_store_factory(
            settings,
            embeddings,
            replace_collection=True,
        )
        vector_store.add_documents(list(chunks))
    except Exception as exc:
        raise IngestionError(
            "Não foi possível gerar ou persistir os vetores no PostgreSQL."
        ) from exc


def ingest_pdf(
    settings: Settings | None = None,
    *,
    loader_cls: Callable[[str], Any] | None = None,
    splitter_cls: Callable[..., Any] | None = None,
    embeddings_factory: Callable[[Settings], Any] = create_embeddings,
    vector_store_factory: Callable[..., Any] = create_vector_store,
) -> int:
    """Run the complete ingestion pipeline and return the stored chunk count."""

    active_settings = settings or load_settings()
    pages = load_pdf(active_settings.pdf_path, loader_cls=loader_cls)
    chunks = split_documents(pages, splitter_cls=splitter_cls)
    persist_chunks(
        chunks,
        active_settings,
        embeddings_factory=embeddings_factory,
        vector_store_factory=vector_store_factory,
    )
    return len(chunks)


def main() -> int:
    try:
        chunk_count = ingest_pdf()
    except (ConfigurationError, IngestionError) as exc:
        print(f"ERRO: {exc}")
        return 1
    except Exception:
        print("ERRO: Falha inesperada durante a ingestão.")
        return 1

    print(f"Ingestão concluída: {chunk_count} chunks armazenados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
