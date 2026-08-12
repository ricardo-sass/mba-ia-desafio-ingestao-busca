"""Interactive terminal chat for the indexed PDF."""

from __future__ import annotations

from typing import Any, Callable

from config import ConfigurationError
from search import SearchError, initialize_search


EXIT_COMMANDS = frozenset({"sair", "exit", "quit"})


def main(
    *,
    service_factory: Callable[[], Any] = initialize_search,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    try:
        service = service_factory()
    except (ConfigurationError, SearchError) as exc:
        output_fn(f"Não foi possível iniciar o chat: {exc}")
        return 1
    except Exception:
        output_fn(
            "Não foi possível iniciar o chat. Verifique a configuração e os serviços."
        )
        return 1

    output_fn("Chat iniciado. Digite sua pergunta ou 'sair' para encerrar.")

    while True:
        try:
            question = input_fn("\nFaça sua pergunta:\n\n")
        except (EOFError, KeyboardInterrupt):
            output_fn("\nChat encerrado.")
            return 0

        question = question.strip()
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            output_fn("Chat encerrado.")
            return 0

        try:
            answer = service.answer(question)
        except SearchError as exc:
            output_fn(f"ERRO: {exc}")
            continue
        except Exception:
            output_fn("ERRO: Não foi possível processar esta pergunta.")
            continue

        output_fn(f"PERGUNTA: {question}\nRESPOSTA: {answer}\n\n---")


if __name__ == "__main__":
    raise SystemExit(main())
