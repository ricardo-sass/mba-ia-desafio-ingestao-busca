from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chat import main
from search import GenerationError, SearchInitializationError


class Inputs:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self, _prompt):
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


class Service:
    def __init__(self, answers=None):
        self.answers = iter(answers or [])
        self.questions = []

    def answer(self, question):
        self.questions.append(question)
        answer = next(self.answers)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class ChatTests(unittest.TestCase):
    def run_chat(self, service, inputs):
        output = []
        code = main(
            service_factory=lambda: service,
            input_fn=Inputs(inputs),
            output_fn=output.append,
        )
        return code, output

    def test_labeled_question_and_answer_and_explicit_exit(self):
        service = Service(["The answer"])

        code, output = self.run_chat(service, ["Question?", "SaIr"])

        self.assertEqual(code, 0)
        self.assertEqual(service.questions, ["Question?"])
        self.assertTrue(any("PERGUNTA: Question?" in line for line in output))
        self.assertTrue(any("RESPOSTA: The answer" in line for line in output))

    def test_blank_input_is_ignored(self):
        service = Service(["answer"])

        self.run_chat(service, ["   ", "real question", "quit"])

        self.assertEqual(service.questions, ["real question"])

    def test_all_exit_commands_are_case_insensitive(self):
        for command in ("sair", "EXIT", "Quit"):
            with self.subTest(command=command):
                service = Service([])
                code, _output = self.run_chat(service, [command])
                self.assertEqual(code, 0)
                self.assertEqual(service.questions, [])

    def test_eof_and_keyboard_interrupt_exit_without_traceback(self):
        for exception in (EOFError(), KeyboardInterrupt()):
            with self.subTest(exception=type(exception).__name__):
                code, output = self.run_chat(Service([]), [exception])
                self.assertEqual(code, 0)
                self.assertTrue(any("encerrado" in line.lower() for line in output))

    def test_recoverable_question_failure_keeps_loop_running(self):
        service = Service([GenerationError("safe failure"), "answer"])

        code, output = self.run_chat(service, ["first", "second", "exit"])

        self.assertEqual(code, 0)
        self.assertEqual(service.questions, ["first", "second"])
        self.assertTrue(any("safe failure" in line for line in output))
        self.assertTrue(any("RESPOSTA: answer" in line for line in output))

    def test_initialization_failure_returns_nonzero_and_no_loop(self):
        output = []

        code = main(
            service_factory=lambda: (_ for _ in ()).throw(
                SearchInitializationError("safe initialization failure")
            ),
            input_fn=lambda _prompt: self.fail("input must not be called"),
            output_fn=output.append,
        )

        self.assertEqual(code, 1)
        self.assertIn("safe initialization failure", output[0])


if __name__ == "__main__":
    unittest.main()
