from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from pi_python.agent.loop import AgentResult
from pi_python.agent.models import Message
from pi_python.agent.providers.base import ProviderError
from pi_python.cli.main import build_parser, run_cli


@dataclass
class FakeAgent:
    outputs: list[str] = field(default_factory=lambda: ["synthetic response"])
    seen_history_lengths: list[int] = field(default_factory=list)

    def run(self, prompt: str, messages: list[Message] | None = None) -> AgentResult:
        self.seen_history_lengths.append(len(messages or []))
        output = self.outputs[min(len(self.seen_history_lengths) - 1, len(self.outputs) - 1)]
        conversation = list(messages or [])
        conversation.append(Message.user(prompt))
        conversation.append(Message.assistant(output))
        return AgentResult(
            output=output,
            messages=conversation,
            iterations=1,
        )


def test_cli_runs_one_shot_prompt(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["--prompt", "hello world"])

    exit_code = run_cli(args, agent=FakeAgent())

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "synthetic response"


def test_cli_persists_and_reuses_named_session(tmp_path: Path, capsys) -> None:
    parser = build_parser()
    agent = FakeAgent(outputs=["first response", "second response"])

    first_args = parser.parse_args(
        ["--prompt", "first prompt", "--session", "demo", "--root", str(tmp_path)]
    )
    second_args = parser.parse_args(
        ["--prompt", "second prompt", "--session", "demo", "--root", str(tmp_path)]
    )

    first_exit_code = run_cli(first_args, agent=agent)
    first_output = capsys.readouterr().out.strip()
    second_exit_code = run_cli(second_args, agent=agent)
    second_output = capsys.readouterr().out.strip()

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert first_output == "first response"
    assert second_output == "second response"
    assert agent.seen_history_lengths == [0, 2]

    session_path = tmp_path / ".pi-python" / "sessions" / "demo.json"
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    assert payload["id"] == "demo"
    assert [message["role"] for message in payload["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_cli_surfaces_provider_errors_cleanly(capsys) -> None:
    class FailingAgent:
        def run(self, prompt: str, messages: list[Message] | None = None) -> AgentResult:
            raise ProviderError("ZAI request failed after 3 attempts: Rate limit exceeded")

    parser = build_parser()
    args = parser.parse_args(["--prompt", "hello world"])

    exit_code = run_cli(args, agent=FailingAgent())
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert (
        captured.err.strip()
        == "Error: ZAI request failed after 3 attempts: Rate limit exceeded"
    )


def test_cli_surfaces_configuration_errors_cleanly(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["--prompt", "hello world", "--api-key", ""])

    exit_code = run_cli(args)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.strip() == (
        "Error: A ZAI API key is required. Pass --api-key or set ZAI_API_KEY."
    )
