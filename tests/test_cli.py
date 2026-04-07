from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
import time

from pi.agent.loop import AgentResult
from pi.agent.models import Message
from pi.agent.providers.base import ProviderError
from pi.cli.main import CLIArgs, app, run_cli


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
        return AgentResult(output=output, messages=conversation, iterations=1)


class CaptureStream(io.StringIO):
    def isatty(self) -> bool:
        return False


def test_cli_runs_one_shot_prompt(capsys) -> None:
    exit_code = run_cli(CLIArgs(prompt="hello world"), agent=FakeAgent())

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "synthetic response"


def test_cli_persists_and_reuses_named_session(tmp_path: Path, capsys) -> None:
    agent = FakeAgent(outputs=["first response", "second response"])

    first_exit_code = run_cli(CLIArgs(prompt="first prompt", session="demo", root=str(tmp_path)), agent=agent)
    first_output = capsys.readouterr().out.strip()
    second_exit_code = run_cli(CLIArgs(prompt="second prompt", session="demo", root=str(tmp_path)), agent=agent)
    second_output = capsys.readouterr().out.strip()

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert first_output == "first response"
    assert second_output == "second response"
    assert agent.seen_history_lengths == [0, 2]

    session_path = tmp_path / ".pi" / "sessions" / "demo.json"
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    assert payload["id"] == "demo"
    assert [message["role"] for message in payload["messages"]] == ["user", "assistant", "user", "assistant"]
    events_path = tmp_path / ".pi" / "sessions" / "demo" / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["type"] == "session"
    assert [entry["type"] for entry in events[1:]] == ["message", "message", "message", "message"]


def test_cli_surfaces_provider_errors_cleanly(capsys) -> None:
    class FailingAgent:
        def run(self, prompt: str, messages: list[Message] | None = None) -> AgentResult:
            raise ProviderError("ZAI request failed after 3 attempts: Rate limit exceeded")

    exit_code = run_cli(CLIArgs(prompt="hello world"), agent=FailingAgent())
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.strip() == "Error: ZAI request failed after 3 attempts: Rate limit exceeded"


def test_typer_entrypoint_invokes_cli() -> None:
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert "--max-iterations" in result.stdout


def test_interactive_cli_accepts_queued_prompts_while_busy(tmp_path: Path) -> None:
    @dataclass
    class SlowAgent:
        seen_history_lengths: list[int] = field(default_factory=list)

        def run(self, prompt: str, messages: list[Message] | None = None, *, on_event=None) -> AgentResult:
            self.seen_history_lengths.append(len(messages or []))
            if on_event is not None:
                on_event("model_start", {"iteration": 1})
                on_event("tool_execution_start", {"tool_name": "bash", "tool_arguments": '{"command":"sleep 0.1"}'})
            time.sleep(0.1)
            if on_event is not None:
                on_event("tool_execution_end", {"tool_name": "bash", "ok": True})
            conversation = list(messages or [])
            conversation.append(Message.user(prompt))
            conversation.append(Message.assistant(f"done {prompt}"))
            return AgentResult(output=f"done {prompt}", messages=conversation, iterations=1)

    prompts = iter(["first", "second", "quit"])

    def fake_input(_: object = "") -> str:
        return next(prompts)

    stdout = CaptureStream()
    stderr = CaptureStream()
    agent = SlowAgent()

    exit_code = run_cli(
        CLIArgs(root=str(tmp_path)),
        agent=agent,
        input_func=fake_input,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert "done first" in stdout.getvalue()
    assert "done second" in stdout.getvalue()
    assert agent.seen_history_lengths == [0, 2]


def test_interactive_cli_reports_tool_failures(tmp_path: Path) -> None:
    class FailingToolAgent:
        def run(self, prompt: str, messages: list[Message] | None = None, *, on_event=None) -> AgentResult:
            if on_event is not None:
                on_event("model_start", {"iteration": 1})
                on_event("tool_execution_start", {"tool_name": "bash", "tool_arguments": '{"command":"false"}'})
                on_event(
                    "tool_execution_end",
                    {"tool_name": "bash", "ok": False, "result": {"error": "command exited with status 1"}},
                )
            conversation = list(messages or [])
            conversation.append(Message.user(prompt))
            conversation.append(Message.assistant("recovered"))
            return AgentResult(output="recovered", messages=conversation, iterations=1)

    prompts = iter(["test tool failure", "quit"])

    def fake_input(_: object = "") -> str:
        return next(prompts)

    stdout = CaptureStream()
    stderr = CaptureStream()

    exit_code = run_cli(
        CLIArgs(root=str(tmp_path)),
        agent=FailingToolAgent(),
        input_func=fake_input,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert "tool bash false" in stdout.getvalue()
    assert "tool! command exited with status 1" in stdout.getvalue()
    assert "recovered" in stdout.getvalue()
