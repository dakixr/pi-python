from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from pi.agent.loop import AgentResult, MaxIterationsExceededError
from pi.agent.models import Message
from pi.agent.providers.base import ProviderError
from pi.cli.main import CLIArgs, InteractiveRenderer, app, run_cli


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
    exit_code = run_cli(CLIArgs(prompt="hello world"), agent=FakeAgent())

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "synthetic response"


class FakeTTY:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        with self._lock:
            self._parts.append(text)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return True

    def getvalue(self) -> str:
        with self._lock:
            return "".join(self._parts)


def test_cli_prints_minimal_tool_trace_in_tty_mode(capsys) -> None:
    class ToolTracingAgent:
        def run(
            self,
            prompt: str,
            messages: list[Message] | None = None,
            *,
            on_event=None,
        ) -> AgentResult:
            if on_event is not None:
                on_event("model_start", {"iteration": 1})
                on_event(
                    "tool_start",
                    {
                        "iteration": 1,
                        "tool_name": "read",
                        "tool_arguments": '{"path":"README.md"}',
                    },
                )
                on_event(
                    "tool_end",
                    {
                        "iteration": 1,
                        "tool_name": "read",
                        "ok": True,
                        "result": {"ok": True, "path": "README.md", "content": "hello"},
                    },
                )
            return AgentResult(
                output="synthetic response",
                messages=[Message.user(prompt), Message.assistant("synthetic response")],
                iterations=1,
            )

    stderr = FakeTTY()
    exit_code = run_cli(CLIArgs(prompt="hello world"), agent=ToolTracingAgent(), stderr=stderr)

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "synthetic response"
    assert "tool read README.md" in stderr.getvalue()


def test_cli_renders_file_diffs_for_mutation_tools_in_tty_mode(capsys) -> None:
    class DiffRenderingAgent:
        def run(
            self,
            prompt: str,
            messages: list[Message] | None = None,
            *,
            on_event=None,
        ) -> AgentResult:
            if on_event is not None:
                on_event("model_start", {"iteration": 1})
                on_event(
                    "tool_start",
                    {
                        "iteration": 1,
                        "tool_name": "edit",
                        "tool_arguments": '{"path":"src/pi/cli/main.py"}',
                    },
                )
                on_event(
                    "tool_end",
                    {
                        "iteration": 1,
                        "tool_name": "edit",
                        "ok": True,
                        "result": {
                            "ok": True,
                            "path": "src/pi/cli/main.py",
                            "edits_applied": 1,
                            "diff": (
                                "--- a/src/pi/cli/main.py\n"
                                "+++ b/src/pi/cli/main.py\n"
                                "@@ -1 +1 @@\n"
                                '-old = "x"\n'
                                '+new = "y"'
                            ),
                            "diff_truncated": False,
                        },
                    },
                )
            return AgentResult(
                output="done",
                messages=[Message.user(prompt), Message.assistant("done")],
                iterations=1,
            )

    stderr = FakeTTY()
    exit_code = run_cli(CLIArgs(prompt="apply patch"), agent=DiffRenderingAgent(), stderr=stderr)

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "done"
    rendered = stderr.getvalue()
    assert "tool edit src/pi/cli/main.py" in rendered
    assert "--- a/src/pi/cli/main.py" in rendered
    assert "+++ b/src/pi/cli/main.py" in rendered
    assert "-old = \"x\"" in rendered
    assert "+new = \"y\"" in rendered


def test_interactive_renderer_keeps_all_tool_calls_visible() -> None:
    stream = FakeTTY()

    with InteractiveRenderer(stream) as renderer:
        renderer.handle_event("model_start", {"iteration": 1})
        renderer.handle_event(
            "tool_start",
            {
                "iteration": 1,
                "tool_name": "read",
                "tool_arguments": '{"path":"README.md"}',
            },
        )
        renderer.handle_event(
            "tool_end",
            {
                "iteration": 1,
                "tool_name": "read",
                "ok": True,
                "result": {"ok": True, "path": "README.md", "content": "hello"},
            },
        )
        renderer.handle_event(
            "tool_start",
            {
                "iteration": 1,
                "tool_name": "read",
                "tool_arguments": '{"path":"src/pi/cli/main.py"}',
            },
        )
        renderer.handle_event(
            "tool_end",
            {
                "iteration": 1,
                "tool_name": "read",
                "ok": True,
                "result": {"ok": True, "path": "src/pi/cli/main.py", "content": "hello"},
            },
        )
        renderer.set_input_buffer("next")

        rendered = stream.getvalue()

    assert "thinking" in rendered
    assert "tool read README.md" in rendered
    assert "tool read src/pi/cli/main.py" in rendered
    assert ">>> next" in rendered
    assert rendered.rfind("tool read src/pi/cli/main.py") < rendered.rfind("thinking")


def test_interactive_renderer_prints_user_separator_as_own_block() -> None:
    stream = FakeTTY()

    with InteractiveRenderer(stream) as renderer:
        renderer.set_input_buffer("queued")
        renderer.print_message("user:\ncheck how this works\n---")

        rendered = stream.getvalue()

    assert "user:\ncheck how this works\n---\n" in rendered


def test_interactive_renderer_collapses_queue_preview_to_one_line() -> None:
    stream = FakeTTY()

    with InteractiveRenderer(stream) as renderer:
        renderer.set_status("thinking")
        renderer.set_queue_messages(["first queued message", "second queued message", "third queued message"])
        rendered = stream.getvalue()

    assert "[3 queued messages]" in rendered
    assert "queued second queued message | third queued message | +1 message" in rendered


def test_interactive_renderer_shows_thinking_spinner() -> None:
    stream = FakeTTY()

    with InteractiveRenderer(stream) as renderer:
        renderer.set_status("thinking")
        time.sleep(0.15)
        rendered = stream.getvalue()

    assert any(f"thinking {frame}" in rendered for frame in ("|", "/", "-", "\\"))


def test_cli_persists_and_reuses_named_session(tmp_path: Path, capsys) -> None:
    agent = FakeAgent(outputs=["first response", "second response"])

    first_exit_code = run_cli(
        CLIArgs(prompt="first prompt", session="demo", root=str(tmp_path)),
        agent=agent,
    )
    first_output = capsys.readouterr().out.strip()
    second_exit_code = run_cli(
        CLIArgs(prompt="second prompt", session="demo", root=str(tmp_path)),
        agent=agent,
    )
    second_output = capsys.readouterr().out.strip()

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert first_output == "first response"
    assert second_output == "second response"
    assert agent.seen_history_lengths == [0, 2]

    session_path = tmp_path / ".pi" / "sessions" / "demo.json"
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

    exit_code = run_cli(CLIArgs(prompt="hello world"), agent=FailingAgent())
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert (
        captured.err.strip()
        == "Error: ZAI request failed after 3 attempts: Rate limit exceeded"
    )


def test_cli_surfaces_configuration_errors_cleanly(capsys) -> None:
    exit_code = run_cli(CLIArgs(prompt="hello world", api_key=""))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.strip() == (
        "Error: A ZAI API key is required. Pass --api-key or set ZAI_API_KEY."
    )


def test_cli_handles_max_iterations_without_traceback(capsys) -> None:
    class StuckAgent:
        def run(self, prompt: str, messages: list[Message] | None = None) -> AgentResult:
            raise MaxIterationsExceededError("Agent exceeded max_iterations=20")

    exit_code = run_cli(CLIArgs(prompt="hello world"), agent=StuckAgent())
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.strip() == "Error: Agent exceeded max_iterations=20"


def test_interactive_cli_continues_after_turn_error(capsys) -> None:
    class RecoveringAgent:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, prompt: str, messages: list[Message] | None = None) -> AgentResult:
            self.calls += 1
            if self.calls == 1:
                raise MaxIterationsExceededError("Agent exceeded max_iterations=20")
            conversation = list(messages or [])
            conversation.append(Message.user(prompt))
            conversation.append(Message.assistant("done"))
            return AgentResult(output="done", messages=conversation, iterations=1)

    prompts = iter(["first task", "second task", "quit"])

    def fake_input(_: str) -> str:
        return next(prompts)

    exit_code = run_cli(CLIArgs(), agent=RecoveringAgent(), input_func=fake_input)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.splitlines() == ["user:", "second task", "---", "done"]
    assert captured.err.strip() == "Error: Agent exceeded max_iterations=20"


def test_interactive_cli_queues_messages_while_turn_is_running(capsys) -> None:
    class QueueingAgent:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []
            self.started = threading.Event()
            self.allow_finish = threading.Event()

        def run(
            self,
            prompt: str,
            messages: list[Message] | None = None,
            *,
            on_event=None,
        ) -> AgentResult:
            self.calls.append((prompt, len(messages or [])))
            if prompt == "first task":
                self.started.set()
                self.allow_finish.wait(timeout=1)
            conversation = list(messages or [])
            conversation.append(Message.user(prompt))
            conversation.append(Message.assistant(f"done {prompt}"))
            return AgentResult(
                output=f"done {prompt}",
                messages=conversation,
                iterations=1,
            )

    agent = QueueingAgent()
    stderr = FakeTTY()
    prompt_calls = 0

    def fake_input(_: str) -> str:
        nonlocal prompt_calls
        prompt_calls += 1
        if prompt_calls == 1:
            return "first task"
        if prompt_calls == 2:
            assert agent.started.wait(timeout=1), "second prompt was not requested while busy"
            return "second task"
        agent.allow_finish.set()
        return "quit"

    result: dict[str, int] = {}

    def run() -> None:
        result["exit_code"] = run_cli(
            CLIArgs(),
            agent=agent,
            input_func=fake_input,
            stderr=stderr,
        )

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=2)
    if thread.is_alive():
        agent.allow_finish.set()
        thread.join(timeout=2)
        raise AssertionError("interactive CLI did not keep accepting input while the turn was running")

    captured = capsys.readouterr()

    assert result["exit_code"] == 0
    assert "queued second task" in stderr.getvalue()
    assert agent.calls == [("first task", 0), ("second task", 2)]
    assert captured.out.splitlines() == [
        "done first task",
        "user:",
        "second task",
        "---",
        "done second task",
    ]


def test_interactive_cli_drops_queued_messages_after_turn_error(capsys) -> None:
    class FailingQueueAgent:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.started = threading.Event()
            self.allow_fail = threading.Event()

        def run(
            self,
            prompt: str,
            messages: list[Message] | None = None,
            *,
            on_event=None,
        ) -> AgentResult:
            self.calls.append(prompt)
            if prompt == "first task":
                self.started.set()
                self.allow_fail.wait(timeout=1)
                raise ProviderError("ZAI request failed: bad payload")
            conversation = list(messages or [])
            conversation.append(Message.user(prompt))
            conversation.append(Message.assistant(f"done {prompt}"))
            return AgentResult(
                output=f"done {prompt}",
                messages=conversation,
                iterations=1,
            )

    agent = FailingQueueAgent()
    stderr = FakeTTY()
    prompt_calls = 0

    def fake_input(_: str) -> str:
        nonlocal prompt_calls
        prompt_calls += 1
        if prompt_calls == 1:
            return "first task"
        if prompt_calls == 2:
            assert agent.started.wait(timeout=1)
            return "second task"
        agent.allow_fail.set()
        return "quit"

    result: dict[str, int] = {}

    def run() -> None:
        result["exit_code"] = run_cli(
            CLIArgs(),
            agent=agent,
            input_func=fake_input,
            stderr=stderr,
        )

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=2)
    if thread.is_alive():
        agent.allow_fail.set()
        thread.join(timeout=2)
        raise AssertionError("interactive CLI did not finish after dropping queued messages")

    captured = capsys.readouterr()

    assert result["exit_code"] == 0
    assert captured.out == ""
    assert agent.calls == ["first task"]
    assert "queued second task" in stderr.getvalue()
    assert "Dropped 1 queued message after turn error." in stderr.getvalue()


def test_typer_entrypoint_invokes_cli() -> None:
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert "--max-iterations" in result.stdout
