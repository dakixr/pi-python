from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Annotated

import typer

from pi.agent.loop import Agent, AgentResult, MaxIterationsExceededError
from pi.agent.models import Message
from pi.agent.providers.base import ProviderError
from pi.agent.providers.zai import ZAIConfig, ZAIProvider
from pi.agent.tools import ToolRegistry
from pi.cli.render import StatusIndicator, format_user_separator
from pi.cli.session import SessionStore

DEFAULT_SYSTEM_PROMPT = (
    "You are pi, a minimal coding agent. "
    "Use only the provided tools, do not invent unavailable capabilities, "
    "and finish with a direct assistant answer."
)
DEFAULT_MODEL = "glm-5.1"
DEFAULT_MAX_ITERATIONS = 20
RUN_ERRORS = (ProviderError, MaxIterationsExceededError)


@dataclass(slots=True)
class CLIArgs:
    prompt: str | None = None
    session: str | None = None
    api_key: str = ""
    model: str = DEFAULT_MODEL
    root: str = os.getcwd()
    max_iterations: int = DEFAULT_MAX_ITERATIONS


def build_agent_from_args(args: CLIArgs) -> Agent:
    if not args.api_key:
        raise ValueError("A ZAI API key is required. Pass --api-key or set ZAI_API_KEY.")
    return Agent(
        provider=ZAIProvider(
            config=ZAIConfig(
                api_key=args.api_key,
                model=args.model,
                debug_log_path=str(Path(args.root) / ".pi" / "logs" / "zai-debug.jsonl"),
            )
        ),
        tools=ToolRegistry(root=Path(args.root)),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_iterations=args.max_iterations,
    )


def execute_turn(
    agent: Agent,
    prompt: str,
    messages: list[Message],
    *,
    stderr: object | None = None,
    indicator: StatusIndicator | None = None,
) -> AgentResult:
    active_indicator = indicator or StatusIndicator(stderr or sys.stderr)
    run_signature = inspect.signature(agent.run)
    with active_indicator:
        if "on_event" in run_signature.parameters:
            return agent.run(prompt, messages=messages, on_event=active_indicator.handle_event)
        return agent.run(prompt, messages=messages)


def run_interactive_cli(
    args: CLIArgs,
    agent: Agent,
    session_messages: list[Message],
    *,
    session_store: SessionStore | None,
    input_func: object,
    stderr: object,
) -> int:
    messages = list(session_messages)
    while True:
        try:
            prompt = input_func(">>> ").strip()
        except EOFError:
            print()
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            return 0
        print(format_user_separator(prompt))
        try:
            result = execute_turn(agent, prompt, messages, stderr=stderr)
        except RUN_ERRORS as exc:
            print(f"Error: {exc}", file=stderr)
            continue
        messages = result.messages
        if session_store and args.session:
            session_store.save(args.session, messages)
        print(result.output)


def run_cli(
    args: CLIArgs,
    agent: Agent | None = None,
    *,
    input_func: object = input,
    stderr: object | None = None,
) -> int:
    stream = stderr or sys.stderr
    try:
        active_agent = agent or build_agent_from_args(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=stream)
        return 1
    session_store = SessionStore(root=Path(args.root)) if args.session else None
    session_messages = session_store.load(args.session).messages if session_store and args.session else []
    if args.prompt:
        try:
            result = execute_turn(active_agent, args.prompt, session_messages, stderr=stream)
        except RUN_ERRORS as exc:
            print(f"Error: {exc}", file=stream)
            return 1
        if session_store and args.session:
            session_store.save(args.session, result.messages)
        print(result.output)
        return 0
    return run_interactive_cli(
        args,
        active_agent,
        session_messages,
        session_store=session_store,
        input_func=input_func,
        stderr=stream,
    )


app = typer.Typer(add_completion=False, help="Minimal Python coding agent CLI.")


@app.command()
def cli(
    prompt: Annotated[str | None, typer.Option("--prompt", help="Run one prompt and exit.")] = None,
    session: Annotated[
        str | None,
        typer.Option("--session", help="Persist conversation under .pi/sessions/<name>.json."),
    ] = None,
    api_key: Annotated[str, typer.Option("--api-key", envvar="ZAI_API_KEY", help="Z.AI API key.")] = "",
    model: Annotated[str, typer.Option("--model", help="Model name to send to the provider.")] = DEFAULT_MODEL,
    root: Annotated[
        Path,
        typer.Option("--root", help="Workspace root for tools.", file_okay=False, dir_okay=True, resolve_path=True),
    ] = Path.cwd(),
    max_iterations: Annotated[
        int,
        typer.Option("--max-iterations", min=1, help="Maximum number of agent iterations per prompt."),
    ] = DEFAULT_MAX_ITERATIONS,
) -> None:
    raise typer.Exit(
        run_cli(
            CLIArgs(
                prompt=prompt,
                session=session,
                api_key=api_key,
                model=model,
                root=str(root),
                max_iterations=max_iterations,
            )
        )
    )


def main(argv: list[str] | None = None) -> int:
    try:
        app(args=argv, prog_name="pi-core", standalone_mode=False)
    except typer.Exit as exc:
        return int(exc.exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
