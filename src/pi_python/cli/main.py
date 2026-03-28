from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pi_python.agent.loop import Agent
from pi_python.agent.providers.base import ProviderError
from pi_python.agent.providers.zai import ZAIConfig, ZAIProvider
from pi_python.agent.tools import ToolRegistry
from pi_python.cli.session import SessionStore

DEFAULT_SYSTEM_PROMPT = (
    "You are pi-python, a minimal coding agent. "
    "Use tools when needed and finish with a direct assistant answer."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pi-python")
    parser.add_argument("--prompt", help="Run one prompt and exit.")
    parser.add_argument(
        "--session",
        help="Load and persist conversation state under .pi-python/sessions/<name>.json.",
    )
    parser.add_argument("--api-key", default=os.getenv("ZAI_API_KEY", ""))
    parser.add_argument("--model", default="glm-5.1")
    parser.add_argument("--root", default=os.getcwd(), help="Workspace root for tools.")
    parser.add_argument("--max-iterations", type=int, default=8)
    return parser


def build_agent_from_args(args: argparse.Namespace) -> Agent:
    if not args.api_key:
        raise ValueError("A ZAI API key is required. Pass --api-key or set ZAI_API_KEY.")

    provider = ZAIProvider(
        config=ZAIConfig(
            api_key=args.api_key,
            model=args.model,
        )
    )
    return Agent(
        provider=provider,
        tools=ToolRegistry(root=Path(args.root)),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_iterations=args.max_iterations,
    )


def run_cli(args: argparse.Namespace, agent: Agent | None = None) -> int:
    try:
        active_agent = agent or build_agent_from_args(args)
        session_store = SessionStore(root=Path(args.root)) if args.session else None
        session_messages = (
            session_store.load(args.session).messages if session_store and args.session else []
        )

        if args.prompt:
            result = active_agent.run(args.prompt, messages=session_messages)
            if session_store and args.session:
                session_store.save(args.session, result.messages)
            print(result.output)
            return 0

        while True:
            try:
                prompt = input(">>> ").strip()
            except EOFError:
                print()
                return 0

            if not prompt:
                continue
            if prompt.lower() in {"exit", "quit"}:
                return 0

            result = active_agent.run(prompt, messages=session_messages)
            session_messages = result.messages
            if session_store and args.session:
                session_store.save(args.session, session_messages)
            print(result.output)
    except (ProviderError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
