from __future__ import annotations

import argparse
import os
from pathlib import Path

from pi_python.agent.loop import Agent, AgentResult
from pi_python.agent.providers.zai import ZAIConfig, ZAIProvider
from pi_python.agent.tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are pi-python, a minimal coding agent. "
    "Use tools when needed and finish with a direct assistant answer."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pi-python")
    parser.add_argument("--prompt", help="Run one prompt and exit.")
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
    active_agent = agent or build_agent_from_args(args)
    if args.prompt:
        result = active_agent.run(args.prompt)
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

        result: AgentResult = active_agent.run(prompt)
        print(result.output)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
