from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

from pi import __version__
from pi.agent.providers.openai_compat import OpenAICompatibleConfig, OpenAICompatibleProvider
from pi.agent.providers.zai import ZAIConfig, ZAIProvider
from pi.ai.sdk import CompletionResult, Context, StreamEvent, Tool, complete, create_agent, run_task, stream

PACKAGE_NAME = "ai"


@dataclass(slots=True, frozen=True)
class OAuthProviderInfo:
    id: str
    name: str


OAUTH_PROVIDERS = (
    OAuthProviderInfo(id="anthropic", name="Anthropic"),
    OAuthProviderInfo(id="github-copilot", name="GitHub Copilot"),
    OAuthProviderInfo(id="google-antigravity", name="Google Antigravity"),
    OAuthProviderInfo(id="google-gemini-cli", name="Google Gemini CLI"),
    OAuthProviderInfo(id="openai-codex", name="OpenAI Codex"),
)


def list_oauth_providers() -> list[OAuthProviderInfo]:
    return list(OAUTH_PROVIDERS)


def run(argv: list[str] | None = None, *, repo: str | Path | None = None) -> int:
    del repo
    parser = argparse.ArgumentParser(prog="pi-ai", description="Native Python AI helpers and provider metadata.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("list", help="List supported OAuth provider metadata.")
    subparsers.add_parser("version", help="Print package version.")
    args = parser.parse_args(list(argv or []))

    if args.command in {None, "list"}:
        print("Available OAuth providers:\n")
        for provider in OAUTH_PROVIDERS:
            print(f"  {provider.id:<20} {provider.name}")
        return 0
    if args.command == "version":
        print(__version__)
        return 0
    parser.print_help(sys.stderr)
    return 1


def upstream_version(*, repo: str | Path | None = None) -> str:
    del repo
    return __version__


__all__ = [
    "OAUTH_PROVIDERS",
    "CompletionResult",
    "Context",
    "OpenAICompatibleConfig",
    "OpenAICompatibleProvider",
    "OAuthProviderInfo",
    "StreamEvent",
    "Tool",
    "ZAIConfig",
    "ZAIProvider",
    "complete",
    "create_agent",
    "list_oauth_providers",
    "run",
    "run_task",
    "stream",
    "upstream_version",
]
