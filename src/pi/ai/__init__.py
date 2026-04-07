from __future__ import annotations

from dataclasses import dataclass

from pi.agent.providers.openai_compat import OpenAICompatibleConfig, OpenAICompatibleProvider
from pi.agent.providers.zai import ZAIConfig, ZAIProvider
from pi.ai.sdk import CompletionResult, Context, StreamEvent, Tool, complete, create_agent, run_task, stream


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
    "run_task",
    "stream",
]
