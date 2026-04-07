from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class PackagePortStatus:
    package: str
    mode: str
    summary: str
    gaps: tuple[str, ...]


def port_status() -> list[PackagePortStatus]:
    return [
        PackagePortStatus(
            package="agent",
            mode="native",
            summary="Native Python agent loop, context compaction, tool hooks, parallel execution, and tool runtime.",
            gaps=(
                "The event model is smaller than the upstream TypeScript agent runtime.",
                "The native runtime still exposes fewer session/runtime services than the full coding-agent product.",
            ),
        ),
        PackagePortStatus(
            package="coding-agent",
            mode="wrapper",
            summary="Python CLI delegates to the upstream TypeScript coding-agent entrypoint.",
            gaps=(
                "Interactive TUI, extensions, skills, sessions, and SDK remain upstream-driven.",
                "Not a pure-Python implementation yet.",
            ),
        ),
        PackagePortStatus(
            package="ai",
            mode="hybrid",
            summary="Native Python SDK surface for completions/agent embedding plus upstream delegation for OAuth/provider CLI flows.",
            gaps=(
                "Only a subset of the upstream provider/auth surface is native today.",
                "OAuth login flows remain upstream-driven.",
            ),
        ),
        PackagePortStatus(
            package="pods",
            mode="hybrid",
            summary="Native Python config models with upstream delegation for the operational CLI.",
            gaps=(
                "SSH orchestration and vLLM lifecycle are still handled upstream.",
                "No native pod execution engine yet.",
            ),
        ),
        PackagePortStatus(
            package="mom",
            mode="hybrid",
            summary="Native Python sandbox parsing with upstream delegation for the Slack bot runtime.",
            gaps=(
                "Slack integration and agent runner remain upstream-driven.",
                "No native long-running bot runtime yet.",
            ),
        ),
        PackagePortStatus(
            package="tui",
            mode="native-subset",
            summary="Native Python text-width and wrapping helpers cover a small useful subset.",
            gaps=(
                "No native differential renderer or widget system.",
                "No terminal image or keybinding stack.",
            ),
        ),
        PackagePortStatus(
            package="web-ui",
            mode="wrapper",
            summary="Python helpers build and export the upstream web-ui package assets.",
            gaps=(
                "No native Python web component implementation.",
                "The browser UI remains upstream TypeScript code.",
            ),
        ),
    ]


def port_status_by_package() -> dict[str, PackagePortStatus]:
    return {status.package: status for status in port_status()}


__all__ = ["PackagePortStatus", "port_status", "port_status_by_package"]
