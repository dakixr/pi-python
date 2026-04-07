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
            mode="native-subset",
            summary="Python CLI now aliases the local native core agent instead of delegating to another repo.",
            gaps=(
                "Interactive TUI, extensions, and session tree management are still not implemented natively.",
                "The surface is intentionally smaller than the historical TypeScript product.",
            ),
        ),
        PackagePortStatus(
            package="ai",
            mode="hybrid",
            summary="Native Python SDK surface for completions/agent embedding plus a local provider metadata CLI.",
            gaps=(
                "Only a subset of the upstream provider/auth surface is native today.",
                "OAuth login flows themselves are not implemented.",
            ),
        ),
        PackagePortStatus(
            package="pods",
            mode="native-subset",
            summary="Native Python config models and a local CLI for managing pod metadata.",
            gaps=(
                "SSH orchestration and vLLM lifecycle are not implemented.",
                "No native pod execution engine yet.",
            ),
        ),
        PackagePortStatus(
            package="mom",
            mode="native-subset",
            summary="Native Python sandbox parsing and a local helper CLI.",
            gaps=(
                "Slack integration and long-running bot behavior are not implemented.",
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
            mode="native-subset",
            summary="Python helpers manage a local placeholder web-ui asset directory.",
            gaps=(
                "No native browser UI implementation exists.",
                "The generated dist is a placeholder rather than a full app build.",
            ),
        ),
    ]


def port_status_by_package() -> dict[str, PackagePortStatus]:
    return {status.package: status for status in port_status()}


__all__ = ["PackagePortStatus", "port_status", "port_status_by_package"]
