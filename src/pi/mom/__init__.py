from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pi.upstream import get_upstream_version, resolve_upstream_installation, run_upstream_cli

PACKAGE_NAME = "mom"


@dataclass(slots=True, frozen=True)
class SandboxConfig:
    type: str
    name: str | None = None


def parse_sandbox(value: str) -> SandboxConfig:
    normalized = value.strip()
    if normalized == "host":
        return SandboxConfig(type="host")
    if normalized.startswith("docker:"):
        name = normalized.partition(":")[2].strip()
        if not name:
            raise ValueError("Docker sandbox requires a container name.")
        return SandboxConfig(type="docker", name=name)
    raise ValueError("Sandbox must be 'host' or 'docker:<name>'.")


def run(argv: list[str] | None = None, *, repo: str | Path | None = None) -> int:
    return run_upstream_cli(PACKAGE_NAME, argv, repo=repo)


def upstream_version(*, repo: str | Path | None = None) -> str:
    return get_upstream_version(PACKAGE_NAME, repo=repo)


__all__ = [
    "PACKAGE_NAME",
    "SandboxConfig",
    "parse_sandbox",
    "resolve_upstream_installation",
    "run",
    "upstream_version",
]
