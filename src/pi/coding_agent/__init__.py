from __future__ import annotations

from pathlib import Path

from pi.upstream import get_upstream_version, resolve_upstream_installation, run_upstream_cli

PACKAGE_NAME = "coding-agent"


def run(argv: list[str] | None = None, *, repo: str | Path | None = None) -> int:
    return run_upstream_cli(PACKAGE_NAME, argv, repo=repo)


def upstream_version(*, repo: str | Path | None = None) -> str:
    return get_upstream_version(PACKAGE_NAME, repo=repo)


__all__ = ["PACKAGE_NAME", "resolve_upstream_installation", "run", "upstream_version"]
