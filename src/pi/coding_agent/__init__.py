from __future__ import annotations

from pathlib import Path

from pi import __version__
from pi.cli.main import main as run_core_cli

PACKAGE_NAME = "coding-agent"


def run(argv: list[str] | None = None, *, repo: str | Path | None = None) -> int:
    del repo
    return run_core_cli(list(argv or []))


def upstream_version(*, repo: str | Path | None = None) -> str:
    del repo
    return __version__


__all__ = ["PACKAGE_NAME", "run", "upstream_version"]
