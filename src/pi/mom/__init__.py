from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

from pi import __version__

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
    del repo
    parser = argparse.ArgumentParser(prog="mom", description="Local MOM sandbox helpers.")
    parser.add_argument("--sandbox", help="Validate sandbox config such as 'host' or 'docker:name'.")
    parser.add_argument("--version", action="store_true", help="Print package version and exit.")
    args = parser.parse_args(list(argv or []))
    if args.version:
        print(__version__)
        return 0
    if args.sandbox:
        sandbox = parse_sandbox(args.sandbox)
        suffix = f":{sandbox.name}" if sandbox.name else ""
        print(f"{sandbox.type}{suffix}")
        return 0
    parser.print_help(sys.stdout)
    return 0


def upstream_version(*, repo: str | Path | None = None) -> str:
    del repo
    return __version__


__all__ = [
    "PACKAGE_NAME",
    "SandboxConfig",
    "parse_sandbox",
    "run",
    "upstream_version",
]
