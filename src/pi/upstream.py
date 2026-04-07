from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import Final

DEFAULT_REPO_CANDIDATES: Final[tuple[Path, ...]] = (
    Path("/tmp/pi-mono-aBS53I"),
    Path("/tmp/pi-mono"),
)

CLI_ENTRYPOINTS: Final[dict[str, str]] = {
    "ai": "packages/ai/src/cli.ts",
    "coding-agent": "packages/coding-agent/src/cli.ts",
    "mom": "packages/mom/src/main.ts",
    "pods": "packages/pods/src/cli.ts",
}


class UpstreamError(RuntimeError):
    pass


class UpstreamRepoNotFoundError(UpstreamError):
    pass


class UpstreamCommandError(UpstreamError):
    pass


@dataclass(slots=True, frozen=True)
class UpstreamInstallation:
    repo: Path
    package_manager: str = "npm"

    @property
    def package_json(self) -> Path:
        return self.repo / "package.json"

    @property
    def node_modules(self) -> Path:
        return self.repo / "node_modules"

    @property
    def tsx_path(self) -> Path:
        suffix = ".cmd" if os.name == "nt" else ""
        return self.node_modules / ".bin" / f"tsx{suffix}"

    def exists(self) -> bool:
        return self.package_json.exists()

    def ensure_dependencies(self) -> None:
        if self.tsx_path.exists():
            return
        if os.environ.get("PI_AUTO_INSTALL_UPSTREAM", "1") in {"0", "false", "False"}:
            raise UpstreamCommandError(
                "Upstream dependencies are not installed and automatic install is disabled."
            )
        self.install_dependencies()

    def install_dependencies(self) -> None:
        command = "ci" if (self.repo / "package-lock.json").exists() else "install"
        self.run_package_manager((command,))

    def run_package_manager(
        self,
        args: tuple[str, ...],
        *,
        cwd: Path | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        process = subprocess.run(
            [self.package_manager, *args],
            cwd=cwd or self.repo,
            text=True,
            capture_output=capture_output,
            check=False,
            env=self._env(),
        )
        if process.returncode != 0:
            detail = process.stderr.strip() if process.stderr else ""
            raise UpstreamCommandError(
                f"{self.package_manager} {' '.join(args)} failed"
                + (f": {detail}" if detail else ".")
            )
        return process

    def run_tsx(self, entrypoint: str, argv: list[str]) -> int:
        self.ensure_dependencies()
        process = subprocess.run(
            [self.package_manager, "exec", "--", "tsx", entrypoint, *argv],
            cwd=self.repo,
            text=True,
            check=False,
            env=self._env(),
        )
        return int(process.returncode)

    def load_package_json(self, relative_path: str = "package.json") -> dict[str, object]:
        path = self.repo / relative_path
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise UpstreamCommandError(f"Missing package metadata: {path}") from exc
        except ValueError as exc:
            raise UpstreamCommandError(f"Invalid package metadata: {path}") from exc
        if not isinstance(payload, dict):
            raise UpstreamCommandError(f"Unexpected package metadata shape: {path}")
        return payload

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("NO_UPDATE_NOTIFIER", "1")
        return env


def resolve_upstream_installation(repo: str | Path | None = None) -> UpstreamInstallation:
    if repo is not None:
        candidate = Path(repo).expanduser().resolve()
        if (candidate / "package.json").exists():
            return UpstreamInstallation(
                repo=candidate,
                package_manager=os.environ.get("PI_NODE_PACKAGE_MANAGER", "npm"),
            )
        raise UpstreamRepoNotFoundError(f"Upstream repo not found at {candidate}")

    env_repo = os.environ.get("PI_MONO_REPO")
    if env_repo:
        return resolve_upstream_installation(env_repo)

    for candidate in DEFAULT_REPO_CANDIDATES:
        resolved = candidate.resolve()
        if (resolved / "package.json").exists():
            return UpstreamInstallation(
                repo=resolved,
                package_manager=os.environ.get("PI_NODE_PACKAGE_MANAGER", "npm"),
            )

    searched = ", ".join(str(path) for path in DEFAULT_REPO_CANDIDATES)
    raise UpstreamRepoNotFoundError(
        "Unable to locate the upstream pi-mono checkout. "
        f"Set PI_MONO_REPO or clone into one of: {searched}"
    )


def run_upstream_cli(package: str, argv: list[str] | None = None, *, repo: str | Path | None = None) -> int:
    entrypoint = CLI_ENTRYPOINTS.get(package)
    if entrypoint is None:
        raise UpstreamCommandError(f"Unknown upstream package: {package}")
    installation = resolve_upstream_installation(repo)
    return installation.run_tsx(entrypoint, list(argv or []))


def get_upstream_version(package: str | None = None, *, repo: str | Path | None = None) -> str:
    installation = resolve_upstream_installation(repo)
    relative = "package.json"
    if package is not None:
        relative = f"packages/{package}/package.json"
    payload = installation.load_package_json(relative)
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise UpstreamCommandError(f"Missing version in {relative}")
    return version


def copy_tree(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination
