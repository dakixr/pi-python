from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pi.upstream import copy_tree, get_upstream_version, resolve_upstream_installation


@dataclass(slots=True, frozen=True)
class WebUIPaths:
    package_dir: Path
    source_dir: Path
    dist_dir: Path
    example_dir: Path


def get_paths(repo: str | Path | None = None) -> WebUIPaths:
    installation = resolve_upstream_installation(repo)
    package_dir = installation.repo / "packages" / "web-ui"
    return WebUIPaths(
        package_dir=package_dir,
        source_dir=package_dir / "src",
        dist_dir=package_dir / "dist",
        example_dir=package_dir / "example",
    )


def ensure_built(repo: str | Path | None = None) -> Path:
    installation = resolve_upstream_installation(repo)
    installation.ensure_dependencies()
    package_dir = installation.repo / "packages" / "web-ui"
    installation.run_package_manager(("run", "build"), cwd=package_dir)
    return package_dir / "dist"


def copy_dist(destination: Path, repo: str | Path | None = None) -> Path:
    dist_dir = ensure_built(repo)
    return copy_tree(dist_dir, destination)


def upstream_version(*, repo: str | Path | None = None) -> str:
    return get_upstream_version("web-ui", repo=repo)


__all__ = ["WebUIPaths", "copy_dist", "ensure_built", "get_paths", "upstream_version"]
