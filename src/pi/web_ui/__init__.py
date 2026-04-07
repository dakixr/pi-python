from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from pi import __version__


@dataclass(slots=True, frozen=True)
class WebUIPaths:
    package_dir: Path
    source_dir: Path
    dist_dir: Path
    example_dir: Path


def get_paths(repo: str | Path | None = None) -> WebUIPaths:
    package_dir = Path(repo).expanduser().resolve() if repo is not None else Path(__file__).resolve().parent
    return WebUIPaths(
        package_dir=package_dir,
        source_dir=package_dir / "src",
        dist_dir=package_dir / "dist",
        example_dir=package_dir / "example",
    )


def ensure_built(repo: str | Path | None = None) -> Path:
    paths = get_paths(repo)
    paths.dist_dir.mkdir(parents=True, exist_ok=True)
    index_html = paths.dist_dir / "index.html"
    if not index_html.exists():
        index_html.write_text(
            "<!doctype html><html><head><meta charset='utf-8'><title>pi web ui</title></head>"
            "<body><main><h1>pi web ui</h1><p>Local placeholder build.</p></main></body></html>",
            encoding="utf-8",
        )
    return paths.dist_dir


def copy_dist(destination: Path, repo: str | Path | None = None) -> Path:
    dist_dir = ensure_built(repo)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dist_dir, destination, dirs_exist_ok=True)
    return destination


def upstream_version(*, repo: str | Path | None = None) -> str:
    del repo
    return __version__


__all__ = ["WebUIPaths", "copy_dist", "ensure_built", "get_paths", "upstream_version"]
