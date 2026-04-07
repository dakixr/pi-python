from __future__ import annotations

from pathlib import Path
import subprocess

from pi.upstream import get_upstream_version, resolve_upstream_installation, run_upstream_cli


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "pi-mono"
    (repo / "package.json").parent.mkdir(parents=True, exist_ok=True)
    (repo / "package.json").write_text('{"name":"pi-monorepo","version":"1.2.3"}', encoding="utf-8")
    (repo / "package-lock.json").write_text("{}", encoding="utf-8")
    packages = repo / "packages"
    for package in ("ai", "coding-agent", "mom", "pods", "web-ui"):
        package_dir = packages / package
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "package.json").write_text('{"version":"9.9.9"}', encoding="utf-8")
    return repo


def test_resolve_upstream_installation_uses_env(monkeypatch, tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    monkeypatch.setenv("PI_MONO_REPO", str(repo))

    installation = resolve_upstream_installation()

    assert installation.repo == repo.resolve()


def test_run_upstream_cli_installs_dependencies_when_missing(monkeypatch, tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code = run_upstream_cli("coding-agent", ["--help"], repo=repo)

    assert exit_code == 0
    assert calls[0] == ["npm", "ci"]
    assert calls[1] == ["npm", "exec", "--", "tsx", "packages/coding-agent/src/cli.ts", "--help"]


def test_run_upstream_cli_skips_install_when_tsx_exists(monkeypatch, tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    tsx_path = repo / "node_modules" / ".bin" / "tsx"
    tsx_path.parent.mkdir(parents=True, exist_ok=True)
    tsx_path.write_text("", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    exit_code = run_upstream_cli("ai", ["list"], repo=repo)

    assert exit_code == 0
    assert calls == [["npm", "exec", "--", "tsx", "packages/ai/src/cli.ts", "list"]]


def test_get_upstream_version_reads_root_and_package_metadata(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)

    assert get_upstream_version(repo=repo) == "1.2.3"
    assert get_upstream_version("web-ui", repo=repo) == "9.9.9"
