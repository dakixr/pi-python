from __future__ import annotations

from pathlib import Path

from pi.ai import list_oauth_providers
from pi.mom import parse_sandbox
from pi.pods import Pod, PodsConfig, PodsConfigStore, default_config_dir
from pi.tui import strip_ansi, truncate_to_width, visible_width, wrap_text_with_ansi
from pi.web_ui import get_paths, upstream_version as web_ui_upstream_version


def test_ai_lists_known_oauth_providers() -> None:
    provider_ids = [provider.id for provider in list_oauth_providers()]
    assert "anthropic" in provider_ids
    assert "openai-codex" in provider_ids


def test_pods_config_tracks_active_pod() -> None:
    config = PodsConfig()
    config.add_pod(Pod(name="gpu-1", ssh="ssh root@example.com"))
    config.add_pod(Pod(name="gpu-2", ssh="ssh root@example.org"))

    assert config.get_active() is not None
    assert config.get_active().name == "gpu-1"

    config.set_active("gpu-2")
    assert config.get_active().name == "gpu-2"


def test_pods_config_store_round_trips(tmp_path: Path) -> None:
    store = PodsConfigStore(tmp_path)
    config = PodsConfig()
    config.add_pod(Pod(name="gpu-1", ssh="ssh root@example.com", models_path="/models"))
    config.set_active("gpu-1")

    path = store.save(config)
    loaded = store.load()

    assert path == tmp_path / "pods.json"
    assert loaded.get_active() is not None
    assert loaded.get_active().ssh == "ssh root@example.com"
    assert loaded.get_active().models_path == "/models"


def test_default_config_dir_uses_env(monkeypatch) -> None:
    monkeypatch.setenv("PI_CONFIG_DIR", "/tmp/pi-config")

    assert default_config_dir() == Path("/tmp/pi-config")


def test_mom_parse_sandbox() -> None:
    assert parse_sandbox("host").type == "host"
    docker = parse_sandbox("docker:mom-sandbox")
    assert docker.type == "docker"
    assert docker.name == "mom-sandbox"


def test_tui_helpers_handle_basic_text() -> None:
    styled = "\x1b[31mhello\x1b[0m world"
    assert strip_ansi(styled) == "hello world"
    assert visible_width(styled) == 11
    assert truncate_to_width(styled, 8) == "hello..."
    assert wrap_text_with_ansi(styled, 5) == ["hello", "world"]


def test_web_ui_get_paths_uses_repo_layout(tmp_path: Path) -> None:
    repo = tmp_path / "pi-mono"
    repo.mkdir()
    (repo / "package.json").write_text("{}", encoding="utf-8")
    package_dir = repo / "packages" / "web-ui"
    package_dir.mkdir(parents=True)
    (package_dir / "package.json").write_text('{"version":"3.4.5"}', encoding="utf-8")

    paths = get_paths(repo)

    assert paths.package_dir == repo / "packages" / "web-ui"
    assert paths.source_dir == repo / "packages" / "web-ui" / "src"
    assert paths.dist_dir == repo / "packages" / "web-ui" / "dist"
    assert web_ui_upstream_version(repo=repo) == "3.4.5"
