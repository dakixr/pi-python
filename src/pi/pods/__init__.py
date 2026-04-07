from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path

from pi.upstream import get_upstream_version, resolve_upstream_installation, run_upstream_cli

PACKAGE_NAME = "pods"


@dataclass(slots=True)
class Pod:
    name: str
    ssh: str
    models_path: str | None = None
    vllm: str = "release"

    def to_dict(self) -> dict[str, object]:
        return {
            "ssh": self.ssh,
            "models_path": self.models_path,
            "vllm": self.vllm,
        }


@dataclass(slots=True)
class PodsConfig:
    active_pod: str | None = None
    pods: dict[str, Pod] = field(default_factory=dict)

    def add_pod(self, pod: Pod) -> None:
        self.pods[pod.name] = pod
        if self.active_pod is None:
            self.active_pod = pod.name

    def set_active(self, name: str) -> None:
        if name not in self.pods:
            raise KeyError(name)
        self.active_pod = name

    def get_active(self) -> Pod | None:
        if self.active_pod is None:
            return None
        return self.pods.get(self.active_pod)

    def to_dict(self) -> dict[str, object]:
        return {
            "active_pod": self.active_pod,
            "pods": {name: pod.to_dict() for name, pod in sorted(self.pods.items())},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PodsConfig":
        pods_payload = payload.get("pods", {})
        pods: dict[str, Pod] = {}
        if isinstance(pods_payload, dict):
            for name, item in pods_payload.items():
                if not isinstance(name, str) or not isinstance(item, dict):
                    continue
                ssh = item.get("ssh")
                if not isinstance(ssh, str) or not ssh.strip():
                    continue
                models_path = item.get("models_path")
                vllm = item.get("vllm", "release")
                pods[name] = Pod(
                    name=name,
                    ssh=ssh,
                    models_path=models_path if isinstance(models_path, str) else None,
                    vllm=vllm if isinstance(vllm, str) and vllm.strip() else "release",
                )
        active_pod = payload.get("active_pod")
        return cls(
            active_pod=active_pod if isinstance(active_pod, str) and active_pod in pods else None,
            pods=pods,
        )


class PodsConfigStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_config_dir()).resolve()
        self.path = self.root / "pods.json"

    def load(self) -> PodsConfig:
        if not self.path.exists():
            return PodsConfig()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid pods config payload in {self.path}")
        return PodsConfig.from_dict(payload)

    def save(self, config: PodsConfig) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        return self.path


def default_config_dir() -> Path:
    return Path(os.environ.get("PI_CONFIG_DIR", "~/.pi")).expanduser()


def run(argv: list[str] | None = None, *, repo: str | Path | None = None) -> int:
    return run_upstream_cli(PACKAGE_NAME, argv, repo=repo)


def upstream_version(*, repo: str | Path | None = None) -> str:
    return get_upstream_version(PACKAGE_NAME, repo=repo)


__all__ = [
    "PACKAGE_NAME",
    "Pod",
    "PodsConfig",
    "PodsConfigStore",
    "default_config_dir",
    "resolve_upstream_installation",
    "run",
    "upstream_version",
]
