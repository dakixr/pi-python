from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys

from pi import __version__

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
    del repo
    parser = argparse.ArgumentParser(prog="pi-pods", description="Local pod configuration management.")
    parser.add_argument("--config-dir", type=Path, default=default_config_dir(), help="Config directory for pods.json.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List configured pods.")

    add_parser = subparsers.add_parser("add", help="Add or replace a pod entry.")
    add_parser.add_argument("name")
    add_parser.add_argument("ssh")
    add_parser.add_argument("--models-path")
    add_parser.add_argument("--vllm", default="release")

    active_parser = subparsers.add_parser("active", help="Set or show the active pod.")
    active_parser.add_argument("name", nargs="?")

    remove_parser = subparsers.add_parser("remove", help="Remove a pod entry.")
    remove_parser.add_argument("name")

    args = parser.parse_args(list(argv or []))
    store = PodsConfigStore(args.config_dir)
    config = store.load()

    if args.command is None or args.command == "list":
        if not config.pods:
            print("No pods configured.")
            return 0
        for name, pod in sorted(config.pods.items()):
            marker = "*" if config.active_pod == name else " "
            models_suffix = f" models={pod.models_path}" if pod.models_path else ""
            print(f"{marker} {name} -> {pod.ssh}{models_suffix} vllm={pod.vllm}")
        return 0

    if args.command == "add":
        config.add_pod(Pod(name=args.name, ssh=args.ssh, models_path=args.models_path, vllm=args.vllm))
        store.save(config)
        print(f"Saved pod {args.name}")
        return 0

    if args.command == "active":
        if args.name is None:
            active = config.get_active()
            print(active.name if active is not None else "")
            return 0
        config.set_active(args.name)
        store.save(config)
        print(args.name)
        return 0

    if args.command == "remove":
        config.pods.pop(args.name, None)
        if config.active_pod == args.name:
            config.active_pod = next(iter(sorted(config.pods)), None)
        store.save(config)
        print(f"Removed pod {args.name}")
        return 0

    parser.print_help(sys.stderr)
    return 1


def upstream_version(*, repo: str | Path | None = None) -> str:
    del repo
    return __version__


__all__ = [
    "PACKAGE_NAME",
    "Pod",
    "PodsConfig",
    "PodsConfigStore",
    "default_config_dir",
    "run",
    "upstream_version",
]
