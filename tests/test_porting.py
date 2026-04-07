from __future__ import annotations

from pi.porting import port_status, port_status_by_package


def test_port_status_covers_expected_packages() -> None:
    statuses = port_status()
    packages = {status.package for status in statuses}

    assert {"agent", "coding-agent", "ai", "pods", "mom", "tui", "web-ui"} <= packages
    assert all(status.gaps for status in statuses)


def test_port_status_by_package_indexes_statuses() -> None:
    indexed = port_status_by_package()

    assert indexed["agent"].mode == "native"
    assert indexed["coding-agent"].mode == "wrapper"
