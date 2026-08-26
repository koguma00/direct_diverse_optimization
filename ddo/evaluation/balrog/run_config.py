"""Resolved BALROG adapter config snapshots consumed by DTC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


RUN_CONFIG_SNAPSHOT_NAME = "resolved_config.yaml"
DEFAULT_BALROG_CONFIG_PATH = Path(__file__).with_name("config.yaml").resolve()


def write_run_config_snapshot(output_dir: str | Path, config: Any) -> Path:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / RUN_CONFIG_SNAPSHOT_NAME
    path.write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
    return path


def resolve_run_config_path(
    base_run_dir: str | Path,
    config_override: str | Path | None = None,
) -> tuple[Path, str]:
    if config_override is not None:
        path = Path(config_override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Config override does not exist: {path}")
        return path, "override"
    snapshot = Path(base_run_dir).expanduser().resolve() / RUN_CONFIG_SNAPSHOT_NAME
    if snapshot.is_file():
        return snapshot, "base_run_snapshot"
    return DEFAULT_BALROG_CONFIG_PATH, "default_fallback"
