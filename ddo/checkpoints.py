"""Validation for released LoRA adapter directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CheckpointAdapterError(ValueError):
    """Raised when a released adapter path is unsafe or incomplete."""


def resolve_paper_adapter(
    adapter_value: str,
    *,
    project_root: Path,
    expected_base_model: str | None = None,
    verify_files: bool = True,
) -> Path:
    """Resolve a repository-relative released adapter directory.

    Configuration loading validates only path containment so configs remain inspectable before
    the separately distributed checkpoint bundle is installed. Execution requires
    the two PEFT files needed for inference.
    """

    adapter_dir = _repo_path(project_root, adapter_value, field="adapter_path")
    if not verify_files:
        return adapter_dir

    model_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    for path in (model_path, config_path):
        if not path.is_file():
            raise CheckpointAdapterError(
                f"released adapter is not installed; missing: {path}. "
                "Place the released adapter as described in README.md."
            )

    adapter_config = _read_json_mapping(config_path)
    actual_base_model = adapter_config.get("base_model_name_or_path")
    if expected_base_model is not None and actual_base_model != expected_base_model:
        raise CheckpointAdapterError(
            "released adapter base model mismatch: "
            f"{actual_base_model} != {expected_base_model}"
        )
    return adapter_dir


def _repo_path(project_root: Path, value: str, *, field: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        raise CheckpointAdapterError(
            f"{field} must be relative to the repository: {value}"
        )
    root = project_root.resolve()
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise CheckpointAdapterError(f"{field} escapes the repository: {value}")
    return resolved


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointAdapterError(f"released adapter has invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CheckpointAdapterError(
            f"released adapter config must contain a JSON object: {path}"
        )
    return value
