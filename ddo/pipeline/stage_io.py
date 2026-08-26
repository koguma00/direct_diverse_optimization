"""Shared I/O helpers for native pipeline stages."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ddo.evaluation.benchmarks.base import BenchmarkAdapter
from ddo.config.schema import DDOConfig
from ddo.schemas import write_json
from ddo.schemas.io import read_json


def config_metadata(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    *,
    stage: str,
) -> dict[str, Any]:
    """Return only configuration owned or consumed by one artifact stage."""

    pooled_stage = stage in {"base", "sft", "collect", "build", "train"}
    task_scope = (
        config.benchmark.task_group
        if pooled_stage and config.benchmark.train_task_filters
        else benchmark.normalize_task_id(config.benchmark.task_filter)
    )
    section_name = {
        "base": "base",
        "sft": "sft",
        "collect": "collection",
        "build": "dataset",
        "train": "training",
        "eval": "evaluation",
    }[stage]
    section = getattr(config, section_name)
    model_fields = {
        "base": ("expert_model", "expert_base_url"),
        "sft": ("target_model",),
        "collect": (
            "expert_model",
            "target_model",
            "expert_base_url",
            "target_base_url",
        ),
        "build": ("target_model", "reference_model"),
        "train": ("target_model", "reference_model"),
        "eval": ("target_model", "target_base_url"),
    }[stage]
    all_models = asdict(config.models)
    runtime_fields = {
        "base": ("seed", "num_workers", "cuda_visible_devices"),
        "sft": ("seed", "cuda_visible_devices"),
        "collect": ("seed", "num_workers", "cuda_visible_devices"),
        "build": ("cuda_visible_devices",),
        "train": ("seed", "cuda_visible_devices"),
        "eval": ("cuda_visible_devices",),
    }[stage]
    payload = {
        "stage": stage,
        "benchmark": {
            "name": benchmark.name,
            "task_scope": task_scope,
            "task_group": config.benchmark.task_group,
            "train_task_filters": config.benchmark.train_task_filters,
            "evaluation_task_filter": (
                benchmark.normalize_task_id(config.benchmark.task_filter)
                if stage == "eval"
                else None
            ),
            "upstream_path": config.benchmark.upstream_path,
        },
        section_name: asdict(section),
        "models": {field: all_models[field] for field in model_fields},
        "runtime": {field: getattr(config.runtime, field) for field in runtime_fields},
    }
    return payload


def single_output_dir(plan: Any) -> Path:
    if not plan.outputs:
        raise RuntimeError(f"stage {plan.stage} did not declare an output directory")
    return Path(plan.outputs[0])


def write_stage_manifest(
    output_dir: Path,
    plan: Any,
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    *,
    status: str = "completed",
) -> None:
    write_json(
        output_dir / "stage_manifest.json",
        {
            "artifact_type": "stage_manifest",
            "backend": "native",
            "stage": plan.stage,
            "summary": plan.summary,
            "inputs": plan.inputs,
            "outputs": plan.outputs,
            "commands": plan.commands,
            "metadata": plan.metadata,
            "status": status,
            "config": config_metadata(config, benchmark, stage=plan.stage),
        },
    )


def sft_checkpoint_id(config: DDOConfig, benchmark: BenchmarkAdapter) -> str:
    """Build a stable logical id independent of a user-facing run name."""

    task = (
        config.benchmark.task_group
        if config.benchmark.train_task_filters
        else benchmark.normalize_task_id(config.benchmark.task_filter)
    ) or "all"
    return "__".join(
        [
            benchmark.name,
            _safe_slug(task),
            "sft",
            _safe_slug(config.models.target_model or "model"),
        ]
    )


def resolve_adapter_path(stage_dir: Path) -> Path | None:
    """Resolve a materialized adapter without relying on its original workspace path."""

    payload: dict[str, Any] = {}
    for name in ("checkpoint_manifest.json", "sft_checkpoint_manifest.json", "checkpoint_ref.json"):
        path = stage_dir / name
        if path.exists():
            payload = read_json(path)
            break

    for field in ("relative_adapter_path", "adapter_path", "native_adapter_path", "checkpoint_dir"):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = stage_dir / candidate
        if candidate.exists() and (candidate / "adapter_config.json").exists():
            return candidate

    trainer_run = stage_dir / "trainer_run"
    checkpoints = [path for path in trainer_run.glob("checkpoint-*") if path.is_dir()]
    checkpoints.sort(key=_checkpoint_sort_key, reverse=True)
    for candidate in [*checkpoints, trainer_run, stage_dir]:
        if (candidate / "adapter_config.json").exists():
            return candidate
    return None


def relative_adapter_path(stage_dir: Path, adapter_path: Path | None) -> str | None:
    if adapter_path is None:
        return None
    try:
        return adapter_path.relative_to(stage_dir).as_posix()
    except ValueError:
        return str(adapter_path)


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.rsplit("-", 1)[-1]
    try:
        return int(suffix), path.name
    except ValueError:
        return -1, path.name


def _safe_slug(value: str) -> str:
    normalized = value.rsplit("/", 1)[-1]
    return "".join(char.lower() if char.isalnum() else "_" for char in normalized).strip("_") or "value"
