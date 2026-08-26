"""Native stage executors backed by the in-repository DDO research code."""

from __future__ import annotations

import json
import os
import shutil
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ddo.evaluation.benchmarks.base import BenchmarkAdapter
from ddo.checkpoints import resolve_paper_adapter
from ddo.config.schema import DDOConfig
from ddo.pipeline.stage_io import (
    config_metadata,
    relative_adapter_path,
    resolve_adapter_path,
    sft_checkpoint_id,
    single_output_dir,
    write_stage_manifest,
)
from ddo.pipeline.normalization import normalize_stage_outputs
from ddo.pipeline.research_commands import (
    _safe_run_name,
    build_stage_command_manifest,
)
from ddo.schemas import (
    PairDatasetArtifact,
    TrajectoryArtifact,
    TrajectoryStep,
    read_json,
    read_jsonl,
    read_trajectories,
    write_json,
    write_jsonl,
    write_pair_dataset,
    write_trajectories,
)


def execute_native_stage_plan(config: DDOConfig, benchmark: BenchmarkAdapter, plan: Any) -> Any:
    if plan.stage == "base":
        return _execute_base(config, benchmark, plan)
    if plan.stage == "sft":
        return _execute_sft(config, benchmark, plan)
    if plan.stage == "collect" and config.collection.method == "none":
        return _execute_noop_collect(config, benchmark, plan)
    if plan.stage == "build" and config.dataset.method == "noop_pairs":
        return _execute_noop_build(config, benchmark, plan)
    if plan.stage == "train" and config.training.method in {"base", "reference"}:
        return _execute_noop_train(config, benchmark, plan)
    if plan.stage in {"collect", "build", "train", "eval"}:
        return _execute_research_stage(config, benchmark, plan)
    raise ValueError(f"unknown native stage: {plan.stage}")


def _execute_base(config: DDOConfig, benchmark: BenchmarkAdapter, plan: Any) -> Any:
    output_dir = benchmark.resolve_base_run_dir(config)
    if output_dir is None:
        raise RuntimeError("native base trajectory collection requires benchmark.base_run_dir")

    output_dir.mkdir(parents=True, exist_ok=True)
    operation_dir = benchmark.resolve_paths(config).work_dir / "base"
    operation_dir.mkdir(parents=True, exist_ok=True)
    if config.base.source_dir or config.base.source_dirs:
        _materialize_base_source(config, benchmark, output_dir)
    existing = _existing_base_artifacts(output_dir)
    if existing is not None:
        manifest = _existing_base_manifest(config, benchmark, output_dir, existing)
        write_json(operation_dir / "native_command_manifest.json", manifest)
        executed_plan = replace(plan, commands=[manifest["commands"][0]["command"]])
        write_stage_manifest(operation_dir, executed_plan, config, benchmark)
        return executed_plan

    manifest = _build_base_command_manifest(config, benchmark, output_dir)
    try:
        manifest["preflight"] = _preflight_model_endpoint(
            config,
            "base",
            _stage_model_id(config, "base"),
        )
    except RuntimeError as exc:
        manifest["status"] = "failed"
        manifest["preflight_error"] = str(exc)
        write_json(operation_dir / "native_command_manifest.json", manifest)
        raise
    write_json(operation_dir / "native_command_manifest.json", manifest)

    result = _run_command(
        manifest["commands"][0],
        operation_dir / "native_command_01.log",
        config=config,
    )
    manifest["results"] = [result]
    manifest["status"] = "command_completed" if result["returncode"] == 0 else "failed"
    write_json(operation_dir / "native_command_manifest.json", manifest)
    if result["returncode"] != 0:
        raise RuntimeError(f"native base command failed: {manifest['commands'][0]['name']}")

    trajectories = _normalize_raw_trajectories(output_dir, config, benchmark)
    try:
        _validate_base_trajectories(
            trajectories,
            output_dir,
            expected_count=config.base.num_episodes,
        )
    except RuntimeError as exc:
        manifest["status"] = "failed"
        manifest["validation_error"] = str(exc)
        write_json(operation_dir / "native_command_manifest.json", manifest)
        raise
    write_trajectories(output_dir / "trajectories.jsonl", trajectories)
    write_trajectories(output_dir / "base_trajectories.jsonl", trajectories)
    write_json(
        output_dir / "base_summary.json",
        {
            "artifact_type": "base_trajectory_summary",
            "backend": "native",
            "benchmark": benchmark.name,
            "task_ids": sorted({trajectory.task_id for trajectory in trajectories}),
            "expert_model": _base_model_id(config),
            "num_trajectories": len(trajectories),
            "num_success": sum(1 for trajectory in trajectories if trajectory.won),
            "raw_trace_count": len(list(output_dir.glob("**/*_llm_trace.json"))),
            "metadata": config_metadata(config, benchmark, stage="base"),
        },
    )

    manifest["status"] = "completed"
    write_json(operation_dir / "native_command_manifest.json", manifest)
    executed_plan = replace(plan, commands=[command["command"] for command in manifest["commands"]])
    write_stage_manifest(operation_dir, executed_plan, config, benchmark)
    return executed_plan


def _materialize_base_source(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    output_dir: Path,
) -> None:
    if config.base.source_dirs:
        _materialize_pooled_base_sources(config, benchmark, output_dir)
        return

    source_dir = _resolve_source_dir(config, config.base.source_dir or "")
    if not source_dir.is_dir():
        raise RuntimeError(f"base.source_dir does not exist: {source_dir}")
    if source_dir == output_dir.resolve():
        return

    for child in source_dir.iterdir():
        target = output_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)

    if (output_dir / "trajectories.jsonl").exists():
        if not (output_dir / "base_trajectories.jsonl").exists():
            shutil.copy2(output_dir / "trajectories.jsonl", output_dir / "base_trajectories.jsonl")
        return

    trajectories = _normalize_raw_trajectories(output_dir, config, benchmark)
    _validate_base_trajectories(trajectories, output_dir)
    write_trajectories(output_dir / "trajectories.jsonl", trajectories)
    write_trajectories(output_dir / "base_trajectories.jsonl", trajectories)
    write_json(
        output_dir / "base_summary.json",
        {
            "artifact_type": "base_trajectory_summary",
            "backend": "native",
            "benchmark": benchmark.name,
            "source_dir": str(source_dir),
            "num_trajectories": len(trajectories),
            "num_success": sum(1 for trajectory in trajectories if trajectory.won),
            "metadata": config_metadata(config, benchmark, stage="base"),
        },
    )


def _materialize_pooled_base_sources(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    output_dir: Path,
) -> None:
    sources_root = output_dir / "sources"
    sources_root.mkdir(parents=True, exist_ok=True)
    trajectories: list[TrajectoryArtifact] = []
    source_records: list[dict[str, Any]] = []

    for index, configured_source in enumerate(config.base.source_dirs, start=1):
        source_dir = _resolve_source_dir(config, configured_source)
        if not source_dir.is_dir():
            raise RuntimeError(f"base.source_dirs entry does not exist: {source_dir}")
        source_label = source_dir.parent.name if source_dir.name == "expert_base" else source_dir.name
        copied_dir = sources_root / f"{index:02d}_{__safe_slug(source_label)}"
        shutil.copytree(source_dir, copied_dir, dirs_exist_ok=True)

        source_trajectories: list[TrajectoryArtifact] | None = None
        for filename in ("trajectories.jsonl", "base_trajectories.jsonl"):
            candidate = copied_dir / filename
            if candidate.exists():
                source_trajectories = read_trajectories(candidate)
                break
        if source_trajectories is None:
            source_trajectories = _normalize_raw_trajectories(copied_dir, config, benchmark)
        _validate_base_trajectories(source_trajectories, copied_dir)
        trajectories.extend(source_trajectories)
        source_records.append(
            {
                "configured_path": configured_source,
                "source_dir": str(source_dir),
                "materialized_dir": str(copied_dir),
                "num_trajectories": len(source_trajectories),
                "task_ids": sorted({trajectory.task_id for trajectory in source_trajectories}),
            }
        )

    trajectories = _filter_pooled_base_trajectories(
        trajectories,
        config.benchmark.train_task_filters,
    )
    duplicate_ids = _duplicates(trajectory.trajectory_id for trajectory in trajectories)
    if duplicate_ids:
        raise RuntimeError(
            "pooled base sources contain duplicate trajectory ids: "
            + ", ".join(duplicate_ids[:10])
        )
    _validate_base_trajectories(trajectories, output_dir)
    write_trajectories(output_dir / "trajectories.jsonl", trajectories)
    write_trajectories(output_dir / "base_trajectories.jsonl", trajectories)
    write_json(
        output_dir / "import_manifest.json",
        {
            "artifact_type": "pooled_base_import",
            "benchmark": benchmark.name,
            "task_group": config.benchmark.task_group,
            "train_task_filters": config.benchmark.train_task_filters,
            "sources": source_records,
            "num_trajectories": len(trajectories),
            "task_ids": sorted({trajectory.task_id for trajectory in trajectories}),
        },
    )
    write_json(
        output_dir / "base_summary.json",
        {
            "artifact_type": "base_trajectory_summary",
            "backend": "native",
            "benchmark": benchmark.name,
            "task_group": config.benchmark.task_group,
            "train_task_filters": config.benchmark.train_task_filters,
            "source_dirs": [record["source_dir"] for record in source_records],
            "num_trajectories": len(trajectories),
            "num_success": sum(1 for trajectory in trajectories if trajectory.won),
            "metadata": config_metadata(config, benchmark, stage="base"),
        },
    )


def _resolve_source_dir(config: DDOConfig, configured_source: str) -> Path:
    source_dir = Path(configured_source).expanduser()
    if not source_dir.is_absolute():
        source_dir = (config.project_root or Path.cwd()) / source_dir
    return source_dir.resolve()


def _filter_pooled_base_trajectories(
    trajectories: list[TrajectoryArtifact],
    task_filters: list[str],
) -> list[TrajectoryArtifact]:
    if not task_filters:
        return trajectories
    selected = [
        trajectory
        for trajectory in trajectories
        if any(_task_ids_match(trajectory.task_id, task_filter) for task_filter in task_filters)
    ]
    missing = [
        task_filter
        for task_filter in task_filters
        if not any(_task_ids_match(trajectory.task_id, task_filter) for trajectory in selected)
    ]
    if missing:
        raise RuntimeError(
            "pooled base sources are missing configured training tasks: " + ", ".join(missing)
        )
    return selected


def _task_ids_match(task_id: str, task_filter: str) -> bool:
    return task_id == task_filter or task_id.rsplit("/", 1)[-1] == task_filter.rsplit("/", 1)[-1]


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _execute_sft(config: DDOConfig, benchmark: BenchmarkAdapter, plan: Any) -> Any:
    if config.sft.method != "expert_success":
        raise NotImplementedError(f"native SFT method is not supported: {config.sft.method}")

    output_dir = single_output_dir(plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    operation_dir = output_dir
    root = _research_root(config)
    base_run_dir = benchmark.resolve_base_run_dir(config)
    if base_run_dir is None:
        raise RuntimeError("native SFT requires benchmark.base_run_dir")
    if not any(base_run_dir.glob("**/*_llm_trace.json")):
        raise RuntimeError(
            "native SFT requires raw *_llm_trace.json files in the base artifact "
            f"to preserve the original prompts: {base_run_dir}"
        )
    train_jsonl = output_dir / "sft_train.jsonl"
    conversion_stats = output_dir / "sft_conversion_stats.json"
    trainer_run_dir = output_dir / "trainer_run"
    manifest = _build_sft_command_manifest(
        config,
        benchmark,
        root,
        output_dir,
        base_run_dir,
        train_jsonl,
        conversion_stats,
        trainer_run_dir,
    )
    write_json(operation_dir / "native_command_manifest.json", manifest)
    executed_plan = replace(
        plan,
        commands=[command["command"] for command in manifest["commands"]],
        warnings=[*plan.warnings, *manifest.get("notes", [])],
    )
    write_stage_manifest(output_dir, executed_plan, config, benchmark)

    results = []
    build_result = _run_command(
        manifest["commands"][0],
        output_dir / "native_command_01.log",
        config=config,
    )
    results.append(build_result)
    if build_result["returncode"] != 0:
        manifest["results"] = results
        manifest["status"] = "failed"
        write_json(operation_dir / "native_command_manifest.json", manifest)
        write_json(output_dir / "native_execution_summary.json", {"commands": results})
        raise RuntimeError("native SFT prompt-preserving conversion failed")

    records = read_jsonl(train_jsonl)
    _validate_sft_records(config, benchmark, records, train_jsonl)
    conversion_payload = read_json(conversion_stats)
    conversion_payload["run_dir"] = "<base-artifact>"
    conversion_payload["run_id"] = _training_scope(config, benchmark)
    conversion_payload["out"] = "sft_train.jsonl"
    write_json(conversion_stats, conversion_payload)
    source_metadata = {
        "kind": "base_artifact_raw_traces",
        "conversion_stats": conversion_payload,
    }
    dataset_summary = _write_sft_dataset(
        output_dir,
        records,
        source_metadata,
        config,
        benchmark,
    )

    train_result = _run_command(
        manifest["commands"][1],
        output_dir / "native_command_02.log",
        config=config,
    )
    results.append(train_result)
    manifest["results"] = results
    manifest["status"] = "completed" if train_result["returncode"] == 0 else "failed"
    write_json(operation_dir / "native_command_manifest.json", manifest)
    write_json(output_dir / "native_execution_summary.json", {"commands": results})
    if train_result["returncode"] != 0:
        raise RuntimeError("native SFT training failed")

    _register_sft_checkpoint(
        config,
        benchmark,
        output_dir=output_dir,
        trainer_run_dir=trainer_run_dir,
        records=records,
        dataset_summary=dataset_summary,
        source_metadata=source_metadata,
        manifest=manifest,
    )
    if not config.sft.validate_only:
        adapter = _native_adapter_path(trainer_run_dir)
        if adapter is None:
            raise RuntimeError(f"SFT completed without a loadable adapter: {trainer_run_dir}")
        _export_release_adapter(
            adapter,
            benchmark.resolve_paths(config).reference_checkpoint,
        )
    write_stage_manifest(output_dir, executed_plan, config, benchmark)
    return executed_plan


def _execute_noop_collect(config: DDOConfig, benchmark: BenchmarkAdapter, plan: Any) -> Any:
    output_dir = single_output_dir(plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories, source_metadata = _load_base_trajectories(config, benchmark)
    write_trajectories(output_dir / "base_trajectories.jsonl", trajectories)
    write_jsonl(output_dir / "branch_sets.jsonl", [])
    write_json(
        output_dir / "collection_summary.json",
        {
            "artifact_type": "collection_summary",
            "backend": "native",
            "benchmark": benchmark.name,
            "task_id": _training_scope(config, benchmark),
            "collection_method": config.collection.method,
            "num_base_trajectories": len(trajectories),
            "num_branch_sets": 0,
            "num_branches": 0,
            "source": source_metadata,
            "noop_stage": True,
            "metadata": config_metadata(config, benchmark, stage="collect"),
        },
    )
    executed_plan = replace(plan, commands=["ddo native collect no-op"])
    _write_noop_command_manifest(
        output_dir,
        config,
        benchmark,
        stage="collect",
        command=executed_plan.commands[0],
        expected_outputs=[
            str(output_dir / "base_trajectories.jsonl"),
            str(output_dir / "branch_sets.jsonl"),
            str(output_dir / "collection_summary.json"),
        ],
    )
    write_stage_manifest(output_dir, executed_plan, config, benchmark)
    return executed_plan


def _execute_noop_build(config: DDOConfig, benchmark: BenchmarkAdapter, plan: Any) -> Any:
    output_dir = single_output_dir(plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_id = _training_scope(config, benchmark)
    dataset = PairDatasetArtifact(
        benchmark=benchmark.name,
        task_id=task_id,
        collection_method=config.collection.method,
        dataset_method=config.dataset.method,
        training_method=config.training.method,
        records=[],
        metadata={
            "backend": "native",
            "noop_stage": True,
            "criterion": config.dataset.criterion,
        },
    )
    write_pair_dataset(output_dir / "dataset.json", dataset)
    write_jsonl(output_dir / "train_pairs.jsonl", [])
    write_jsonl(output_dir / "train_pairs_noop.jsonl", [])
    write_json(
        output_dir / "dataset_summary.json",
        {
            "artifact_type": "dataset_summary",
            "backend": "native",
            "benchmark": benchmark.name,
            "task_id": task_id,
            "collection_method": config.collection.method,
            "dataset_method": config.dataset.method,
            "training_method": config.training.method,
            "num_records": 0,
            "pair_types": {},
            "noop_stage": True,
            "metadata": config_metadata(config, benchmark, stage="build"),
        },
    )
    executed_plan = replace(plan, commands=["ddo native build noop_pairs control dataset"])
    _write_noop_command_manifest(
        output_dir,
        config,
        benchmark,
        stage="build",
        command=executed_plan.commands[0],
        expected_outputs=[
            str(output_dir / "dataset.json"),
            str(output_dir / "train_pairs.jsonl"),
            str(output_dir / "train_pairs_noop.jsonl"),
            str(output_dir / "dataset_summary.json"),
        ],
    )
    write_stage_manifest(output_dir, executed_plan, config, benchmark)
    return executed_plan


def _execute_noop_train(config: DDOConfig, benchmark: BenchmarkAdapter, plan: Any) -> Any:
    output_dir = single_output_dir(plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = benchmark.resolve_paths(config)
    task_id = _training_scope(config, benchmark)
    sft_dir = paths.reference_checkpoint
    sft_adapter = None
    if config.training.method == "reference":
        sft_adapter = resolve_adapter_path(sft_dir)
        if sft_adapter is None:
            raise RuntimeError(f"native reference checkpoint requires a materialized SFT adapter: {sft_dir}")

    base_model = config.models.target_model
    if sft_adapter is not None:
        base_model = str(sft_adapter)
    adapter_config = {
        "artifact_type": "lora_adapter_config",
        "backend": "native",
        "base_model_name_or_path": base_model,
        "peft_type": "NONE" if config.training.method == "base" else "LORA",
        "finetune_type": config.training.finetune_type,
        "materialized_weights": False,
    }
    trainer_state = {
        "artifact_type": "trainer_state",
        "backend": "native",
        "status": "base_model" if config.training.method == "base" else "sft_reference",
        "global_step": 0,
        "learning_rate": config.training.learning_rate,
        "num_train_epochs": config.training.num_train_epochs,
    }
    checkpoint_manifest = {
        "artifact_type": (
            "base_checkpoint_manifest"
            if config.training.method == "base"
            else "sft_reference_checkpoint_manifest"
        ),
        "backend": "native",
        "checkpoint_id": _native_checkpoint_id(config, benchmark),
        "benchmark": benchmark.name,
        "task_id": task_id,
        "collection_method": config.collection.method,
        "dataset_method": config.dataset.method,
        "training_method": config.training.method,
        "target_model": config.models.target_model,
        "reference_model": config.models.reference_model or config.models.target_model,
        "num_train_records": 0,
        "adapter_config": "adapter_config.json" if config.training.method != "base" else None,
        "trainer_state": "trainer_state.json",
        "materialized_weights": False,
        "adapter_source_stage": "sft" if sft_adapter is not None else None,
        "relative_adapter_path": None,
        "adapter_path": None,
        "adapter_loading_status": "base_model" if config.training.method == "base" else "sft_dependency",
    }
    write_json(output_dir / "adapter_config.json", adapter_config)
    write_json(output_dir / "trainer_state.json", trainer_state)
    write_json(output_dir / "checkpoint_manifest.json", checkpoint_manifest)
    executed_plan = replace(plan, commands=[f"ddo native train {config.training.method} control checkpoint"])
    _write_noop_command_manifest(
        output_dir,
        config,
        benchmark,
        stage="train",
        command=executed_plan.commands[0],
        expected_outputs=[
            str(output_dir / "checkpoint_manifest.json"),
            str(output_dir / "trainer_state.json"),
        ],
    )
    write_stage_manifest(output_dir, executed_plan, config, benchmark)
    return executed_plan


def _execute_research_stage(config: DDOConfig, benchmark: BenchmarkAdapter, plan: Any) -> Any:
    output_dir = single_output_dir(plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = _research_root(config)
    manifest = _native_manifest_from_research_commands(config, benchmark, plan, root, output_dir)
    operation_dir = benchmark.resolve_paths(config).work_dir / plan.stage
    operation_dir.mkdir(parents=True, exist_ok=True)
    if plan.stage == "eval" and config.evaluation.server_mode == "managed":
        with _managed_evaluation_server(config, benchmark, operation_dir) as endpoint:
            _patch_evaluation_commands(
                manifest,
                base_url=endpoint["base_url"],
                model_id=endpoint["model_id"],
            )
            manifest["managed_server"] = endpoint
            manifest["preflight"] = _preflight_endpoint(endpoint["base_url"], endpoint["model_id"])
            return _execute_research_manifest(config, benchmark, plan, output_dir, operation_dir, manifest)

    try:
        preflight = _stage_preflight(config, plan.stage)
    except RuntimeError as exc:
        manifest["status"] = "failed"
        manifest["preflight_error"] = str(exc)
        write_json(operation_dir / "native_command_manifest.json", manifest)
        raise
    if preflight is not None:
        manifest["preflight"] = preflight
    return _execute_research_manifest(config, benchmark, plan, output_dir, operation_dir, manifest)


def _execute_research_manifest(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    plan: Any,
    output_dir: Path,
    operation_dir: Path,
    manifest: dict[str, Any],
) -> Any:
    write_json(operation_dir / "native_command_manifest.json", manifest)
    executed_plan = replace(
        plan,
        commands=[command["command"] for command in manifest.get("commands", [])],
        warnings=[*plan.warnings, *manifest.get("notes", [])],
    )
    write_stage_manifest(operation_dir, executed_plan, config, benchmark, status="running")
    try:
        results = _run_native_manifest_commands(manifest, operation_dir, config=config)
    except Exception:
        manifest["status"] = "failed"
        write_json(operation_dir / "native_command_manifest.json", manifest)
        write_stage_manifest(operation_dir, executed_plan, config, benchmark, status="failed")
        raise
    manifest["results"] = results
    manifest["status"] = "completed"
    write_json(operation_dir / "native_command_manifest.json", manifest)
    normalization = normalize_stage_outputs(
        config,
        benchmark,
        plan,
        output_dir,
        manifest,
        summary_dir=operation_dir,
    )
    if normalization.get("status") == "skipped":
        manifest["status"] = "failed"
        manifest["normalization_error"] = normalization
        write_json(operation_dir / "native_command_manifest.json", manifest)
        write_stage_manifest(operation_dir, executed_plan, config, benchmark, status="failed")
        raise RuntimeError(
            f"native {plan.stage} command completed but required outputs were missing: "
            f"{normalization.get('reason')}"
        )
    if plan.stage == "train" and not config.training.validate_only:
        adapter = _native_adapter_path(output_dir / "trainer_run")
        if adapter is None:
            raise RuntimeError(f"training completed without a loadable adapter: {output_dir}")
        _export_release_adapter(adapter, benchmark.resolve_paths(config).output_checkpoint)
    write_stage_manifest(operation_dir, executed_plan, config, benchmark)
    return executed_plan



def _build_base_command_manifest(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    output_dir: Path,
) -> dict[str, Any]:
    root = _research_root(config)
    command = (
        _webshop_base_command(config, benchmark, root, output_dir)
        if benchmark.name == "webshop"
        else _balrog_base_command(config, benchmark, root, output_dir)
    )
    return {
        "artifact_type": "native_command_manifest",
        "backend": "native",
        "stage": "base",
        "status": "prepared",
        "research_root": str(root),
        "run_commands": True,
        "commands": [command],
        "expected_outputs": [
            str(output_dir),
            str(output_dir / "trajectories.jsonl"),
            str(output_dir / "base_trajectories.jsonl"),
            str(output_dir / "base_summary.json"),
        ],
        "config": config_metadata(config, benchmark, stage="base"),
    }


def _webshop_base_command(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    script = _required_script(root, "ddo/evaluation/webshop/eval.py")
    argv = [
        sys.executable,
        str(script),
        "--output-root",
        str(output_dir.parent),
        "--run-id",
        output_dir.name,
        "--task-id",
        _task_filter(config, benchmark) or "webshop",
        "--episode-start",
        "0",
        "--episode-end",
        str(config.base.num_episodes),
        "--seed",
        str(config.base.seed),
        "--seed-mode",
        config.base.seed_mode,
        "--num-workers",
        str(config.runtime.num_workers),
        "--parallel-workers",
        "--model-id",
        _base_model_id(config),
        "--temperature",
        str(config.base.temperature),
        "--top-p",
        str(config.base.top_p),
        "--max-tokens",
        str(config.base.max_tokens),
        "--max-text-history",
        str(config.base.max_text_history),
        "--client-timeout",
        str(config.base.client_timeout),
        "--client-max-retries",
        str(config.base.client_max_retries),
        "--client-delay",
        str(config.base.client_delay),
        "--success-threshold",
        str(config.dataset.success_threshold),
    ]
    if config.base.max_steps_per_episode is not None:
        argv.extend(["--max-steps", str(config.base.max_steps_per_episode)])
    argv.extend(
        [
            "--llm-seed",
            str(config.base.seed),
            "--llm-seed-mode",
            "per_episode",
        ]
    )
    if config.models.expert_base_url:
        argv.extend(["--base-url", config.models.expert_base_url])
    return _command("native_collect_webshop_base", root, argv)


def _balrog_base_command(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    script = _required_script(root, "ddo/evaluation/balrog/eval.py")
    env_name = "babaisai" if benchmark.name == "babaisai" else "babyai"
    argv = [
        sys.executable,
        str(script),
        "--benchmark", env_name,
        "--task", _task_filter(config, benchmark) or "",
        "--output-dir", str(output_dir),
        "--num-episodes", str(config.base.num_episodes),
        "--num-workers", str(config.runtime.num_workers),
        "--seed", str(config.base.seed),
        "--seed-mode", config.base.seed_mode,
        "--model-id", _base_model_id(config),
        "--temperature", str(config.base.temperature),
        "--top-p", str(config.base.top_p),
        "--max-tokens", str(config.base.max_tokens),
        "--max-text-history", str(config.base.max_text_history),
        "--client-timeout", str(config.base.client_timeout),
        "--client-max-retries", str(config.base.client_max_retries),
        "--llm-seed-base", str(config.base.seed),
    ]
    if config.base.max_steps_per_episode is not None:
        argv.extend(["--max-steps", str(config.base.max_steps_per_episode)])
    if config.models.expert_base_url:
        argv.extend(["--base-url", config.models.expert_base_url])
    return _command("native_collect_balrog_base", root, argv)


def _build_sft_command_manifest(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    root: Path,
    output_dir: Path,
    base_run_dir: Path,
    train_jsonl: Path,
    conversion_stats: Path,
    trainer_run_dir: Path,
) -> dict[str, Any]:
    build_argv = [
        sys.executable,
        str(_required_script(root, "ddo/data/build_sft_from_raw.py")),
        "--run-dir",
        str(base_run_dir),
        "--out",
        str(train_jsonl),
        "--stats-out",
        str(conversion_stats),
        "--success-only",
        "--criterion",
        "webshop_reward_threshold" if benchmark.name == "webshop" else "legacy_success",
    ]
    if benchmark.name == "webshop":
        build_argv.extend(["--success-threshold", str(config.dataset.success_threshold)])
    if not config.benchmark.train_task_filters:
        task_filter = _task_filter(config, benchmark)
        if task_filter:
            build_argv.extend(["--task-filter", task_filter])
    if config.sft.source_shuffle_seed is not None:
        build_argv.extend(["--shuffle-seed", config.sft.source_shuffle_seed])

    train_argv = [
        sys.executable,
        str(_required_script(root, "ddo/training/train_trl_sft.py")),
        "--model-name-or-path",
        config.models.target_model,
        "--train-jsonl",
        str(train_jsonl),
        "--output-dir",
        str(trainer_run_dir),
        "--run-name",
        f"{_safe_run_name(config, benchmark)}__sft",
        "--learning-rate",
        str(config.sft.learning_rate),
        "--num-train-epochs",
        str(config.sft.num_train_epochs),
        "--max-steps",
        str(config.sft.max_steps),
        "--warmup-ratio",
        str(config.sft.warmup_ratio),
        "--per-device-train-batch-size",
        str(config.sft.per_device_train_batch_size),
        "--gradient-accumulation-steps",
        str(config.sft.gradient_accumulation_steps),
        "--max-length",
        str(config.sft.max_length),
        "--dataloader-num-workers",
        str(config.sft.dataloader_num_workers),
        "--logging-steps",
        str(config.sft.logging_steps),
        "--save-steps",
        str(config.sft.save_steps),
        "--save-total-limit",
        _optional_int_arg(config.sft.save_total_limit),
        "--torch-dtype",
        config.sft.torch_dtype,
        "--lora-r",
        str(config.sft.lora_r),
        "--lora-alpha",
        str(config.sft.lora_alpha),
        "--lora-dropout",
        str(config.sft.lora_dropout),
        "--seed",
        str(config.sft.seed if config.sft.seed is not None else config.runtime.seed),
    ]
    if config.sft.save_epochs_fraction is not None:
        train_argv.extend(["--save-epochs-fraction", str(config.sft.save_epochs_fraction)])
    if config.sft.stop_after_epochs is not None:
        train_argv.extend(["--stop-after-epochs", str(config.sft.stop_after_epochs)])
    if config.sft.train_adapter_path:
        train_argv.extend(["--train-adapter-path", config.sft.train_adapter_path])
    if config.sft.resume_from_checkpoint:
        train_argv.extend(["--resume-from-checkpoint", config.sft.resume_from_checkpoint])
    if config.sft.bf16:
        train_argv.append("--bf16")
    if config.sft.fp16:
        train_argv.append("--fp16")
    train_argv.append(
        "--gradient-checkpointing"
        if config.sft.gradient_checkpointing
        else "--no-gradient-checkpointing"
    )
    if config.sft.lora_target_modules:
        train_argv.extend(["--lora-target-modules", config.sft.lora_target_modules])
    if config.sft.validate_only:
        train_argv.append("--validate-only")

    return {
        "artifact_type": "native_command_manifest",
        "backend": "native",
        "stage": "sft",
        "status": "prepared",
        "research_root": str(root),
        "run_commands": True,
        "commands": [
            _command("native_build_prompt_preserving_sft", root, build_argv),
            _command("native_train_sft_reference", root, train_argv),
        ],
        "expected_outputs": [
            str(train_jsonl),
            str(conversion_stats),
            str(output_dir / "sft_dataset.json"),
            str(trainer_run_dir / "run_metadata.json"),
            str(trainer_run_dir / "trainer_state.json"),
            str(output_dir / "checkpoint_manifest.json"),
            str(output_dir / "sft_checkpoint_manifest.json"),
        ],
        "notes": [],
        "config": config_metadata(config, benchmark, stage="sft"),
    }


def _load_base_trajectories(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
) -> tuple[list[TrajectoryArtifact], dict[str, Any]]:
    base_run_dir = benchmark.resolve_base_run_dir(config)
    if base_run_dir is None:
        raise RuntimeError("native SFT requires benchmark.base_run_dir with normalized base trajectories")
    for filename in ("trajectories.jsonl", "base_trajectories.jsonl"):
        candidate = base_run_dir / filename
        if candidate.exists():
            trajectories = read_trajectories(candidate)
            if not trajectories:
                raise RuntimeError(f"native SFT found no trajectories in {candidate}")
            return trajectories, {"kind": "file", "path": str(candidate)}
    raise RuntimeError(f"native SFT could not find trajectories.jsonl under base_run_dir: {base_run_dir}")


def _validate_sft_records(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    records: list[dict[str, Any]],
    source: Path,
) -> None:
    if not records:
        raise RuntimeError(f"native SFT conversion produced no rows: {source}")
    unsuccessful = [record for record in records if not bool(record.get("trajectory_success"))]
    if unsuccessful:
        raise RuntimeError(
            "native SFT expert_success conversion retained unsuccessful trajectories: "
            f"{len(unsuccessful)} rows"
        )
    task_ids = sorted({str(record.get("task_id") or "") for record in records})
    expected = config.benchmark.train_task_filters or [
        str(benchmark.normalize_task_id(config.benchmark.task_filter) or "")
    ]
    missing = [
        task_filter
        for task_filter in expected
        if task_filter
        and not any(_task_ids_match(task_id, task_filter) for task_id in task_ids)
    ]
    if missing:
        raise RuntimeError("native SFT conversion is missing tasks: " + ", ".join(missing))


def _write_sft_dataset(
    output_dir: Path,
    records: list[dict[str, Any]],
    source_metadata: dict[str, Any],
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
) -> dict[str, Any]:
    write_jsonl(output_dir / "sft_train.jsonl", records)
    summary = {
        "artifact_type": "sft_dataset",
        "backend": "native",
        "benchmark": benchmark.name,
        "task_id": _training_scope(config, benchmark),
        "task_ids": sorted({str(record.get("task_id") or "") for record in records}),
        "sft_method": config.sft.method,
        "num_records": len(records),
        "num_trajectories": len({str(record.get("trajectory_id")) for record in records}),
        "source": source_metadata,
        "train_jsonl": "sft_train.jsonl",
        "metadata": config_metadata(config, benchmark, stage="sft"),
    }
    write_json(output_dir / "sft_dataset.json", summary)
    return summary


def _register_sft_checkpoint(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    *,
    output_dir: Path,
    trainer_run_dir: Path,
    records: list[dict[str, Any]],
    dataset_summary: dict[str, Any],
    source_metadata: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    run_metadata_path = trainer_run_dir / "run_metadata.json"
    trainer_state_path = trainer_run_dir / "trainer_state.json"
    if config.sft.validate_only:
        run_metadata_payload = {
            "artifact_type": "sft_run_metadata",
            "backend": "native",
            "status": "validated",
            "model_name_or_path": config.models.target_model,
            "input": {
                "train_jsonl": str(output_dir / "sft_train.jsonl"),
                "source": source_metadata,
            },
            "dataset_summary": dataset_summary,
        }
        trainer_state_payload = {
            "artifact_type": "sft_trainer_state",
            "backend": "native",
            "status": "validated",
            "global_step": 0,
        }
        write_json(run_metadata_path, run_metadata_payload)
        write_json(trainer_state_path, trainer_state_payload)
    else:
        if not run_metadata_path.exists() or not trainer_state_path.exists():
            raise RuntimeError(f"native SFT trainer did not write expected metadata under: {trainer_run_dir}")
        run_metadata_payload = read_json(run_metadata_path)
        trainer_state_payload = read_json(trainer_state_path)

    adapter_path = None if config.sft.validate_only else _native_adapter_path(trainer_run_dir)
    if not config.sft.validate_only and adapter_path is None:
        raise RuntimeError(f"native SFT trainer produced no loadable adapter under: {trainer_run_dir}")
    adapter_config = {
        "artifact_type": "sft_adapter_config",
        "backend": "native",
        "base_model_name_or_path": config.models.target_model,
        "peft_type": "LORA" if config.sft.finetune_type == "lora" else config.sft.finetune_type.upper(),
        "finetune_type": config.sft.finetune_type,
        "materialized_weights": adapter_path is not None,
        "r": config.sft.lora_r,
        "lora_alpha": config.sft.lora_alpha,
        "lora_dropout": config.sft.lora_dropout,
        "lora_target_modules": config.sft.lora_target_modules,
    }
    checkpoint_manifest = {
        "artifact_type": "sft_checkpoint_manifest",
        "backend": "native",
        "checkpoint_id": sft_checkpoint_id(config, benchmark),
        "benchmark": benchmark.name,
        "task_id": _training_scope(config, benchmark),
        "task_ids": sorted({str(record.get("task_id") or "") for record in records}),
        "target_model": config.models.target_model,
        "expert_model": config.models.expert_model,
        "num_train_records": len(records),
        "adapter_config": "adapter_config.json",
        "trainer_state": "trainer_state.json",
        "native_command_manifest": "native_command_manifest.json",
        "trainer_run_path": "trainer_run",
        "relative_adapter_path": relative_adapter_path(output_dir, adapter_path),
        "adapter_path": relative_adapter_path(output_dir, adapter_path),
        "adapter_loading_status": (
            "validate_only" if config.sft.validate_only else "available" if adapter_path is not None else "missing"
        ),
        "materialized_weights": adapter_path is not None,
        "run_metadata": run_metadata_payload,
        "trainer_state_payload": trainer_state_payload,
        "command_results": manifest.get("results", []),
    }
    write_json(output_dir / "adapter_config.json", adapter_config)
    write_json(output_dir / "run_metadata.json", run_metadata_payload)
    write_json(output_dir / "trainer_state.json", trainer_state_payload)
    write_json(output_dir / "checkpoint_manifest.json", checkpoint_manifest)
    write_json(output_dir / "sft_checkpoint_manifest.json", checkpoint_manifest)
    return checkpoint_manifest


def _export_release_adapter(source: Path, target: Path) -> None:
    required = ("adapter_model.safetensors", "adapter_config.json")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise RuntimeError(
            f"adapter export source is missing required files under {source}: {', '.join(missing)}"
        )
    target.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(path.name for path in target.iterdir() if path.name not in required)
    if unexpected:
        raise RuntimeError(
            f"release checkpoint directory must contain only PEFT inference files: {target}; "
            f"unexpected: {', '.join(unexpected)}"
        )
    for name in required:
        temporary = target / f".{name}.tmp"
        shutil.copy2(source / name, temporary)
        temporary.replace(target / name)


def _native_adapter_path(trainer_run_dir: Path) -> Path | None:
    latest_checkpoint = _latest_checkpoint_dir(trainer_run_dir)
    if latest_checkpoint is not None:
        return latest_checkpoint
    if (trainer_run_dir / "adapter_config.json").exists():
        return trainer_run_dir
    return None


def _latest_checkpoint_dir(run_dir: Path) -> Path | None:
    checkpoints = [path for path in run_dir.glob("checkpoint-*") if path.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_sort_key)


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.rsplit("-", 1)[-1]
    try:
        return int(suffix), path.name
    except ValueError:
        return -1, path.name


def _native_manifest_from_research_commands(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    plan: Any,
    root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = build_stage_command_manifest(config, benchmark, plan, root)
    if plan.stage == "train":
        _rewrite_training_manifest_output_dir(manifest, output_dir / "trainer_run", output_dir)
    return manifest


def _rewrite_training_manifest_output_dir(
    manifest: dict[str, Any],
    trainer_run_dir: Path,
    output_dir: Path,
) -> None:
    for command in manifest.get("commands", []):
        argv = command.get("argv")
        if not isinstance(argv, list):
            continue
        if "--output-dir" not in argv:
            continue
        index = argv.index("--output-dir")
        if index + 1 >= len(argv):
            continue
        argv[index + 1] = str(trainer_run_dir)
        command["command"] = _argv_to_command(argv)
    manifest["expected_outputs"] = [
        str(trainer_run_dir / "run_metadata.json"),
        str(trainer_run_dir / "trainer_state.json"),
        str(output_dir / "native_command_manifest.json"),
    ]


def _run_native_manifest_commands(
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    config: DDOConfig,
) -> list[dict[str, Any]]:
    commands = manifest.get("commands") or []
    if not commands:
        raise RuntimeError(f"native stage {manifest.get('stage')} has no runnable commands")
    results: list[dict[str, Any]] = []
    for index, command in enumerate(commands, start=1):
        if not command.get("runnable", True):
            raise RuntimeError(
                f"native command is not marked runnable: {command.get('name')}. "
                f"notes={manifest.get('notes', [])}"
            )
        if command.get("executor"):
            raise RuntimeError(
                "native backend does not use smoke fallback executors; "
                f"unsupported executor for {command.get('name')}: {command.get('executor')}"
            )
        if not command.get("argv"):
            raise RuntimeError(f"native command has no argv: {command.get('name')}")
        result = _run_command(
            command,
            output_dir / f"native_command_{index:02d}.log",
            config=config,
        )
        results.append(result)
        if result["returncode"] != 0:
            write_json(output_dir / "native_execution_summary.json", {"commands": results})
            raise RuntimeError(f"native command failed: {command['name']}")
    write_json(output_dir / "native_execution_summary.json", {"commands": results})
    return results


def _write_noop_command_manifest(
    output_dir: Path,
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    *,
    stage: str,
    command: str,
    expected_outputs: list[str],
) -> None:
    result = {
        "name": f"native_{stage}_control",
        "returncode": 0,
        "log_path": None,
    }
    write_json(
        output_dir / "native_command_manifest.json",
        {
            "artifact_type": "native_command_manifest",
            "backend": "native",
            "stage": stage,
            "status": "completed",
            "research_root": str(_research_root(config)),
            "run_commands": False,
            "commands": [
                {
                    "name": result["name"],
                    "cwd": str(_research_root(config)),
                    "argv": [],
                    "command": command,
                    "executor": "native_control",
                    "runnable": True,
                }
            ],
            "expected_outputs": expected_outputs,
            "results": [result],
            "notes": ["control stage materialized without external subprocess"],
            "config": config_metadata(config, benchmark, stage=stage),
        },
    )
    write_json(output_dir / "native_execution_summary.json", {"commands": [result]})


def _run_command(
    command: dict[str, Any], log_path: Path, *, config: DDOConfig
) -> dict[str, Any]:
    argv = command.get("argv") or []
    if len(argv) >= 2 and str(argv[1]).endswith(".py"):
        script = Path(str(argv[1]))
        if not script.is_file():
            raise RuntimeError(
                f"native command script does not exist: {script}. "
                "Install ignored external dependencies as documented in benchmarks/README.md"
            )
    environment = _subprocess_environment(config)
    command["environment"] = environment
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command["argv"],
            cwd=command["cwd"],
            env={**os.environ, **environment, "PYTHONUNBUFFERED": "1"},
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "name": command["name"],
        "returncode": completed.returncode,
        "log_path": str(log_path),
        "environment": environment,
    }


def _normalize_raw_trajectories(
    output_dir: Path,
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
) -> list[TrajectoryArtifact]:
    trajectories = []
    for trace_path in sorted(output_dir.glob("**/*_llm_trace.json")):
        if _is_generated_branch_trace(trace_path):
            continue
        trajectory = _trajectory_from_trace(trace_path, output_dir, config, benchmark)
        if trajectory is not None:
            trajectories.append(trajectory)
    return trajectories


def _validate_base_trajectories(
    trajectories: list[TrajectoryArtifact],
    output_dir: Path,
    *,
    expected_count: int | None = None,
) -> None:
    if not trajectories:
        raise RuntimeError(f"native base produced no compatible raw traces under: {output_dir}")
    if expected_count is not None and len(trajectories) != int(expected_count):
        raise RuntimeError(
            "native base trajectory count does not match the requested episode count: "
            f"expected={int(expected_count)}, actual={len(trajectories)}, output={output_dir}"
        )
    nonempty_actions = sum(
        1
        for trajectory in trajectories
        for step in trajectory.steps
        if str(step.action or "").strip()
    )
    if nonempty_actions:
        return
    abort_reasons = _base_abort_reasons(output_dir)
    detail = f"; abort_reasons={abort_reasons}" if abort_reasons else ""
    raise RuntimeError(
        "native base produced trajectories with zero non-empty actions. "
        "The model endpoint likely failed or every rollout aborted"
        f"{detail}"
    )


def _existing_base_artifacts(output_dir: Path) -> dict[str, Any] | None:
    trajectories_path = output_dir / "trajectories.jsonl"
    if not trajectories_path.exists():
        return None
    trajectories = read_trajectories(trajectories_path)
    _validate_base_trajectories(trajectories, output_dir)
    return {
        "trajectories_path": str(trajectories_path),
        "base_trajectories_path": str(output_dir / "base_trajectories.jsonl"),
        "base_summary_path": str(output_dir / "base_summary.json"),
        "import_manifest_path": (
            str(output_dir / "import_manifest.json")
            if (output_dir / "import_manifest.json").exists()
            else None
        ),
        "num_trajectories": len(trajectories),
        "num_success": sum(1 for trajectory in trajectories if trajectory.won),
        "num_nonempty_actions": sum(
            1
            for trajectory in trajectories
            for step in trajectory.steps
            if str(step.action or "").strip()
        ),
    }


def _existing_base_manifest(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    output_dir: Path,
    existing: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_type": "native_command_manifest",
        "backend": "native",
        "stage": "base",
        "status": "completed",
        "reuse_existing_output": True,
        "research_root": str(_research_root(config)),
        "run_commands": False,
        "commands": [
            {
                "name": "native_reuse_existing_base_trajectories",
                "cwd": str(_research_root(config)),
                "argv": [],
                "command": f"reuse existing base trajectories at {output_dir}",
                "runnable": False,
            }
        ],
        "expected_outputs": [
            str(output_dir / "trajectories.jsonl"),
            str(output_dir / "base_trajectories.jsonl"),
            str(output_dir / "base_summary.json"),
        ],
        "existing_artifacts": existing,
        "config": config_metadata(config, benchmark, stage="base"),
    }


def _base_abort_reasons(output_dir: Path) -> dict[str, int]:
    reasons: dict[str, int] = {}
    for trace_path in sorted(output_dir.glob("**/*_llm_trace.json")):
        if _is_generated_branch_trace(trace_path):
            continue
        try:
            payload = _read_json_object(trace_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        calls = payload.get("calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            extras = call.get("extras")
            if not isinstance(extras, dict):
                continue
            reason = str(extras.get("abort_reason") or extras.get("abort_leaf_last") or "").strip()
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
    return dict(sorted(reasons.items()))


def _stage_preflight(config: DDOConfig, stage: str) -> dict[str, Any] | None:
    model_id = _stage_model_id(config, stage)
    if model_id is None:
        return None
    return _preflight_model_endpoint(config, stage, model_id)


def _adapter_lora_rank(adapter_path: Path) -> int:
    config_path = adapter_path / "adapter_config.json"
    try:
        payload = _read_json_object(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read LoRA adapter config: {config_path}") from exc
    rank = payload.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"Invalid LoRA rank in {config_path}: {rank!r}")
    return rank


@contextmanager
def _managed_evaluation_server(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    output_dir: Path,
):
    host = config.evaluation.server_host
    port = config.evaluation.server_port
    _require_available_port(host, port)
    model_id = _managed_eval_model_id(config, benchmark)
    adapter_path = _evaluation_adapter_path(config, benchmark)
    base_url = f"http://{host}:{port}/v1"
    log_path = output_dir / "managed_vllm.log"
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config.models.target_model,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        config.evaluation.server_dtype,
        "--tensor-parallel-size",
        str(config.evaluation.server_tensor_parallel_size),
        "--max-model-len",
        str(config.evaluation.server_max_model_len),
        "--max-num-seqs",
        str(config.evaluation.server_max_num_seqs),
        "--max-num-batched-tokens",
        str(config.evaluation.server_max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(config.evaluation.server_gpu_memory_utilization),
        "--generation-config",
        "vllm",
    ]
    if config.evaluation.server_enforce_eager:
        argv.append("--enforce-eager")
    if config.evaluation.server_reasoning_parser:
        argv.extend(["--reasoning-parser", config.evaluation.server_reasoning_parser])
    if adapter_path is None:
        argv.extend(["--served-model-name", model_id])
    else:
        argv.extend(
            [
                "--enable-lora",
                "--max-loras",
                "1",
                "--max-lora-rank",
                str(_adapter_lora_rank(adapter_path)),
                "--lora-modules",
                f"{model_id}={adapter_path}",
            ]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    environment = _subprocess_environment(config)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            argv,
            cwd=str(_research_root(config)),
            env={**os.environ, **environment},
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            _wait_for_managed_server(
                process,
                base_url=base_url,
                model_id=model_id,
                timeout=config.evaluation.server_ready_timeout,
                poll_interval=config.evaluation.server_poll_interval,
                log_path=log_path,
            )
            yield {
                "mode": "managed",
                "base_url": base_url,
                "model_id": model_id,
                "adapter_path": str(adapter_path) if adapter_path is not None else None,
                "command": _argv_to_command(argv),
                "environment": environment,
                "log_path": str(log_path),
            }
        finally:
            _terminate_process_group(process)


def _evaluation_adapter_path(config: DDOConfig, benchmark: BenchmarkAdapter) -> Path | None:
    if config.evaluation.adapter_path:
        if config.project_root is None:
            raise RuntimeError("paper checkpoint inference requires a project root")
        return resolve_paper_adapter(
            config.evaluation.adapter_path,
            project_root=config.project_root,
            expected_base_model=config.models.target_model,
            verify_files=True,
        )
    if config.training.method == "base":
        return None
    paths = benchmark.resolve_paths(config)
    stage_dir = (
        paths.reference_checkpoint
        if config.training.method == "reference"
        else paths.output_checkpoint
    )
    adapter = resolve_adapter_path(stage_dir)
    if adapter is None:
        raise RuntimeError(f"eval requires a materialized adapter under: {stage_dir}")
    return adapter


def _managed_eval_model_id(config: DDOConfig, benchmark: BenchmarkAdapter) -> str:
    return f"ddo_{_safe_run_name(config, benchmark)}"


def _patch_evaluation_commands(
    manifest: dict[str, Any],
    *,
    base_url: str,
    model_id: str,
) -> None:
    for command in manifest.get("commands", []):
        if not command.get("patch_model_endpoint", True):
            continue
        argv = command.get("argv")
        if not isinstance(argv, list):
            continue
        if "--model-id" in argv:
            _replace_option(argv, "--model-id", model_id)
            if "--base-url" in argv:
                _replace_option(argv, "--base-url", base_url)
            else:
                argv.extend(["--base-url", base_url])
        else:
            _replace_assignment(argv, "client.model_id", model_id)
            _replace_assignment(argv, "client.base_url", base_url)
        command["command"] = _argv_to_command(argv)


def _replace_option(argv: list[str], option: str, value: str) -> None:
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise RuntimeError(f"command option has no value: {option}")
    argv[index + 1] = value


def _replace_assignment(argv: list[str], key: str, value: str) -> None:
    prefix = f"{key}="
    for index, item in enumerate(argv):
        if item.startswith(prefix):
            argv[index] = f"{prefix}{value}"
            return
    argv.append(f"{prefix}{value}")


def _require_available_port(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"managed eval server port is unavailable: {host}:{port}") from exc


def _wait_for_managed_server(
    process: subprocess.Popen[str],
    *,
    base_url: str,
    model_id: str,
    timeout: int,
    poll_interval: float,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"managed eval server exited with code {returncode}; see {log_path}"
            )
        try:
            _preflight_endpoint(base_url, model_id)
            return
        except RuntimeError as exc:
            last_error = exc
        time.sleep(poll_interval)
    raise RuntimeError(
        f"managed eval server was not ready within {timeout}s; see {log_path}; last_error={last_error}"
    )


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def _stage_model_id(config: DDOConfig, stage: str) -> str | None:
    if stage in {"base", "collect"}:
        return _base_model_id(config)
    if stage == "eval":
        return config.models.target_model
    return None


def _preflight_model_endpoint(
    config: DDOConfig,
    stage: str,
    model_id: str | None,
) -> dict[str, Any]:
    endpoint_field = "target_base_url" if stage == "eval" else "expert_base_url"
    return _preflight_configured_endpoint(
        getattr(config.models, endpoint_field),
        endpoint_field,
        model_id,
    )


def _preflight_configured_endpoint(
    configured_url: str | None,
    endpoint_field: str,
    model_id: str | None,
) -> dict[str, Any]:
    base_url = str(configured_url or "").strip()
    if not base_url:
        return {
            "status": "skipped",
            "reason": f"models.{endpoint_field} is not configured",
            "requested_model_id": model_id,
        }
    return _preflight_endpoint(base_url, model_id)


def _preflight_endpoint(base_url: str, model_id: str | None) -> dict[str, Any]:
    base_url = base_url.strip()

    models_url = f"{base_url.rstrip('/')}/models"
    try:
        payload = _fetch_model_endpoint_payload(models_url)
    except Exception as exc:
        if isinstance(exc, (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError)):
            raise RuntimeError(
                "native model endpoint preflight failed: "
                f"could not read {models_url} for model_id={model_id!r} "
                f"({type(exc).__name__}: {exc}). "
                "Start the configured model server or set the matching models.*_base_url endpoint."
            ) from exc
        raise

    served_model_ids = _served_model_ids(payload)
    if model_id and served_model_ids and model_id not in served_model_ids:
        raise RuntimeError(
            "native model endpoint preflight failed: "
            f"{models_url} is reachable, but model_id={model_id!r} is not served. "
            f"served_model_ids={served_model_ids}"
        )
    return {
        "status": "ok",
        "models_url": models_url,
        "requested_model_id": model_id,
        "served_model_ids": served_model_ids,
    }


def _fetch_model_endpoint_payload(models_url: str) -> Any:
    request = Request(models_url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def _served_model_ids(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        data = payload.get("data")
    else:
        data = payload
    if not isinstance(data, list):
        return []

    model_ids: list[str] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if isinstance(model_id, str) and model_id:
            model_ids.append(model_id)
    return sorted(set(model_ids))


def _trajectory_from_trace(
    trace_path: Path,
    output_dir: Path,
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
) -> TrajectoryArtifact | None:
    trace_payload = _read_json_object(trace_path)
    calls = trace_payload.get("calls")
    if not isinstance(calls, list):
        return None

    stem = trace_path.name[: -len("_llm_trace.json")]
    episode_path = trace_path.with_name(f"{stem}.json")
    episode_log = _read_json_object(episode_path) if episode_path.exists() else {}
    task_id = str(
        trace_payload.get("task_or_env_params")
        or episode_log.get("task")
        or benchmark.normalize_task_id(config.benchmark.task_filter)
        or benchmark.name
    )
    trajectory_id = str(trace_payload.get("episode_id") or stem)
    steps = [
        _step_from_call(index, call, trace_path.relative_to(output_dir))
        for index, call in enumerate(calls)
        if isinstance(call, dict)
    ]
    return TrajectoryArtifact(
        benchmark=benchmark.name,
        task_id=task_id,
        trajectory_id=trajectory_id,
        model_id=str(trace_payload.get("model_id") or _base_model_id(config)),
        seed=_as_int(trace_payload.get("seed"), default=config.base.seed),
        won=_trajectory_won(episode_log, trace_payload, calls),
        score=_trajectory_score(episode_log, calls),
        steps=steps,
        metadata={
            "source": "native_raw_trace",
            "trace_path": str(trace_path.relative_to(output_dir)),
            "episode_path": str(episode_path.relative_to(output_dir)) if episode_path.exists() else None,
            "csv_path": str(trace_path.with_name(f"{stem}.csv").relative_to(output_dir)),
            "env_name": trace_payload.get("env_name"),
            "schema_version": trace_payload.get("schema_version"),
            "episode_idx": trace_payload.get("episode_idx"),
            "base_seed": trace_payload.get("base_seed") or episode_log.get("base_seed"),
            "seed_mode": trace_payload.get("seed_mode") or episode_log.get("seed_mode"),
        },
    )


def _step_from_call(index: int, call: dict[str, Any], relative_trace_path: Path) -> TrajectoryStep:
    reward = _optional_float(call.get("reward"))
    progression = _optional_float(call.get("progression"))
    return TrajectoryStep(
        step_index=_as_int(call.get("episode_step"), default=index),
        observation=str(call.get("observation") or call.get("observation_pre") or ""),
        action=str(call.get("action") or call.get("action_executed") or call.get("action_model") or ""),
        raw_model_output=None if call.get("raw_output") is None else str(call.get("raw_output")),
        reward=reward,
        done=bool(call.get("done") or call.get("terminated")),
        metadata={
            "source_trace_path": str(relative_trace_path),
            "call_idx": call.get("call_idx"),
            "thought": call.get("thought"),
            "feedback": call.get("feedback"),
            "observation_post": call.get("observation_post"),
            "progression": progression,
            "token_usage": call.get("token_usage"),
            "won": call.get("won"),
            "lost": call.get("lost"),
            "termination_reason": call.get("termination_reason"),
            "action_defaulted": call.get("action_defaulted"),
        },
    )


def _trajectory_won(
    episode_log: dict[str, Any],
    trace_payload: dict[str, Any],
    calls: list[Any],
) -> bool:
    for key in ("success", "won"):
        value = episode_log.get(key)
        if value is not None:
            return bool(value)
    value = trace_payload.get("won")
    if value is not None:
        return bool(value)
    episode_progress = _optional_float(episode_log.get("progression"))
    if episode_progress is not None and episode_progress >= 1.0:
        return True
    return any(isinstance(call, dict) and bool(call.get("won")) for call in calls)


def _trajectory_score(episode_log: dict[str, Any], calls: list[Any]) -> float | None:
    for key in ("score", "progression", "episode_return", "reward"):
        score = _optional_float(episode_log.get(key))
        if score is not None:
            return score
    for call in reversed(calls):
        if not isinstance(call, dict):
            continue
        for key in ("progression", "reward"):
            score = _optional_float(call.get(key))
            if score is not None:
                return score
    return None


def _research_root(config: DDOConfig) -> Path:
    root = config.project_root or Path.cwd()
    if not (root / "ddo").exists():
        raise RuntimeError(f"native research scripts are missing under repository root: {root}")
    return root.resolve()


def _external_script(root: Path, relative_path: str) -> Path:
    """Resolve an ignored user-managed dependency without requiring it during planning."""
    return root / relative_path


def _required_script(root: Path, relative_path: str) -> Path:
    script = root / relative_path
    if not script.exists():
        raise RuntimeError(f"native research script does not exist: {script}")
    return script


def _command(name: str, cwd: Path, argv: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "cwd": str(cwd),
        "argv": argv,
        "command": _argv_to_command(argv),
        "runnable": True,
    }


def _argv_to_command(argv: list[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def _subprocess_environment(config: DDOConfig) -> dict[str, str]:
    """Return the explicit, non-secret runtime environment recorded with commands."""

    environment: dict[str, str] = {}
    if config.runtime.cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = config.runtime.cuda_visible_devices

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = str(Path(conda_prefix) / "lib")
        inherited = os.environ.get("LD_LIBRARY_PATH", "")
        entries = [entry for entry in inherited.split(os.pathsep) if entry != conda_lib]
        environment["LD_LIBRARY_PATH"] = os.pathsep.join([conda_lib, *entries])
    return environment


def _native_checkpoint_id(config: DDOConfig, benchmark: BenchmarkAdapter) -> str:
    task_id = _training_scope(config, benchmark)
    return "__".join(
        [
            benchmark.name,
            __safe_slug(task_id),
            config.collection.method,
            config.dataset.method,
            config.training.method,
            __safe_slug(config.models.target_model or "model"),
            str(config.runtime.seed),
        ]
    )


def __safe_slug(value: str) -> str:
    normalized = value.rsplit("/", 1)[-1]
    return "".join(char.lower() if char.isalnum() else "_" for char in normalized).strip("_") or "value"


def _task_filter(config: DDOConfig, benchmark: BenchmarkAdapter) -> str | None:
    task = benchmark.normalize_task_id(config.benchmark.task_filter)
    if not task:
        return None
    if benchmark.name == "babaisai" and task.startswith("babaisai/"):
        return f"env/{task.split('/', 1)[1]}"
    return task


def _training_scope(config: DDOConfig, benchmark: BenchmarkAdapter) -> str:
    if config.benchmark.train_task_filters:
        return config.benchmark.task_group or "all"
    return benchmark.normalize_task_id(config.benchmark.task_filter) or "all"


def _base_model_id(config: DDOConfig) -> str:
    return config.models.expert_model or config.models.target_model or "Qwen/Qwen3-1.7B"


def _is_generated_branch_trace(trace_path: Path) -> bool:
    return "__dtc_" in trace_path.name or any(part.startswith("_dtc") for part in trace_path.parts)


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _as_int(value: Any, *, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int_arg(value: int | None) -> str:
    return "none" if value is None else str(value)
