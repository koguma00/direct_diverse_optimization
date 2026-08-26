"""Normalize vendored research-program outputs into reusable stage artifacts."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from ddo.evaluation.benchmarks.base import BenchmarkAdapter
from ddo.config.schema import DDOConfig
from ddo.pipeline.stage_io import relative_adapter_path
from ddo.schemas import (
    BranchRecord,
    BranchSetArtifact,
    PairDatasetArtifact,
    PairRecord,
    read_jsonl,
    write_branch_sets,
    write_json,
    write_jsonl,
    write_pair_dataset,
)
from ddo.schemas.io import read_json


def normalize_stage_outputs(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    plan: Any,
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    backend: str = "native",
    summary_dir: Path | None = None,
) -> dict[str, Any]:
    if plan.stage == "base":
        summary = _normalize_base(output_dir, backend=backend)
    elif plan.stage == "collect":
        summary = _normalize_collect(config, benchmark, output_dir, backend=backend)
    elif plan.stage == "build":
        summary = _normalize_dataset(config, benchmark, output_dir, backend=backend)
    elif plan.stage == "train":
        summary = _normalize_train(config, benchmark, output_dir, manifest, backend=backend)
    elif plan.stage == "eval":
        summary = _normalize_eval(config, benchmark, output_dir, backend=backend)
    else:
        raise ValueError(f"unknown stage: {plan.stage}")
    write_json((summary_dir or output_dir) / "normalization_summary.json", summary)
    return summary


def _normalize_base(output_dir: Path, *, backend: str) -> dict[str, Any]:
    for filename in ("trajectories.jsonl", "base_trajectories.jsonl"):
        trajectories = output_dir / filename
        if trajectories.exists():
            rows = read_jsonl(trajectories)
            return {
                "artifact_type": _normalization_artifact_type(backend),
                "backend": backend,
                "stage": "base",
                "status": "available_native_only",
                "source": str(trajectories),
                "outputs": [str(trajectories)],
                "num_trajectories": len(rows),
            }

    trace_paths = sorted(output_dir.glob("*_llm_trace.json"))
    if trace_paths:
        return {
            "artifact_type": _normalization_artifact_type(backend),
            "backend": backend,
            "stage": "base",
            "status": "available_raw_traces",
            "sources": [str(path) for path in trace_paths],
            "outputs": [str(output_dir)],
            "num_trajectories": len(trace_paths),
        }
    return _skipped("missing base trajectory artifacts", backend=backend, candidates=[str(output_dir)])


def _normalize_collect(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    output_dir: Path,
    *,
    backend: str,
) -> dict[str, Any]:
    native_branch_sets = output_dir / "branch_sets.jsonl"
    if native_branch_sets.exists():
        rows = read_jsonl(native_branch_sets)
        return {
            "artifact_type": _normalization_artifact_type(backend),
            "backend": backend,
            "stage": "collect",
            "status": "available_native_only",
            "source": str(native_branch_sets),
            "outputs": [str(native_branch_sets)],
            "num_branch_sets": len(rows),
            "num_branches": sum(len(row.get("branches", [])) for row in rows),
        }
    if config.collection.method == "dtc":
        return _normalize_dtc_collect(config, benchmark, output_dir, backend=backend)
    return _skipped("unsupported collection method", backend=backend, method=config.collection.method)



def _normalize_dtc_collect(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    output_dir: Path,
    *,
    backend: str,
) -> dict[str, Any]:
    branch_index = output_dir / "_dtc" / "branch_index.jsonl"
    if not branch_index.exists():
        return _skipped("missing DTC branch index", backend=backend, path=str(branch_index))

    rows = [
        row
        for row in read_jsonl(branch_index)
        if str(row.get("status") or "generated") == "generated"
    ]
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        base_id = str(row.get("base_traj_id") or row.get("source_trajectory_id") or "").strip()
        divergence_step = _as_int(row.get("divergence_step"), 0)
        state_key = str(row.get("state_key") or "").strip()
        if not base_id:
            continue
        grouped.setdefault((base_id, divergence_step, state_key), []).append(row)

    branch_sets: list[BranchSetArtifact] = []
    for (base_id, divergence_step, state_key), group in sorted(grouped.items()):
        task_id = str(group[0].get("task_id") or benchmark.normalize_task_id(config.benchmark.task_filter) or "")
        branches = [
            _branch_from_dtc_row(row, output_dir, divergence_step)
            for row in group
        ]
        branch_sets.append(
            BranchSetArtifact(
                benchmark=benchmark.name,
                task_id=task_id,
                collection_method=config.collection.method,
                source_trajectory_id=base_id,
                divergence_step=divergence_step,
                prompt_text=_prompt_from_trace(
                    benchmark.resolve_paths(config).base_trajectories,
                    base_id,
                    divergence_step,
                    state_key,
                ),
                branches=branches,
                metadata={
                    "backend": backend,
                    "source": str(branch_index),
                    "state_key": state_key,
                },
            )
        )

    write_branch_sets(output_dir / "branch_sets.jsonl", branch_sets)
    return {
        "artifact_type": _normalization_artifact_type(backend),
        "backend": backend,
        "stage": "collect",
        "status": "normalized",
        "source": str(branch_index),
        "outputs": [str(output_dir / "branch_sets.jsonl")],
        "num_branch_sets": len(branch_sets),
        "num_branches": sum(len(branch_set.branches) for branch_set in branch_sets),
    }



def _normalize_dataset(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    output_dir: Path,
    *,
    backend: str,
) -> dict[str, Any]:
    sources = _pair_sources(config, output_dir)
    existing = [path for path in sources if path.exists()]
    if not existing:
        return _skipped(
            "missing trainer-ready pair JSONL outputs",
            backend=backend,
            candidates=[str(path) for path in sources],
        )

    records: list[PairRecord] = []
    task_ids: set[str] = set()
    task_id = (
        config.benchmark.task_group
        if config.benchmark.train_task_filters
        else benchmark.normalize_task_id(config.benchmark.task_filter)
    ) or ""
    for source in existing:
        for row in read_jsonl(source):
            records.append(_pair_record_from_row(row, source))
            row_task_id = str(row.get("task_id") or "")
            if row_task_id:
                task_ids.add(row_task_id)
            if not config.benchmark.train_task_filters:
                task_id = row_task_id or task_id

    dataset = PairDatasetArtifact(
        benchmark=benchmark.name,
        task_id=task_id,
        collection_method=config.collection.method,
        dataset_method=config.dataset.method,
        training_method=config.training.method,
        records=records,
        metadata={
            "backend": backend,
            "sources": [str(path) for path in existing],
            "task_group": config.benchmark.task_group,
            "task_ids": sorted(task_ids),
        },
    )
    write_pair_dataset(output_dir / "dataset.json", dataset)
    write_jsonl(output_dir / "train_pairs.jsonl", records)
    return {
        "artifact_type": _normalization_artifact_type(backend),
        "backend": backend,
        "stage": "build",
        "status": "normalized",
        "sources": [str(path) for path in existing],
        "outputs": [str(output_dir / "dataset.json"), str(output_dir / "train_pairs.jsonl")],
        "num_records": len(records),
    }



def _normalize_train(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    backend: str,
) -> dict[str, Any]:
    run_metadata, trainer_state = _training_outputs(manifest)
    existing = [path for path in (run_metadata, trainer_state) if path is not None and path.exists()]
    if not existing:
        return _skipped(
            "missing trainer outputs",
            backend=backend,
            candidates=[str(path) for path in (run_metadata, trainer_state) if path is not None],
        )

    run_metadata_payload = read_json(run_metadata) if run_metadata is not None and run_metadata.exists() else {}
    trainer_state_payload = read_json(trainer_state) if trainer_state is not None and trainer_state.exists() else {}
    checkpoint_info = _checkpoint_info(config, run_metadata, trainer_state)
    if not config.training.validate_only and checkpoint_info["adapter_path"] is None:
        raise RuntimeError(
            f"trainer completed without a loadable adapter under: {checkpoint_info['run_dir']}"
        )
    task_id = _training_scope(config, benchmark)
    checkpoint_manifest = {
        "artifact_type": "lora_checkpoint_manifest",
        "backend": backend,
        "checkpoint_id": _checkpoint_id(config, benchmark, checkpoint_info),
        "benchmark": benchmark.name,
        "task_id": task_id,
        "collection_method": config.collection.method,
        "dataset_method": config.dataset.method,
        "training_method": config.training.method,
        "target_model": config.models.target_model,
        "reference_model": config.models.reference_model or config.models.target_model,
        "run_metadata_path": _relative_path(output_dir, run_metadata),
        "trainer_state_path": _relative_path(output_dir, trainer_state),
        "trainer_run_path": _relative_path(output_dir, checkpoint_info["run_dir"]),
        "selected_checkpoint_path": _relative_path(output_dir, checkpoint_info["checkpoint_dir"]),
        "relative_adapter_path": relative_adapter_path(output_dir, checkpoint_info["adapter_path"]),
        "adapter_path": relative_adapter_path(output_dir, checkpoint_info["adapter_path"]),
        "adapter_loading_status": checkpoint_info["adapter_loading_status"],
        "materialized_weights": checkpoint_info["adapter_path"] is not None,
        "run_metadata": run_metadata_payload,
        "trainer_state": trainer_state_payload,
    }
    write_json(output_dir / "checkpoint_manifest.json", checkpoint_manifest)
    return {
        "artifact_type": _normalization_artifact_type(backend),
        "backend": backend,
        "stage": "train",
        "status": "normalized",
        "sources": [str(path) for path in existing],
        "outputs": [str(output_dir / "checkpoint_manifest.json")],
        "checkpoint_id": checkpoint_manifest["checkpoint_id"],
        "adapter_path": checkpoint_manifest["relative_adapter_path"],
    }


def _normalize_eval(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    output_dir: Path,
    *,
    backend: str,
) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        existing = read_json(metrics_path)
        if _has_normalized_eval_metric(existing):
            return {
                "artifact_type": _normalization_artifact_type(backend),
                "backend": backend,
                "stage": "eval",
                "status": "available_native_only",
                "sources": [str(metrics_path)],
                "outputs": [str(metrics_path)],
            }

    candidates = _eval_summary_candidates(output_dir)
    performance_sources = [
        path for path in candidates if "performance" in path.parent.name.lower()
    ]
    source = performance_sources[0] if performance_sources else next(
        (path for path in candidates if path.exists()),
        None,
    )
    if source is None:
        raise RuntimeError(f"evaluation produced no benchmark summary under: {output_dir}")
    payload = read_json(source)
    environment_payload = _first_environment_payload(payload)
    success_value = _first_metric_value(
        _metric_value(
            environment_payload,
            [
                "success_rate",
                "success",
                "mean_success",
                "average_success",
                "win_rate",
            ],
        ),
        _success_from_counts(environment_payload),
        _percent_metric_value(
            environment_payload,
            ["average_progress", "progression_percentage", "mean_progress", "progress"],
        ),
    )
    diversity_sources = [
        path.parent
        for path in candidates
        if path.name == "summary.json" and "diversity" in path.parent.name.lower()
    ]
    diversity_sources = list(dict.fromkeys(diversity_sources))
    diversity_metrics = _effective_diversity_metrics(
        benchmark,
        diversity_sources,
        budget=config.evaluation.diversity_rollouts,
        webshop_success_threshold=config.evaluation.webshop_success_threshold,
        webshop_trajectory_class=config.evaluation.webshop_trajectory_class,
    )
    requested = list(dict.fromkeys(config.evaluation.suites))
    metric_values = {
        "success": _metric_payload(
            success_value,
            std=_percent_metric_value(
                environment_payload,
                ["standard_error", "stderr", "std_error"],
            ),
        ),
        **diversity_metrics,
    }
    for suite in requested:
        metric_values.setdefault(suite, _metric_payload(None))
    metrics = {
        "artifact_type": "evaluation_metrics",
        "backend": backend,
        "experiment": config.experiment.name,
        "benchmark": benchmark.name,
        "task_id": benchmark.normalize_task_id(config.benchmark.task_filter),
        "collection_method": config.collection.method,
        "dataset_method": config.dataset.method,
        "training_method": config.training.method,
        "source": str(source),
        "diversity_sources": [str(path) for path in diversity_sources],
        "metrics": {suite: metric_values[suite] for suite in requested},
    }
    write_json(metrics_path, metrics)
    _write_metrics_csv(output_dir / "metrics.csv", metrics)
    return {
        "artifact_type": _normalization_artifact_type(backend),
        "backend": backend,
        "stage": "eval",
        "status": "normalized",
        "sources": [str(source), *[str(path) for path in diversity_sources]],
        "outputs": [str(metrics_path)],
    }



def _first_environment_payload(payload: dict[str, Any]) -> dict[str, Any]:
    environments = payload.get("environments")
    if isinstance(environments, dict):
        first = next((value for value in environments.values() if isinstance(value, dict)), None)
        if first is not None:
            return first
    return payload


def _effective_diversity_metrics(
    benchmark: BenchmarkAdapter,
    run_dirs: list[Path],
    *,
    budget: int,
    webshop_success_threshold: float,
    webshop_trajectory_class: str,
) -> dict[str, dict[str, Any]]:
    names = ("esd", "h_esd", "wsd", "h_wsd")
    if not run_dirs:
        return {name: _metric_payload(None) for name in names}

    from ddo.metrics.effective_diversity import aggregate_run_dir_effective_diversity

    filter_name = {
        "babyai": "babyai_turn_noise",
        "babaisai": "babaisai_noop_loop_prune",
        "webshop": "webshop_strategy",
    }.get(benchmark.name, "none")
    stats = [
        aggregate_run_dir_effective_diversity(
            run_dir,
            budget=budget,
            filter_name=filter_name,
            webshop_success_threshold=webshop_success_threshold,
            webshop_trajectory_class=webshop_trajectory_class,
        )
        for run_dir in run_dirs
    ]
    values = {
        "esd": [row.effective_esd for row in stats],
        "h_esd": [row.effective_h_esd for row in stats],
        "wsd": [row.effective_wsd for row in stats],
        "h_wsd": [row.effective_h_wsd for row in stats],
    }
    return {
        name: {
            "status": "ok",
            "mean": mean(rows),
            "std": pstdev(rows) if len(rows) > 1 else 0.0,
            "per_seed": rows,
        }
        for name, rows in values.items()
    }


def _eval_summary_candidates(output_dir: Path) -> list[Path]:
    direct = [output_dir / "summary.json", output_dir / "results.json", output_dir / "eval_summary.json"]
    nested = [
        path
        for path in sorted(output_dir.rglob("*.json"))
        if _is_benchmark_eval_summary_candidate(path)
    ]
    seen: set[Path] = set()
    candidates: list[Path] = []
    for path in [*direct, *nested]:
        if path in seen:
            continue
        seen.add(path)
        candidates.append(path)
    return candidates


def _has_normalized_eval_metric(metrics: dict[str, Any]) -> bool:
    values = metrics.get("metrics")
    if not isinstance(values, dict):
        return False
    return any(isinstance(payload, dict) and payload.get("status") == "ok" for payload in values.values())


def _is_benchmark_eval_summary_candidate(path: Path) -> bool:
    if path.name in {"native_command_manifest.json", "normalization_summary.json"}:
        return False
    if path.name in {"stage_manifest.json", "checkpoint_manifest.json", "run_metadata.json"}:
        return False
    return path.name in {"summary.json", "results.json", "eval_summary.json"} or path.name.endswith("_summary.json")


def _branch_from_dtc_row(
    row: dict[str, Any],
    output_dir: Path,
    divergence_step: int,
) -> BranchRecord:
    trace = _resolve_optional_trace(output_dir, row.get("output_trace"), row.get("alt_traj_id"))
    action, response_text, won, score = _branch_details_from_trace(trace, divergence_step)
    action = str(row.get("action") or row.get("alt_action") or row.get("selected_action") or action or "")
    response_text = str(row.get("response_text") or response_text or f"Action: {action}")
    return BranchRecord(
        branch_id=str(row.get("branch_id") or row.get("alt_traj_id") or ""),
        trajectory_id=str(row.get("alt_traj_id") or ""),
        divergence_step=divergence_step,
        action=action,
        won=won,
        score=score,
        response_text=response_text,
        metadata={key: value for key, value in row.items() if key not in {"response_text"}},
    )



def _pair_record_from_row(row: dict[str, Any], source: Path) -> PairRecord:
    pair_type = str(row.get("pair_type") or "win_lose")
    return PairRecord(
        prompt_text=_text(row.get("prompt")),
        chosen_text=_completion_text(row.get("chosen")),
        rejected_text=_completion_text(row.get("rejected")),
        pair_type=pair_type,
        target_prob=_target_prob(row, pair_type),
        weight=_optional_float(row.get("pair_weight") or row.get("weight")) or 1.0,
        metadata={
            "source": str(source),
            **{key: value for key, value in row.items() if key not in {"prompt", "chosen", "rejected"}},
        },
    )


def _pair_sources(config: DDOConfig, output_dir: Path) -> list[Path]:
    task_slug = _task_slug(config.benchmark.task_filter)
    if config.dataset.method in {"rto_pairs", "tiedpo_rk_pairs", "tiedpo_dav_pairs", "tiedpo_pairs"}:
        return [
            output_dir / "train_pairs_DDO_balanced_geometric_win_lose.jsonl",
            output_dir / "train_pairs_DDO_balanced_geometric_win_win.jsonl",
        ]
    if config.dataset.method in {"divpo_pairs", "divpo_freq_pairs", "divpo_prob_pairs"}:
        return [output_dir / "train_pairs_DivPO.jsonl"]
    return [output_dir / f"{task_slug}_win_lose.jsonl"]


def _training_outputs(manifest: dict[str, Any]) -> tuple[Path | None, Path | None]:
    outputs = [Path(path) for path in manifest.get("expected_outputs", [])]
    run_metadata = next((path for path in outputs if path.name == "run_metadata.json"), None)
    trainer_state = next((path for path in outputs if path.name == "trainer_state.json"), None)
    return run_metadata, trainer_state


def _checkpoint_info(
    config: DDOConfig,
    run_metadata: Path | None,
    trainer_state: Path | None,
) -> dict[str, Path | str | None]:
    run_dir = _training_run_dir(run_metadata, trainer_state)
    if config.training.validate_only:
        return {
            "run_dir": run_dir,
            "checkpoint_dir": None,
            "adapter_path": None,
            "adapter_loading_status": "validate_only",
        }
    if run_dir is None or not run_dir.exists():
        return {
            "run_dir": run_dir,
            "checkpoint_dir": None,
            "adapter_path": None,
            "adapter_loading_status": "missing",
        }
    checkpoint_dir = _latest_checkpoint_dir(run_dir)
    if checkpoint_dir is not None:
        return {
            "run_dir": run_dir,
            "checkpoint_dir": checkpoint_dir,
            "adapter_path": checkpoint_dir,
            "adapter_loading_status": "checkpoint_found",
        }
    if (run_dir / "adapter_config.json").exists():
        return {
            "run_dir": run_dir,
            "checkpoint_dir": None,
            "adapter_path": run_dir,
            "adapter_loading_status": "run_dir_only",
        }
    return {
        "run_dir": run_dir,
        "checkpoint_dir": None,
        "adapter_path": None,
        "adapter_loading_status": "missing_adapter_weights",
    }


def _checkpoint_id(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    checkpoint_info: dict[str, Path | str | None],
) -> str:
    task = _safe_slug(_training_scope(config, benchmark))
    model = _safe_slug(config.models.target_model or "model")
    return "__".join(
        [
            benchmark.name,
            task,
            config.collection.method,
            config.dataset.method,
            config.training.method,
            model,
            str(config.runtime.seed),
        ]
    )


def _training_run_dir(run_metadata: Path | None, trainer_state: Path | None) -> Path | None:
    for path in (run_metadata, trainer_state):
        if path is not None:
            return path.parent
    return None


def _latest_checkpoint_dir(run_dir: Path) -> Path | None:
    checkpoints = [
        path
        for path in run_dir.glob("checkpoint-*")
        if path.is_dir() and (path / "adapter_config.json").exists()
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_sort_key)


def _relative_path(root: Path, path: Path | str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return str(candidate)


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.rsplit("-", 1)[-1]
    try:
        return int(suffix), path.name
    except ValueError:
        return -1, path.name


def _safe_slug(value: str) -> str:
    normalized = value.rsplit("/", 1)[-1]
    return "".join(char.lower() if char.isalnum() else "_" for char in normalized).strip("_") or "value"


def _prompt_from_trace(
    output_dir: Path,
    base_id: str,
    divergence_step: int,
    fallback: str,
) -> str:
    trace = _resolve_optional_trace(output_dir, None, base_id)
    if trace is None:
        return fallback
    try:
        payload = read_json(trace)
    except (OSError, ValueError):
        return fallback
    calls = payload.get("calls")
    if not isinstance(calls, list) or divergence_step >= len(calls):
        return fallback
    call = calls[divergence_step]
    if not isinstance(call, dict):
        return fallback
    instruction = str(call.get("instruction") or "").strip()
    observation = str(call.get("observation") or "").strip()
    return "\n\n".join(part for part in (instruction, observation) if part) or fallback


def _branch_details_from_trace(trace: Path | None, divergence_step: int) -> tuple[str, str, bool, float | None]:
    if trace is None:
        return "", "", False, None
    try:
        payload = read_json(trace)
    except (OSError, ValueError):
        return "", "", False, None
    calls = payload.get("calls")
    if not isinstance(calls, list) or not calls:
        return "", "", False, None
    call_index = min(max(divergence_step, 0), len(calls) - 1)
    call = calls[call_index] if isinstance(calls[call_index], dict) else {}
    terminal = calls[-1] if isinstance(calls[-1], dict) else {}
    action = str(call.get("action") or "").strip()
    response_text = str(call.get("raw_output") or "").strip() or f"Action: {action}"
    score = _optional_float(terminal.get("progression") or terminal.get("reward"))
    won = bool(payload.get("won", score is not None and score >= 1.0))
    return action, response_text, won, score


def _resolve_optional_trace(output_dir: Path, raw_path: Any, trajectory_id: Any) -> Path | None:
    if raw_path:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = output_dir / path
        if path.exists():
            return path
    if trajectory_id:
        matches = sorted(output_dir.glob(f"**/{trajectory_id}_llm_trace.json"))
        if matches:
            return matches[0]
    return None


def _completion_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("action_text") or value.get("content") or value.get("text") or "").strip()
    return _text(value)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    if value is None:
        return ""
    return str(value)


def _target_prob(row: dict[str, Any], pair_type: str) -> float:
    explicit = _optional_float(row.get("chosen_target_prob") or row.get("target_prob"))
    if explicit is not None:
        return explicit
    if pair_type == "win_win":
        return 0.5
    return 1.0


def _metric_value(payload: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = _optional_float(payload.get(key))
        if value is not None:
            return value
    nested_metrics = payload.get("metrics")
    if isinstance(nested_metrics, dict):
        for key in keys:
            value = nested_metrics.get(key)
            if isinstance(value, dict):
                numeric = _optional_float(value.get("mean") or value.get("value"))
            else:
                numeric = _optional_float(value)
            if numeric is not None:
                return numeric
    success_count = _optional_float(payload.get("success_count") or payload.get("num_success"))
    total_count = _optional_float(payload.get("episode_count") or payload.get("num_episodes") or payload.get("total"))
    if success_count is not None and total_count:
        return success_count / total_count
    return None


def _percent_metric_value(payload: dict[str, Any], keys: list[str]) -> float | None:
    value = _metric_value(payload, keys)
    if value is None:
        return None
    if value > 1.0:
        return value / 100.0
    return value


def _success_from_counts(payload: dict[str, Any]) -> float | None:
    success_count = _first_metric_value(
        _optional_float(payload.get("success_count")),
        _optional_float(payload.get("num_success")),
    )
    total_count = _first_metric_value(
        _optional_float(payload.get("episode_count")),
        _optional_float(payload.get("num_episodes")),
        _optional_float(payload.get("total")),
    )
    if success_count is None or not total_count:
        return None
    return success_count / total_count


def _first_metric_value(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _metric_payload(value: float | None, *, std: float | None = None) -> dict[str, Any]:
    if value is None:
        return {"status": "missing", "mean": None, "std": None, "per_seed": []}
    return {"status": "ok", "mean": value, "std": std, "per_seed": []}


def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows = [
        {
            "suite": suite,
            "status": values.get("status"),
            "mean": values.get("mean"),
            "std": values.get("std"),
        }
        for suite, values in metrics["metrics"].items()
    ]
    _write_csv(path, ["suite", "status", "mean", "std"], rows)



def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def _training_scope(config: DDOConfig, benchmark: BenchmarkAdapter) -> str:
    if config.benchmark.train_task_filters:
        return config.benchmark.task_group or "all"
    return benchmark.normalize_task_id(config.benchmark.task_filter) or "all"


def _task_slug(task_filter: str | None) -> str:
    if not task_filter:
        return "all"
    return str(task_filter).rsplit("/", 1)[-1].replace("-", "_")


def _as_int(value: Any, default: int) -> int:
    try:
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


def _normalization_artifact_type(backend: str) -> str:
    return f"{backend}_normalization_summary"


def _skipped(reason: str, *, backend: str, **metadata: Any) -> dict[str, Any]:
    return {
        "artifact_type": _normalization_artifact_type(backend),
        "backend": backend,
        "status": "skipped",
        "reason": reason,
        **metadata,
    }
