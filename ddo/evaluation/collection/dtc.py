"""Divergence Tree Collection planning."""

from __future__ import annotations

from ddo.evaluation.benchmarks.base import BenchmarkAdapter
from ddo.config.schema import DDOConfig
from ddo.pipeline.runner import StagePlan


class DTCCollector:
    name = "dtc"

    def plan(self, config: DDOConfig, benchmark: BenchmarkAdapter) -> StagePlan:
        paths = benchmark.resolve_paths(config)
        task = (
            config.benchmark.task_group
            if config.benchmark.train_task_filters
            else benchmark.normalize_task_id(config.benchmark.task_filter)
        )
        base_run_dir = benchmark.resolve_base_run_dir(config)
        output_dir = paths.dtc
        command = (
            "ddo stage collect "
            f"--benchmark {benchmark.name} "
            "--collection-method dtc "
            f"--task-filter {task or '<all>'} "
            f"--expert-model {config.models.expert_model} "
            f"--divergence-count {config.collection.divergence_count} "
            f"--alt-budget {config.collection.alt_budget} "
            f"--alt-mode {config.collection.alt_mode} "
            f"--step-sampling-mode {config.collection.step_sampling_mode} "
            f"--num-workers {config.runtime.num_workers}"
        )
        if config.collection.success_only:
            command += " --success-only"
        return StagePlan(
            stage="collect",
            summary=f"Collect DTC branch sets for {benchmark.name}",
            inputs=[
                f"base_run_dir={base_run_dir}",
                f"expert_model={config.models.expert_model}",
                f"task_filter={task}",
            ],
            outputs=[str(output_dir)],
            commands=[command],
            warnings=benchmark.dry_run_notes(config),
            metadata={
                "divergence_count": config.collection.divergence_count,
                "alt_budget": config.collection.alt_budget,
                "alt_mode": config.collection.alt_mode,
                "step_sampling_mode": config.collection.step_sampling_mode,
                "success_only": config.collection.success_only,
                "train_task_filters": config.benchmark.train_task_filters,
            },
        )
