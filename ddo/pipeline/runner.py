"""Pipeline orchestration for DDO stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ddo.evaluation.benchmarks.base import BenchmarkAdapter
from ddo.config.schema import DDOConfig
from ddo.pipeline.native import execute_native_stage_plan
from ddo.registries import (
    benchmark_registry,
    collection_registry,
    dataset_registry,
    evaluation_registry,
    register_defaults,
    training_registry,
)

ALL_STAGES = ("base", "sft", "collect", "build", "train", "eval")


@dataclass(frozen=True)
class StagePlan:
    stage: str
    summary: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineRunner:
    def __init__(self, config: DDOConfig) -> None:
        register_defaults()
        self.config = config
        self.benchmark: BenchmarkAdapter = benchmark_registry.get(config.benchmark.name)

    def plan_stage(self, stage: str) -> StagePlan:
        if stage not in ALL_STAGES:
            raise ValueError(f"unknown stage: {stage}. Expected one of: {', '.join(ALL_STAGES)}")
        if stage == "base":
            return self._plan_base_stage()
        if stage == "sft":
            return self._plan_sft_stage()
        if stage == "collect":
            collector = collection_registry.get(self.config.collection.method)
            return collector.plan(self.config, self.benchmark)
        if stage == "build":
            builder = dataset_registry.get(self.config.dataset.method)
            return builder.plan(self.config, self.benchmark)
        if stage == "train":
            trainer = training_registry.get(self.config.training.method)
            return trainer.plan(self.config, self.benchmark)
        if stage == "eval":
            evaluator = evaluation_registry.get("default")
            return evaluator.plan(self.config, self.benchmark)
        raise ValueError(f"unknown stage: {stage}")

    def plan_run(self) -> list[StagePlan]:
        return [self.plan_stage(stage) for stage in ALL_STAGES]

    def execute_stage(self, stage: str) -> StagePlan:
        plan = self.plan_stage(stage)
        if self.config.runtime.dry_run:
            return plan
        return execute_native_stage_plan(self.config, self.benchmark, plan)

    def execute_run(self) -> list[StagePlan]:
        if self.config.runtime.dry_run:
            return self.plan_run()
        return [self.execute_stage(stage) for stage in ALL_STAGES]

    def _plan_base_stage(self) -> StagePlan:
        task = self._training_scope()
        base_run_dir = self.benchmark.resolve_base_run_dir(self.config)
        inputs = [
            f"expert_model={self.config.models.expert_model}",
            f"expert_base_url={self.config.models.expert_base_url}",
            f"task_filter={task}",
        ]
        commands = [
            "ddo stage base "
            f"--benchmark {self.benchmark.name} "
            f"--task-filter {task or '<all>'} "
            f"--expert-model {self.config.models.expert_model}"
        ]
        summary = f"Collect expert base trajectories for {self.benchmark.name}"
        output = str(base_run_dir) if base_run_dir is not None else "<unset benchmark.base_run_dir>"
        return StagePlan(
            stage="base",
            summary=summary,
            inputs=inputs,
            outputs=[output],
            commands=commands,
            warnings=self.benchmark.dry_run_notes(self.config),
            metadata={
                "base_run_dir": self.config.benchmark.base_run_dir,
                "resolved_base_run_dir": str(base_run_dir) if base_run_dir is not None else None,
                "num_episodes": self.config.base.num_episodes,
                "max_steps_per_episode": self.config.base.max_steps_per_episode,
                "seed": self.config.base.seed,
                "seed_mode": self.config.base.seed_mode,
                "agent_type": self.config.base.agent_type,
                "temperature": self.config.base.temperature,
                "top_p": self.config.base.top_p,
                "max_tokens": self.config.base.max_tokens,
                "max_text_history": self.config.base.max_text_history,
                "source_dir": self.config.base.source_dir,
                "source_dirs": self.config.base.source_dirs,
                "task_group": self.config.benchmark.task_group,
                "train_task_filters": self.config.benchmark.train_task_filters,
            },
        )

    def _plan_sft_stage(self) -> StagePlan:
        task = self._training_scope()
        base_run_dir = self.benchmark.resolve_base_run_dir(self.config)
        paths = self.benchmark.resolve_paths(self.config)
        output_dir = paths.work_dir / "sft"
        command = (
            "ddo stage sft "
            f"--benchmark {self.benchmark.name} "
            f"--task-filter {task or '<all>'} "
            f"--target-model {self.config.models.target_model} "
            f"--expert-model {self.config.models.expert_model}"
        )
        return StagePlan(
            stage="sft",
            summary=f"Train SFT reference checkpoint from expert successes for {self.benchmark.name}",
            inputs=[
                f"base_run_dir={base_run_dir}",
                f"target_model={self.config.models.target_model}",
                f"task_filter={task}",
            ],
            outputs=[str(output_dir)],
            commands=[command],
            warnings=self.benchmark.dry_run_notes(self.config),
            metadata={
                "sft_method": self.config.sft.method,
                "finetune_type": self.config.sft.finetune_type,
                "learning_rate": self.config.sft.learning_rate,
                "num_train_epochs": self.config.sft.num_train_epochs,
                "validate_only": self.config.sft.validate_only,
                "reference_checkpoint": str(paths.reference_checkpoint),
                "task_group": self.config.benchmark.task_group,
                "train_task_filters": self.config.benchmark.train_task_filters,
            },
        )

    def _training_scope(self) -> str | None:
        if self.config.benchmark.train_task_filters:
            return self.config.benchmark.task_group
        return self.benchmark.normalize_task_id(self.config.benchmark.task_filter)

def render_stage_plan(plan: StagePlan) -> str:
    lines = [f"[{plan.stage}] {plan.summary}"]
    for label, values in (
        ("inputs", plan.inputs),
        ("outputs", plan.outputs),
        ("commands", plan.commands),
        ("warnings", plan.warnings),
    ):
        if values:
            lines.append(f"{label}:")
            lines.extend(f"  - {value}" for value in values)
    return "\n".join(lines)
