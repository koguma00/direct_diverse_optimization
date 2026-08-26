"""Shared dataset builder planning helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ddo.evaluation.benchmarks.base import BenchmarkAdapter
from ddo.config.schema import DDOConfig
from ddo.pipeline.runner import StagePlan


@dataclass(frozen=True)
class PairDatasetPlanner:
    name: str
    label: str
    training_method: str

    def plan(self, config: DDOConfig, benchmark: BenchmarkAdapter) -> StagePlan:
        paths = benchmark.resolve_paths(config)
        collection_dir = paths.dtc
        dataset_dir = paths.preference_dataset
        return StagePlan(
            stage="build",
            summary=f"Build {self.label} dataset from {benchmark.name} {config.collection.method} artifacts",
            inputs=[
                str(collection_dir),
                f"reference_model={config.models.reference_model or config.models.target_model}",
            ],
            outputs=[str(dataset_dir)],
            commands=[
                "ddo stage build "
                f"--benchmark {benchmark.name} "
                f"--collection-method {config.collection.method} "
                f"--dataset-method {self.name} "
                f"--training-method {config.training.method}"
            ],
            metadata={
                "criterion": config.dataset.criterion,
                "alpha": config.dataset.alpha,
                "target_beta": config.dataset.target_beta,
                "action_freq_scope": config.dataset.action_freq_scope,
                "tie_target": config.dataset.tie_target,
            },
        )
