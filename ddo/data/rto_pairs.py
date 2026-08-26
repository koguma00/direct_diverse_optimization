"""RTO pair dataset planning."""

from __future__ import annotations

from ddo.evaluation.benchmarks.base import BenchmarkAdapter
from ddo.config.schema import DDOConfig
from ddo.pipeline.runner import StagePlan


class RTOPairBuilder:
    name = "rto_pairs"

    def plan(self, config: DDOConfig, benchmark: BenchmarkAdapter) -> StagePlan:
        paths = benchmark.resolve_paths(config)
        collection_dir = paths.dtc
        dataset_dir = paths.preference_dataset
        return StagePlan(
            stage="build",
            summary=f"Build RTO reference-relative target pairs from {benchmark.name} {config.collection.method} artifacts",
            inputs=[
                str(collection_dir),
                f"reference_model={config.models.reference_model or config.models.target_model}",
            ],
            outputs=[str(dataset_dir)],
            commands=[
                "ddo stage build "
                f"--benchmark {benchmark.name} "
                f"--collection-method {config.collection.method} "
                "--dataset-method rto_pairs "
                f"--training-method {config.training.method} "
                f"--target-model {config.models.target_model}"
            ],
            metadata={
                "criterion": config.dataset.criterion,
                "alpha": config.dataset.alpha,
                "target_beta": config.dataset.target_beta,
            },
        )
