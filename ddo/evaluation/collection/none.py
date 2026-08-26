"""No-op collection stage for base-model and SFT-reference runs."""

from __future__ import annotations

from ddo.evaluation.benchmarks.base import BenchmarkAdapter
from ddo.config.schema import DDOConfig
from ddo.pipeline.runner import StagePlan


class NoOpCollector:
    name = "none"

    def plan(self, config: DDOConfig, benchmark: BenchmarkAdapter) -> StagePlan:
        paths = benchmark.resolve_paths(config)
        output_dir = paths.work_dir / "collection" / self.name
        return StagePlan(
            stage="collect",
            summary=f"Skip branch collection for {benchmark.name}",
            inputs=[str(benchmark.resolve_base_run_dir(config))],
            outputs=[str(output_dir)],
            commands=["ddo stage collect --collection-method none"],
            warnings=benchmark.dry_run_notes(config),
            metadata={"noop_stage": True},
        )
