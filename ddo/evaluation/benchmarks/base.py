"""Shared benchmark adapter interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ddo.config.schema import DDOConfig


@dataclass(frozen=True)
class BenchmarkPaths:
    upstream_path: Path
    base_trajectories: Path
    dtc: Path
    preference_dataset: Path
    reference_checkpoint: Path
    output_checkpoint: Path
    work_dir: Path
    results_dir: Path


class BenchmarkAdapter:
    """Base class for project-specific benchmark integration."""

    name: str
    upstream_dir_name: str
    default_task_filter: str | None = None

    def resolve_paths(self, config: DDOConfig) -> BenchmarkPaths:
        project_root = config.project_root or Path.cwd()
        upstream = config.benchmark.upstream_path
        upstream_path = (
            Path(upstream).expanduser().resolve()
            if upstream
            else project_root / "benchmarks" / self.upstream_dir_name
        )
        def repository_path(value: str) -> Path:
            path = Path(value).expanduser()
            return path.resolve() if path.is_absolute() else project_root / path

        return BenchmarkPaths(
            upstream_path=upstream_path,
            base_trajectories=repository_path(config.paths.base_trajectories),
            dtc=repository_path(config.paths.dtc),
            preference_dataset=repository_path(config.paths.preference_dataset),
            reference_checkpoint=repository_path(config.paths.reference_checkpoint),
            output_checkpoint=repository_path(config.paths.output_checkpoint),
            work_dir=repository_path(config.paths.work_dir),
            results_dir=repository_path(config.paths.results_dir),
        )

    def resolve_base_run_dir(self, config: DDOConfig) -> Path | None:
        """Resolve the project-owned expert trajectory directory."""

        if config.benchmark.base_run_dir is None:
            return None
        path = Path(config.benchmark.base_run_dir).expanduser()
        if path.is_absolute():
            return path
        return (config.project_root or Path.cwd()) / path

    def normalize_task_id(self, task_id: str | None) -> str | None:
        return task_id or self.default_task_filter

    def dry_run_notes(self, config: DDOConfig) -> list[str]:
        paths = self.resolve_paths(config)
        notes = [f"upstream checkout: {paths.upstream_path}"]
        if not paths.upstream_path.exists():
            notes.append("upstream checkout is missing; clone the official benchmark before real execution")
        return notes
