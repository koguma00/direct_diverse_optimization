"""Evaluation suite selection."""

from __future__ import annotations

from ddo.evaluation.benchmarks.base import BenchmarkAdapter
from ddo.config.schema import DDOConfig
from ddo.pipeline.runner import StagePlan


SUPPORTED_EVAL_SUITES = {
    "success": "Task success",
    "esd": "Episode-level strategy diversity",
    "h_esd": "History-normalized episode-level strategy diversity",
}


class EvaluationPlanner:
    def plan(self, config: DDOConfig, benchmark: BenchmarkAdapter) -> StagePlan:
        unknown = sorted(set(config.evaluation.suites) - set(SUPPORTED_EVAL_SUITES))
        warnings = benchmark.dry_run_notes(config)
        if unknown:
            warnings.append(f"unsupported evaluation suites requested: {', '.join(unknown)}")

        paths = benchmark.resolve_paths(config)
        checkpoint_dir = paths.output_checkpoint
        checkpoint_input = (
            f"paper_adapter_path={config.evaluation.adapter_path}"
            if config.evaluation.adapter_path
            else str(checkpoint_dir)
        )
        eval_dir = paths.results_dir
        suites = ", ".join(config.evaluation.suites)
        return StagePlan(
            stage="eval",
            summary=(
                f"Evaluate {config.training.method} checkpoint on {benchmark.name} "
                f"with {config.evaluation.protocol} protocol: {suites}"
            ),
            inputs=[
                checkpoint_input,
                f"target_model={config.models.target_model}",
                f"task_filter={benchmark.normalize_task_id(config.benchmark.task_filter)}",
            ],
            outputs=[str(eval_dir)],
            commands=[
                "ddo stage eval "
                f"--benchmark {benchmark.name} "
                f"--collection-method {config.collection.method} "
                f"--dataset-method {config.dataset.method} "
                f"--training-method {config.training.method} "
                f"--target-model {config.models.target_model}"
            ],
            warnings=warnings,
            metadata={
                "suites": config.evaluation.suites,
                "protocol": config.evaluation.protocol,
                "rollout_budget": config.evaluation.rollout_budget,
                "performance_rollouts": config.evaluation.performance_rollouts,
                "diversity_rollouts": config.evaluation.diversity_rollouts,
                "temperature": config.evaluation.temperature,
                "top_p": config.evaluation.top_p,
                "max_tokens": config.evaluation.max_tokens,
                "max_text_history": config.evaluation.max_text_history,
                "seeds": config.evaluation.seeds,
                "server_mode": config.evaluation.server_mode,
                "server_host": config.evaluation.server_host,
                "server_port": config.evaluation.server_port,
                "adapter_path": config.evaluation.adapter_path,
                "supported_suites": sorted(SUPPORTED_EVAL_SUITES),
            },
        )
