"""Shared LoRA training plan helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ddo.evaluation.benchmarks.base import BenchmarkAdapter
from ddo.config.schema import DDOConfig
from ddo.pipeline.runner import StagePlan


@dataclass(frozen=True)
class LoRATrainingPlanner:
    name: str
    label: str

    def plan(self, config: DDOConfig, benchmark: BenchmarkAdapter) -> StagePlan:
        paths = benchmark.resolve_paths(config)
        dataset_dir = paths.preference_dataset
        sft_dir = paths.reference_checkpoint
        run_dir = paths.work_dir / "train"
        return StagePlan(
            stage="train",
            summary=f"Train LoRA checkpoint with {self.label} on {benchmark.name}",
            inputs=[
                str(dataset_dir),
                str(sft_dir),
                f"target_model={config.models.target_model}",
                f"reference_model={config.models.reference_model or config.models.target_model}",
            ],
            outputs=[str(run_dir)],
            commands=[
                "ddo stage train "
                f"--benchmark {benchmark.name} "
                f"--collection-method {config.collection.method} "
                f"--dataset-method {config.dataset.method} "
                f"--training-method {self.name} "
                f"--target-model {config.models.target_model}"
            ],
            metadata={
                "finetune_type": config.training.finetune_type,
                "beta": config.training.beta,
                "ddo_objective": config.training.ddo_objective,
                "learning_rate": config.training.learning_rate,
                "num_train_epochs": config.training.num_train_epochs,
                "max_steps": config.training.max_steps,
                "warmup_ratio": config.training.warmup_ratio,
                "per_device_train_batch_size": config.training.per_device_train_batch_size,
                "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
                "max_prompt_length": config.training.max_prompt_length,
                "max_length": config.training.max_length,
                "save_steps": config.training.save_steps,
                "save_total_limit": config.training.save_total_limit,
                "bf16": config.training.bf16,
                "fp16": config.training.fp16,
                "gradient_checkpointing": config.training.gradient_checkpointing,
                "torch_dtype": config.training.torch_dtype,
                "lora_r": config.training.lora_r,
                "lora_alpha": config.training.lora_alpha,
                "lora_dropout": config.training.lora_dropout,
                "lora_target_modules": config.training.lora_target_modules,
                "seed": config.training.seed,
                "train_adapter_path": config.training.train_adapter_path,
                "ref_adapter_path": config.training.ref_adapter_path,
                "validate_only": config.training.validate_only,
                "output_checkpoint": str(paths.output_checkpoint),
            },
        )
