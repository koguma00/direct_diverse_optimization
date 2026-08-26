"""Typed configuration objects for the DDO pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a config file is missing required fields or is inconsistent."""


@dataclass(frozen=True)
class ExperimentConfig:
    kind: str
    name: str = "unnamed_experiment"
    start_from: str = "scratch"


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    task_filter: str | None = None
    task_group: str | None = None
    train_task_filters: list[str] = field(default_factory=list)
    base_run_dir: str | None = None
    upstream_path: str | None = None


@dataclass(frozen=True)
class CollectionConfig:
    method: str
    divergence_count: int = 5
    alt_budget: int = 3
    alt_mode: str = "random"
    step_sampling_mode: str = "uniform"
    success_only: bool = True
    rollout_max_steps: int | None = None
    rollout_extra_steps: int = 10
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 8192
    max_text_history: int = 16
    client_timeout: float = 300.0
    client_max_retries: int = 1
    client_delay: float = 1.0
    external_retry_attempts: int = 1


@dataclass(frozen=True)
class BaseConfig:
    source_dir: str | None = None
    source_dirs: list[str] = field(default_factory=list)
    num_episodes: int = 20
    max_steps_per_episode: int | None = None
    seed: int = 42000
    seed_mode: str = "per_episode"
    agent_type: str = "thought_action"
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 8192
    max_text_history: int = 16
    client_timeout: float = 300.0
    client_max_retries: int = 1
    client_delay: float = 1.0


@dataclass(frozen=True)
class DatasetConfig:
    method: str
    criterion: str = "terminal_progression"
    usable_trajectories: str = "base-alt"
    quality_score: str = "max_same_obs_action_run"
    quality_threshold: int | None = 1
    max_families: int = 0
    include_base_alt_win_pseudo_pairs: bool = False
    alpha: float = 0.5
    target_beta: float = 0.1
    ref_device: str | None = None
    reference_response_scope: str = "action_only"
    reference_normalize: str = "sum"
    action_freq_scope: str = "state"
    probability_response_scope: str = "action_only"
    probability_normalize: str = "mean"
    tie_target: float = 0.5
    success_threshold: float = 0.9


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    finetune_type: str = "lora"
    beta: float = 0.1
    dpo_rk_alpha: float = math.log(3.0)
    dpo_d_nu: float = 1.0
    ddo_objective: str = "reference_relative"
    lambda_div: float = 1.0
    lambda_floor: float = 0.0
    train_adapter_path: str | None = None
    ref_adapter_path: str | None = None
    learning_rate: float = 5e-6
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_prompt_length: int = 2048
    max_length: int = 2304
    logging_steps: int = 10
    save_steps: int = 200
    save_epochs_fraction: float | None = None
    stop_after_epochs: float | None = None
    save_total_limit: int | None = 8
    dataloader_num_workers: int = 8
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    torch_dtype: str = "auto"
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str | None = None
    seed: int | None = None
    resume_from_checkpoint: str | None = None
    validate_only: bool = False


@dataclass(frozen=True)
class SFTConfig:
    method: str = "expert_success"
    source_shuffle_seed: str | None = None
    finetune_type: str = "lora"
    learning_rate: float = 5e-6
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 2304
    logging_steps: int = 10
    save_steps: int = 200
    save_epochs_fraction: float | None = None
    stop_after_epochs: float | None = None
    save_total_limit: int | None = 8
    dataloader_num_workers: int = 8
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    torch_dtype: str = "auto"
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str | None = None
    seed: int | None = None
    train_adapter_path: str | None = None
    resume_from_checkpoint: str | None = None
    validate_only: bool = False


@dataclass(frozen=True)
class EvaluationConfig:
    protocol: str = "standard"
    suites: list[str] = field(default_factory=lambda: ["success", "coverage"])
    rollout_budget: int = 20
    performance_rollouts: int = 50
    diversity_rollouts: int = 20
    max_steps_per_episode: int = 100
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 8192
    max_text_history: int = 16
    seeds: list[int] = field(
        default_factory=lambda: [20260305, 20260306, 20260307]
    )
    webshop_diversity_sessions: list[int] = field(default_factory=lambda: [500, 501, 502])
    webshop_performance_session_start: int = 500
    webshop_performance_num_workers: int = 2
    webshop_diversity_num_workers: int = 1
    webshop_num_products: int = 0
    webshop_human_goals: bool = True
    webshop_max_search_queries: int = 6
    webshop_max_tokens: int = 256
    webshop_max_text_history: int = 8
    webshop_client_timeout: int = 900
    webshop_client_max_retries: int = 1
    webshop_client_delay: float = 2.0
    webshop_disable_thinking: bool = True
    webshop_success_threshold: float = 0.9
    webshop_trajectory_class: str = "purchase_action_type_trace"
    llm_seed_base: int | None = None
    webshop_diversity_llm_seed_base: int | None = None
    adapter_path: str | None = None
    server_mode: str = "managed"
    server_host: str = "127.0.0.1"
    server_port: int = 18920
    server_ready_timeout: int = 900
    server_poll_interval: float = 5.0
    server_max_model_len: int = 32768
    server_max_num_seqs: int = 8
    server_max_num_batched_tokens: int = 4096
    server_gpu_memory_utilization: float = 0.85
    server_reasoning_parser: str | None = "qwen3"
    server_dtype: str = "bfloat16"
    server_tensor_parallel_size: int = 1
    server_enforce_eager: bool = False
    client_timeout: int = 60
    client_max_retries: int = 1
    client_delay: float = 1.0



@dataclass(frozen=True)
class ModelConfig:
    profile: str | None = None
    expert_model: str | None = None
    target_model: str | None = None
    reference_model: str | None = None
    expert_base_url: str | None = None
    target_base_url: str | None = None


@dataclass(frozen=True)
class PathsConfig:
    base_trajectories: str = "trajectories/base"
    dtc: str = "trajectories/dtc"
    preference_dataset: str = "datasets/preference"
    reference_checkpoint: str = "checkpoints/reference"
    output_checkpoint: str = "checkpoints/output"
    work_dir: str = ".runs/train"
    results_dir: str = "results/eval"


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 20260305
    num_workers: int = 8
    dry_run: bool = False
    cuda_visible_devices: str | None = None


@dataclass(frozen=True)
class DDOConfig:
    experiment: ExperimentConfig
    benchmark: BenchmarkConfig
    base: BaseConfig
    collection: CollectionConfig
    dataset: DatasetConfig
    training: TrainingConfig
    sft: SFTConfig = field(default_factory=SFTConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    project_root: Path | None = None
    source_path: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        project_root: Path | None = None,
        source_path: Path | None = None,
    ) -> "DDOConfig":
        _require_section(raw, "experiment")
        _require_section(raw, "benchmark")
        _require_section(raw, "collection")
        _require_section(raw, "dataset")
        _require_section(raw, "training")

        config = cls(
            experiment=_build_dataclass(ExperimentConfig, raw["experiment"], "experiment"),
            benchmark=_build_dataclass(BenchmarkConfig, raw["benchmark"], "benchmark"),
            base=_build_dataclass(BaseConfig, raw.get("base", {}), "base"),
            collection=_build_dataclass(CollectionConfig, raw["collection"], "collection"),
            dataset=_build_dataclass(DatasetConfig, raw["dataset"], "dataset"),
            training=_build_dataclass(TrainingConfig, raw["training"], "training"),
            sft=_build_dataclass(SFTConfig, raw.get("sft", {}), "sft"),
            evaluation=_build_dataclass(EvaluationConfig, raw.get("evaluation", {}), "evaluation"),
            models=_build_dataclass(ModelConfig, raw.get("models", {}), "models"),
            paths=_build_dataclass(PathsConfig, raw.get("paths", {}), "paths"),
            runtime=_build_dataclass(RuntimeConfig, raw.get("runtime", {}), "runtime"),
            project_root=project_root,
            source_path=source_path,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.experiment.start_from not in {"scratch", "dtc", "dataset"}:
            raise ConfigError("experiment.start_from must be scratch, dtc, or dataset")
        if not self.models.target_model:
            raise ConfigError("models.target_model is required")
        if not self.models.expert_model:
            raise ConfigError("models.expert_model is required")
        if self.evaluation.server_mode == "external" and not self.models.target_base_url:
            raise ConfigError("external evaluation requires models.target_base_url")
        if self.collection.divergence_count <= 0:
            raise ConfigError("collection.divergence_count must be positive")
        if self.collection.alt_budget <= 0:
            raise ConfigError("collection.alt_budget must be positive")
        if self.collection.rollout_max_steps is not None and self.collection.rollout_max_steps <= 0:
            raise ConfigError("collection.rollout_max_steps must be positive when set")
        if self.collection.rollout_extra_steps < 0:
            raise ConfigError("collection.rollout_extra_steps must be non-negative")
        if self.collection.temperature < 0:
            raise ConfigError("collection.temperature must be non-negative")
        if not 0 < self.collection.top_p <= 1:
            raise ConfigError("collection.top_p must be in (0, 1]")
        if self.collection.max_tokens <= 0 or self.collection.max_text_history <= 0:
            raise ConfigError("collection.max_tokens and max_text_history must be positive")
        if self.collection.client_timeout <= 0:
            raise ConfigError("collection.client_timeout must be positive")
        if self.collection.client_max_retries < 0 or self.collection.external_retry_attempts < 0:
            raise ConfigError("collection retry counts must be non-negative")
        if self.collection.client_delay < 0:
            raise ConfigError("collection.client_delay must be non-negative")
        if self.collection.step_sampling_mode not in {"uniform", "random"}:
            raise ConfigError("collection.step_sampling_mode must be 'uniform' or 'random'")
        allowed_alt_modes = {"random", "request", "random_teacher"}
        if self.benchmark.name == "webshop":
            allowed_alt_modes.add("structured")
        if self.collection.alt_mode not in allowed_alt_modes:
            raise ConfigError(
                f"collection.alt_mode for {self.benchmark.name} must be one of: "
                + ", ".join(sorted(allowed_alt_modes))
            )
        if self.base.num_episodes <= 0:
            raise ConfigError("base.num_episodes must be positive")
        if not isinstance(self.base.source_dirs, list) or not all(
            isinstance(source, str) for source in self.base.source_dirs
        ):
            raise ConfigError("base.source_dirs must be a list of paths")
        if self.base.source_dir and self.base.source_dirs:
            raise ConfigError("base.source_dir and base.source_dirs are mutually exclusive")
        if any(not source.strip() for source in self.base.source_dirs):
            raise ConfigError("base.source_dirs must not contain empty paths")
        if self.benchmark.train_task_filters:
            if not isinstance(self.benchmark.train_task_filters, list) or not all(
                isinstance(task, str) and task.strip()
                for task in self.benchmark.train_task_filters
            ):
                raise ConfigError("benchmark.train_task_filters must be a list of task ids")
            if not self.benchmark.task_group:
                raise ConfigError("benchmark.task_group is required with train_task_filters")
            if len(set(self.benchmark.train_task_filters)) != len(
                self.benchmark.train_task_filters
            ):
                raise ConfigError("benchmark.train_task_filters must not contain duplicates")
            if self.benchmark.task_filter not in self.benchmark.train_task_filters:
                raise ConfigError(
                    "benchmark.task_filter must be one of benchmark.train_task_filters"
                )
        if self.base.max_steps_per_episode is not None and self.base.max_steps_per_episode <= 0:
            raise ConfigError("base.max_steps_per_episode must be positive when set")
        if not isinstance(self.base.seed, int) or isinstance(self.base.seed, bool) or self.base.seed < 0:
            raise ConfigError("base.seed must be a non-negative integer")
        if self.base.seed_mode not in {"fixed", "per_episode"}:
            raise ConfigError("base.seed_mode must be 'fixed' or 'per_episode'")
        if self.base.agent_type != "thought_action":
            raise ConfigError("base.agent_type must be 'thought_action'")
        if self.base.temperature < 0:
            raise ConfigError("base.temperature must be non-negative")
        if not 0 < self.base.top_p <= 1:
            raise ConfigError("base.top_p must be in (0, 1]")
        if self.base.max_tokens <= 0:
            raise ConfigError("base.max_tokens must be positive")
        if self.base.max_text_history <= 0:
            raise ConfigError("base.max_text_history must be positive")
        if self.base.client_timeout <= 0:
            raise ConfigError("base.client_timeout must be positive")
        if self.base.client_max_retries < 0:
            raise ConfigError("base.client_max_retries must be non-negative")
        if self.base.client_delay < 0:
            raise ConfigError("base.client_delay must be non-negative")
        if self.dataset.criterion not in {
            "terminal_progression",
            "textworld_task_progression_v1",
            "webshop_reward_threshold",
        }:
            raise ConfigError("dataset.criterion is not supported")
        if self.dataset.quality_score != "max_same_obs_action_run":
            raise ConfigError("dataset.quality_score must be 'max_same_obs_action_run'")
        if not 0.0 <= self.dataset.alpha <= 1.0:
            raise ConfigError("dataset.alpha must be in [0, 1]")
        if self.dataset.target_beta <= 0:
            raise ConfigError("dataset.target_beta must be positive")
        allowed_trajectory_modes = {"base-alt", "base-alt-alt"}
        if self.dataset.usable_trajectories not in allowed_trajectory_modes:
            allowed = ", ".join(sorted(allowed_trajectory_modes))
            raise ConfigError(
                f"dataset.usable_trajectories for {self.dataset.method} must be one of: {allowed}"
            )
        if self.dataset.quality_threshold is not None and self.dataset.quality_threshold < 0:
            raise ConfigError("dataset.quality_threshold must be non-negative when set")
        if self.dataset.max_families < 0:
            raise ConfigError("dataset.max_families must be non-negative")
        if self.dataset.ref_device not in {None, "auto", "cpu", "cuda"}:
            raise ConfigError("dataset.ref_device must be null, 'auto', 'cpu', or 'cuda'")
        if self.dataset.reference_response_scope not in {"full", "action_only"}:
            raise ConfigError("dataset.reference_response_scope must be 'full' or 'action_only'")
        if self.dataset.reference_normalize not in {"mean", "sum"}:
            raise ConfigError("dataset.reference_normalize must be 'mean' or 'sum'")
        if self.dataset.action_freq_scope not in {"state", "task"}:
            raise ConfigError("dataset.action_freq_scope must be 'state' or 'task'")
        if self.dataset.probability_response_scope not in {"full", "action_only"}:
            raise ConfigError("dataset.probability_response_scope must be 'full' or 'action_only'")
        if self.dataset.probability_normalize not in {"mean", "sum"}:
            raise ConfigError("dataset.probability_normalize must be 'mean' or 'sum'")
        if not 0.0 <= self.dataset.tie_target <= 1.0:
            raise ConfigError("dataset.tie_target must be in [0, 1]")
        if not 0.0 <= self.dataset.success_threshold <= 1.0:
            raise ConfigError("dataset.success_threshold must be in [0, 1]")
        if self.training.finetune_type not in {"full", "lora"}:
            raise ConfigError("training.finetune_type must be 'full' or 'lora'")
        if self.training.num_train_epochs <= 0:
            raise ConfigError("training.num_train_epochs must be positive")
        if self.training.warmup_ratio < 0:
            raise ConfigError("training.warmup_ratio must be non-negative")
        if not 0.0 <= self.training.adam_beta1 < 1.0:
            raise ConfigError("training.adam_beta1 must be in [0, 1)")
        if not 0.0 <= self.training.adam_beta2 < 1.0:
            raise ConfigError("training.adam_beta2 must be in [0, 1)")
        if self.training.adam_epsilon <= 0:
            raise ConfigError("training.adam_epsilon must be positive")
        if self.training.weight_decay < 0:
            raise ConfigError("training.weight_decay must be non-negative")
        if self.training.max_grad_norm <= 0:
            raise ConfigError("training.max_grad_norm must be positive")
        if self.training.lr_scheduler_type != "linear":
            raise ConfigError("training.lr_scheduler_type must be linear")
        if self.training.per_device_train_batch_size <= 0:
            raise ConfigError("training.per_device_train_batch_size must be positive")
        if self.training.gradient_accumulation_steps <= 0:
            raise ConfigError("training.gradient_accumulation_steps must be positive")
        if self.training.max_prompt_length <= 0:
            raise ConfigError("training.max_prompt_length must be positive")
        if self.training.max_length <= 0:
            raise ConfigError("training.max_length must be positive")
        if self.training.max_length < self.training.max_prompt_length:
            raise ConfigError("training.max_length must be >= training.max_prompt_length")
        if self.training.logging_steps <= 0:
            raise ConfigError("training.logging_steps must be positive")
        if self.training.save_steps <= 0:
            raise ConfigError("training.save_steps must be positive")
        if (
            self.training.save_epochs_fraction is not None
            and self.training.save_epochs_fraction <= 0
        ):
            raise ConfigError("training.save_epochs_fraction must be positive when set")
        if self.training.stop_after_epochs is not None and self.training.stop_after_epochs <= 0:
            raise ConfigError("training.stop_after_epochs must be positive when set")
        if self.training.save_total_limit is not None and self.training.save_total_limit <= 0:
            raise ConfigError("training.save_total_limit must be positive when set")
        if self.training.dataloader_num_workers < 0:
            raise ConfigError("training.dataloader_num_workers must be non-negative")
        if self.training.bf16 and self.training.fp16:
            raise ConfigError("training.bf16 and training.fp16 cannot both be true")
        if self.training.torch_dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ConfigError(
                "training.torch_dtype must be 'auto', 'bfloat16', 'float16', or 'float32'"
            )
        if self.training.lora_r <= 0:
            raise ConfigError("training.lora_r must be positive")
        if self.training.lora_alpha <= 0:
            raise ConfigError("training.lora_alpha must be positive")
        if self.training.lora_dropout < 0:
            raise ConfigError("training.lora_dropout must be non-negative")
        if self.training.dpo_rk_alpha <= 0:
            raise ConfigError("training.dpo_rk_alpha must be positive")
        if self.training.dpo_d_nu <= 0:
            raise ConfigError("training.dpo_d_nu must be positive")
        if self.training.ddo_objective not in {"reference_relative", "squared_gap", "upward_floor"}:
            raise ConfigError("training.ddo_objective must be reference_relative, squared_gap, or upward_floor")
        if self.training.lambda_div < 0:
            raise ConfigError("training.lambda_div must be non-negative")
        if self.training.lambda_floor < 0:
            raise ConfigError("training.lambda_floor must be non-negative")
        if self.sft.finetune_type != "lora":
            raise ConfigError("sft.finetune_type must be 'lora'")
        if self.sft.source_shuffle_seed is not None and (
            not isinstance(self.sft.source_shuffle_seed, str)
            or not self.sft.source_shuffle_seed.strip()
        ):
            raise ConfigError("sft.source_shuffle_seed must be a non-empty string when set")
        if self.sft.num_train_epochs <= 0:
            raise ConfigError("sft.num_train_epochs must be positive")
        if self.sft.warmup_ratio < 0:
            raise ConfigError("sft.warmup_ratio must be non-negative")
        if self.sft.per_device_train_batch_size <= 0:
            raise ConfigError("sft.per_device_train_batch_size must be positive")
        if self.sft.gradient_accumulation_steps <= 0:
            raise ConfigError("sft.gradient_accumulation_steps must be positive")
        if self.sft.max_length <= 0:
            raise ConfigError("sft.max_length must be positive")
        if self.sft.logging_steps <= 0 or self.sft.save_steps <= 0:
            raise ConfigError("sft.logging_steps and sft.save_steps must be positive")
        if self.sft.save_epochs_fraction is not None and self.sft.save_epochs_fraction <= 0:
            raise ConfigError("sft.save_epochs_fraction must be positive when set")
        if self.sft.stop_after_epochs is not None and self.sft.stop_after_epochs <= 0:
            raise ConfigError("sft.stop_after_epochs must be positive when set")
        if self.sft.save_total_limit is not None and self.sft.save_total_limit <= 0:
            raise ConfigError("sft.save_total_limit must be positive when set")
        if self.sft.dataloader_num_workers < 0:
            raise ConfigError("sft.dataloader_num_workers must be non-negative")
        if self.sft.bf16 and self.sft.fp16:
            raise ConfigError("sft.bf16 and sft.fp16 cannot both be true")
        if self.sft.torch_dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ConfigError("sft.torch_dtype must be 'auto', 'bfloat16', 'float16', or 'float32'")
        if self.sft.lora_r <= 0 or self.sft.lora_alpha <= 0:
            raise ConfigError("sft.lora_r and sft.lora_alpha must be positive")
        if self.sft.lora_dropout < 0:
            raise ConfigError("sft.lora_dropout must be non-negative")
        if self.evaluation.rollout_budget <= 0:
            raise ConfigError("evaluation.rollout_budget must be positive")
        if self.evaluation.protocol != "standard":
            raise ConfigError("evaluation.protocol must be standard")
        if self.evaluation.performance_rollouts <= 0:
            raise ConfigError("evaluation.performance_rollouts must be positive")
        if self.evaluation.diversity_rollouts <= 0:
            raise ConfigError("evaluation.diversity_rollouts must be positive")
        if self.evaluation.max_steps_per_episode <= 0:
            raise ConfigError("evaluation.max_steps_per_episode must be positive")
        if self.evaluation.temperature < 0:
            raise ConfigError("evaluation.temperature must be non-negative")
        if not 0 < self.evaluation.top_p <= 1:
            raise ConfigError("evaluation.top_p must be in (0, 1]")
        if self.evaluation.max_tokens <= 0:
            raise ConfigError("evaluation.max_tokens must be positive")
        if self.evaluation.max_text_history <= 0:
            raise ConfigError("evaluation.max_text_history must be positive")
        if not self.evaluation.seeds or not all(
            isinstance(seed, int) and not isinstance(seed, bool) for seed in self.evaluation.seeds
        ):
            raise ConfigError("evaluation.seeds must be a non-empty integer list")
        if len(self.evaluation.seeds) < 3:
            raise ConfigError("evaluation.seeds must contain at least three seeds")
        if len(set(self.evaluation.seeds)) != len(self.evaluation.seeds):
            raise ConfigError("evaluation.seeds must not contain duplicates")
        if not self.evaluation.webshop_diversity_sessions or not all(
            isinstance(session, int) and not isinstance(session, bool)
            for session in self.evaluation.webshop_diversity_sessions
        ):
            raise ConfigError(
                "evaluation.webshop_diversity_sessions must be a non-empty integer list"
            )
        if (
            not isinstance(self.evaluation.webshop_performance_session_start, int)
            or isinstance(self.evaluation.webshop_performance_session_start, bool)
            or self.evaluation.webshop_performance_session_start < 0
        ):
            raise ConfigError(
                "evaluation.webshop_performance_session_start must be a non-negative integer"
            )
        if self.evaluation.webshop_performance_num_workers <= 0:
            raise ConfigError("evaluation.webshop_performance_num_workers must be positive")
        if self.evaluation.webshop_diversity_num_workers <= 0:
            raise ConfigError("evaluation.webshop_diversity_num_workers must be positive")
        if self.evaluation.webshop_num_products < 0:
            raise ConfigError("evaluation.webshop_num_products must be non-negative")
        if not isinstance(self.evaluation.webshop_human_goals, bool):
            raise ConfigError("evaluation.webshop_human_goals must be boolean")
        if self.evaluation.webshop_max_search_queries <= 0:
            raise ConfigError("evaluation.webshop_max_search_queries must be positive")
        if self.evaluation.webshop_max_tokens <= 0:
            raise ConfigError("evaluation.webshop_max_tokens must be positive")
        if self.evaluation.webshop_max_text_history <= 0:
            raise ConfigError("evaluation.webshop_max_text_history must be positive")
        if (
            self.evaluation.webshop_client_timeout <= 0
            or self.evaluation.webshop_client_max_retries < 0
            or self.evaluation.webshop_client_delay < 0
        ):
            raise ConfigError(
                "evaluation WebShop client timeout must be positive and retries/delay non-negative"
            )
        if not isinstance(self.evaluation.webshop_disable_thinking, bool):
            raise ConfigError("evaluation.webshop_disable_thinking must be boolean")
        if self.evaluation.llm_seed_base is not None and (
            not isinstance(self.evaluation.llm_seed_base, int)
            or isinstance(self.evaluation.llm_seed_base, bool)
            or self.evaluation.llm_seed_base < 0
        ):
            raise ConfigError(
                "evaluation.llm_seed_base must be null or a non-negative integer"
            )
        if self.evaluation.webshop_diversity_llm_seed_base is not None and (
            not isinstance(self.evaluation.webshop_diversity_llm_seed_base, int)
            or isinstance(self.evaluation.webshop_diversity_llm_seed_base, bool)
            or self.evaluation.webshop_diversity_llm_seed_base < 0
        ):
            raise ConfigError(
                "evaluation.webshop_diversity_llm_seed_base must be null or a non-negative integer"
            )
        if not 0.0 <= self.evaluation.webshop_success_threshold <= 1.0:
            raise ConfigError("evaluation.webshop_success_threshold must be in [0, 1]")
        if self.evaluation.webshop_trajectory_class not in {
            "purchase_signature",
            "action_trace",
            "action_type_trace",
            "purchase_action_type_trace",
        }:
            raise ConfigError(
                "evaluation.webshop_trajectory_class must be purchase_signature, "
                "action_trace, action_type_trace, or purchase_action_type_trace"
            )
        if self.evaluation.adapter_path is not None and (
            not isinstance(self.evaluation.adapter_path, str)
            or not self.evaluation.adapter_path.strip()
        ):
            raise ConfigError("evaluation.adapter_path must be a non-empty path when set")
        if self.evaluation.adapter_path is not None:
            if self.evaluation.server_mode != "managed":
                raise ConfigError(
                    "evaluation.adapter_path requires evaluation.server_mode=managed"
                )
            if self.training.method == "base":
                raise ConfigError("base-model evaluation must not declare an adapter path")
            if self.project_root is None:
                raise ConfigError("adapter path validation requires a project root")
            from ddo.checkpoints import (
                CheckpointAdapterError,
                resolve_paper_adapter,
            )

            try:
                resolve_paper_adapter(
                    self.evaluation.adapter_path,
                    project_root=self.project_root,
                    verify_files=False,
                )
            except CheckpointAdapterError as exc:
                raise ConfigError(str(exc)) from exc
        if self.evaluation.server_mode not in {"managed", "external"}:
            raise ConfigError("evaluation.server_mode must be 'managed' or 'external'")
        if not 0 < self.evaluation.server_port <= 65535:
            raise ConfigError("evaluation.server_port must be between 1 and 65535")
        if self.evaluation.server_ready_timeout <= 0 or self.evaluation.server_poll_interval <= 0:
            raise ConfigError("evaluation server timeout and poll interval must be positive")
        if self.evaluation.server_max_model_len <= 0:
            raise ConfigError("evaluation.server_max_model_len must be positive")
        if self.evaluation.server_max_num_seqs <= 0:
            raise ConfigError("evaluation.server_max_num_seqs must be positive")
        if self.evaluation.server_max_num_batched_tokens <= 0:
            raise ConfigError("evaluation.server_max_num_batched_tokens must be positive")
        if not 0 < self.evaluation.server_gpu_memory_utilization <= 1:
            raise ConfigError("evaluation.server_gpu_memory_utilization must be in (0, 1]")
        if self.evaluation.server_dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ConfigError(
                "evaluation.server_dtype must be 'auto', 'bfloat16', 'float16', or 'float32'"
            )
        if self.evaluation.server_tensor_parallel_size <= 0:
            raise ConfigError("evaluation.server_tensor_parallel_size must be positive")
        if self.evaluation.client_timeout <= 0 or self.evaluation.client_max_retries < 0:
            raise ConfigError("evaluation client timeout must be positive and retries non-negative")
        if self.evaluation.client_delay < 0:
            raise ConfigError("evaluation.client_delay must be non-negative")
        if self.runtime.num_workers <= 0:
            raise ConfigError("runtime.num_workers must be positive")
        if self.runtime.cuda_visible_devices is not None:
            device_ids = [item.strip() for item in self.runtime.cuda_visible_devices.split(",")]
            if (
                not device_ids
                or any(not item.isdigit() for item in device_ids)
                or len(set(device_ids)) != len(device_ids)
            ):
                raise ConfigError(
                    "runtime.cuda_visible_devices must be a comma-separated list of unique "
                    "non-negative GPU indices"
                )
        if not self.models.target_model:
            raise ConfigError("models.target_model must be set directly or through a model profile")
        if not self.models.expert_model:
            raise ConfigError("models.expert_model must be set directly or through a model profile")


def _require_section(raw: dict[str, Any], name: str) -> None:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing required mapping section: {name}")


def _build_dataclass(cls: type[Any], value: dict[str, Any], section: str) -> Any:
    if not isinstance(value, dict):
        raise ConfigError(f"{section} must be a mapping")
    valid_fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = sorted(set(value) - valid_fields)
    if unknown:
        joined = ", ".join(unknown)
        raise ConfigError(f"unknown {section} field(s): {joined}")
    try:
        return cls(**value)
    except TypeError as exc:
        raise ConfigError(f"invalid {section} section: {exc}") from exc
