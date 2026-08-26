"""Public YAML loaders for DDO training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml

from ddo.config.schema import (
    BaseConfig,
    BenchmarkConfig,
    CollectionConfig,
    ConfigError,
    DatasetConfig,
    DDOConfig,
    EvaluationConfig,
    ModelConfig,
    PathsConfig,
    RuntimeConfig,
    SFTConfig,
    TrainingConfig,
)


PAPER_TARGET_MODEL = "Qwen/Qwen3-1.7B"
PAPER_EXPERT_MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"
PAPER_PREFERENCE_EPOCHS = {"babyai": 5.0, "babaisai": 5.0, "webshop": 15.0}

TASK_ALIASES = {
    "babyai": {
        "goto": "BabyAI-MixedTrainLocal-v0/goto",
        "pickup": "BabyAI-MixedTrainLocal-v0/pickup",
        "open": "BabyAI-MixedTrainLocal-v0/open",
        "comp": "BabyAI-MixedTrainLocal-v0/pick_up_seq_go_to",
    },
    "babaisai": {
        "basic": "babaisai/basic",
        "room": "babaisai/room",
        "stop": "babaisai/stop",
        "flex": "babaisai/flex",
    },
    "webshop": {"webshop": "webshop"},
}

METHODS = {
    "base": ("none", "noop_pairs", "base"),
    "reference": ("none", "noop_pairs", "reference"),
    "dpo": ("dtc", "dpo_pairs", "dpo"),
    "divfreq": ("dtc", "divpo_freq_pairs", "divpo_freq"),
    "divprob": ("dtc", "divpo_prob_pairs", "divpo_prob"),
    "tiedpo_rk": ("dtc", "tiedpo_rk_pairs", "tiedpo_rk"),
    "tiedpo_dav": ("dtc", "tiedpo_dav_pairs", "tiedpo_dav"),
    "ddo": ("dtc", "rto_pairs", "ddo"),
}


def load_training_config(path: str | Path, *, dry_run: bool = False) -> DDOConfig:
    source = Path(path).expanduser().resolve()
    raw = _read_yaml(source)
    _validate_top_level(raw, evaluation=False)
    benchmark, task, method = _identity(raw)
    start_from = str(raw.get("start_from", "scratch")).strip().lower()
    if start_from not in {"scratch", "dtc", "dataset"}:
        raise ConfigError("start_from must be scratch, dtc, or dataset")
    if method in {"base", "reference"} and start_from != "scratch":
        raise ConfigError(f"method={method} requires start_from=scratch")

    task_slug = _task_slug(task)
    paths = _training_paths(raw, benchmark, task_slug, method)
    collection_method, dataset_method, training_method = METHODS[method]
    decoding = _shared_decoding(raw)
    lora = _shared_lora(raw)

    base = asdict(BaseConfig())
    base.update(decoding)
    base = _merge_dataclass_options(base, raw, "base", BaseConfig)

    collection = asdict(CollectionConfig(method=collection_method))
    collection.update(decoding)
    collection = _merge_dataclass_options(
        collection, raw, "collection", CollectionConfig, excluded={"method"}
    )
    collection["method"] = collection_method

    dataset = _dataset_config(dataset_method)
    dataset = _merge_dataclass_options(
        dataset, raw, "dataset", DatasetConfig, excluded={"method"}
    )
    if dataset["criterion"] == "auto":
        dataset["criterion"] = (
            "webshop_reward_threshold" if benchmark == "webshop" else "terminal_progression"
        )
    dataset["method"] = dataset_method
    _apply_ddo_aliases(dataset, raw)

    sft = asdict(
        SFTConfig(
            method="expert_success",
            num_train_epochs=10.0,
            bf16=True,
            torch_dtype="bfloat16",
        )
    )
    sft.update(lora)
    sft_section = _normalize_stage_section(raw, "sft", SFTConfig, allow_optimizer=False)
    sft.update(sft_section)

    training = asdict(
        TrainingConfig(
            method=training_method,
            train_adapter_path=paths["reference_checkpoint"],
            ref_adapter_path=paths["reference_checkpoint"],
            num_train_epochs=PAPER_PREFERENCE_EPOCHS[benchmark],
            bf16=True,
            torch_dtype="bfloat16",
        )
    )
    training.update(lora)
    preference = _normalize_stage_section(
        raw, "preference", TrainingConfig, allow_optimizer=True
    )
    training.update(preference)
    training["method"] = training_method

    models = _models(raw, require_expert_url=start_from == "scratch")
    runtime = _runtime(raw, dry_run=dry_run)
    benchmark_mapping = _benchmark_mapping(raw, benchmark, task)
    benchmark_mapping["base_run_dir"] = paths["base_trajectories"]
    experiment_name = str(
        raw.get("experiment_name") or f"{benchmark}_{task_slug}_{method}"
    ).strip()
    if not experiment_name:
        raise ConfigError("experiment_name must be non-empty when set")

    mapping: dict[str, Any] = {
        "experiment": {
            "kind": "paper_training",
            "name": experiment_name,
            "start_from": start_from,
        },
        "benchmark": benchmark_mapping,
        "base": base,
        "collection": collection,
        "dataset": dataset,
        "training": training,
        "sft": sft,
        "evaluation": {},
        "models": models,
        "paths": paths,
        "runtime": runtime,
    }
    return DDOConfig.from_mapping(
        mapping,
        project_root=_project_root(),
        source_path=source,
    )


def load_evaluation_config(path: str | Path, *, dry_run: bool = False) -> DDOConfig:
    source = Path(path).expanduser().resolve()
    raw = _read_yaml(source)
    _validate_top_level(raw, evaluation=True)
    benchmark, task, method = _identity(raw)
    task_slug = _task_slug(task)

    checkpoint_value = raw.get("checkpoint")
    checkpoint = None if checkpoint_value is None else _relative_path(checkpoint_value, "checkpoint")
    if method == "base" and checkpoint is not None:
        raise ConfigError("base evaluation must not set checkpoint")
    if method != "base" and checkpoint is None:
        checkpoint = str(Path("checkpoints") / benchmark / task_slug / method)

    paths = _evaluation_paths(raw, benchmark, task_slug, method, checkpoint)
    models = _models(raw, require_expert_url=False)
    evaluation = asdict(EvaluationConfig())
    evaluation.update(_shared_decoding(raw))
    evaluation_section = _normalize_evaluation_section(raw)
    evaluation.update(evaluation_section)
    if "adapter_path" in evaluation_section:
        configured_adapter = evaluation_section["adapter_path"]
        if checkpoint is not None and configured_adapter != checkpoint:
            raise ConfigError("checkpoint and evaluation.adapter_path must match when both are set")
        checkpoint = configured_adapter
    evaluation["adapter_path"] = checkpoint

    if method == "base" and evaluation["adapter_path"] is not None:
        raise ConfigError("base evaluation must not set evaluation.adapter_path")

    runtime = _runtime(raw, dry_run=dry_run)
    training_method = METHODS[method][2]
    experiment_name = str(
        raw.get("experiment_name") or f"{benchmark}_{task_slug}_{method}"
    ).strip()
    if not experiment_name:
        raise ConfigError("experiment_name must be non-empty when set")

    mapping: dict[str, Any] = {
        "experiment": {"kind": "paper_evaluation", "name": experiment_name},
        "benchmark": _benchmark_mapping(raw, benchmark, task),
        "base": {},
        "collection": {"method": "none"},
        "dataset": {"method": "noop_pairs"},
        "training": {"method": training_method},
        "sft": {},
        "evaluation": evaluation,
        "models": models,
        "paths": paths,
        "runtime": runtime,
    }
    return DDOConfig.from_mapping(
        mapping,
        project_root=_project_root(),
        source_path=source,
    )


def training_stages(config: DDOConfig) -> tuple[str, ...]:
    if config.training.method == "base":
        return ("base",)
    if config.training.method == "reference":
        return ("base", "sft")
    return {
        "scratch": ("base", "sft", "collect", "build", "train"),
        "dtc": ("build", "train"),
        "dataset": ("train",),
    }[config.experiment.start_from]


def validate_training_inputs(config: DDOConfig) -> None:
    if config.experiment.start_from == "scratch" or config.training.method in {"base", "reference"}:
        return
    root = config.project_root or Path.cwd()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    reference = resolve(config.paths.reference_checkpoint)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        if not (reference / name).is_file():
            raise ConfigError(f"missing reference checkpoint file: {reference / name}")

    if config.experiment.start_from == "dtc":
        base = resolve(config.paths.base_trajectories)
        dtc = resolve(config.paths.dtc)
        if not any(base.glob("**/*_llm_trace.json")):
            raise ConfigError(f"no base trajectory traces found under: {base}")
        if not (dtc / "_dtc" / "branch_index.jsonl").is_file():
            raise ConfigError(f"missing DTC branch index: {dtc / '_dtc' / 'branch_index.jsonl'}")
        if not any(dtc.glob("**/*__dtc_*_llm_trace.json")):
            raise ConfigError(f"no DTC branch traces found under: {dtc}")
        return

    dataset = resolve(config.paths.preference_dataset)
    required = _required_dataset_files(config.dataset.method, _task_slug(config.benchmark.task_filter or ""))
    for name in required:
        if not (dataset / name).is_file():
            raise ConfigError(f"missing preference dataset file: {dataset / name}")


def _required_dataset_files(method: str, task_slug: str) -> tuple[str, ...]:
    if method in {"rto_pairs", "tiedpo_rk_pairs", "tiedpo_dav_pairs", "tiedpo_pairs"}:
        return (
            "train_pairs_DDO_balanced_geometric_win_lose.jsonl",
            "train_pairs_DDO_balanced_geometric_win_win.jsonl",
        )
    if method in {"divpo_pairs", "divpo_freq_pairs", "divpo_prob_pairs"}:
        return ("train_pairs_DivPO.jsonl",)
    return (f"{task_slug}_win_lose.jsonl",)


def _training_paths(raw: dict[str, Any], benchmark: str, task: str, method: str) -> dict[str, str]:
    defaults = asdict(
        PathsConfig(
            base_trajectories=str(Path("trajectories/base") / benchmark / task),
            dtc=str(Path("trajectories/dtc") / benchmark / task),
            preference_dataset=str(Path("datasets") / benchmark / task / method),
            reference_checkpoint=str(Path("checkpoints") / benchmark / task / "reference"),
            output_checkpoint=str(Path("checkpoints") / benchmark / task / method),
            work_dir=str(Path(".runs") / benchmark / task / method),
            results_dir=str(Path("results") / benchmark / task / method),
        )
    )
    return _path_options(defaults, raw)


def _evaluation_paths(
    raw: dict[str, Any],
    benchmark: str,
    task: str,
    method: str,
    checkpoint: str | None,
) -> dict[str, str]:
    results_value = raw.get("results_dir", str(Path("results") / benchmark / task / method))
    results_dir = _relative_path(results_value, "results_dir")
    defaults = asdict(
        PathsConfig(
            base_trajectories=str(Path("trajectories/base") / benchmark / task),
            dtc=str(Path("trajectories/dtc") / benchmark / task),
            preference_dataset=str(Path("datasets") / benchmark / task / method),
            reference_checkpoint=str(Path("checkpoints") / benchmark / task / "reference"),
            output_checkpoint=checkpoint or str(Path("checkpoints") / benchmark / task / method),
            work_dir=str(Path(".runs/eval") / benchmark / task / method),
            results_dir=results_dir,
        )
    )
    paths = _path_options(defaults, raw)
    if "results_dir" in raw and paths["results_dir"] != results_dir:
        raise ConfigError("results_dir and paths.results_dir must match when both are set")
    return paths


def _path_options(defaults: dict[str, str], raw: dict[str, Any]) -> dict[str, str]:
    configured = _optional_mapping(raw, "paths")
    _reject_unknown(configured, {field.name for field in fields(PathsConfig)}, "paths")
    defaults.update(configured)
    return {name: _relative_path(value, f"paths.{name}") for name, value in defaults.items()}


def _benchmark_mapping(raw: dict[str, Any], benchmark: str, task: str) -> dict[str, Any]:
    mapping = asdict(BenchmarkConfig(name=benchmark, task_filter=task))
    options = _optional_mapping(raw, "benchmark_options")
    _reject_unknown(options, {field.name for field in fields(BenchmarkConfig)} - {"name", "task_filter"}, "benchmark_options")
    mapping.update(options)
    mapping["name"] = benchmark
    mapping["task_filter"] = task
    return mapping


def _identity(raw: dict[str, Any]) -> tuple[str, str, str]:
    benchmark = str(raw.get("benchmark", "")).strip().lower()
    if benchmark not in TASK_ALIASES:
        raise ConfigError(f"unsupported benchmark: {benchmark}")
    task_value = str(raw.get("task", "")).strip()
    task = TASK_ALIASES[benchmark].get(task_value, task_value)
    if not task:
        raise ConfigError("task is required")
    method = str(raw.get("method", "")).strip().lower()
    if method not in METHODS:
        raise ConfigError(f"unsupported method: {method}")
    return benchmark, task, method


def _dataset_config(method: str) -> dict[str, Any]:
    config = asdict(DatasetConfig(method=method))
    if method in {"dpo_pairs", "tiedpo_rk_pairs", "tiedpo_dav_pairs", "rto_pairs"}:
        config["quality_threshold"] = 999999
    if method == "divpo_freq_pairs":
        config["action_freq_scope"] = "task"
    if method == "divpo_prob_pairs":
        config.update(
            ref_device="cuda",
            probability_response_scope="action_only",
            probability_normalize="mean",
        )
    if method in {"tiedpo_rk_pairs", "tiedpo_dav_pairs"}:
        config.update(alpha=0.5, target_beta=0.1, ref_device="cuda", tie_target=0.5)
    if method == "rto_pairs":
        config.update(
            alpha=0.5,
            target_beta=0.1,
            reference_response_scope="action_only",
            ref_device="cuda",
        )
    return config


def _apply_ddo_aliases(dataset: dict[str, Any], raw: dict[str, Any]) -> None:
    ddo = _optional_mapping(raw, "ddo")
    _reject_unknown(ddo, {"alpha", "reference_scoring"}, "ddo")
    if "alpha" in ddo:
        _set_alias(dataset, "alpha", ddo["alpha"], "ddo.alpha", raw.get("dataset"))
    if "reference_scoring" in ddo:
        _set_alias(
            dataset,
            "reference_response_scope",
            ddo["reference_scoring"],
            "ddo.reference_scoring",
            raw.get("dataset"),
        )


def _normalize_stage_section(
    raw: dict[str, Any],
    section_name: str,
    cls: type[Any],
    *,
    allow_optimizer: bool,
) -> dict[str, Any]:
    section = dict(_optional_mapping(raw, section_name))
    aliases = {"epochs", "precision"}
    if allow_optimizer:
        aliases.add("optimizer")
    valid = {field.name for field in fields(cls)} - {"method"}
    _reject_unknown(section, valid | aliases, section_name)

    normalized = {key: value for key, value in section.items() if key in valid}
    if "epochs" in section:
        _set_alias(normalized, "num_train_epochs", section["epochs"], f"{section_name}.epochs", section)
    if "precision" in section:
        for key, value in _precision_options(section["precision"], f"{section_name}.precision").items():
            _set_alias(normalized, key, value, f"{section_name}.precision", section)
    if allow_optimizer and "optimizer" in section:
        optimizer = section["optimizer"]
        if not isinstance(optimizer, dict):
            raise ConfigError(f"{section_name}.optimizer must be a mapping")
        optimizer_aliases = {
            "name": None,
            "beta1": "adam_beta1",
            "beta2": "adam_beta2",
            "epsilon": "adam_epsilon",
            "weight_decay": "weight_decay",
            "max_grad_norm": "max_grad_norm",
            "scheduler": "lr_scheduler_type",
        }
        _reject_unknown(optimizer, set(optimizer_aliases), f"{section_name}.optimizer")
        if optimizer.get("name", "adamw") != "adamw":
            raise ConfigError(f"{section_name}.optimizer.name must be adamw")
        for source_key, target_key in optimizer_aliases.items():
            if target_key is not None and source_key in optimizer:
                _set_alias(
                    normalized,
                    target_key,
                    optimizer[source_key],
                    f"{section_name}.optimizer.{source_key}",
                    section,
                )
    return normalized


def _normalize_evaluation_section(raw: dict[str, Any]) -> dict[str, Any]:
    section = dict(_optional_mapping(raw, "evaluation"))
    aliases = {
        "success_rollouts": "performance_rollouts",
        "webshop_sessions": "webshop_diversity_sessions",
        "diversity_seed_count": None,
    }
    valid = {field.name for field in fields(EvaluationConfig)}
    _reject_unknown(section, valid | set(aliases), "evaluation")
    normalized = {key: value for key, value in section.items() if key in valid}
    for source_key, target_key in aliases.items():
        if target_key is not None and source_key in section:
            _set_alias(normalized, target_key, section[source_key], f"evaluation.{source_key}", section)
    if "diversity_seed_count" in section:
        count = int(section["diversity_seed_count"])
        seeds = normalized.get("seeds", EvaluationConfig().seeds)
        if count != len(seeds):
            raise ConfigError("evaluation.diversity_seed_count must equal len(evaluation.seeds)")
    return normalized


def _merge_dataclass_options(
    defaults: dict[str, Any],
    raw: dict[str, Any],
    section_name: str,
    cls: type[Any],
    *,
    excluded: set[str] | None = None,
) -> dict[str, Any]:
    section = _optional_mapping(raw, section_name)
    excluded = excluded or set()
    valid = {field.name for field in fields(cls)} - excluded
    _reject_unknown(section, valid, section_name)
    defaults.update(section)
    return defaults


def _shared_decoding(raw: dict[str, Any]) -> dict[str, Any]:
    section = _optional_mapping(raw, "decoding")
    valid = {"temperature", "top_p", "max_tokens", "max_text_history"}
    _reject_unknown(section, valid, "decoding")
    return dict(section)


def _shared_lora(raw: dict[str, Any]) -> dict[str, Any]:
    section = _optional_mapping(raw, "lora")
    if not section:
        return {}
    valid = {"rank", "alpha", "dropout", "target_modules"}
    _reject_unknown(section, valid, "lora")
    normalized: dict[str, Any] = {}
    aliases = {
        "rank": "lora_r",
        "alpha": "lora_alpha",
        "dropout": "lora_dropout",
    }
    for source_key, target_key in aliases.items():
        if source_key in section:
            normalized[target_key] = section[source_key]
    if "target_modules" in section:
        targets = section["target_modules"]
        if not isinstance(targets, list) or not targets or not all(
            isinstance(target, str) and target for target in targets
        ):
            raise ConfigError("lora.target_modules must be a non-empty string list")
        normalized["lora_target_modules"] = ",".join(targets)
    return normalized


def _precision_options(value: Any, name: str) -> dict[str, Any]:
    precision = str(value).strip().lower()
    if precision == "bf16":
        return {"bf16": True, "fp16": False, "torch_dtype": "bfloat16"}
    if precision == "fp16":
        return {"bf16": False, "fp16": True, "torch_dtype": "float16"}
    if precision in {"fp32", "float32"}:
        return {"bf16": False, "fp16": False, "torch_dtype": "float32"}
    if precision == "auto":
        return {"bf16": False, "fp16": False, "torch_dtype": "auto"}
    raise ConfigError(f"{name} must be bf16, fp16, fp32, or auto")


def _models(raw: dict[str, Any], *, require_expert_url: bool) -> dict[str, Any]:
    section = dict(_optional_mapping(raw, "models"))
    aliases = {"target": "target_model", "expert": "expert_model", "reference": "reference_model"}
    valid = {field.name for field in fields(ModelConfig)}
    _reject_unknown(section, valid | set(aliases), "models")
    normalized: dict[str, Any] = {
        "target_model": PAPER_TARGET_MODEL,
        "expert_model": PAPER_EXPERT_MODEL,
    }
    for key in valid:
        if key in section:
            normalized[key] = section[key]
    for source_key, target_key in aliases.items():
        if source_key in section:
            _set_alias(normalized, target_key, section[source_key], f"models.{source_key}", section)
    normalized.setdefault("reference_model", normalized["target_model"])
    if normalized.get("reference_model") is None:
        normalized["reference_model"] = normalized["target_model"]
    if require_expert_url and not normalized.get("expert_base_url"):
        raise ConfigError("models.expert_base_url is required when start_from=scratch")
    return normalized


def _runtime(raw: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    section = dict(_optional_mapping(raw, "runtime"))
    valid = {field.name for field in fields(RuntimeConfig)} - {"dry_run"}
    _reject_unknown(section, valid, "runtime")
    runtime = asdict(RuntimeConfig())
    runtime.update(section)
    if "seed" in raw:
        if "seed" in section and section["seed"] != raw["seed"]:
            raise ConfigError("seed and runtime.seed must match when both are set")
        runtime["seed"] = raw["seed"]
    if "num_workers" in raw:
        if "num_workers" in section and section["num_workers"] != raw["num_workers"]:
            raise ConfigError("num_workers and runtime.num_workers must match when both are set")
        runtime["num_workers"] = raw["num_workers"]
    if "gpu_ids" in raw:
        gpu_ids = _gpu_ids(raw)
        cuda_visible_devices = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        configured = section.get("cuda_visible_devices")
        if configured is not None and configured != cuda_visible_devices:
            raise ConfigError(
                "gpu_ids and runtime.cuda_visible_devices must identify the same GPUs"
            )
        runtime["cuda_visible_devices"] = cuda_visible_devices
    runtime["dry_run"] = dry_run
    return runtime


def _gpu_ids(raw: dict[str, Any]) -> list[int]:
    gpu_ids = raw.get("gpu_ids")
    if not isinstance(gpu_ids, list) or not gpu_ids or not all(
        isinstance(gpu_id, int) and gpu_id >= 0 for gpu_id in gpu_ids
    ):
        raise ConfigError("gpu_ids must be a non-empty list of non-negative integers")
    return gpu_ids


def _set_alias(
    target: dict[str, Any],
    key: str,
    value: Any,
    alias_name: str,
    direct_section: Any,
) -> None:
    if isinstance(direct_section, dict) and key in direct_section and direct_section[key] != value:
        raise ConfigError(f"{alias_name} conflicts with direct field {key}")
    target[key] = value


def _validate_top_level(raw: dict[str, Any], *, evaluation: bool) -> None:
    common = {
        "benchmark",
        "task",
        "method",
        "experiment_name",
        "paths",
        "gpu_ids",
        "num_workers",
        "seed",
        "models",
        "decoding",
        "runtime",
        "benchmark_options",
    }
    if evaluation:
        valid = common | {"checkpoint", "results_dir", "evaluation"}
    else:
        valid = common | {
            "start_from",
            "base",
            "collection",
            "dataset",
            "sft",
            "preference",
            "ddo",
            "lora",
        }
    _reject_unknown(raw, valid, "top-level")


def _reject_unknown(value: dict[str, Any], valid: set[str], section: str) -> None:
    unknown = sorted(set(value) - valid)
    if unknown:
        raise ConfigError(f"unknown {section} field(s): {', '.join(unknown)}")


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read config: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config must contain a YAML mapping")
    return raw


def _optional_mapping(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError(f"{name} must stay inside the repository: {value}")
    return str(path)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _task_slug(task: str) -> str:
    return task.rsplit("/", 1)[-1].replace("-", "_")
