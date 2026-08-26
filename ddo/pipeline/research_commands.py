"""Concrete commands for the DDO research programs vendored in this repository."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any

from ddo.evaluation.benchmarks.base import BenchmarkAdapter
from ddo.config.schema import DDOConfig
from ddo.pipeline.stage_io import (
    config_metadata,
    resolve_adapter_path,
    single_output_dir,
)


def build_stage_command_manifest(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    plan: Any,
    research_root: Path,
) -> dict[str, Any]:
    commands, expected_outputs, notes = _stage_commands(config, benchmark, plan, research_root)
    return {
        "artifact_type": "native_command_manifest",
        "backend": "native",
        "stage": plan.stage,
        "status": "prepared" if commands else "unsupported",
        "research_root": str(research_root),
        "run_commands": True,
        "commands": commands,
        "expected_outputs": expected_outputs,
        "notes": notes,
        "config": config_metadata(config, benchmark, stage=plan.stage),
    }


def _stage_commands(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    plan: Any,
    research_root: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    if plan.stage == "collect":
        return _collection_commands(config, benchmark, plan, research_root)
    if plan.stage == "build":
        return _dataset_commands(config, benchmark, plan, research_root)
    if plan.stage == "train":
        return _training_commands(config, benchmark, plan, research_root)
    if plan.stage == "eval":
        return _evaluation_commands(config, benchmark, plan, research_root)
    raise ValueError(f"unknown stage: {plan.stage}")


def _collection_commands(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    plan: Any,
    research_root: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    output_dir = single_output_dir(plan)
    base_run_dir = _resolved_base_run_dir(config, benchmark)
    if base_run_dir is None:
        return [], [], ["native collection requires benchmark.base_run_dir"]

    task_filter = _collection_task_filter(config, benchmark)
    base_args = [
        "--base-run-dir",
        str(base_run_dir),
        "--output-root",
        str(output_dir),
        "--divergence-count",
        str(config.collection.divergence_count),
        "--step-sampling-mode",
        config.collection.step_sampling_mode,
        "--num-workers",
        str(config.runtime.num_workers),
    ]
    if task_filter:
        if benchmark.name == "webshop":
            base_args.extend(["--task-id", task_filter])
        else:
            base_args.extend(["--task-filter", task_filter])
    if config.collection.success_only:
        base_args.append("--success-only")
    if benchmark.name == "webshop":
        base_args.extend(["--success-threshold", str(config.dataset.success_threshold)])
    if config.collection.rollout_max_steps is not None:
        base_args.extend(["--rollout-max-steps", str(config.collection.rollout_max_steps)])
    if config.collection.rollout_extra_steps is not None and benchmark.name != "webshop":
        base_args.extend(["--rollout-extra-steps", str(config.collection.rollout_extra_steps)])

    if config.collection.method == "dtc":
        if benchmark.name == "webshop":
            script = _required_script(research_root, "ddo/evaluation/webshop/collect_divergence_tree.py")
            argv = [
                sys.executable,
                str(script),
                *base_args,
                "--parallel-workers",
                "--alt-mode",
                _webshop_alt_mode(config.collection.alt_mode),
                "--alt-budget",
                str(config.collection.alt_budget),
                "--temperature",
                str(config.collection.temperature),
                "--top-p",
                str(config.collection.top_p),
                "--max-tokens",
                str(config.collection.max_tokens),
                "--max-text-history",
                str(config.collection.max_text_history),
                "--client-timeout",
                str(config.collection.client_timeout),
                "--client-max-retries",
                str(config.collection.client_max_retries),
                "--client-delay",
                str(config.collection.client_delay),
                "--model-id",
                _collection_model_id(config),
                "--seed",
                str(config.base.seed),
            ]
            if config.models.expert_base_url:
                argv.extend(["--base-url", config.models.expert_base_url])
        else:
            script = _required_script(research_root, "ddo/evaluation/dtc/collect_divergence_tree.py")
            argv = [
                sys.executable,
                str(script),
                *base_args,
                "--alt-mode",
                config.collection.alt_mode,
                "--alt-budget",
                str(config.collection.alt_budget),
                "--resume",
                "--client-timeout-override",
                str(config.collection.client_timeout),
                "--client-max-retries-override",
                str(config.collection.client_max_retries),
                "--external-retry-attempts",
                str(config.collection.external_retry_attempts),
            ]
    else:
        return [], [], [f"unsupported native collection method: {config.collection.method}"]

    return [_command("native_collect", research_root, argv)], [str(output_dir)], []


def _dataset_commands(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    plan: Any,
    research_root: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    output_dir = single_output_dir(plan)
    paths = benchmark.resolve_paths(config)
    collection_dir = paths.dtc
    common = [
        "--base-run-dir",
        str(paths.base_trajectories),
        "--usable-traj",
        config.dataset.usable_trajectories,
        "--criterion",
        config.dataset.criterion,
    ]
    if config.dataset.success_threshold is not None:
        common.extend(["--success-threshold", str(config.dataset.success_threshold)])

    if config.dataset.method in {"rto_pairs", "tiedpo_rk_pairs", "tiedpo_dav_pairs", "tiedpo_pairs"}:
        script = _required_script(
            research_root,
            "ddo/data/build_DDO_balanced_geometric_legacy_pairs_from_dtc.py",
        )
        win_lose = output_dir / "train_pairs_DDO_balanced_geometric_win_lose.jsonl"
        win_win = output_dir / "train_pairs_DDO_balanced_geometric_win_win.jsonl"
        builder_output_dir = output_dir
        builder_win_lose = win_lose
        builder_win_win = win_win
        argv = [
            sys.executable,
            str(script),
            "--dtc-run-dir",
            str(collection_dir),
            *common,
            "--quality-score",
            config.dataset.quality_score,
            "--alpha",
            str(config.dataset.alpha),
            "--target-mode",
            "reference_relative",
            "--target-beta",
            str(config.dataset.target_beta),
            "--ref-model-name-or-path",
            config.models.reference_model or config.models.target_model,
            "--out-dir",
            str(builder_output_dir),
            "--aggregate-win-lose-out",
            str(builder_win_lose),
            "--aggregate-win-win-out",
            str(builder_win_win),
        ]
        if config.dataset.quality_threshold is not None:
            argv.extend(
                ["--max-same-obs-action-run-threshold", str(config.dataset.quality_threshold)]
            )
        if config.dataset.max_families:
            argv.extend(["--max-families", str(config.dataset.max_families)])
        task_filter = _task_filter(config, benchmark)
        if task_filter:
            argv.extend(["--task-filter", task_filter])
        reference_adapter = _reference_adapter_path(config, benchmark)
        if reference_adapter is None:
            raise RuntimeError(
                f"{config.dataset.method} requires the materialized SFT reference adapter"
            )
        argv.extend(["--ref-adapter-path", reference_adapter])
        argv.extend(
            [
                "--ref-torch-dtype",
                _reference_torch_dtype(config),
                "--ref-response-scope",
                config.dataset.reference_response_scope,
                "--ref-normalize",
                _reference_normalize_arg(config.dataset.reference_normalize),
                "--max-prompt-length",
                str(config.training.max_prompt_length),
                "--max-length",
                str(config.training.max_length),
            ]
        )
        argv.extend(["--ref-device", config.dataset.ref_device or "auto"])
        commands = [_command("native_build_reference_relative_pairs", research_root, argv)]
        expected = [str(win_lose), str(win_win)]
        notes: list[str] = []
        return commands, expected, notes

    if config.dataset.method == "dpo_pairs":
        script = _required_script(research_root, "ddo/data/build_filtered_legacy_pairs_from_dtc.py")
        task_slug = _task_slug(_task_filter(config, benchmark))
        win_lose = output_dir / f"{task_slug}_win_lose.jsonl"
        win_win = output_dir / f"{task_slug}_win_win.jsonl"
        argv = [
            sys.executable,
            str(script),
            "--dtc-run-dir",
            str(collection_dir),
            *common,
            "--quality-score",
            config.dataset.quality_score,
            "--out-dir",
            str(output_dir),
        ]
        if config.dataset.quality_threshold is not None:
            argv.extend(
                ["--max-same-obs-action-run-threshold", str(config.dataset.quality_threshold)]
            )
        if config.dataset.max_families:
            argv.extend(["--max-families", str(config.dataset.max_families)])
        if config.dataset.include_base_alt_win_pseudo_pairs:
            argv.append("--include-base-alt-win-pseudo-pairs")
        commands = [_command("native_build_dpo_pairs", research_root, argv)]
        expected = [str(win_lose), str(win_win)]
        notes: list[str] = []
        return commands, expected, notes

    if config.dataset.method in {"divpo_pairs", "divpo_freq_pairs", "divpo_prob_pairs"}:
        script = _required_script(research_root, "ddo/data/build_DivPO_legacy_pairs_from_dtc.py")
        aggregate = output_dir / "train_pairs_DivPO.jsonl"
        probability_mode = config.dataset.method == "divpo_prob_pairs"
        argv = [
            sys.executable,
            str(script),
            "--dtc-run-dir",
            str(collection_dir),
            *common,
            "--diversity-criterion",
            "prob" if probability_mode else "action_freq",
            "--out-dir",
            str(output_dir),
            "--aggregate-out",
            str(aggregate),
        ]
        if config.dataset.max_families:
            argv.extend(["--max-families", str(config.dataset.max_families)])
        if config.dataset.include_base_alt_win_pseudo_pairs:
            argv.append("--include-base-alt-win-pseudo-pairs")
        if probability_mode:
            argv.extend(
                [
                    "--prob-model-name-or-path",
                    config.models.reference_model or config.models.target_model,
                    "--prob-device",
                    config.dataset.ref_device or "auto",
                    "--prob-torch-dtype",
                    _reference_torch_dtype(config),
                    "--prob-response-scope",
                    config.dataset.probability_response_scope,
                    "--prob-normalize",
                    config.dataset.probability_normalize,
                    "--prob-max-prompt-length",
                    str(config.training.max_prompt_length),
                    "--prob-max-length",
                    str(config.training.max_length),
                ]
            )
            reference_adapter = _reference_adapter_path(config, benchmark)
            if reference_adapter is None:
                raise RuntimeError("divpo_prob_pairs requires the materialized SFT reference adapter")
            argv.extend(["--prob-adapter-path", reference_adapter])
        else:
            argv.extend(["--action-freq-scope", config.dataset.action_freq_scope])
        commands = [_command("native_build_divpo_pairs", research_root, argv)]
        expected = [str(aggregate)]
        notes: list[str] = []
        return commands, expected, notes

    return [], [], [f"unsupported native dataset method: {config.dataset.method}"]



def _training_commands(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    plan: Any,
    research_root: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    output_dir = single_output_dir(plan)
    dataset_dir = benchmark.resolve_paths(config).preference_dataset
    if (config.dataset.method == "rto_pairs" and config.training.method in {"rto", "ddo"} and config.training.ddo_objective == "reference_relative"):
        return _balanced_geometric_training_commands(
            config,
            benchmark,
            research_root,
            output_dir,
            dataset_dir,
        )
    mode = _training_mode(config.training.method)
    ddo_loss_variant = "symmetric_gap"
    if config.training.method == "ddo" and config.training.ddo_objective != "reference_relative":
        mode = "ddo"
        ddo_loss_variant = {"squared_gap": "symmetric_gap", "upward_floor": "upward_floor"}[config.training.ddo_objective]
    win_lose, win_win = _pair_inputs(config, benchmark, dataset_dir)
    trainer_run_dir = output_dir / "trainer_run"
    argv = [
        sys.executable,
        str(_required_script(research_root, "ddo/training/train_trl_preference.py")),
        "--mode",
        mode,
        "--model-name-or-path",
        config.models.target_model,
        "--ref-model-name-or-path",
        config.models.reference_model or config.models.target_model,
        "--finetune-type",
        config.training.finetune_type,
        "--win-lose-jsonl",
        str(win_lose),
        "--output-dir",
        str(trainer_run_dir),
        "--run-name",
        _safe_run_name(config, benchmark),
        "--beta",
        str(config.training.beta),
        "--dpo-rk-alpha",
        str(config.training.dpo_rk_alpha),
        "--dpo-d-nu",
        str(config.training.dpo_d_nu),
        "--lambda-div",
        str(config.training.lambda_div),
        "--ddo-loss-variant",
        ddo_loss_variant,
        "--lambda-floor",
        str(config.training.lambda_floor),
        "--learning-rate",
        str(config.training.learning_rate),
        "--num-train-epochs",
        str(config.training.num_train_epochs),
        "--max-steps",
        str(config.training.max_steps),
        "--warmup-ratio",
        str(config.training.warmup_ratio),
        "--adam-beta1",
        str(config.training.adam_beta1),
        "--adam-beta2",
        str(config.training.adam_beta2),
        "--adam-epsilon",
        str(config.training.adam_epsilon),
        "--weight-decay",
        str(config.training.weight_decay),
        "--max-grad-norm",
        str(config.training.max_grad_norm),
        "--lr-scheduler-type",
        config.training.lr_scheduler_type,
        "--per-device-train-batch-size",
        str(config.training.per_device_train_batch_size),
        "--gradient-accumulation-steps",
        str(config.training.gradient_accumulation_steps),
        "--max-prompt-length",
        str(config.training.max_prompt_length),
        "--max-length",
        str(config.training.max_length),
        "--logging-steps",
        str(config.training.logging_steps),
        "--save-steps",
        str(config.training.save_steps),
        "--save-total-limit",
        _optional_int_arg(config.training.save_total_limit),
        "--dataloader-num-workers",
        str(config.training.dataloader_num_workers),
        "--torch-dtype",
        config.training.torch_dtype,
        "--lora-r",
        str(config.training.lora_r),
        "--lora-alpha",
        str(config.training.lora_alpha),
        "--lora-dropout",
        str(config.training.lora_dropout),
        "--seed",
        str(config.training.seed if config.training.seed is not None else config.runtime.seed),
    ]
    if config.training.save_epochs_fraction is not None:
        argv.extend(["--save-epochs-fraction", str(config.training.save_epochs_fraction)])
    if config.training.stop_after_epochs is not None:
        argv.extend(["--stop-after-epochs", str(config.training.stop_after_epochs)])
    if win_win is not None:
        argv.extend(["--win-win-jsonl", str(win_win)])
    train_adapter, ref_adapter = _training_adapter_paths(config, benchmark)
    if train_adapter is None or ref_adapter is None:
        raise RuntimeError("preference training requires the materialized SFT reference adapter")
    if train_adapter:
        argv.extend(["--train-adapter-path", train_adapter])
    if ref_adapter:
        argv.extend(["--ref-adapter-path", ref_adapter])
    if config.training.resume_from_checkpoint:
        argv.extend(["--resume-from-checkpoint", config.training.resume_from_checkpoint])
    if config.training.bf16:
        argv.append("--bf16")
    if config.training.fp16:
        argv.append("--fp16")
    if config.training.gradient_checkpointing:
        argv.append("--gradient-checkpointing")
    if config.training.lora_target_modules:
        argv.extend(["--lora-target-modules", config.training.lora_target_modules])
    if config.training.validate_only:
        argv.append("--validate-only")

    expected = [
        str(trainer_run_dir / "run_metadata.json"),
        str(trainer_run_dir / "trainer_state.json"),
        str(output_dir / "native_command_manifest.json"),
    ]
    return [_command("native_train_preference_lora", research_root, argv)], expected, []



def _balanced_geometric_training_commands(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    research_root: Path,
    output_dir: Path,
    dataset_dir: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    trainer_run_dir = output_dir / "trainer_run"
    argv = [
        sys.executable,
        str(_required_script(research_root, "ddo/training/train_trl_preference_DDO_balanced_geometric.py")),
        "--dataset-dir",
        str(dataset_dir),
        "--model-name-or-path",
        config.models.target_model,
        "--finetune-type",
        config.training.finetune_type,
        "--output-dir",
        str(trainer_run_dir),
        "--run-name",
        _safe_run_name(config, benchmark),
        "--beta",
        str(config.training.beta),
        "--learning-rate",
        str(config.training.learning_rate),
        "--num-train-epochs",
        str(config.training.num_train_epochs),
        "--max-steps",
        str(config.training.max_steps),
        "--warmup-ratio",
        str(config.training.warmup_ratio),
        "--adam-beta1",
        str(config.training.adam_beta1),
        "--adam-beta2",
        str(config.training.adam_beta2),
        "--adam-epsilon",
        str(config.training.adam_epsilon),
        "--weight-decay",
        str(config.training.weight_decay),
        "--max-grad-norm",
        str(config.training.max_grad_norm),
        "--lr-scheduler-type",
        config.training.lr_scheduler_type,
        "--per-device-train-batch-size",
        str(config.training.per_device_train_batch_size),
        "--gradient-accumulation-steps",
        str(config.training.gradient_accumulation_steps),
        "--max-prompt-length",
        str(config.training.max_prompt_length),
        "--max-length",
        str(config.training.max_length),
        "--logging-steps",
        str(config.training.logging_steps),
        "--save-steps",
        str(config.training.save_steps),
        "--save-total-limit",
        _optional_int_arg(config.training.save_total_limit),
        "--dataloader-num-workers",
        str(config.training.dataloader_num_workers),
        "--torch-dtype",
        config.training.torch_dtype,
        "--lora-r",
        str(config.training.lora_r),
        "--lora-alpha",
        str(config.training.lora_alpha),
        "--lora-dropout",
        str(config.training.lora_dropout),
        "--seed",
        str(config.training.seed if config.training.seed is not None else config.runtime.seed),
    ]
    train_adapter, ref_adapter = _training_adapter_paths(config, benchmark)
    if train_adapter is None or ref_adapter is None:
        raise RuntimeError("DDO training requires the materialized SFT reference adapter")
    if ref_adapter is None:
        argv.extend(["--ref-model-name-or-path", config.models.reference_model or config.models.target_model])
    if config.training.save_epochs_fraction is not None:
        argv.extend(["--save-epochs-fraction", str(config.training.save_epochs_fraction)])
    if config.training.stop_after_epochs is not None:
        argv.extend(["--stop-after-epochs", str(config.training.stop_after_epochs)])
    if train_adapter:
        argv.extend(["--train-adapter-path", train_adapter])
    if ref_adapter:
        argv.extend(["--ref-adapter-path", ref_adapter])
    if config.training.resume_from_checkpoint:
        argv.extend(["--resume-from-checkpoint", config.training.resume_from_checkpoint])
    if config.training.bf16:
        argv.append("--bf16")
    if config.training.fp16:
        argv.append("--fp16")
    if config.training.gradient_checkpointing:
        argv.append("--gradient-checkpointing")
    if config.training.lora_target_modules:
        argv.extend(["--lora-target-modules", config.training.lora_target_modules])
    if config.training.validate_only:
        argv.append("--validate-only")

    expected = [
        str(trainer_run_dir / "run_metadata.json"),
        str(trainer_run_dir / "trainer_state.json"),
        str(output_dir / "native_command_manifest.json"),
    ]
    return [_command("native_train_balanced_geometric_lora", research_root, argv)], expected, []


def _evaluation_commands(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    plan: Any,
    research_root: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    output_dir = single_output_dir(plan)
    task_filter = _task_filter(config, benchmark)
    base_url = config.models.target_base_url
    if benchmark.name == "webshop":
        script = _required_script(research_root, "ddo/evaluation/webshop/eval.py")
        run_prefix = _safe_run_name(config, benchmark)

        def webshop_argv(
            *,
            run_id: str,
            episode_count: int,
            session_start: int,
            session_mode: str,
            llm_seed: int | None,
            num_workers: int,
        ) -> list[str]:
            argv = [
                sys.executable,
                str(script),
                "--output-root",
                str(output_dir),
                "--run-id",
                run_id,
                "--task-id",
                task_filter or "webshop",
                "--episode-start",
                "0",
                "--episode-end",
                str(episode_count),
                "--session-start",
                str(session_start),
                "--session-mode",
                session_mode,
                "--seed",
                str(config.runtime.seed),
                "--seed-mode",
                "per_episode" if session_mode == "per_episode" else "fixed",
                "--num-workers",
                str(num_workers),
                "--parallel-workers",
                "--num-products",
                str(config.evaluation.webshop_num_products),
                "--human-goals",
                "1" if config.evaluation.webshop_human_goals else "0",
                "--max-search-queries",
                str(config.evaluation.webshop_max_search_queries),
                "--model-id",
                config.models.target_model,
                "--max-steps",
                str(config.evaluation.max_steps_per_episode),
                "--max-text-history",
                str(config.evaluation.webshop_max_text_history),
                "--temperature",
                str(config.evaluation.temperature),
                "--top-p",
                str(config.evaluation.top_p),
                "--max-tokens",
                str(config.evaluation.webshop_max_tokens),
                "--client-timeout",
                str(config.evaluation.webshop_client_timeout),
                "--client-max-retries",
                str(config.evaluation.webshop_client_max_retries),
                "--client-delay",
                str(config.evaluation.webshop_client_delay),
                "--success-threshold",
                str(config.evaluation.webshop_success_threshold),
            ]
            if config.evaluation.webshop_disable_thinking:
                argv.append("--vllm-disable-thinking")
            if llm_seed is None:
                argv.extend(["--llm-seed-mode", "none"])
            else:
                argv.extend(["--llm-seed", str(llm_seed), "--llm-seed-mode", "per_episode"])
            if base_url:
                argv.extend(["--base-url", base_url])
            return argv

        commands = [
            _command(
                "native_eval_webshop_performance",
                research_root,
                webshop_argv(
                    run_id=f"{run_prefix}_performance",
                    episode_count=config.evaluation.performance_rollouts,
                    session_start=config.evaluation.webshop_performance_session_start,
                    session_mode="per_episode",
                    llm_seed=config.evaluation.llm_seed_base,
                    num_workers=config.evaluation.webshop_performance_num_workers,
                ),
            )
        ]
        diversity_seed_base = config.evaluation.webshop_diversity_llm_seed_base
        for position, session_id in enumerate(config.evaluation.webshop_diversity_sessions):
            commands.append(
                _command(
                    f"native_eval_webshop_diversity_session{session_id}",
                    research_root,
                    webshop_argv(
                        run_id=f"{run_prefix}_diversity_session{session_id}",
                        episode_count=config.evaluation.diversity_rollouts,
                        session_start=session_id,
                        session_mode="fixed",
                        llm_seed=(
                            diversity_seed_base + position * 100
                            if diversity_seed_base is not None
                            else (
                                None
                                if config.evaluation.llm_seed_base is None
                                else config.evaluation.llm_seed_base + (position + 1) * 100
                            )
                        ),
                        num_workers=config.evaluation.webshop_diversity_num_workers,
                    ),
                )
            )
    else:
        script = _required_script(research_root, "ddo/evaluation/balrog/eval.py")
        env_name = "babaisai" if benchmark.name == "babaisai" else "babyai"
        commands = []
        eval_jobs = [
            (
                "performance",
                "per_episode",
                config.evaluation.seeds[0],
                config.evaluation.performance_rollouts,
            ),
            *(
                ("diversity", "fixed", seed, config.evaluation.diversity_rollouts)
                for seed in config.evaluation.seeds
            ),
        ]
        for mode, seed_mode, seed, episodes in eval_jobs:
            run_postfix = (
                f"{_safe_run_name(config, benchmark)}_"
                f"{mode}{episodes}_seed{seed}_{mode}"
            )
            argv = [
                sys.executable,
                str(script),
                "--benchmark", env_name,
                "--task", task_filter or "",
                "--output-dir", str(output_dir / run_postfix),
                "--num-episodes", str(episodes),
                "--num-workers", str(config.runtime.num_workers),
                "--seed", str(seed),
                "--seed-mode", seed_mode,
                "--model-id", config.models.target_model,
                "--max-steps", str(config.evaluation.max_steps_per_episode),
                "--max-text-history", str(config.evaluation.max_text_history),
                "--temperature", str(config.evaluation.temperature),
                "--top-p", str(config.evaluation.top_p),
                "--max-tokens", str(config.evaluation.max_tokens),
                "--client-timeout", str(config.evaluation.client_timeout),
                "--client-max-retries", str(config.evaluation.client_max_retries),
            ]
            if config.evaluation.llm_seed_base is not None:
                argv.extend(["--llm-seed-base", str(config.evaluation.llm_seed_base)])
            if base_url:
                argv.extend(["--base-url", base_url])
            commands.append(_command(f"native_eval_{mode}_seed{seed}", research_root, argv))
    for command in commands:
        command["server_mode"] = config.evaluation.server_mode
    notes = [
        "evaluation server is started and stopped by the eval stage"
        if config.evaluation.server_mode == "managed"
        else "evaluation uses the model endpoint configured in models.target_base_url"
    ]
    return commands, [str(output_dir)], notes



def _command(name: str, cwd: Path, argv: list[str], *, runnable: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "cwd": str(cwd),
        "argv": argv,
        "command": " ".join(shlex.quote(part) for part in argv),
        "runnable": runnable,
    }

def _resolved_base_run_dir(config: DDOConfig, benchmark: BenchmarkAdapter) -> Path | None:
    return benchmark.resolve_base_run_dir(config)


def _external_script(research_root: Path, relative_path: str) -> Path:
    """Resolve an ignored user-managed dependency without requiring it during planning."""
    return research_root / relative_path


def _required_script(research_root: Path, relative_path: str) -> Path:
    script = research_root / relative_path
    if not script.exists():
        raise RuntimeError(f"research backend script does not exist: {script}")
    return script


def _pair_inputs(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
    dataset_dir: Path,
) -> tuple[Path, Path | None]:
    task_slug = _task_slug(_task_filter(config, benchmark))
    if config.dataset.method in {"rto_pairs", "tiedpo_rk_pairs", "tiedpo_dav_pairs", "tiedpo_pairs"}:
        win_lose = dataset_dir / "train_pairs_DDO_balanced_geometric_win_lose.jsonl"
        if config.training.method == "dpo":
            return win_lose, None
        return win_lose, dataset_dir / "train_pairs_DDO_balanced_geometric_win_win.jsonl"
    if config.dataset.method in {"divpo_pairs", "divpo_freq_pairs", "divpo_prob_pairs"}:
        return dataset_dir / "train_pairs_DivPO.jsonl", None
    return dataset_dir / f"{task_slug}_win_lose.jsonl", None


def _training_mode(training_method: str) -> str:
    return {
        "rto": "ddo_bg_v2",
        "ddo": "ddo_bg_v2",
        "dpo": "dpo",
        "divpo": "divpo",
        "divpo_freq": "divpo",
        "divpo_prob": "divpo",
        "tiedpo": "dpo_d",
        "tiedpo_rk": "dpo_rk",
        "tiedpo_dav": "dpo_d",
    }.get(training_method, training_method)


def _reference_adapter_path(config: DDOConfig, benchmark: BenchmarkAdapter) -> str | None:
    configured = config.training.ref_adapter_path or config.training.train_adapter_path
    if configured:
        return configured
    sft_dir = benchmark.resolve_paths(config).reference_checkpoint
    adapter = resolve_adapter_path(sft_dir)
    return str(adapter) if adapter is not None else None


def _training_adapter_paths(
    config: DDOConfig,
    benchmark: BenchmarkAdapter,
) -> tuple[str | None, str | None]:
    sft_adapter = _reference_adapter_path(config, benchmark)
    return (
        config.training.train_adapter_path or sft_adapter,
        config.training.ref_adapter_path or sft_adapter,
    )


def _reference_torch_dtype(config: DDOConfig) -> str:
    if config.sft.torch_dtype != "auto":
        return config.sft.torch_dtype
    if config.sft.bf16:
        return "bfloat16"
    if config.sft.fp16:
        return "float16"
    return "auto"


def _optional_int_arg(value: int | None) -> str:
    return "none" if value is None else str(value)


def _task_filter(config: DDOConfig, benchmark: BenchmarkAdapter) -> str | None:
    task = benchmark.normalize_task_id(config.benchmark.task_filter)
    if not task:
        return None
    if benchmark.name == "babaisai" and task.startswith("babaisai/"):
        return f"env/{task.split('/', 1)[1]}"
    return task


def _collection_task_filter(config: DDOConfig, benchmark: BenchmarkAdapter) -> str | None:
    if config.benchmark.train_task_filters:
        return None
    return _task_filter(config, benchmark)


def _task_slug(task_filter: str | None) -> str:
    if not task_filter:
        return "all"
    return task_filter.rsplit("/", 1)[-1].replace("-", "_")


def _safe_run_name(config: DDOConfig, _benchmark: BenchmarkAdapter) -> str:
    raw = config.experiment.name
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in raw)


def _webshop_alt_mode(alt_mode: str) -> str:
    return alt_mode


def _reference_normalize_arg(value: str) -> str:
    return "avg" if value == "mean" else value


def _collection_model_id(config: DDOConfig) -> str:
    return config.models.expert_model or config.models.target_model or "Qwen/Qwen3-1.7B"
