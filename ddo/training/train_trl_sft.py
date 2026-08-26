#!/usr/bin/env python3
"""Research SFT trainer for conversational DTC datasets.

This entrypoint follows the current P1 SFT dataset contract:
- input rows live under data/dataset/sft/<run_id>/<usable_traj>/train.jsonl
- each row contains conversational `prompt` and `completion` message lists

The implementation prioritizes the common Qwen <=9B post-training pattern:
- TRL `SFTTrainer`
- LoRA only
- tokenizer chat template path
- completion-only loss on prompt-completion data
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DATASET_ROOT = REPO_ROOT / "data" / "dataset"
CANONICAL_SFT_DATASET_ROOT = CANONICAL_DATASET_ROOT / "sft"
CANONICAL_CHECKPOINT_ROOT = REPO_ROOT / "data" / "checkpoints" / "manual"
CANONICAL_TRAIN_FILENAME = "train.jsonl"
DEFAULT_MODEL_NAME_OR_PATH = "Qwen/Qwen3.5-0.8B"
DEFAULT_USABLE_TRAJ = "base-traj"
DEFAULT_ASSISTANT_LOSS_MODE = "full"
ACTION_LINE_MARKER = "Action:"
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]
LORA_TARGET_PROFILES: dict[str, list[str]] = {
    "qv": [
        "q_proj",
        "v_proj",
    ],
    "qvko": list(DEFAULT_LORA_TARGET_MODULES),
    "qvko+mlp": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}


@dataclass(frozen=True)
class TrainInputPaths:
    train_jsonl: Path
    sample_plan_jsonl: Path | None


class SFTSchemaError(ValueError):
    pass


def _task_short_name(task_id: str) -> str:
    return task_id.rsplit("/", maxsplit=1)[-1]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _use_qwen35_text_backbone(config: Any) -> bool:
    return getattr(config, "model_type", None) == "qwen3_5" and getattr(config, "text_config", None) is not None


def _qwen35_causal_lm_class():
    try:
        from transformers.models.qwen3_5 import Qwen3_5ForCausalLM
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Qwen3.5 training requires a Transformers build with qwen3_5 support."
        ) from exc
    return Qwen3_5ForCausalLM


def _resolve_torch_dtype(args: argparse.Namespace) -> str | None:
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 cannot be set together")

    if args.torch_dtype != "auto":
        return {
            "bfloat16": "bfloat16",
            "float16": "float16",
            "float32": "float32",
        }[args.torch_dtype]

    if args.bf16:
        return "bfloat16"
    if args.fp16:
        return "float16"
    return None


def _parse_optional_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"none", "null"}:
        return None
    return int(normalized)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _parse_lora_target_modules(value: str | None) -> str | list[str]:
    if value is None or not value.strip():
        return list(DEFAULT_LORA_TARGET_MODULES)

    normalized = value.strip()
    if normalized == "all-linear":
        return "all-linear"
    profile = LORA_TARGET_PROFILES.get(normalized)
    if profile is not None:
        return list(profile)
    if normalized.replace(" ", "") == "qvko+gate_proj,up_proj,down_proj":
        return list(LORA_TARGET_PROFILES["qvko+mlp"])
    return [item.strip() for item in normalized.split(",") if item.strip()]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SFTSchemaError(f"{path}:{line_idx} invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SFTSchemaError(f"{path}:{line_idx} row must be an object")
            rows.append(row)
    if not rows:
        raise SFTSchemaError(f"{path} has no valid JSONL rows")
    return rows


def _validate_messages(
    messages: Any,
    *,
    source: Path,
    line_idx: int,
    field_name: str,
    allowed_roles: set[str] | None = None,
) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise SFTSchemaError(f"{source}:{line_idx} field '{field_name}' must be a non-empty list")

    normalized: list[dict[str, str]] = []
    for msg_idx, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise SFTSchemaError(
                f"{source}:{line_idx} field '{field_name}[{msg_idx}]' must be an object"
            )
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if not role or not content:
            raise SFTSchemaError(
                f"{source}:{line_idx} field '{field_name}[{msg_idx}]' must contain non-empty role/content"
            )
        if allowed_roles is not None and role not in allowed_roles:
            allowed = ", ".join(sorted(allowed_roles))
            raise SFTSchemaError(
                f"{source}:{line_idx} field '{field_name}[{msg_idx}].role' must be one of: {allowed}"
            )
        normalized.append({"role": role, "content": content})
    return normalized


def _validate_sft_row(row: dict[str, Any], source: Path, line_idx: int) -> dict[str, Any]:
    if "prompt" not in row:
        raise SFTSchemaError(f"{source}:{line_idx} missing required field: prompt")
    if "completion" not in row:
        raise SFTSchemaError(f"{source}:{line_idx} missing required field: completion")

    prompt = _validate_messages(
        row["prompt"],
        source=source,
        line_idx=line_idx,
        field_name="prompt",
    )
    completion = _validate_messages(
        row["completion"],
        source=source,
        line_idx=line_idx,
        field_name="completion",
        allowed_roles={"assistant"},
    )

    validated = dict(row)
    validated["prompt"] = prompt
    validated["completion"] = completion
    return validated


def _resolve_input_paths(args: argparse.Namespace) -> TrainInputPaths:
    if args.train_jsonl:
        return TrainInputPaths(
            train_jsonl=Path(args.train_jsonl),
            sample_plan_jsonl=Path(args.sample_plan_jsonl) if args.sample_plan_jsonl else None,
        )
    if not args.dataset_run_id:
        raise ValueError("Provide either --dataset-run-id or --train-jsonl")
    usable_traj = args.usable_traj.strip()
    if not usable_traj:
        raise ValueError("--usable-traj must be a non-empty string")
    return TrainInputPaths(
        train_jsonl=CANONICAL_SFT_DATASET_ROOT / args.dataset_run_id.strip() / usable_traj / CANONICAL_TRAIN_FILENAME,
        sample_plan_jsonl=Path(args.sample_plan_jsonl) if args.sample_plan_jsonl else None,
    )


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if not args.run_name or not args.run_name.strip():
        raise ValueError("--run-name must be a non-empty string")

    if args.output_dir:
        candidate = Path(args.output_dir)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        elif candidate.parts[:1] == ("data",):
            resolved = (REPO_ROOT / candidate).resolve()
        else:
            resolved = (CANONICAL_CHECKPOINT_ROOT / candidate).resolve()
    else:
        resolved = (CANONICAL_CHECKPOINT_ROOT / args.run_name.strip()).resolve()

    checkpoint_root = CANONICAL_CHECKPOINT_ROOT.resolve()
    project_root = REPO_ROOT.resolve()
    if not _is_relative_to(resolved, project_root):
        raise ValueError(
            f"--output-dir must resolve under repository root {project_root} "
            f"(default checkpoint root: {checkpoint_root})"
        )
    return resolved


def load_sft_rows(train_jsonl: Path) -> list[dict[str, Any]]:
    raw_rows = _read_jsonl(train_jsonl)
    return [
        _validate_sft_row(row, train_jsonl, line_idx)
        for line_idx, row in enumerate(raw_rows, start=1)
    ]


def load_sample_plan(sample_plan_jsonl: Path, *, row_count: int) -> list[dict[str, Any]]:
    plan_rows = _read_jsonl(sample_plan_jsonl)
    normalized_rows: list[dict[str, Any]] = []
    for line_idx, row in enumerate(plan_rows, start=1):
        row_idx = row.get("row_idx")
        if not isinstance(row_idx, int):
            raise SFTSchemaError(f"{sample_plan_jsonl}:{line_idx} missing integer row_idx")
        if row_idx < 0 or row_idx >= row_count:
            raise SFTSchemaError(
                f"{sample_plan_jsonl}:{line_idx} row_idx {row_idx} out of range for {row_count} source rows"
            )
        normalized_rows.append({"row_idx": row_idx, **row})
    if not normalized_rows:
        raise SFTSchemaError(f"{sample_plan_jsonl} has no valid plan rows")
    return normalized_rows


def apply_sample_plan(rows: list[dict[str, Any]], sample_plan_rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if sample_plan_rows is None:
        return list(rows)
    return [rows[int(plan_row["row_idx"])] for plan_row in sample_plan_rows]


def _resolve_task_id(rows: list[dict[str, Any]], requested_task_id: str | None) -> str | None:
    if requested_task_id is None:
        return None

    normalized = requested_task_id.strip()
    if not normalized:
        raise ValueError("--task-id must be a non-empty string")

    available_task_ids = sorted({str(row.get("task_id") or "unknown") for row in rows})
    if normalized in available_task_ids:
        return normalized

    suffix_matches = [task_id for task_id in available_task_ids if _task_short_name(task_id) == normalized]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        raise ValueError(
            f"--task-id '{normalized}' is ambiguous. Matching task ids: {', '.join(suffix_matches)}"
        )

    raise ValueError(
        f"--task-id '{normalized}' was not found in the dataset. "
        f"Available task ids: {', '.join(available_task_ids)}"
    )


def filter_sft_rows_by_task(rows: list[dict[str, Any]], requested_task_id: str | None) -> tuple[list[dict[str, Any]], str | None]:
    resolved_task_id = _resolve_task_id(rows, requested_task_id)
    if resolved_task_id is None:
        return list(rows), None

    filtered_rows = [row for row in rows if str(row.get("task_id") or "unknown") == resolved_task_id]
    if not filtered_rows:
        raise ValueError(f"--task-id '{resolved_task_id}' matched zero rows")
    return filtered_rows, resolved_task_id


def _trajectory_group_key(row: dict[str, Any]) -> str:
    trajectory_id = str(row.get("trajectory_id") or "").strip()
    if trajectory_id:
        return trajectory_id
    family_id = str(row.get("family_id") or "").strip()
    if family_id:
        return family_id
    raise ValueError("--row-balance trajectory requires each row to define trajectory_id or family_id")


def _resample_rows_to_target_count(group_rows: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    if not group_rows:
        return []
    if target_count <= 0:
        return []

    source_count = len(group_rows)
    if target_count == source_count:
        return list(group_rows)
    if target_count > source_count:
        return [group_rows[idx % source_count] for idx in range(target_count)]

    # Select evenly spaced rows to avoid front-loading early steps when shrinking long trajectories.
    return [
        group_rows[min(source_count - 1, ((2 * idx + 1) * source_count) // (2 * target_count))]
        for idx in range(target_count)
    ]


def rebalance_sft_rows_by_trajectory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[_trajectory_group_key(row)].append(row)

    total_row_count = len(rows)
    trajectory_ids = sorted(grouped_rows)
    trajectory_count = len(trajectory_ids)
    base_target = total_row_count // trajectory_count
    remainder = total_row_count % trajectory_count

    balanced_rows: list[dict[str, Any]] = []
    for idx, trajectory_id in enumerate(trajectory_ids):
        target_count = base_target + (1 if idx < remainder else 0)
        balanced_rows.extend(_resample_rows_to_target_count(grouped_rows[trajectory_id], target_count))
    return balanced_rows


def preprocess_sft_rows(
    rows: list[dict[str, Any]],
    *,
    task_id: str | None,
    row_balance: str,
) -> tuple[list[dict[str, Any]], str | None]:
    filtered_rows, resolved_task_id = filter_sft_rows_by_task(rows, task_id)
    if row_balance == "none":
        return filtered_rows, resolved_task_id
    if row_balance == "trajectory":
        return rebalance_sft_rows_by_trajectory(filtered_rows), resolved_task_id
    raise ValueError(f"Unsupported --row-balance value: {row_balance}")


def summarize_sample_plan(sample_plan_rows: list[dict[str, Any]] | None, *, source_row_count: int) -> dict[str, Any] | None:
    if sample_plan_rows is None:
        return None
    distinct_row_count = len({int(row["row_idx"]) for row in sample_plan_rows})
    mode_counts = Counter(str(row.get("mode") or "unknown") for row in sample_plan_rows)
    return {
        "plan_row_count": len(sample_plan_rows),
        "source_row_count": source_row_count,
        "distinct_source_row_count": distinct_row_count,
        "resample_ratio": len(sample_plan_rows) / source_row_count if source_row_count else 0.0,
        "reuse_ratio": len(sample_plan_rows) / distinct_row_count if distinct_row_count else 0.0,
        "mode_counts": dict(sorted(mode_counts.items())),
    }


def build_sft_train_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt": row["prompt"],
            "completion": row["completion"],
        }
        for row in rows
    ]


def _render_chat_text(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    )


def _tokenize_text_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    return input_ids, offsets


def _find_loss_start_char(full_text: str, prompt_text: str, assistant_loss_mode: str) -> int:
    prompt_end = len(prompt_text)
    if assistant_loss_mode == "full":
        return prompt_end
    if assistant_loss_mode == "action_only":
        action_idx = full_text.rfind(ACTION_LINE_MARKER, prompt_end)
        if action_idx < 0:
            raise ValueError(
                f"assistant completion is missing required action marker {ACTION_LINE_MARKER!r} "
                f"for assistant_loss_mode={assistant_loss_mode}"
            )
        return action_idx
    raise ValueError(f"Unsupported --assistant-loss-mode value: {assistant_loss_mode}")


def _build_completion_mask(offsets: list[tuple[int, int]], loss_start_char: int) -> list[int]:
    return [1 if end > loss_start_char else 0 for _, end in offsets]


def build_tokenized_sft_dataset(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    assistant_loss_mode: str,
) -> list[dict[str, Any]]:
    tokenized_rows: list[dict[str, Any]] = []
    for row in rows:
        prompt_text = _render_chat_text(
            tokenizer,
            row["prompt"],
            add_generation_prompt=True,
        )
        full_text = _render_chat_text(
            tokenizer,
            row["prompt"] + row["completion"],
            add_generation_prompt=False,
        )

        prompt_ids, _ = _tokenize_text_with_offsets(tokenizer, prompt_text)
        full_ids, full_offsets = _tokenize_text_with_offsets(tokenizer, full_text)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "Tokenized prompt is not a prefix of tokenized prompt+completion. "
                "Verify chat-template and tokenizer consistency."
            )

        loss_start_char = _find_loss_start_char(full_text, prompt_text, assistant_loss_mode)
        completion_mask = _build_completion_mask(full_offsets, loss_start_char)
        if not any(completion_mask):
            raise ValueError(
                f"assistant_loss_mode={assistant_loss_mode} masked every token in the example"
            )
        tokenized_rows.append(
            {
                "input_ids": full_ids,
                "completion_mask": completion_mask,
            }
        )
    return tokenized_rows


def estimate_steps_per_epoch(
    *,
    row_count: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if per_device_train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    effective_rows_per_step = per_device_train_batch_size * gradient_accumulation_steps
    return max(1, math.ceil(row_count / effective_rows_per_step))


def summarize_sft_rows(rows: list[dict[str, Any]], train_jsonl: Path) -> dict[str, Any]:
    task_counts = Counter(str(row.get("task_id") or "unknown") for row in rows)
    trajectory_kind_counts = Counter(str(row.get("trajectory_kind") or "unknown") for row in rows)
    return {
        "train_jsonl": str(train_jsonl),
        "row_count": len(rows),
        "family_count": len({str(row.get("family_id") or "") for row in rows if row.get("family_id")}),
        "trajectory_count": len({str(row.get("trajectory_id") or "") for row in rows if row.get("trajectory_id")}),
        "task_counts": dict(sorted(task_counts.items())),
        "trajectory_kind_counts": dict(sorted(trajectory_kind_counts.items())),
        "max_prompt_messages": max((len(row["prompt"]) for row in rows), default=0),
        "max_completion_messages": max((len(row["completion"]) for row in rows), default=0),
    }


def build_train_summary(
    *,
    output_dir: Path,
    train_jsonl: Path,
    row_count: int,
    global_step: int,
    train_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = dict(train_metrics or {})
    return {
        "output_dir": str(output_dir),
        "train_jsonl": str(train_jsonl),
        "row_count": row_count,
        "global_step": int(global_step),
        "train_runtime": float(metrics.get("train_runtime", 0.0) or 0.0),
        "train_loss": float(metrics.get("train_loss", 0.0) or 0.0),
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def build_log_history(log_history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in log_history or []:
        if not isinstance(entry, dict):
            continue
        normalized.append(dict(entry))
    return normalized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA SFT trainer for DTC conversational datasets")

    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_NAME_OR_PATH)

    parser.add_argument(
        "--dataset-run-id",
        default=None,
        help=(
            "Canonical SFT dataset run id. Resolves input to "
            "data/dataset/sft/<run_id>/<usable_traj>/train.jsonl."
        ),
    )
    parser.add_argument(
        "--usable-traj",
        default=DEFAULT_USABLE_TRAJ,
        help=(
            "Canonical SFT subset under data/dataset/sft/<run_id>/. "
            "Ignored when --train-jsonl is used."
        ),
    )
    parser.add_argument(
        "--train-jsonl",
        default=None,
        help="Explicit conversational SFT JSONL path. Overrides canonical dataset resolution.",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help=(
            "Optional task filter for task-specific SFT. Accepts a canonical task id such as "
            "'BabyAI-MixedTrainLocal-v0/goto' or a unique short suffix such as 'goto'."
        ),
    )
    parser.add_argument(
        "--row-balance",
        choices=("none", "trajectory"),
        default="none",
        help=(
            "Optional row preprocessing before training. 'trajectory' keeps the total row count "
            "fixed while equalizing each trajectory's contribution within the selected dataset."
        ),
    )
    parser.add_argument(
        "--sample-plan-jsonl",
        default=None,
        help=(
            "Optional JSONL sampling plan. Each row must contain integer `row_idx` that indexes the "
            "post-filter/post-row-balance dataset rows. Allows deterministic resampling without copying train.jsonl."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Run output directory under data/. "
            "If omitted, defaults to data/checkpoints/manual/<run_name>. "
            "Relative non-data paths are interpreted under data/checkpoints/manual/."
        ),
    )
    parser.add_argument("--run-name", default="qwen35_0p8b_sft_lora")
    parser.add_argument(
        "--train-adapter-path",
        default=None,
        help=(
            "Existing LoRA adapter directory used to initialize trainable SFT adapter weights. "
            "This is not a trainer-state resume; use --resume-from-checkpoint for that."
        ),
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the SFT dataset schema and print dataset stats without importing training dependencies.",
    )

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)

    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dataloader-num-workers", type=int, default=8)

    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--save-epochs-fraction",
        type=float,
        default=None,
        help=(
            "If set, override --save-steps using this fraction of one epoch. "
            "For example 0.5 saves every half epoch."
        ),
    )
    parser.add_argument(
        "--stop-after-epochs",
        type=float,
        default=None,
        help="Stop after this many epochs while preserving the configured scheduler horizon.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=_parse_optional_int,
        default=8,
        help="Maximum checkpoints to keep. Use 'none' to disable checkpoint deletion.",
    )

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing (default: false).",
    )
    parser.add_argument(
        "--packing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable TRL packing for SFT sequences (default: false).",
    )
    parser.add_argument(
        "--completion-only-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train only on completion tokens from the prompt-completion dataset (default: true).",
    )
    parser.add_argument(
        "--assistant-loss-mode",
        choices=("full", "action_only"),
        default=DEFAULT_ASSISTANT_LOSS_MODE,
        help=(
            "Mask assistant tokens before loss computation. 'full' keeps the entire assistant completion trainable. "
            "'action_only' keeps only the final Action line trainable."
        ),
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Model loading dtype. If 'auto', follows --bf16/--fp16 when set.",
    )

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default=",".join(DEFAULT_LORA_TARGET_MODULES),
        help=(
            "LoRA target modules. Defaults to q_proj,k_proj,v_proj,o_proj for direct "
            "vLLM/BALROG compatibility. Supported aliases: 'qv', 'qvko', "
            "'qvko+mlp'. Use 'all-linear' or pass a comma-separated list for "
            "explicit experiments."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cuda-memory-log-jsonl",
        default=None,
        help="Optional JSONL path for CUDA memory records during training.",
    )
    parser.add_argument(
        "--cuda-memory-log-every-steps",
        type=int,
        default=0,
        help="If >0 and --cuda-memory-log-jsonl is set, record CUDA memory every N optimizer steps.",
    )
    parser.add_argument(
        "--cuda-memory-log-reset-peak-per-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset CUDA peak memory stats at each step when CUDA memory logging is enabled.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_paths = _resolve_input_paths(args)
    raw_rows = load_sft_rows(input_paths.train_jsonl)
    raw_dataset_summary = summarize_sft_rows(raw_rows, input_paths.train_jsonl)
    rows, resolved_task_id = preprocess_sft_rows(
        raw_rows,
        task_id=args.task_id,
        row_balance=args.row_balance,
    )
    preplan_row_count = len(rows)
    sample_plan_rows = load_sample_plan(input_paths.sample_plan_jsonl, row_count=preplan_row_count) if input_paths.sample_plan_jsonl else None
    rows = apply_sample_plan(rows, sample_plan_rows)
    dataset_summary = summarize_sft_rows(rows, input_paths.train_jsonl)
    sample_plan_summary = summarize_sample_plan(sample_plan_rows, source_row_count=preplan_row_count)
    steps_per_epoch = estimate_steps_per_epoch(
        row_count=len(rows),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    resolved_save_steps = args.save_steps
    if args.save_epochs_fraction is not None:
        if args.save_epochs_fraction <= 0:
            raise ValueError("--save-epochs-fraction must be positive")
        resolved_save_steps = max(1, math.ceil(steps_per_epoch * args.save_epochs_fraction))
    resolved_stop_after_steps: int | None = None
    if args.stop_after_epochs is not None:
        if args.stop_after_epochs <= 0:
            raise ValueError("--stop-after-epochs must be positive")
        resolved_stop_after_steps = max(1, math.ceil(steps_per_epoch * args.stop_after_epochs))

    if args.validate_only:
        payload = {
            "model_name_or_path": args.model_name_or_path,
            "input": {
                "dataset_run_id": args.dataset_run_id,
                "usable_traj": args.usable_traj,
                "task_id": args.task_id,
                "resolved_task_id": resolved_task_id,
                "row_balance": args.row_balance,
                "assistant_loss_mode": args.assistant_loss_mode,
                "sample_plan_jsonl": str(input_paths.sample_plan_jsonl) if input_paths.sample_plan_jsonl else None,
            },
            "lora_target_modules": _parse_lora_target_modules(args.lora_target_modules),
            "raw_dataset_summary": raw_dataset_summary,
            "dataset_summary": dataset_summary,
            "sample_plan_summary": sample_plan_summary,
            "steps_per_epoch": steps_per_epoch,
            "resolved_save_steps": resolved_save_steps,
            "resolved_stop_after_steps": resolved_stop_after_steps,
        }
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers.trainer_callback import ProgressCallback, TrainerCallback
        from trl import SFTConfig, SFTTrainer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SFT training requires `torch`, `datasets`, `transformers`, `peft`, and `trl`. "
            "For schema-only checks, run with --validate-only."
        ) from exc

    torch_dtype_name = _resolve_torch_dtype(args)
    model_load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch_dtype_name is not None:
        model_load_kwargs["torch_dtype"] = getattr(torch, torch_dtype_name)

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for SFT")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "Tokenizer must provide a chat template for conversational SFT training. "
            "Use an instruct/chat model tokenizer for this entrypoint."
        )

    if _use_qwen35_text_backbone(config):
        model = _qwen35_causal_lm_class().from_pretrained(
            args.model_name_or_path,
            config=config.text_config,
            **model_load_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_load_kwargs,
        )

    if args.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if args.assistant_loss_mode != DEFAULT_ASSISTANT_LOSS_MODE and not args.completion_only_loss:
        raise ValueError("--assistant-loss-mode requires --completion-only-loss")

    train_adapter_path = Path(args.train_adapter_path) if args.train_adapter_path else None
    if train_adapter_path is not None:
        if not train_adapter_path.exists():
            raise FileNotFoundError(f"--train-adapter-path does not exist: {train_adapter_path}")
        if not (train_adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing adapter_config.json under --train-adapter-path: {train_adapter_path}")
        model = PeftModel.from_pretrained(model, str(train_adapter_path), is_trainable=True)

    train_dataset = Dataset.from_list(
        build_tokenized_sft_dataset(
            rows,
            tokenizer,
            assistant_loss_mode=args.assistant_loss_mode,
        )
    )

    peft_config = None
    if train_adapter_path is None:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=_parse_lora_target_modules(args.lora_target_modules),
        )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_steps=resolved_save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        completion_only_loss=args.completion_only_loss,
        packing=args.packing,
        seed=args.seed,
        report_to="none",
        eval_strategy="no",
    )

    run_metadata = {
        "model_name_or_path": args.model_name_or_path,
        "input": {
            "train_jsonl": str(input_paths.train_jsonl),
            "dataset_run_id": args.dataset_run_id,
            "usable_traj": args.usable_traj,
            "task_id": args.task_id,
            "resolved_task_id": resolved_task_id,
            "row_balance": args.row_balance,
            "sample_plan_jsonl": str(input_paths.sample_plan_jsonl) if input_paths.sample_plan_jsonl else None,
        },
        "trainer": {
            "train_adapter_path": str(train_adapter_path) if train_adapter_path else None,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "warmup_ratio": args.warmup_ratio,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "steps_per_epoch_estimate": steps_per_epoch,
            "max_length": args.max_length,
            "dataloader_num_workers": args.dataloader_num_workers,
            "packing": args.packing,
            "completion_only_loss": args.completion_only_loss,
            "assistant_loss_mode": args.assistant_loss_mode,
            "gradient_checkpointing": args.gradient_checkpointing,
            "bf16": args.bf16,
            "fp16": args.fp16,
            "torch_dtype": torch_dtype_name,
            "save_steps_requested": args.save_steps,
            "save_steps_resolved": resolved_save_steps,
            "save_epochs_fraction": args.save_epochs_fraction,
            "stop_after_epochs": args.stop_after_epochs,
            "stop_after_steps_resolved": resolved_stop_after_steps,
            "save_total_limit": args.save_total_limit,
            "cuda_memory_log_jsonl": args.cuda_memory_log_jsonl,
            "cuda_memory_log_every_steps": args.cuda_memory_log_every_steps,
            "cuda_memory_log_reset_peak_per_step": args.cuda_memory_log_reset_peak_per_step,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": _parse_lora_target_modules(args.lora_target_modules),
        },
        "raw_dataset_summary": raw_dataset_summary,
        "dataset_summary": dataset_summary,
        "sample_plan_summary": sample_plan_summary,
    }
    _write_json(output_dir / "run_metadata.json", run_metadata)

    class SilentProgressCallback(ProgressCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
            del args, state, logs, kwargs
            return control

    class StopAfterEpochCallback(TrainerCallback):
        def __init__(self, *, stop_after_epochs: float) -> None:
            self._stop_after_epochs = stop_after_epochs

        def _should_stop(self, state: Any) -> bool:
            epoch = float(getattr(state, "epoch", 0.0) or 0.0)
            return epoch + 1e-9 >= self._stop_after_epochs

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if self._should_stop(state):
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if self._should_stop(state):
                control.should_training_stop = True
            return control

    class CUDAMemoryLogCallback(TrainerCallback):
        def __init__(
            self,
            *,
            torch_module: Any,
            output_path: Path,
            every_steps: int,
            reset_peak_per_step: bool,
        ) -> None:
            self._torch = torch_module
            self._output_path = output_path
            self._every_steps = every_steps
            self._reset_peak_per_step = reset_peak_per_step
            self._enabled = self._torch.cuda.is_available()

        def _record(self, *, label: str, state: Any) -> None:
            if not self._enabled:
                return
            self._torch.cuda.synchronize()
            stats = self._torch.cuda.memory_stats()
            payload = {
                "label": label,
                "global_step": int(getattr(state, "global_step", 0) or 0),
                "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
                "allocated_bytes": int(self._torch.cuda.memory_allocated()),
                "reserved_bytes": int(self._torch.cuda.memory_reserved()),
                "max_allocated_bytes": int(self._torch.cuda.max_memory_allocated()),
                "max_reserved_bytes": int(self._torch.cuda.max_memory_reserved()),
                "active_bytes_current": int(stats.get("active_bytes.all.current", 0)),
                "active_bytes_peak": int(stats.get("active_bytes.all.peak", 0)),
                "inactive_split_bytes_current": int(stats.get("inactive_split_bytes.all.current", 0)),
                "requested_bytes_current": int(stats.get("requested_bytes.all.current", 0)),
            }
            _append_jsonl(self._output_path, payload)

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if self._enabled:
                self._torch.cuda.reset_peak_memory_stats()
            self._record(label="train_begin", state=state)
            return control

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if self._enabled and self._reset_peak_per_step:
                self._torch.cuda.reset_peak_memory_stats()
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if self._every_steps > 0 and int(getattr(state, "global_step", 0) or 0) % self._every_steps == 0:
                self._record(label="step_end", state=state)
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            self._record(label="save", state=state)
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            self._record(label="train_end", state=state)
            return control

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(SilentProgressCallback)
    if args.stop_after_epochs is not None:
        trainer.add_callback(StopAfterEpochCallback(stop_after_epochs=args.stop_after_epochs))
    if args.cuda_memory_log_jsonl:
        if args.cuda_memory_log_every_steps < 0:
            raise ValueError("--cuda-memory-log-every-steps must be >= 0")
        memory_log_path = Path(args.cuda_memory_log_jsonl)
        if memory_log_path.exists():
            memory_log_path.unlink()
        trainer.add_callback(
            CUDAMemoryLogCallback(
                torch_module=torch,
                output_path=memory_log_path,
                every_steps=args.cuda_memory_log_every_steps,
                reset_peak_per_step=args.cuda_memory_log_reset_peak_per_step,
            )
        )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    _write_json(
        output_dir / "log_history.json",
        build_log_history(getattr(trainer.state, "log_history", None)),
    )

    train_summary = build_train_summary(
        output_dir=output_dir,
        train_jsonl=input_paths.train_jsonl,
        row_count=len(rows),
        global_step=int(trainer.state.global_step),
        train_metrics=getattr(train_result, "metrics", None),
    )
    _write_json(output_dir / "train_summary.json", train_summary)


if __name__ == "__main__":
    main()
