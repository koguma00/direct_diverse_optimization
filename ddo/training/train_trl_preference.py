#!/usr/bin/env python3
"""Research preference trainer for DPO, DPO-RK/DPO-D, DivPO, direct DDO, and DDO-bg-v2.

Objectives:
- DPO: train on win_lose pairs with the DPO logistic objective.
- DPO-RK/DPO-D: train on win_lose pairs as wins and optional win_win pairs as ties
  with Rao-Kupper or Davidson tie-aware preference likelihoods.
- DivPO: train on diversity-selected win_lose pairs with the same DPO logistic objective.
- DDO: train on win_lose + win_win pairs with a hybrid objective.
- DDO-bg-v2: train on flat mixed win_lose + win_win rows with a soft-target DPO objective.

DDO implementation note:
- Win-Lose pairs optimize the DPO preference term.
- Win-Win pairs optimize either the legacy symmetric reward-gap balancing term
  or an optional upward-matching + success-floor variant.
- The reward-like score is r_theta(x, y) = log pi_theta(y|x) - log pi_ref(y|x).

DDO-bg-v2 implementation note:
- Reuses balanced-geometric builder rows with precomputed `chosen_target_prob`.
- Uses a single row-wise soft-label DPO loss on the flat mixed dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DATASET_ROOT = REPO_ROOT / "data" / "dataset"
CANONICAL_DPO_DATASET_ROOT = CANONICAL_DATASET_ROOT / "dpo"
CANONICAL_DDO_DATASET_ROOT = CANONICAL_DATASET_ROOT / "ddo"
CANONICAL_CHECKPOINT_ROOT = REPO_ROOT / "data" / "checkpoints" / "manual"
CANONICAL_PAIR_FILENAME = "train_pairs.jsonl"

DEFAULT_ATTENTION_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]
DEFAULT_TEXT_LORA_TARGET_MODULES = [
    *DEFAULT_ATTENTION_LORA_TARGET_MODULES,
    "gate_proj",
    "up_proj",
    "down_proj",
]
LORA_TARGET_PROFILES: dict[str, list[str]] = {
    "qv": [
        "q_proj",
        "v_proj",
    ],
    "qvko": list(DEFAULT_ATTENTION_LORA_TARGET_MODULES),
    "qvko+mlp": list(DEFAULT_TEXT_LORA_TARGET_MODULES),
}

# Keep the default LoRA surface on the standard attention/MLP projections.
# Qwen3.5-specific linear-attention projections remain available via
# --lora-target-modules when explicitly needed for an experiment.
DEFAULT_QWEN3_5_LORA_TARGET_MODULES = list(DEFAULT_TEXT_LORA_TARGET_MODULES)

PAIR_TYPE_WIN_LOSE = "win_lose"
PAIR_TYPE_WIN_WIN = "win_win"
ROW_FIELD_CHOSEN_TARGET_PROB = "chosen_target_prob"
PAIR_TYPE_TO_ID = {
    PAIR_TYPE_WIN_LOSE: 0,
    PAIR_TYPE_WIN_WIN: 1,
}

MODE_DPO = "dpo"
MODE_DIVPO = "divpo"
MODE_DPO_RK = "dpo_rk"
MODE_DPO_D = "dpo_d"
MODE_DDO = "ddo"
MODE_DDO_BG_V2 = "ddo_bg_v2"
TIE_AWARE_DPO_MODES = {MODE_DPO_RK, MODE_DPO_D}
MODES_REQUIRING_WIN_WIN = {MODE_DDO, MODE_DDO_BG_V2}

DDO_LOSS_VARIANT_SYMMETRIC_GAP = "symmetric_gap"
DDO_LOSS_VARIANT_UPWARD_FLOOR = "upward_floor"

REQUIRED_PAIR_FIELDS = (
    "pair_type",
    "prompt",
    "chosen",
    "rejected",
)

REQUIRED_ACTION_FIELDS = (
    "action_text",
    "success",
)


@dataclass(frozen=True)
class TrainInputPaths:
    win_lose_jsonl: Path
    win_win_jsonl: Path | None


class PairSchemaError(ValueError):
    pass


def _parse_optional_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"none", "null"}:
        return None
    return int(normalized)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_multimodal_config(config: Any) -> bool:
    return hasattr(config, "vision_config")


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


def _default_lora_target_modules(config: Any) -> list[str]:
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3_5":
        return list(DEFAULT_QWEN3_5_LORA_TARGET_MODULES)
    if getattr(config, "text_config", None) is not None and getattr(config.text_config, "model_type", None) == "qwen3_5_text":
        return list(DEFAULT_QWEN3_5_LORA_TARGET_MODULES)
    return list(DEFAULT_TEXT_LORA_TARGET_MODULES)


def _parse_lora_target_modules(value: str | None, config: Any) -> list[str]:
    if value is None or not value.strip():
        return _default_lora_target_modules(config)
    normalized = value.strip()
    profile = LORA_TARGET_PROFILES.get(normalized)
    if profile is not None:
        return list(profile)
    if normalized.replace(" ", "") == "qvko+gate_proj,up_proj,down_proj":
        return list(LORA_TARGET_PROFILES["qvko+mlp"])
    return [item.strip() for item in normalized.split(",") if item.strip()]


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PairSchemaError(f"{path}:{line_idx} invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise PairSchemaError(f"{path}:{line_idx} row must be an object")
            rows.append(row)
    if not rows:
        raise PairSchemaError(f"{path} has no valid JSONL rows")
    return rows


def _canonical_pair_path(root: Path, run_id: str) -> Path:
    run_id = run_id.strip()
    if not run_id:
        raise ValueError("dataset run id must be a non-empty string")
    return root / run_id / CANONICAL_PAIR_FILENAME


def _resolve_input_paths(args: argparse.Namespace) -> TrainInputPaths:
    dataset_run_id = getattr(args, "dataset_run_id", None)
    win_lose_jsonl = getattr(args, "win_lose_jsonl", None)
    win_win_jsonl = getattr(args, "win_win_jsonl", None)

    if win_lose_jsonl:
        resolved_win_lose = Path(win_lose_jsonl)
    elif dataset_run_id and args.mode != MODE_DIVPO:
        resolved_win_lose = _canonical_pair_path(CANONICAL_DPO_DATASET_ROOT, dataset_run_id)
    else:
        raise ValueError(
            "Provide --win-lose-jsonl, or use --dataset-run-id for dpo/ddo canonical datasets"
        )

    resolved_win_win: Path | None = None
    if args.mode in MODES_REQUIRING_WIN_WIN or args.mode in TIE_AWARE_DPO_MODES:
        if win_win_jsonl:
            resolved_win_win = Path(win_win_jsonl)
        elif dataset_run_id:
            if args.mode == MODE_DDO_BG_V2:
                raise ValueError(
                    "ddo_bg_v2 requires explicit --win-win-jsonl"
                )
            resolved_win_win = _canonical_pair_path(CANONICAL_DDO_DATASET_ROOT, dataset_run_id)
        elif args.mode in MODES_REQUIRING_WIN_WIN:
            raise ValueError(
                f"{args.mode} mode requires --win-win-jsonl"
            )

    return TrainInputPaths(
        win_lose_jsonl=resolved_win_lose,
        win_win_jsonl=resolved_win_win,
    )


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if not getattr(args, "run_name", None):
        raise ValueError("--run-name must be a non-empty string")

    raw_output_dir = getattr(args, "output_dir", None)
    if raw_output_dir:
        candidate = Path(raw_output_dir)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        elif candidate.parts[:1] == ("data",):
            resolved = (REPO_ROOT / candidate).resolve()
        else:
            resolved = (CANONICAL_CHECKPOINT_ROOT / candidate).resolve()
    else:
        resolved = (CANONICAL_CHECKPOINT_ROOT / args.run_name).resolve()

    checkpoint_root = CANONICAL_CHECKPOINT_ROOT.resolve()
    project_root = REPO_ROOT.resolve()
    if not _is_relative_to(resolved, project_root):
        raise ValueError(
            f"--output-dir must resolve under repository root {project_root} "
            f"(default checkpoint root: {checkpoint_root})"
        )
    return resolved


def _checkpoint_step(path: Path) -> int | None:
    if not path.is_dir():
        return None
    if not path.name.startswith("checkpoint-"):
        return None
    suffix = path.name.removeprefix("checkpoint-")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    latest_path: Path | None = None
    latest_step = -1
    if not output_dir.exists():
        return None
    for child in output_dir.iterdir():
        step = _checkpoint_step(child)
        if step is None:
            continue
        if step > latest_step:
            latest_path = child
            latest_step = step
    return latest_path


def _resolve_resume_from_checkpoint(output_dir: Path, resume_from_checkpoint: str | None) -> Path | None:
    if resume_from_checkpoint is None:
        return None
    normalized = resume_from_checkpoint.strip()
    if not normalized:
        return None
    if normalized == "latest":
        latest_checkpoint = _find_latest_checkpoint(output_dir)
        if latest_checkpoint is None:
            raise ValueError(
                f"--resume-from-checkpoint latest requested, but no checkpoint-* directory exists under {output_dir}"
            )
        return latest_checkpoint

    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = output_dir / candidate
    resolved = candidate.resolve()
    if not resolved.exists():
        raise ValueError(f"--resume-from-checkpoint path does not exist: {resolved}")
    return resolved


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


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


def build_log_history(log_history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in log_history or []:
        if not isinstance(entry, dict):
            continue
        normalized.append(dict(entry))
    return normalized


def build_train_summary(
    *,
    mode: str,
    output_dir: Path,
    win_lose_jsonl: Path,
    win_win_jsonl: Path | None,
    train_record_count: int,
    win_lose_pair_count: int,
    win_win_pair_count: int,
    global_step: int,
    train_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = dict(train_metrics or {})
    return {
        "mode": mode,
        "output_dir": str(output_dir),
        "win_lose_jsonl": str(win_lose_jsonl),
        "win_win_jsonl": str(win_win_jsonl) if win_win_jsonl else None,
        "train_record_count": int(train_record_count),
        "win_lose_pair_count": int(win_lose_pair_count),
        "win_win_pair_count": int(win_win_pair_count),
        "global_step": int(global_step),
        "train_runtime": float(metrics.get("train_runtime", 0.0) or 0.0),
        "train_loss": float(metrics.get("train_loss", 0.0) or 0.0),
    }


def _validate_pair_row(
    row: dict[str, Any],
    source: Path,
    line_idx: int,
    *,
    require_target_prob: bool = False,
) -> dict[str, Any]:
    missing = [k for k in REQUIRED_PAIR_FIELDS if k not in row]
    if missing:
        raise PairSchemaError(
            f"{source}:{line_idx} missing required fields: {', '.join(missing)}"
        )

    if not isinstance(row["prompt"], str) or not row["prompt"].strip():
        raise PairSchemaError(f"{source}:{line_idx} field 'prompt' must be non-empty string")

    for side in ("chosen", "rejected"):
        if not isinstance(row[side], dict):
            raise PairSchemaError(f"{source}:{line_idx} field '{side}' must be an object")
        missing_action = [k for k in REQUIRED_ACTION_FIELDS if k not in row[side]]
        if missing_action:
            raise PairSchemaError(
                f"{source}:{line_idx} field '{side}' missing: {', '.join(missing_action)}"
            )
        if not isinstance(row[side]["action_text"], str) or not row[side]["action_text"].strip():
            raise PairSchemaError(
                f"{source}:{line_idx} field '{side}.action_text' must be non-empty string"
            )

    validated = dict(row)
    if ROW_FIELD_CHOSEN_TARGET_PROB in row:
        try:
            chosen_target_prob = float(row[ROW_FIELD_CHOSEN_TARGET_PROB])
        except (TypeError, ValueError) as exc:
            raise PairSchemaError(
                f"{source}:{line_idx} field '{ROW_FIELD_CHOSEN_TARGET_PROB}' must be a float in [0, 1]"
            ) from exc
        if not (0.0 <= chosen_target_prob <= 1.0):
            raise PairSchemaError(
                f"{source}:{line_idx} field '{ROW_FIELD_CHOSEN_TARGET_PROB}' must be in [0, 1]"
            )
        validated[ROW_FIELD_CHOSEN_TARGET_PROB] = chosen_target_prob
    elif require_target_prob:
        raise PairSchemaError(
            f"{source}:{line_idx} missing required field: {ROW_FIELD_CHOSEN_TARGET_PROB}"
        )
    else:
        validated[ROW_FIELD_CHOSEN_TARGET_PROB] = 1.0

    return validated


def _as_pref_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_type": row["pair_type"],
        "prompt": row["prompt"],
        "chosen": row["chosen"]["action_text"],
        "rejected": row["rejected"]["action_text"],
        ROW_FIELD_CHOSEN_TARGET_PROB: float(row.get(ROW_FIELD_CHOSEN_TARGET_PROB, 1.0)),
    }


def _extract_pair_type(
    rows: list[dict[str, Any]],
    source: Path,
    *,
    require_target_prob: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    win_lose: list[dict[str, Any]] = []
    win_win: list[dict[str, Any]] = []
    for idx, raw_row in enumerate(rows, start=1):
        row = _validate_pair_row(
            raw_row,
            source,
            idx,
            require_target_prob=require_target_prob,
        )
        pair_type = row["pair_type"]
        if pair_type == PAIR_TYPE_WIN_LOSE:
            win_lose.append(row)
        elif pair_type == PAIR_TYPE_WIN_WIN:
            win_win.append(row)
        else:
            raise PairSchemaError(
                f"{source}:{idx} unsupported pair_type='{pair_type}' (expected win_lose/win_win)"
            )
    return win_lose, win_win


def _load_pair_rows(mode: str, paths: TrainInputPaths) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    win_lose_raw = _read_jsonl(paths.win_lose_jsonl)
    require_target_prob = mode == MODE_DDO_BG_V2
    win_lose_rows, win_lose_win_win_rows = _extract_pair_type(
        win_lose_raw,
        paths.win_lose_jsonl,
        require_target_prob=require_target_prob,
    )

    if win_lose_win_win_rows:
        raise PairSchemaError(
            f"{paths.win_lose_jsonl} contains win_win pairs; win_lose input must be win_lose only"
        )

    if not win_lose_rows:
        raise PairSchemaError(f"{paths.win_lose_jsonl} has no win_lose rows")

    if mode in {MODE_DPO, MODE_DIVPO} or (mode in TIE_AWARE_DPO_MODES and paths.win_win_jsonl is None):
        return win_lose_rows, []

    if paths.win_win_jsonl is None:
        raise ValueError(f"{mode} mode requires --win-win-jsonl")

    win_win_raw = _read_jsonl(paths.win_win_jsonl)
    win_win_win_lose_rows, win_win_rows = _extract_pair_type(
        win_win_raw,
        paths.win_win_jsonl,
        require_target_prob=require_target_prob,
    )

    if win_win_win_lose_rows:
        raise PairSchemaError(
            f"{paths.win_win_jsonl} contains win_lose pairs; win_win input must be win_win only"
        )

    if not win_win_rows:
        raise PairSchemaError(f"{paths.win_win_jsonl} has no win_win rows")

    return win_lose_rows, win_win_rows


def _build_dpo_dataset(win_lose_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_as_pref_record(row) for row in win_lose_rows]


def _build_ddo_bg_v2_dataset(
    win_lose_rows: list[dict[str, Any]],
    win_win_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [_as_pref_record(row) for row in win_lose_rows] + [_as_pref_record(row) for row in win_win_rows]


def _build_tie_aware_dpo_dataset(
    win_lose_rows: list[dict[str, Any]],
    win_win_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [_as_pref_record(row) for row in win_lose_rows] + [_as_pref_record(row) for row in win_win_rows]


def _build_ddo_dataset(
    win_lose_rows: list[dict[str, Any]],
    win_win_rows: list[dict[str, Any]],
    win_win_repeats: int,
) -> list[dict[str, Any]]:
    if win_win_repeats < 1:
        raise ValueError("win_win_repeats must be >= 1")

    records = [_as_pref_record(row) for row in win_lose_rows]
    for _ in range(win_win_repeats):
        records.extend(_as_pref_record(row) for row in win_win_rows)
    return records


def load_preference_records(
    mode: str,
    paths: TrainInputPaths,
    win_win_repeats: int,
) -> list[dict[str, Any]]:
    win_lose_rows, win_win_rows = _load_pair_rows(mode, paths)
    if mode in {MODE_DPO, MODE_DIVPO}:
        return _build_dpo_dataset(win_lose_rows)
    if mode in TIE_AWARE_DPO_MODES:
        return _build_tie_aware_dpo_dataset(win_lose_rows, win_win_rows)
    if mode == MODE_DDO_BG_V2:
        return _build_ddo_bg_v2_dataset(win_lose_rows, win_win_rows)
    return _build_ddo_dataset(win_lose_rows, win_win_rows, win_win_repeats)


def _append_eos_token(completion_ids: list[int], eos_token_id: int | None, max_completion_length: int) -> list[int]:
    if max_completion_length <= 0:
        raise ValueError("max_length is too small to keep any completion tokens")
    if eos_token_id is None:
        return completion_ids[:max_completion_length]
    if max_completion_length == 1:
        return [eos_token_id]
    return completion_ids[: max_completion_length - 1] + [eos_token_id]


def _tokenize_response_pair(
    record: dict[str, Any],
    *,
    tokenizer: Any,
    max_prompt_length: int,
    max_length: int,
) -> dict[str, Any]:
    prompt_ids = tokenizer(record["prompt"], add_special_tokens=False)["input_ids"]
    prompt_ids = prompt_ids[-max_prompt_length:]
    if len(prompt_ids) >= max_length:
        prompt_ids = prompt_ids[-(max_length - 1) :]

    def build_side(text: str) -> tuple[list[int], list[int], list[int]]:
        completion_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        max_completion_length = max_length - len(prompt_ids)
        completion_ids = _append_eos_token(completion_ids, tokenizer.eos_token_id, max_completion_length)
        input_ids = prompt_ids + completion_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + completion_ids
        return input_ids, attention_mask, labels

    chosen_input_ids, chosen_attention_mask, chosen_labels = build_side(record["chosen"])
    rejected_input_ids, rejected_attention_mask, rejected_labels = build_side(record["rejected"])

    return {
        "pair_type_id": PAIR_TYPE_TO_ID[record["pair_type"]],
        ROW_FIELD_CHOSEN_TARGET_PROB: float(record.get(ROW_FIELD_CHOSEN_TARGET_PROB, 1.0)),
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
        "rejected_labels": rejected_labels,
    }


def _tokenize_preference_records(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_prompt_length: int,
    max_length: int,
) -> list[dict[str, Any]]:
    return [
        _tokenize_response_pair(
            record,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            max_length=max_length,
        )
        for record in records
    ]


def _pad_pair_batch_tensors(inputs: dict[str, Any], pad_token_id: int) -> tuple[Any, Any, Any]:
    import torch
    import torch.nn.functional as F

    max_len = max(
        inputs["chosen_input_ids"].shape[1],
        inputs["rejected_input_ids"].shape[1],
    )

    def _pad_tensor(tensor: Any, pad_value: int) -> Any:
        pad_width = max_len - tensor.shape[1]
        if pad_width <= 0:
            return tensor
        return F.pad(tensor, (0, pad_width), value=pad_value)

    concatenated_input_ids = torch.cat(
        [
            _pad_tensor(inputs["chosen_input_ids"], pad_token_id),
            _pad_tensor(inputs["rejected_input_ids"], pad_token_id),
        ],
        dim=0,
    )
    concatenated_attention_mask = torch.cat(
        [
            _pad_tensor(inputs["chosen_attention_mask"], 0),
            _pad_tensor(inputs["rejected_attention_mask"], 0),
        ],
        dim=0,
    )
    concatenated_labels = torch.cat(
        [
            _pad_tensor(inputs["chosen_labels"], -100),
            _pad_tensor(inputs["rejected_labels"], -100),
        ],
        dim=0,
    )
    return concatenated_input_ids, concatenated_attention_mask, concatenated_labels


def _compute_loss_terms(
    chosen_rewards: Any,
    rejected_rewards: Any,
    pair_type_ids: Any,
    chosen_target_probs: Any,
    beta: float,
    lambda_div: float,
    ddo_loss_variant: str,
    lambda_floor: float,
    mode: str,
    dpo_rk_alpha: float = math.log(3.0),
    dpo_d_nu: float = 1.0,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    reward_gap = chosen_rewards - rejected_rewards
    zero = reward_gap.new_zeros(())
    pref_mask = pair_type_ids == PAIR_TYPE_TO_ID[PAIR_TYPE_WIN_LOSE]
    div_mask = pair_type_ids == PAIR_TYPE_TO_ID[PAIR_TYPE_WIN_WIN]

    if mode in TIE_AWARE_DPO_MODES:
        d_theta = beta * reward_gap
        if mode == MODE_DPO_RK:
            if dpo_rk_alpha <= 0.0:
                raise ValueError("dpo_rk_alpha must be positive")
            alpha = float(dpo_rk_alpha)
            log_tie_factor = math.log(math.expm1(2.0 * alpha))
            pref_row_loss = -F.logsigmoid(d_theta - alpha)
            div_row_loss = (
                -log_tie_factor
                - F.logsigmoid(-d_theta - alpha)
                - F.logsigmoid(d_theta - alpha)
            )
        else:
            if dpo_d_nu <= 0.0:
                raise ValueError("dpo_d_nu must be positive")
            log_2nu = math.log(2.0 * float(dpo_d_nu))
            log_denominator = torch.logsumexp(
                torch.stack(
                    [
                        torch.zeros_like(d_theta),
                        -d_theta,
                        d_theta.new_full(d_theta.shape, log_2nu) - (0.5 * d_theta),
                    ],
                    dim=0,
                ),
                dim=0,
            )
            pref_row_loss = log_denominator
            div_row_loss = log_denominator - (log_2nu - (0.5 * d_theta))

        row_loss = torch.where(pref_mask, pref_row_loss, div_row_loss)
        pref_loss = row_loss[pref_mask].mean() if pref_mask.any() else zero
        div_loss = row_loss[div_mask].mean() if div_mask.any() else zero
        pref_reward_gap = reward_gap[pref_mask].mean() if pref_mask.any() else zero
        div_reward_gap = reward_gap[div_mask].mean() if div_mask.any() else zero
        return {
            "loss": row_loss.mean() if row_loss.numel() else zero,
            "pref_loss": pref_loss,
            "div_loss": div_loss,
            "div_match_loss": zero,
            "div_floor_loss": zero,
            "pref_reward_gap": pref_reward_gap,
            "div_reward_gap": div_reward_gap,
            "div_reward_gap_sq": reward_gap[div_mask].square().mean() if div_mask.any() else zero,
            "div_reward_hi": chosen_rewards[div_mask].maximum(rejected_rewards[div_mask]).mean() if div_mask.any() else zero,
            "div_reward_lo": chosen_rewards[div_mask].minimum(rejected_rewards[div_mask]).mean() if div_mask.any() else zero,
            "div_floor_target": zero,
            "div_target_prob_mean": chosen_target_probs[div_mask].mean() if div_mask.any() else zero,
            "num_pref_pairs": int(pref_mask.sum().item()),
            "num_div_pairs": int(div_mask.sum().item()),
        }

    if mode == MODE_DDO_BG_V2:
        row_loss = (
            chosen_target_probs * -F.logsigmoid(beta * reward_gap)
            + (1.0 - chosen_target_probs) * -F.logsigmoid(-beta * reward_gap)
        )
        pref_loss = row_loss[pref_mask].mean() if pref_mask.any() else zero
        div_loss = row_loss[div_mask].mean() if div_mask.any() else zero
        pref_reward_gap = reward_gap[pref_mask].mean() if pref_mask.any() else zero
        div_reward_gap = reward_gap[div_mask].mean() if div_mask.any() else zero
        return {
            "loss": row_loss.mean() if row_loss.numel() else zero,
            "pref_loss": pref_loss,
            "div_loss": div_loss,
            "div_match_loss": zero,
            "div_floor_loss": zero,
            "pref_reward_gap": pref_reward_gap,
            "div_reward_gap": div_reward_gap,
            "div_reward_gap_sq": reward_gap[div_mask].square().mean() if div_mask.any() else zero,
            "div_reward_hi": chosen_rewards[div_mask].maximum(rejected_rewards[div_mask]).mean() if div_mask.any() else zero,
            "div_reward_lo": chosen_rewards[div_mask].minimum(rejected_rewards[div_mask]).mean() if div_mask.any() else zero,
            "div_floor_target": zero,
            "div_target_prob_mean": chosen_target_probs[div_mask].mean() if div_mask.any() else zero,
            "num_pref_pairs": int(pref_mask.sum().item()),
            "num_div_pairs": int(div_mask.sum().item()),
        }

    pref_loss = zero
    div_loss = zero
    div_match_loss = zero
    div_floor_loss = zero
    pref_reward_gap = zero
    div_reward_gap = zero
    div_reward_gap_sq = zero
    div_reward_hi = zero
    div_reward_lo = zero
    div_floor_target = zero

    if pref_mask.any():
        pref_gap = reward_gap[pref_mask]
        pref_loss = -F.logsigmoid(beta * pref_gap).mean()
        pref_reward_gap = pref_gap.mean()

    if div_mask.any():
        div_chosen = chosen_rewards[div_mask]
        div_rejected = rejected_rewards[div_mask]
        div_gap = reward_gap[div_mask]
        div_reward_gap = div_gap.mean()
        div_reward_gap_sq = div_gap.square().mean()
        div_hi = div_chosen.maximum(div_rejected)
        div_lo = div_chosen.minimum(div_rejected)
        div_reward_hi = div_hi.mean()
        div_reward_lo = div_lo.mean()
        if ddo_loss_variant == DDO_LOSS_VARIANT_SYMMETRIC_GAP:
            div_match_loss = div_gap.square().mean()
            div_loss = div_match_loss
        elif ddo_loss_variant == DDO_LOSS_VARIANT_UPWARD_FLOOR:
            # Pull the lower-successful branch up toward the stronger one,
            # without letting the stronger branch be pushed down by this term.
            div_match_loss = (div_lo - div_hi.detach()).square().mean()
            if pref_mask.any():
                div_floor_target = chosen_rewards[pref_mask].detach().mean()
            else:
                div_floor_target = div_hi.detach().mean()
            div_floor_loss = (
                (div_floor_target - div_chosen).clamp_min(0.0).square() +
                (div_floor_target - div_rejected).clamp_min(0.0).square()
            ).mean() * 0.5
            div_loss = div_match_loss + (lambda_floor * div_floor_loss)
        else:
            raise ValueError(f"Unsupported DDO loss variant: {ddo_loss_variant}")

    total_loss = pref_loss + (lambda_div * div_loss)
    return {
        "loss": total_loss,
        "pref_loss": pref_loss,
        "div_loss": div_loss,
        "div_match_loss": div_match_loss,
        "div_floor_loss": div_floor_loss,
        "pref_reward_gap": pref_reward_gap,
        "div_reward_gap": div_reward_gap,
        "div_reward_gap_sq": div_reward_gap_sq,
        "div_reward_hi": div_reward_hi,
        "div_reward_lo": div_reward_lo,
        "div_floor_target": div_floor_target,
        "div_target_prob_mean": zero,
        "num_pref_pairs": int(pref_mask.sum().item()),
        "num_div_pairs": int(div_mask.sum().item()),
    }


class PreferenceDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def _pad(self, sequences: list[list[int]], pad_value: int):
        import torch

        max_len = max(len(seq) for seq in sequences)
        padded = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch = {
            "pair_type_ids": torch.tensor([feature["pair_type_id"] for feature in features], dtype=torch.long),
            "chosen_target_probs": torch.tensor(
                [feature[ROW_FIELD_CHOSEN_TARGET_PROB] for feature in features],
                dtype=torch.float32,
            ),
        }
        for prefix in ("chosen", "rejected"):
            batch[f"{prefix}_input_ids"] = self._pad(
                [feature[f"{prefix}_input_ids"] for feature in features],
                self.pad_token_id,
            )
            batch[f"{prefix}_attention_mask"] = self._pad(
                [feature[f"{prefix}_attention_mask"] for feature in features],
                0,
            )
            batch[f"{prefix}_labels"] = self._pad(
                [feature[f"{prefix}_labels"] for feature in features],
                -100,
            )
        return batch

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research preference trainer for DPO/DPO-RK/DPO-D/DivPO/DDO/DDO-bg-v2")

    parser.add_argument(
        "--mode",
        choices=(MODE_DPO, MODE_DIVPO, MODE_DPO_RK, MODE_DPO_D, MODE_DDO, MODE_DDO_BG_V2),
        required=True,
    )
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--ref-model-name-or-path", default=None)
    parser.add_argument("--finetune-type", choices=("full", "lora"), default="full")
    parser.add_argument(
        "--train-adapter-path",
        default=None,
        help=(
            "Optional PEFT adapter path used to initialize the trainable policy model. "
            "Requires --finetune-type lora."
        ),
    )
    parser.add_argument(
        "--ref-adapter-path",
        default=None,
        help=(
            "Optional PEFT adapter path used to initialize the frozen reference model. "
            "Defaults to --train-adapter-path when that is set."
        ),
    )

    parser.add_argument(
        "--dataset-run-id",
        default=None,
        help=(
            "Canonical dataset run id. "
            "Resolves DPO input to data/dataset/dpo/<run_id>/train_pairs.jsonl "
            "and DDO input to data/dataset/ddo/<run_id>/train_pairs.jsonl. "
            "DivPO should use --win-lose-jsonl explicitly. "
            "DDO-bg-v2 should use explicit --win-lose-jsonl/--win-win-jsonl."
        ),
    )
    parser.add_argument(
        "--win-lose-jsonl",
        default=None,
        help="Explicit DPO win_lose JSONL path. Overrides the canonical path for this input only.",
    )
    parser.add_argument(
        "--win-win-jsonl",
        default=None,
        help="Explicit DDO win_win JSONL path. Overrides the canonical path for this input only.",
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
    parser.add_argument("--run-name", default="preference-trainer")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint directory to resume from. Use 'latest' to pick the largest checkpoint-* under --output-dir.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate JSONL inputs and print record counts without importing training dependencies",
    )

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--dpo-rk-alpha",
        type=float,
        default=math.log(3.0),
        help=(
            "DPO-RK tie threshold alpha. The paper's balanced-tie default is log(3), "
            "equivalent to nu_RK=3."
        ),
    )
    parser.add_argument(
        "--dpo-d-nu",
        type=float,
        default=1.0,
        help="DPO-D Davidson tie parameter nu. The paper's balanced-tie default is 1.",
    )
    parser.add_argument("--lambda-div", type=float, default=1.0)
    parser.add_argument(
        "--ddo-loss-variant",
        choices=(DDO_LOSS_VARIANT_SYMMETRIC_GAP, DDO_LOSS_VARIANT_UPWARD_FLOOR),
        default=DDO_LOSS_VARIANT_SYMMETRIC_GAP,
        help=(
            "DDO win-win loss variant. "
            "'symmetric_gap' preserves the legacy squared reward-gap loss. "
            "'upward_floor' raises the weaker successful branch toward the stronger one "
            "and adds an absolute success floor."
        ),
    )
    parser.add_argument(
        "--lambda-floor",
        type=float,
        default=0.0,
        help=(
            "Relative weight of the success-floor term inside --ddo-loss-variant upward_floor. "
            "Ignored for symmetric_gap."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-type", default="linear")

    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)

    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=2304)

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument(
        "--save-epochs-fraction",
        type=float,
        default=None,
        help=(
            "If set, override --save-steps using this multiple of one epoch. "
            "For example 5 saves checkpoints near epochs 5, 10, 15, and 20 in a 20-epoch run."
        ),
    )
    parser.add_argument(
        "--stop-after-epochs",
        type=float,
        default=None,
        help=(
            "If set, request a checkpoint save and stop training after this many "
            "estimated epochs while keeping the configured total-epoch schedule."
        ),
    )
    parser.add_argument(
        "--save-total-limit",
        type=_parse_optional_int,
        default=8,
        help="Maximum checkpoints to keep. Use 'none' to disable checkpoint deletion.",
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=8)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Model loading dtype. If 'auto', follows --bf16/--fp16 when set.",
    )

    parser.add_argument(
        "--win-win-repeats",
        type=int,
        default=1,
        help="DDO only: multiplicative weight for win_win pairs by repeating each pair once per repeat.",
    )
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default=None,
        help=(
            "Comma-separated LoRA target module names. If omitted, defaults are inferred from model config. "
            "Supported aliases: 'qv', 'qvko', 'qvko+mlp'."
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
    win_lose_rows, win_win_rows = _load_pair_rows(args.mode, input_paths)
    records = load_preference_records(args.mode, input_paths, args.win_win_repeats)
    steps_per_epoch = estimate_steps_per_epoch(
        row_count=len(records),
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
    if args.dpo_rk_alpha <= 0.0:
        raise ValueError("--dpo-rk-alpha must be positive")
    if args.dpo_d_nu <= 0.0:
        raise ValueError("--dpo-d-nu must be positive")

    if args.validate_only:
        payload = {
            "mode": args.mode,
            "dataset_run_id": args.dataset_run_id,
            "finetune_type": args.finetune_type,
            "win_lose_jsonl": str(input_paths.win_lose_jsonl),
            "win_win_jsonl": str(input_paths.win_win_jsonl) if input_paths.win_win_jsonl else None,
            "win_lose_pairs": len(win_lose_rows),
            "win_win_pairs": len(win_win_rows),
            "train_records": len(records),
            "steps_per_epoch": steps_per_epoch,
            "resolved_save_steps": resolved_save_steps,
            "resolved_stop_after_steps": resolved_stop_after_steps,
            "dpo_rk_alpha": args.dpo_rk_alpha,
            "dpo_d_nu": args.dpo_d_nu,
        }
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    if not args.model_name_or_path:
        raise ValueError("--model-name-or-path is required unless --validate-only is set")
    if args.train_adapter_path and args.finetune_type != "lora":
        raise ValueError(
            "--train-adapter-path requires --finetune-type lora"
        )
    if args.ref_adapter_path and args.finetune_type != "lora":
        raise ValueError(
            "--ref-adapter-path requires --finetune-type lora"
        )
    if args.finetune_type == "full" and not args.ref_model_name_or_path:
        raise ValueError(
            "--ref-model-name-or-path is required for --finetune-type full"
        )
    if args.mode != MODE_DDO:
        if args.ddo_loss_variant != DDO_LOSS_VARIANT_SYMMETRIC_GAP:
            raise ValueError("--ddo-loss-variant only applies to --mode ddo")
        if args.lambda_floor != 0.0:
            raise ValueError("--lambda-floor only applies to --mode ddo")
        if args.win_win_repeats != 1:
            raise ValueError("--win-win-repeats only applies to --mode ddo")
    if args.ddo_loss_variant == DDO_LOSS_VARIANT_SYMMETRIC_GAP and args.lambda_floor != 0.0:
        raise ValueError("--lambda-floor only applies to --ddo-loss-variant upward_floor")
    if args.train_adapter_path and not Path(args.train_adapter_path).exists():
        raise ValueError(f"--train-adapter-path does not exist: {args.train_adapter_path}")
    if args.ref_adapter_path and not Path(args.ref_adapter_path).exists():
        raise ValueError(f"--ref-adapter-path does not exist: {args.ref_adapter_path}")
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_from_checkpoint = _resolve_resume_from_checkpoint(output_dir, args.resume_from_checkpoint)
    torch_dtype_name = _resolve_torch_dtype(args)

    run_metadata = {
        "mode": args.mode,
        "model_name_or_path": args.model_name_or_path,
        "ref_model_name_or_path": args.ref_model_name_or_path,
        "train_adapter_path": args.train_adapter_path,
        "ref_adapter_path": args.ref_adapter_path,
        "finetune_type": args.finetune_type,
        "input": {
            "dataset_run_id": args.dataset_run_id,
            "win_lose_jsonl": str(input_paths.win_lose_jsonl),
            "win_win_jsonl": str(input_paths.win_win_jsonl) if input_paths.win_win_jsonl else None,
            "win_lose_pair_count": len(win_lose_rows),
            "win_win_pair_count": len(win_win_rows),
            "train_record_count": len(records),
            "win_win_repeats": args.win_win_repeats,
        },
        "trainer": {
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "warmup_ratio": args.warmup_ratio,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_epsilon": args.adam_epsilon,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "lr_scheduler_type": args.lr_scheduler_type,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_prompt_length": args.max_prompt_length,
            "max_length": args.max_length,
            "logging_steps": args.logging_steps,
            "save_steps": resolved_save_steps,
            "save_steps_requested": args.save_steps,
            "save_epochs_fraction": args.save_epochs_fraction,
            "steps_per_epoch_estimate": steps_per_epoch,
            "stop_after_epochs": args.stop_after_epochs,
            "stop_after_steps": resolved_stop_after_steps,
            "save_total_limit": args.save_total_limit,
            "dataloader_num_workers": args.dataloader_num_workers,
            "gradient_checkpointing": args.gradient_checkpointing,
            "bf16": args.bf16,
            "fp16": args.fp16,
            "torch_dtype": torch_dtype_name,
            "seed": args.seed,
            "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
            "cuda_memory_log_jsonl": args.cuda_memory_log_jsonl,
            "cuda_memory_log_every_steps": args.cuda_memory_log_every_steps,
            "cuda_memory_log_reset_peak_per_step": args.cuda_memory_log_reset_peak_per_step,
        },
        "objective": {
            "beta": args.beta,
            "dpo_rk_alpha": args.dpo_rk_alpha,
            "dpo_d_nu": args.dpo_d_nu,
            "lambda_div": args.lambda_div,
            "ddo_loss_variant": args.ddo_loss_variant,
            "lambda_floor": args.lambda_floor,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        },
    }

    try:
        import torch
        import torch.nn.functional as F
        from datasets import Dataset
        from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        from transformers.trainer_callback import TrainerCallback
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Training requires `datasets`, `transformers`, and `peft`. "
            "For schema-only checks, run with --validate-only."
        ) from exc

    class PreferenceTrainer(Trainer):
        def __init__(
            self,
            *trainer_args,
            ref_model: Any,
            beta: float,
            lambda_div: float,
            ddo_loss_variant: str,
            lambda_floor: float,
            dpo_rk_alpha: float,
            dpo_d_nu: float,
            pad_token_id: int,
            **trainer_kwargs,
        ):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.ref_model = ref_model
            self.beta = beta
            self.lambda_div = lambda_div
            self.ddo_loss_variant = ddo_loss_variant
            self.lambda_floor = lambda_floor
            self.dpo_rk_alpha = dpo_rk_alpha
            self.dpo_d_nu = dpo_d_nu
            self.pad_token_id = pad_token_id
            self._stored_metrics: dict[str, list[float]] = defaultdict(list)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad_(False)

        def _store_metrics(self, metrics: dict[str, float]) -> None:
            for key, value in metrics.items():
                self._stored_metrics[key].append(float(value))

        def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
            for key, values in list(self._stored_metrics.items()):
                if values:
                    logs[key] = sum(values) / len(values)
            self._stored_metrics.clear()
            super().log(logs, start_time)

        @staticmethod
        def _sequence_logps(logits: Any, labels: Any) -> Any:
            shift_logits = logits[:, :-1, :].float()
            shift_labels = labels[:, 1:]
            valid_mask = shift_labels != -100
            safe_labels = shift_labels.masked_fill(~valid_mask, 0)
            token_logps = F.log_softmax(shift_logits, dim=-1).gather(
                dim=-1,
                index=safe_labels.unsqueeze(-1),
            ).squeeze(-1)
            token_logps = token_logps.masked_fill(~valid_mask, 0.0)
            return token_logps.sum(dim=-1)

        def _paired_logps(self, model: Any, inputs: dict[str, Any]) -> tuple[Any, Any]:
            (
                concatenated_input_ids,
                concatenated_attention_mask,
                concatenated_labels,
            ) = _pad_pair_batch_tensors(inputs, self.pad_token_id)
            outputs = model(
                input_ids=concatenated_input_ids,
                attention_mask=concatenated_attention_mask,
            )
            sequence_logps = self._sequence_logps(outputs.logits, concatenated_labels)
            batch_size = inputs["chosen_input_ids"].shape[0]
            return sequence_logps[:batch_size], sequence_logps[batch_size:]

        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, num_items_in_batch: Any = None):
            del num_items_in_batch
            chosen_logps, rejected_logps = self._paired_logps(model, inputs)
            with torch.no_grad():
                ref_chosen_logps, ref_rejected_logps = self._paired_logps(self.ref_model, inputs)

            chosen_rewards = chosen_logps - ref_chosen_logps
            rejected_rewards = rejected_logps - ref_rejected_logps
            terms = _compute_loss_terms(
                chosen_rewards,
                rejected_rewards,
                inputs["pair_type_ids"],
                inputs["chosen_target_probs"],
                beta=self.beta,
                lambda_div=self.lambda_div,
                ddo_loss_variant=self.ddo_loss_variant,
                lambda_floor=self.lambda_floor,
                mode=args.mode,
                dpo_rk_alpha=self.dpo_rk_alpha,
                dpo_d_nu=self.dpo_d_nu,
            )
            reward_gap = chosen_rewards - rejected_rewards

            completion_tokens = (
                (inputs["chosen_labels"][:, 1:] != -100).sum() +
                (inputs["rejected_labels"][:, 1:] != -100).sum()
            )
            self._store_metrics(
                {
                    "loss_pref": terms["pref_loss"].detach().item(),
                    "loss_div": terms["div_loss"].detach().item(),
                    "loss_div_match": terms["div_match_loss"].detach().item(),
                    "loss_div_floor": terms["div_floor_loss"].detach().item(),
                    "reward/chosen": chosen_rewards.mean().detach().item(),
                    "reward/rejected": rejected_rewards.mean().detach().item(),
                    "reward_gap/pref": terms["pref_reward_gap"].detach().item(),
                    "reward_gap/div": terms["div_reward_gap"].detach().item(),
                    "reward_gap_sq/div": terms["div_reward_gap_sq"].detach().item(),
                    "reward_hi/div": terms["div_reward_hi"].detach().item(),
                    "reward_lo/div": terms["div_reward_lo"].detach().item(),
                    "reward_floor_target/div": terms["div_floor_target"].detach().item(),
                    "target_prob/div_mean": terms["div_target_prob_mean"].detach().item(),
                    "num_pref_pairs": terms["num_pref_pairs"],
                    "num_div_pairs": terms["num_div_pairs"],
                    "num_tokens": completion_tokens.detach().item(),
                }
            )

            outputs = {
                "chosen_rewards": chosen_rewards.detach(),
                "rejected_rewards": rejected_rewards.detach(),
                "reward_gap": reward_gap.detach(),
            }
            if return_outputs:
                return terms["loss"], outputs
            return terms["loss"]

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
            allocated_bytes = int(self._torch.cuda.memory_allocated())
            reserved_bytes = int(self._torch.cuda.memory_reserved())
            active_bytes_current = int(stats.get("active_bytes.all.current", 0))
            max_allocated_bytes = int(self._torch.cuda.max_memory_allocated())
            max_reserved_bytes = int(self._torch.cuda.max_memory_reserved())
            current_device = int(self._torch.cuda.current_device())
            payload = {
                "label": label,
                "global_step": int(getattr(state, "global_step", 0) or 0),
                "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
                "device_index": current_device,
                "device_name": self._torch.cuda.get_device_name(current_device),
                "total_memory_bytes": int(self._torch.cuda.get_device_properties(current_device).total_memory),
                "allocated_bytes": allocated_bytes,
                "reserved_bytes": reserved_bytes,
                "max_allocated_bytes": max_allocated_bytes,
                "max_reserved_bytes": max_reserved_bytes,
                "active_bytes_current": active_bytes_current,
                "active_bytes_peak": int(stats.get("active_bytes.all.peak", 0)),
                "inactive_split_bytes_current": int(stats.get("inactive_split_bytes.all.current", 0)),
                "requested_bytes_current": int(stats.get("requested_bytes.all.current", 0)),
                "cache_bytes_estimate": max(0, reserved_bytes - allocated_bytes),
                "reserved_minus_active_bytes": max(0, reserved_bytes - active_bytes_current),
                "peak_cache_bytes_estimate": max(0, max_reserved_bytes - max_allocated_bytes),
            }
            _append_jsonl(self._output_path, payload)

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if self._enabled:
                self._torch.cuda.reset_peak_memory_stats()
            self._record(label="train_begin", state=state)
            return control

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, state, kwargs
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

    class StopAfterStepCallback(TrainerCallback):
        def __init__(self, *, stop_after_steps: int) -> None:
            self._stop_after_steps = stop_after_steps

        def _should_stop(self, state: Any) -> bool:
            return int(getattr(state, "global_step", 0) or 0) >= self._stop_after_steps

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

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    resolved_lora_target_modules = _parse_lora_target_modules(args.lora_target_modules, config)
    run_metadata["lora"]["target_modules"] = resolved_lora_target_modules
    _write_json(output_dir / "run_metadata.json", run_metadata)

    model_load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch_dtype_name is not None:
        model_load_kwargs["torch_dtype"] = getattr(torch, torch_dtype_name)

    if _use_qwen35_text_backbone(config):
        processing_class = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)
    elif _is_multimodal_config(config):
        processing_class = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        processing_class = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, trust_remote_code=True)

    if args.train_adapter_path:
        if _use_qwen35_text_backbone(config) or _is_multimodal_config(config):
            raise ValueError(
                "--train-adapter-path currently supports text-only causal LM backbones"
            )
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.train_adapter_path,
            is_trainable=True,
            **model_load_kwargs,
        )
    elif _use_qwen35_text_backbone(config):
        model = _qwen35_causal_lm_class().from_pretrained(
            args.model_name_or_path,
            config=config.text_config,
            **model_load_kwargs,
        )
    elif _is_multimodal_config(config):
        model = AutoModelForImageTextToText.from_pretrained(args.model_name_or_path, **model_load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_load_kwargs)

    tokenizer = processing_class.tokenizer if hasattr(processing_class, "tokenizer") else processing_class
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for preference training")
    if getattr(processing_class, "pad_token", None) is None and hasattr(processing_class, "tokenizer"):
        processing_class.tokenizer.pad_token = tokenizer.pad_token

    tokenized_records = _tokenize_preference_records(
        records,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
    )
    train_dataset = Dataset.from_list(tokenized_records)

    if args.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    ref_model_path = args.ref_model_name_or_path or args.model_name_or_path
    if args.finetune_type == "lora" and not args.train_adapter_path:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=resolved_lora_target_modules,
        )
        model = get_peft_model(model, peft_config)
        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if args.ref_adapter_path or args.train_adapter_path:
        resolved_ref_adapter_path = args.ref_adapter_path or args.train_adapter_path
        if resolved_ref_adapter_path is None:
            raise ValueError("internal error: expected an adapter path for the reference model")
        if _use_qwen35_text_backbone(config) or _is_multimodal_config(config):
            raise ValueError(
                "--ref-adapter-path currently supports text-only causal LM backbones"
            )
        ref_model = AutoPeftModelForCausalLM.from_pretrained(
            resolved_ref_adapter_path,
            is_trainable=False,
            **model_load_kwargs,
        )
    elif _use_qwen35_text_backbone(config):
        ref_model = _qwen35_causal_lm_class().from_pretrained(
            ref_model_path,
            config=config.text_config,
            **model_load_kwargs,
        )
    elif _is_multimodal_config(config):
        ref_model = AutoModelForImageTextToText.from_pretrained(ref_model_path, **model_load_kwargs)
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, **model_load_kwargs)
    ref_model.eval()
    if hasattr(ref_model, "config"):
        ref_model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=resolved_save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = PreferenceTrainer(
        model=model,
        ref_model=ref_model,
        beta=args.beta,
        lambda_div=args.lambda_div,
        ddo_loss_variant=args.ddo_loss_variant,
        lambda_floor=args.lambda_floor,
        dpo_rk_alpha=args.dpo_rk_alpha,
        dpo_d_nu=args.dpo_d_nu,
        pad_token_id=tokenizer.pad_token_id,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=PreferenceDataCollator(tokenizer.pad_token_id),
    )
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
    if resolved_stop_after_steps is not None:
        trainer.add_callback(StopAfterStepCallback(stop_after_steps=resolved_stop_after_steps))

    if hasattr(trainer, "_move_model_to_device"):
        trainer._move_model_to_device(ref_model, trainer.args.device)

    train_result = trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None
    )
    trainer.save_state()
    trainer.save_model(str(output_dir))
    processing_class.save_pretrained(str(output_dir))
    _write_json(
        output_dir / "log_history.json",
        build_log_history(getattr(trainer.state, "log_history", None)),
    )
    train_summary = build_train_summary(
        mode=args.mode,
        output_dir=output_dir,
        win_lose_jsonl=input_paths.win_lose_jsonl,
        win_win_jsonl=input_paths.win_win_jsonl,
        train_record_count=len(records),
        win_lose_pair_count=len(win_lose_rows),
        win_win_pair_count=len(win_win_rows),
        global_step=int(trainer.state.global_step),
        train_metrics=getattr(train_result, "metrics", None),
    )
    _write_json(output_dir / "train_summary.json", train_summary)


if __name__ == "__main__":
    main()
