#!/usr/bin/env python3
"""Research trainer for balanced-geometric DDO preference optimization."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddo.training.train_trl_preference import (
    PAIR_TYPE_TO_ID,
    PAIR_TYPE_WIN_LOSE,
    PAIR_TYPE_WIN_WIN,
    PreferenceDataCollator,
    _append_jsonl,
    estimate_steps_per_epoch,
    _is_multimodal_config,
    _pad_pair_batch_tensors,
    _parse_lora_target_modules,
    _qwen35_causal_lm_class,
    _parse_optional_int,
    _resolve_output_dir,
    _resolve_resume_from_checkpoint,
    _resolve_torch_dtype,
    _tokenize_response_pair,
    _use_qwen35_text_backbone,
    _write_json,
    build_log_history,
)


CANONICAL_WIN_LOSE_FILENAME = "train_pairs_DDO_balanced_geometric_win_lose.jsonl"
CANONICAL_WIN_WIN_FILENAME = "train_pairs_DDO_balanced_geometric_win_win.jsonl"
CANONICAL_DDO_V3_WIN_LOSE_FILENAME = "train_pairs_DDO_v3_win_lose.jsonl"
CANONICAL_DDO_V3_WIN_WIN_FILENAME = "train_pairs_DDO_v3_win_win.jsonl"

ROW_FIELD_PAIR_WEIGHT = "pair_weight"
ROW_FIELD_CHOSEN_TARGET_PROB = "chosen_target_prob"
ROW_FIELD_TARGET_MODE = "target_mode"
ROW_FIELD_TARGET_BETA = "target_beta"
ROW_FIELD_TARGET_MARGIN = "target_margin"
TARGET_MODE_REFERENCE_RELATIVE = "reference_relative"

REQUIRED_PAIR_FIELDS = (
    "pair_type",
    "prompt",
    "chosen",
    "rejected",
    ROW_FIELD_PAIR_WEIGHT,
    ROW_FIELD_CHOSEN_TARGET_PROB,
)

REQUIRED_ACTION_FIELDS = (
    "action_text",
    "success",
)


@dataclass(frozen=True)
class TrainInputPaths:
    dataset_dir: Path | None
    win_lose_jsonl: Path
    win_win_jsonl: Path


class BalancedGeometricSchemaError(ValueError):
    pass


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
                raise BalancedGeometricSchemaError(f"{path}:{line_idx} invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise BalancedGeometricSchemaError(f"{path}:{line_idx} row must be an object")
            rows.append(row)
    if not rows:
        raise BalancedGeometricSchemaError(f"{path} has no valid JSONL rows")
    return rows


def _validate_probability(value: Any, *, source: Path, line_idx: int, field_name: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise BalancedGeometricSchemaError(
            f"{source}:{line_idx} field '{field_name}' must be a float in [0, 1]"
        ) from exc
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise BalancedGeometricSchemaError(
            f"{source}:{line_idx} field '{field_name}' must be a finite float in [0, 1]"
        )
    return probability


def _validate_gt_zero_float(value: Any, *, source: Path, line_idx: int, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BalancedGeometricSchemaError(
            f"{source}:{line_idx} field '{field_name}' must be a float > 0"
        ) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise BalancedGeometricSchemaError(
            f"{source}:{line_idx} field '{field_name}' must be a finite float > 0"
        )
    return number


def _validate_pair_row(row: dict[str, Any], source: Path, line_idx: int) -> dict[str, Any]:
    missing = [field for field in REQUIRED_PAIR_FIELDS if field not in row]
    if missing:
        raise BalancedGeometricSchemaError(
            f"{source}:{line_idx} missing required fields: {', '.join(missing)}"
        )

    if not isinstance(row["prompt"], str) or not row["prompt"].strip():
        raise BalancedGeometricSchemaError(f"{source}:{line_idx} field 'prompt' must be non-empty string")

    validated = dict(row)
    for side in ("chosen", "rejected"):
        side_value = row[side]
        if not isinstance(side_value, dict):
            raise BalancedGeometricSchemaError(f"{source}:{line_idx} field '{side}' must be an object")
        missing_action = [field for field in REQUIRED_ACTION_FIELDS if field not in side_value]
        if missing_action:
            raise BalancedGeometricSchemaError(
                f"{source}:{line_idx} field '{side}' missing: {', '.join(missing_action)}"
            )
        if not isinstance(side_value["action_text"], str) or not side_value["action_text"].strip():
            raise BalancedGeometricSchemaError(
                f"{source}:{line_idx} field '{side}.action_text' must be non-empty string"
            )

    validated[ROW_FIELD_PAIR_WEIGHT] = _validate_gt_zero_float(
        row[ROW_FIELD_PAIR_WEIGHT],
        source=source,
        line_idx=line_idx,
        field_name=ROW_FIELD_PAIR_WEIGHT,
    )
    validated[ROW_FIELD_CHOSEN_TARGET_PROB] = _validate_probability(
        row[ROW_FIELD_CHOSEN_TARGET_PROB],
        source=source,
        line_idx=line_idx,
        field_name=ROW_FIELD_CHOSEN_TARGET_PROB,
    )
    return validated


def _extract_rows_for_split(
    rows: list[dict[str, Any]],
    *,
    source: Path,
    expected_pair_type: str,
) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    for idx, raw_row in enumerate(rows, start=1):
        row = _validate_pair_row(raw_row, source, idx)
        pair_type = row["pair_type"]
        if pair_type != expected_pair_type:
            raise BalancedGeometricSchemaError(
                f"{source}:{idx} expected pair_type='{expected_pair_type}', got '{pair_type}'"
            )
        if pair_type == PAIR_TYPE_WIN_LOSE and not math.isclose(
            row[ROW_FIELD_CHOSEN_TARGET_PROB],
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise BalancedGeometricSchemaError(
                f"{source}:{idx} win_lose rows must set chosen_target_prob=1.0"
            )
        extracted.append(row)
    if not extracted:
        raise BalancedGeometricSchemaError(f"{source} has no {expected_pair_type} rows")
    return extracted


def _resolve_dataset_file(dataset_dir: Path, filenames: tuple[str, ...], *, required: bool) -> Path | None:
    existing = [dataset_dir / filename for filename in filenames if (dataset_dir / filename).exists()]
    if len(existing) > 1:
        names = ", ".join(path.name for path in existing)
        raise ValueError(f"{dataset_dir} contains multiple matching dataset files: {names}")
    if existing:
        return existing[0]
    if required:
        return dataset_dir / filenames[0]
    return None


def _resolve_input_paths(args: argparse.Namespace) -> TrainInputPaths:
    dataset_dir_value = getattr(args, "dataset_dir", None)
    win_lose_value = getattr(args, "win_lose_jsonl", None)
    win_win_value = getattr(args, "win_win_jsonl", None)

    if dataset_dir_value:
        dataset_dir = Path(dataset_dir_value)
        if win_lose_value or win_win_value:
            raise ValueError(
                "Use either --dataset-dir or explicit --win-lose-jsonl/--win-win-jsonl, not both"
            )
        win_lose_jsonl = _resolve_dataset_file(
            dataset_dir,
            (CANONICAL_WIN_LOSE_FILENAME, CANONICAL_DDO_V3_WIN_LOSE_FILENAME),
            required=True,
        )
        win_win_jsonl = _resolve_dataset_file(
            dataset_dir,
            (CANONICAL_WIN_WIN_FILENAME, CANONICAL_DDO_V3_WIN_WIN_FILENAME),
            required=True,
        )
        if win_lose_jsonl is None or win_win_jsonl is None:
            raise ValueError("internal error: required dataset paths were not resolved")
        return TrainInputPaths(
            dataset_dir=dataset_dir,
            win_lose_jsonl=win_lose_jsonl,
            win_win_jsonl=win_win_jsonl,
        )

    if not win_lose_value or not win_win_value:
        raise ValueError("Provide --dataset-dir or both --win-lose-jsonl and --win-win-jsonl")
    return TrainInputPaths(
        dataset_dir=None,
        win_lose_jsonl=Path(win_lose_value),
        win_win_jsonl=Path(win_win_value),
    )


def _as_balanced_pref_record(row: dict[str, Any]) -> dict[str, Any]:
    record = {
        "pair_type": row["pair_type"],
        "prompt": row["prompt"],
        "chosen": row["chosen"]["action_text"],
        "rejected": row["rejected"]["action_text"],
        ROW_FIELD_PAIR_WEIGHT: row[ROW_FIELD_PAIR_WEIGHT],
        ROW_FIELD_CHOSEN_TARGET_PROB: row[ROW_FIELD_CHOSEN_TARGET_PROB],
    }
    for field in (ROW_FIELD_TARGET_MODE, ROW_FIELD_TARGET_BETA, ROW_FIELD_TARGET_MARGIN):
        if field in row:
            record[field] = row[field]
    return record


def load_balanced_preference_records(
    paths: TrainInputPaths,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    win_lose_raw = _read_jsonl(paths.win_lose_jsonl)
    win_win_raw = _read_jsonl(paths.win_win_jsonl)

    win_lose_rows = _extract_rows_for_split(
        win_lose_raw,
        source=paths.win_lose_jsonl,
        expected_pair_type=PAIR_TYPE_WIN_LOSE,
    )
    win_win_rows = _extract_rows_for_split(
        win_win_raw,
        source=paths.win_win_jsonl,
        expected_pair_type=PAIR_TYPE_WIN_WIN,
    )
    return (
        [_as_balanced_pref_record(row) for row in win_lose_rows],
        [_as_balanced_pref_record(row) for row in win_win_rows],
    )


def validate_ddo_v3_target_beta(win_win_rows: list[dict[str, Any]], *, train_beta: float) -> dict[str, Any]:
    reference_relative_rows = [
        row for row in win_win_rows
        if row.get(ROW_FIELD_TARGET_MODE) == TARGET_MODE_REFERENCE_RELATIVE
    ]
    if not reference_relative_rows:
        return {
            "reference_relative_rows": 0,
            "target_beta": None,
            "train_beta": float(train_beta),
        }
    if not math.isfinite(train_beta) or train_beta <= 0.0:
        raise BalancedGeometricSchemaError("DDO-v3 reference-relative targets require --beta to be a finite float > 0")

    observed_target_beta: float | None = None
    for idx, row in enumerate(reference_relative_rows, start=1):
        if ROW_FIELD_TARGET_BETA not in row:
            raise BalancedGeometricSchemaError(
                f"DDO-v3 reference-relative win_win row {idx} is missing {ROW_FIELD_TARGET_BETA}"
            )
        try:
            target_beta = float(row[ROW_FIELD_TARGET_BETA])
        except (TypeError, ValueError) as exc:
            raise BalancedGeometricSchemaError(
                f"DDO-v3 reference-relative win_win row {idx} has non-float {ROW_FIELD_TARGET_BETA}"
            ) from exc
        if not math.isfinite(target_beta) or target_beta <= 0.0:
            raise BalancedGeometricSchemaError(
                f"DDO-v3 reference-relative win_win row {idx} has invalid {ROW_FIELD_TARGET_BETA}"
            )
        if observed_target_beta is None:
            observed_target_beta = target_beta
        elif not math.isclose(observed_target_beta, target_beta, rel_tol=1e-9, abs_tol=1e-12):
            raise BalancedGeometricSchemaError(
                f"DDO-v3 reference-relative rows have mixed target_beta values: "
                f"{observed_target_beta} and {target_beta}"
            )
        if not math.isclose(target_beta, train_beta, rel_tol=1e-9, abs_tol=1e-12):
            raise BalancedGeometricSchemaError(
                f"DDO-v3 requires target_beta to match training beta: "
                f"target_beta={target_beta}, train_beta={train_beta}"
            )

    return {
        "reference_relative_rows": len(reference_relative_rows),
        "target_beta": observed_target_beta,
        "train_beta": float(train_beta),
    }


def _tokenize_balanced_records(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_prompt_length: int,
    max_length: int,
) -> list[dict[str, Any]]:
    tokenized: list[dict[str, Any]] = []
    for record in records:
        tokenized_row = _tokenize_response_pair(
            record,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            max_length=max_length,
        )
        tokenized_row[ROW_FIELD_PAIR_WEIGHT] = float(record[ROW_FIELD_PAIR_WEIGHT])
        tokenized_row[ROW_FIELD_CHOSEN_TARGET_PROB] = float(record[ROW_FIELD_CHOSEN_TARGET_PROB])
        tokenized.append(tokenized_row)
    return tokenized


class BalancedPairedDataset:
    def __init__(
        self,
        *,
        win_lose_features: list[dict[str, Any]],
        win_win_features: list[dict[str, Any]],
        seed: int,
    ) -> None:
        if not win_lose_features:
            raise ValueError("win_lose_features must be non-empty")
        if not win_win_features:
            raise ValueError("win_win_features must be non-empty")
        self.win_lose_features = win_lose_features
        self.win_win_features = win_win_features
        self.seed = seed
        self.length = max(len(win_lose_features), len(win_win_features))
        self._win_lose_order = list(range(len(win_lose_features)))
        self._win_win_order = list(range(len(win_win_features)))
        self.refresh_orders(seed)

    def refresh_orders(self, seed: int) -> None:
        rng = random.Random(seed)
        self._win_lose_order = list(range(len(self.win_lose_features)))
        self._win_win_order = list(range(len(self.win_win_features)))
        rng.shuffle(self._win_lose_order)
        rng.shuffle(self._win_win_order)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        win_lose_idx = self._win_lose_order[index % len(self._win_lose_order)]
        win_win_idx = self._win_win_order[index % len(self._win_win_order)]
        payload = {
            "win_lose": self.win_lose_features[win_lose_idx],
            "win_win": self.win_win_features[win_win_idx],
        }
        return payload


class BalancedPreferenceDataCollator:
    def __init__(self, pad_token_id: int):
        self._pair_collator = PreferenceDataCollator(pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        win_lose_features = [feature["win_lose"] for feature in features]
        win_win_features = [feature["win_win"] for feature in features]
        batch = {
            "win_lose": self._pair_collator(win_lose_features),
            "win_win": self._pair_collator(win_win_features),
            "win_lose_pair_weights": torch.tensor(
                [feature[ROW_FIELD_PAIR_WEIGHT] for feature in win_lose_features],
                dtype=torch.float32,
            ),
            "win_win_pair_weights": torch.tensor(
                [feature[ROW_FIELD_PAIR_WEIGHT] for feature in win_win_features],
                dtype=torch.float32,
            ),
            "win_win_target_probs": torch.tensor(
                [feature[ROW_FIELD_CHOSEN_TARGET_PROB] for feature in win_win_features],
                dtype=torch.float32,
            ),
        }
        return batch


def _compute_balanced_geometric_loss_terms(
    *,
    win_lose_chosen_rewards: Any,
    win_lose_rejected_rewards: Any,
    win_lose_pair_weights: Any,
    win_win_chosen_rewards: Any,
    win_win_rejected_rewards: Any,
    win_win_target_probs: Any,
    win_win_pair_weights: Any,
    beta: float,
) -> dict[str, Any]:
    import torch.nn.functional as F

    zero = win_lose_chosen_rewards.new_zeros(())

    win_lose_gap = win_lose_chosen_rewards - win_lose_rejected_rewards
    win_lose_row_loss = -F.logsigmoid(beta * win_lose_gap)
    win_lose_loss = (win_lose_pair_weights * win_lose_row_loss).mean() if win_lose_row_loss.numel() else zero

    win_win_gap = win_win_chosen_rewards - win_win_rejected_rewards
    win_win_pos_loss = -F.logsigmoid(beta * win_win_gap)
    win_win_neg_loss = -F.logsigmoid(-beta * win_win_gap)
    win_win_row_loss = (
        win_win_target_probs * win_win_pos_loss +
        (1.0 - win_win_target_probs) * win_win_neg_loss
    )
    win_win_loss = (win_win_pair_weights * win_win_row_loss).mean() if win_win_row_loss.numel() else zero

    total_loss = win_lose_loss + win_win_loss
    return {
        "loss": total_loss,
        "win_lose_loss": win_lose_loss,
        "win_win_loss": win_win_loss,
        "win_lose_gap": win_lose_gap.mean() if win_lose_gap.numel() else zero,
        "win_win_gap": win_win_gap.mean() if win_win_gap.numel() else zero,
        "win_win_target_mean": win_win_target_probs.mean() if win_win_target_probs.numel() else zero,
        "num_win_lose_pairs": int(win_lose_gap.numel()),
        "num_win_win_pairs": int(win_win_gap.numel()),
    }


def build_train_summary(
    *,
    output_dir: Path,
    input_paths: TrainInputPaths,
    win_lose_pair_count: int,
    win_win_pair_count: int,
    paired_steps_per_epoch: int,
    global_step: int,
    train_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = dict(train_metrics or {})
    return {
        "output_dir": str(output_dir),
        "dataset_dir": str(input_paths.dataset_dir) if input_paths.dataset_dir else None,
        "win_lose_jsonl": str(input_paths.win_lose_jsonl),
        "win_win_jsonl": str(input_paths.win_win_jsonl),
        "win_lose_pair_count": int(win_lose_pair_count),
        "win_win_pair_count": int(win_win_pair_count),
        "paired_steps_per_epoch": int(paired_steps_per_epoch),
        "global_step": int(global_step),
        "train_runtime": float(metrics.get("train_runtime", 0.0) or 0.0),
        "train_loss": float(metrics.get("train_loss", 0.0) or 0.0),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Balanced-geometric DDO preference trainer")

    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--win-lose-jsonl", default=None)
    parser.add_argument("--win-win-jsonl", default=None)

    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--ref-model-name-or-path", default=None)
    parser.add_argument("--finetune-type", choices=("full", "lora"), default="full")
    parser.add_argument(
        "--train-adapter-path",
        default=None,
        help="Optional PEFT adapter path used to initialize the trainable policy model.",
    )
    parser.add_argument(
        "--ref-adapter-path",
        default=None,
        help="Optional PEFT adapter path used to initialize the frozen reference model.",
    )

    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default="ddo-balanced-geometric")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--validate-only", action="store_true")

    parser.add_argument("--beta", type=float, default=0.1)
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
        help="If set, stop training after this many epochs while still using the requested scheduler horizon.",
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
    )
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default=None,
        help="Comma-separated LoRA target module names.",
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
    win_lose_rows, win_win_rows = load_balanced_preference_records(input_paths)
    paired_steps_per_epoch = max(len(win_lose_rows), len(win_win_rows))
    optimizer_steps_per_epoch = estimate_steps_per_epoch(
        row_count=paired_steps_per_epoch,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    resolved_save_steps = args.save_steps
    if args.save_epochs_fraction is not None:
        if args.save_epochs_fraction <= 0:
            raise ValueError("--save-epochs-fraction must be positive")
        resolved_save_steps = max(1, math.ceil(optimizer_steps_per_epoch * args.save_epochs_fraction))
    resolved_stop_after_steps: int | None = None
    if args.stop_after_epochs is not None:
        if args.stop_after_epochs <= 0:
            raise ValueError("--stop-after-epochs must be positive")
        resolved_stop_after_steps = max(1, math.ceil(optimizer_steps_per_epoch * args.stop_after_epochs))
    ddo_v3_target_beta_check = validate_ddo_v3_target_beta(win_win_rows, train_beta=args.beta)

    if args.validate_only:
        payload = {
            "dataset_dir": str(input_paths.dataset_dir) if input_paths.dataset_dir else None,
            "win_lose_jsonl": str(input_paths.win_lose_jsonl),
            "win_win_jsonl": str(input_paths.win_win_jsonl),
            "win_lose_pairs": len(win_lose_rows),
            "win_win_pairs": len(win_win_rows),
            "paired_steps_per_epoch": paired_steps_per_epoch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "resolved_save_steps": resolved_save_steps,
            "resolved_stop_after_steps": resolved_stop_after_steps,
            "ddo_v3_target_beta_check": ddo_v3_target_beta_check,
        }
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    if not args.model_name_or_path:
        raise ValueError("--model-name-or-path is required unless --validate-only is set")
    if args.train_adapter_path and args.finetune_type != "lora":
        raise ValueError("--train-adapter-path requires --finetune-type lora")
    if args.ref_adapter_path and args.finetune_type != "lora":
        raise ValueError("--ref-adapter-path requires --finetune-type lora")
    if args.finetune_type == "full" and not args.ref_model_name_or_path:
        raise ValueError("--ref-model-name-or-path is required for --finetune-type full")
    if args.train_adapter_path and not Path(args.train_adapter_path).exists():
        raise ValueError(f"--train-adapter-path does not exist: {args.train_adapter_path}")
    if args.ref_adapter_path and not Path(args.ref_adapter_path).exists():
        raise ValueError(f"--ref-adapter-path does not exist: {args.ref_adapter_path}")

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_from_checkpoint = _resolve_resume_from_checkpoint(output_dir, args.resume_from_checkpoint)
    torch_dtype_name = _resolve_torch_dtype(args)

    run_metadata = {
        "dataset_dir": str(input_paths.dataset_dir) if input_paths.dataset_dir else None,
        "input": {
            "win_lose_jsonl": str(input_paths.win_lose_jsonl),
            "win_win_jsonl": str(input_paths.win_win_jsonl),
            "win_lose_pair_count": len(win_lose_rows),
            "win_win_pair_count": len(win_win_rows),
            "paired_steps_per_epoch": paired_steps_per_epoch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        },
        "model_name_or_path": args.model_name_or_path,
        "ref_model_name_or_path": args.ref_model_name_or_path,
        "train_adapter_path": args.train_adapter_path,
        "ref_adapter_path": args.ref_adapter_path,
        "finetune_type": args.finetune_type,
        "objective": {
            "beta": args.beta,
            "ddo_v3_target_beta_check": ddo_v3_target_beta_check,
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
            "stop_after_epochs": args.stop_after_epochs,
            "stop_after_steps": resolved_stop_after_steps,
            "save_total_limit": args.save_total_limit,
            "dataloader_num_workers": args.dataloader_num_workers,
            "gradient_checkpointing": args.gradient_checkpointing,
            "bf16": args.bf16,
            "fp16": args.fp16,
            "torch_dtype": torch_dtype_name,
            "seed": args.seed,
            "cuda_memory_log_jsonl": args.cuda_memory_log_jsonl,
            "cuda_memory_log_every_steps": args.cuda_memory_log_every_steps,
            "cuda_memory_log_reset_peak_per_step": args.cuda_memory_log_reset_peak_per_step,
            "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        },
    }

    try:
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
        import torch
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Training requires `transformers`, `torch`, and `peft`. For schema checks, run with --validate-only."
        ) from exc

    class BalancedDatasetShuffleCallback(TrainerCallback):
        def __init__(self, dataset: BalancedPairedDataset, *, seed: int) -> None:
            self.dataset = dataset
            self.seed = seed
            self._last_epoch = None

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            self.dataset.refresh_orders(self.seed)
            self._last_epoch = -1
            return control

        def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            epoch = int(math.floor(float(getattr(state, "epoch", 0.0) or 0.0)))
            if epoch != self._last_epoch:
                self.dataset.refresh_orders(self.seed + epoch + 1)
                self._last_epoch = epoch
            return control

    class StopAfterStepCallback(TrainerCallback):
        def __init__(self, *, stop_after_steps: int) -> None:
            self._stop_after_steps = stop_after_steps

        def _maybe_stop(self, state: Any, control: Any) -> Any:
            if int(getattr(state, "global_step", 0) or 0) >= self._stop_after_steps:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            return self._maybe_stop(state, control)

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            return self._maybe_stop(state, control)

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
            if self._every_steps <= 0:
                return control
            if int(getattr(state, "global_step", 0) or 0) % self._every_steps == 0:
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

    class BalancedGeometricTrainer(Trainer):
        def __init__(
            self,
            *trainer_args,
            ref_model: Any,
            beta: float,
            pad_token_id: int,
            **trainer_kwargs,
        ):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.ref_model = ref_model
            self.beta = beta
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
            concatenated_input_ids, concatenated_attention_mask, concatenated_labels = _pad_pair_batch_tensors(
                inputs,
                self.pad_token_id,
            )
            outputs = model(
                input_ids=concatenated_input_ids,
                attention_mask=concatenated_attention_mask,
            )
            sequence_logps = self._sequence_logps(outputs.logits, concatenated_labels)
            batch_size = inputs["chosen_input_ids"].shape[0]
            return sequence_logps[:batch_size], sequence_logps[batch_size:]

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ):
            del num_items_in_batch
            win_lose_inputs = inputs["win_lose"]
            win_win_inputs = inputs["win_win"]

            win_lose_chosen_logps, win_lose_rejected_logps = self._paired_logps(model, win_lose_inputs)
            win_win_chosen_logps, win_win_rejected_logps = self._paired_logps(model, win_win_inputs)
            with torch.no_grad():
                ref_win_lose_chosen_logps, ref_win_lose_rejected_logps = self._paired_logps(self.ref_model, win_lose_inputs)
                ref_win_win_chosen_logps, ref_win_win_rejected_logps = self._paired_logps(self.ref_model, win_win_inputs)

            win_lose_chosen_rewards = win_lose_chosen_logps - ref_win_lose_chosen_logps
            win_lose_rejected_rewards = win_lose_rejected_logps - ref_win_lose_rejected_logps
            win_win_chosen_rewards = win_win_chosen_logps - ref_win_win_chosen_logps
            win_win_rejected_rewards = win_win_rejected_logps - ref_win_win_rejected_logps

            terms = _compute_balanced_geometric_loss_terms(
                win_lose_chosen_rewards=win_lose_chosen_rewards,
                win_lose_rejected_rewards=win_lose_rejected_rewards,
                win_lose_pair_weights=inputs["win_lose_pair_weights"].to(win_lose_chosen_rewards.dtype),
                win_win_chosen_rewards=win_win_chosen_rewards,
                win_win_rejected_rewards=win_win_rejected_rewards,
                win_win_target_probs=inputs["win_win_target_probs"].to(win_win_chosen_rewards.dtype),
                win_win_pair_weights=inputs["win_win_pair_weights"].to(win_win_chosen_rewards.dtype),
                beta=self.beta,
            )

            completion_tokens = (
                (win_lose_inputs["chosen_labels"][:, 1:] != -100).sum() +
                (win_lose_inputs["rejected_labels"][:, 1:] != -100).sum() +
                (win_win_inputs["chosen_labels"][:, 1:] != -100).sum() +
                (win_win_inputs["rejected_labels"][:, 1:] != -100).sum()
            )
            self._store_metrics(
                {
                    "loss_wl": terms["win_lose_loss"].detach().item(),
                    "loss_geo": terms["win_win_loss"].detach().item(),
                    "reward_gap/wl": terms["win_lose_gap"].detach().item(),
                    "reward_gap/ww": terms["win_win_gap"].detach().item(),
                    "target_prob/ww_mean": terms["win_win_target_mean"].detach().item(),
                    "num_wl_pairs": terms["num_win_lose_pairs"],
                    "num_ww_pairs": terms["num_win_win_pairs"],
                    "num_tokens": completion_tokens.detach().item(),
                }
            )

            outputs = {
                "win_lose_chosen_rewards": win_lose_chosen_rewards.detach(),
                "win_lose_rejected_rewards": win_lose_rejected_rewards.detach(),
                "win_win_chosen_rewards": win_win_chosen_rewards.detach(),
                "win_win_rejected_rewards": win_win_rejected_rewards.detach(),
            }
            if return_outputs:
                return terms["loss"], outputs
            return terms["loss"]

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
            raise ValueError("--train-adapter-path currently supports text-only causal LM backbones")
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

    tokenized_win_lose_records = _tokenize_balanced_records(
        win_lose_rows,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
    )
    tokenized_win_win_records = _tokenize_balanced_records(
        win_win_rows,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
    )
    train_dataset = BalancedPairedDataset(
        win_lose_features=tokenized_win_lose_records,
        win_win_features=tokenized_win_win_records,
        seed=args.seed,
    )

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
            raise ValueError("--ref-adapter-path currently supports text-only causal LM backbones")
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

    trainer = BalancedGeometricTrainer(
        model=model,
        ref_model=ref_model,
        beta=args.beta,
        pad_token_id=tokenizer.pad_token_id,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=BalancedPreferenceDataCollator(tokenizer.pad_token_id),
    )
    trainer.add_callback(BalancedDatasetShuffleCallback(train_dataset, seed=args.seed))
    if resolved_stop_after_steps is not None:
        trainer.add_callback(StopAfterStepCallback(stop_after_steps=resolved_stop_after_steps))
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
        output_dir=output_dir,
        input_paths=input_paths,
        win_lose_pair_count=len(win_lose_rows),
        win_win_pair_count=len(win_win_rows),
        paired_steps_per_epoch=paired_steps_per_epoch,
        global_step=int(trainer.state.global_step),
        train_metrics=getattr(train_result, "metrics", None),
    )
    _write_json(output_dir / "train_summary.json", train_summary)


if __name__ == "__main__":
    main()
