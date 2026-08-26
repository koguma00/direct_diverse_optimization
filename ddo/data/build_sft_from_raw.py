#!/usr/bin/env python3
"""Build Hugging Face SFT JSONL records from raw thought-action runs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


USABLE_TRAJ_BASE = "base-traj"
SUCCESS_CRITERION_DEFAULT = "legacy_success"
SUCCESS_CRITERION_TEXTWORLD_SFT_V1 = "textworld_task_progression_sft_v1"
SUCCESS_CRITERION_WEBSHOP_REWARD_THRESHOLD = "webshop_reward_threshold"
WEBSHOP_SUCCESS_THRESHOLD = 0.9


def _task_slug(task_id: str) -> str:
    return task_id.rsplit("/", maxsplit=1)[-1]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_success(last_call: dict[str, Any]) -> bool:
    done = bool(last_call.get("done"))
    won = bool(last_call.get("won"))
    progression = _as_float(last_call.get("progression"), default=0.0)
    reward = _as_float(last_call.get("reward"), default=0.0)
    return won or progression >= 1.0 or (done and reward > 0.0)


def _parse_success_with_criterion(task_id: str, last_call: dict[str, Any], criterion: str) -> bool:
    return _parse_success_with_threshold(
        task_id,
        last_call,
        criterion,
        success_threshold=WEBSHOP_SUCCESS_THRESHOLD,
    )


def _parse_success_with_threshold(
    task_id: str,
    last_call: dict[str, Any],
    criterion: str,
    *,
    success_threshold: float,
) -> bool:
    if criterion == SUCCESS_CRITERION_DEFAULT:
        return _parse_success(last_call)
    if criterion == SUCCESS_CRITERION_TEXTWORLD_SFT_V1:
        task_slug = _task_slug(task_id)
        progression = _as_float(last_call.get("progression"), default=0.0)
        if task_slug == "the_cooking_game":
            return progression >= (5.0 / 17.0) - 1e-9
        if task_slug in {"treasure_hunter", "coin_collector"}:
            return progression >= 1.0 - 1e-9
        return _parse_success(last_call)
    if criterion == SUCCESS_CRITERION_WEBSHOP_REWARD_THRESHOLD:
        progression = _as_float(last_call.get("progression"), default=0.0)
        reward = _as_float(last_call.get("reward"), default=0.0)
        return max(progression, reward) >= float(success_threshold)
    raise ValueError(f"Unsupported success criterion: {criterion}")


def _threshold_slug(value: float) -> str:
    return str(float(value)).rstrip("0").rstrip(".").replace(".", "p")


def _usable_traj_label(*, success_only: bool, criterion: str, success_threshold: float) -> str:
    if not success_only:
        return USABLE_TRAJ_BASE
    if criterion == SUCCESS_CRITERION_WEBSHOP_REWARD_THRESHOLD:
        return f"{USABLE_TRAJ_BASE}-success-only-{criterion}_geq_{_threshold_slug(success_threshold)}"
    return f"{USABLE_TRAJ_BASE}-success-only-{criterion}"


def _find_latest_run_dir(run_root: Path) -> Path:
    run_dirs = sorted(path for path in run_root.iterdir() if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No raw run directories found under: {run_root}")
    return run_dirs[-1]


def _normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        return []

    messages: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if not role or not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def _fallback_prompt_messages(call: dict[str, Any]) -> list[dict[str, str]]:
    instruction = str(call.get("instruction") or "").strip()
    observation = str(call.get("observation") or "").strip()
    if not instruction or not observation:
        return []

    return [
        {"role": "user", "content": instruction},
        {
            "role": "user",
            "content": (
                f"Current Observation:\n{observation}\n\n"
                "Your response should use the following format:\n\n"
                "Thought: <your thoughts>\n"
                "Action: <your next action>"
            ),
        },
    ]


def _prompt_messages_from_call(call: dict[str, Any]) -> list[dict[str, str]]:
    extras = call.get("extras") or {}
    balrog_raw = extras.get("balrog_raw") or {}
    prompt_messages = _normalize_messages(balrog_raw.get("messages"))
    if prompt_messages:
        return prompt_messages
    return _fallback_prompt_messages(call)


def _build_completion_text(call: dict[str, Any]) -> str:
    extras = call.get("extras") or {}
    balrog_raw = extras.get("balrog_raw") or {}
    response = balrog_raw.get("response") or {}

    raw_completion = str(response.get("raw_completion") or call.get("raw_output") or "").strip()
    action = str(call.get("action") or "").strip()
    thought = str(call.get("thought") or "").strip()

    if raw_completion and raw_completion != action:
        return raw_completion
    if thought:
        return f"Thought: {thought}\n\nAction: {action}"
    if action:
        return f"Action: {action}"
    return raw_completion


def _collect_trace_paths(run_dir: Path) -> list[Path]:
    trace_paths = sorted(run_dir.glob("**/*_llm_trace.json"))
    return [
        path
        for path in trace_paths
        if "_dtc" not in path.parts and "__dtc_" not in path.name
    ]


def _build_rows_from_trace(
    trace_path: Path,
    *,
    source_root: Path,
    success_only: bool,
    criterion: str,
    success_threshold: float,
) -> list[dict[str, Any]]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    calls = payload.get("calls") or []
    if not isinstance(calls, list) or not calls:
        return []

    task_id = str(payload.get("task_or_env_params") or "").strip()
    seed = _as_int(payload.get("seed"), default=-1)
    episode_id = str(payload.get("episode_id") or trace_path.stem.replace("_llm_trace", "")).strip()
    last_call = calls[-1]
    trajectory_success = _parse_success_with_threshold(
        task_id,
        last_call,
        criterion,
        success_threshold=success_threshold,
    )
    if success_only and not trajectory_success:
        return []
    trajectory_progress = _as_float(last_call.get("progression"), default=0.0)
    trajectory_total_reward = _as_float(last_call.get("reward"), default=0.0)
    trajectory_total_steps = len(calls)

    rows: list[dict[str, Any]] = []
    for idx, call in enumerate(calls):
        prompt_messages = _prompt_messages_from_call(call)
        completion_text = _build_completion_text(call)
        action = str(call.get("action") or "").strip()
        if not prompt_messages or not completion_text or not action:
            continue

        rows.append(
            {
                "prompt": prompt_messages,
                "completion": [{"role": "assistant", "content": completion_text}],
                "task_id": task_id,
                "seed": seed,
                "family_id": episode_id,
                "trajectory_id": episode_id,
                "trajectory_kind": "base",
                "episode_step": _as_int(call.get("episode_step"), default=idx),
                "divergence_step": None,
                "state_key": None,
                "action_text": action,
                "trajectory_success": trajectory_success,
                "trajectory_progress": trajectory_progress,
                "trajectory_total_reward": trajectory_total_reward,
                "trajectory_total_steps": trajectory_total_steps,
                "source_trace": trace_path.relative_to(source_root).as_posix(),
            }
        )
    return rows


def build_sft_rows_from_raw(
    *,
    run_dir: Path,
    task_filter: str | None = None,
    success_only: bool = False,
    criterion: str = SUCCESS_CRITERION_DEFAULT,
    success_threshold: float = WEBSHOP_SUCCESS_THRESHOLD,
    shuffle_seed: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trajectory_count = 0
    task_counts: dict[str, int] = {}
    skipped_unsuccessful = 0

    normalized_filter = task_filter.strip() if task_filter else None
    for trace_path in tqdm(_collect_trace_paths(run_dir), desc="Raw SFT Traces", unit="trace"):
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        task_id = str(payload.get("task_or_env_params") or "").strip()
        task_slug = _task_slug(task_id) if task_id else ""
        if normalized_filter and task_slug != normalized_filter and task_id != normalized_filter:
            continue

        trajectory_rows = _build_rows_from_trace(
            trace_path,
            source_root=run_dir,
            success_only=success_only,
            criterion=criterion,
            success_threshold=success_threshold,
        )
        if not trajectory_rows:
            if success_only:
                skipped_unsuccessful += 1
            continue
        rows.extend(trajectory_rows)
        trajectory_count += 1
        task_counts[task_slug or "<missing>"] = task_counts.get(task_slug or "<missing>", 0) + 1

    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)

    stats = {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "usable_traj": _usable_traj_label(
            success_only=success_only,
            criterion=criterion,
            success_threshold=success_threshold,
        ),
        "task_filter": normalized_filter,
        "success_only": success_only,
        "criterion": criterion,
        "success_threshold": float(success_threshold),
        "trajectory_count": trajectory_count,
        "skipped_unsuccessful": skipped_unsuccessful,
        "output_rows": len(rows),
        "task_counts": task_counts,
        "shuffle_seed": shuffle_seed,
    }
    return rows, stats


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Hugging Face SFT prompt/completion JSONL from raw runs")
    parser.add_argument(
        "--run-root",
        default="data/raw/trajectories/balrog/textworld/thought_action",
        help="Root directory containing per-run raw thought-action directories",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Specific raw run directory. If omitted, the latest run under --run-root is used.",
    )
    parser.add_argument(
        "--task-filter",
        default=None,
        help="Optional task slug filter (for example: treasure_hunter)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSONL path (default: data/dataset/sft/<run_id>/base-traj/train.jsonl)",
    )
    parser.add_argument(
        "--stats-out",
        default=None,
        help="Output JSON path for conversion stats (default: data/dataset/sft/<run_id>/base-traj/conversion_stats.json)",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Keep only trajectories that satisfy the selected success criterion.",
    )
    parser.add_argument(
        "--criterion",
        default=SUCCESS_CRITERION_DEFAULT,
        choices=[
            SUCCESS_CRITERION_DEFAULT,
            SUCCESS_CRITERION_TEXTWORLD_SFT_V1,
            SUCCESS_CRITERION_WEBSHOP_REWARD_THRESHOLD,
        ],
        help="Success criterion used when --success-only is enabled.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=WEBSHOP_SUCCESS_THRESHOLD,
        help="Reward/progression threshold used by criterion=webshop_reward_threshold.",
    )
    parser.add_argument(
        "--shuffle-seed",
        default=None,
        help="Optional string seed for a deterministic one-time row shuffle.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root)
    run_dir = Path(args.run_dir) if args.run_dir else _find_latest_run_dir(run_root)
    run_id = run_dir.name

    usable_traj = _usable_traj_label(
        success_only=args.success_only,
        criterion=args.criterion,
        success_threshold=args.success_threshold,
    )
    out_path = (
        Path(args.out)
        if args.out
        else Path("data/dataset/sft") / run_id / usable_traj / "train.jsonl"
    )
    stats_path = (
        Path(args.stats_out)
        if args.stats_out
        else Path("data/dataset/sft") / run_id / usable_traj / "conversion_stats.json"
    )

    rows, stats = build_sft_rows_from_raw(
        run_dir=run_dir,
        task_filter=args.task_filter,
        success_only=args.success_only,
        criterion=args.criterion,
        success_threshold=args.success_threshold,
        shuffle_seed=args.shuffle_seed,
    )
    stats["out"] = str(out_path)

    _write_jsonl(out_path, rows)
    _write_json(stats_path, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
