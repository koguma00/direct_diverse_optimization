#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


BABYAI_TURN_LEFT = "turn left"
BABYAI_TURN_RIGHT = "turn right"
BABYAI_TURN_ACTIONS = {BABYAI_TURN_LEFT, BABYAI_TURN_RIGHT}
BABAISAI_INVERSE_ACTION = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


@dataclass(frozen=True)
class EffectiveDiversityStats:
    episode_count: int
    valid_episode_count: int
    success_count: int
    raw_unique_success: int
    effective_unique_success: int
    raw_entropy_effective_success: float
    effective_entropy_effective_success: float
    raw_esd: float
    raw_h_esd: float
    raw_wsd: float
    raw_h_wsd: float
    effective_esd: float
    effective_h_esd: float
    effective_wsd: float
    effective_h_wsd: float
    changed_success_trajectories: int
    removed_turn_actions: int


def is_successful_episode(episode: dict[str, Any]) -> bool:
    for key in ("success", "won"):
        if key in episode:
            value = _optional_bool(episode.get(key))
            if value is not None:
                return value

    progression = _optional_float(episode.get("progression"))
    if progression is not None:
        return progression >= 1.0

    calls = episode.get("calls")
    if isinstance(calls, list):
        for call in reversed(calls):
            if not isinstance(call, dict):
                continue
            won = _optional_bool(call.get("won"))
            if won is not None:
                return won
            progression = _optional_float(call.get("progression"))
            if progression is not None:
                return progression >= 1.0

    # Some older summaries only contain episode_return. Requiring a full unit
    # return avoids treating shaped partial rewards as task success.
    episode_return = _optional_float(episode.get("episode_return"))
    return episode_return is not None and episode_return >= 1.0


def is_valid_episode(episode: dict[str, Any]) -> bool:
    if _optional_bool(episode.get("aborted")) is True:
        return False
    if str(episode.get("abort_reason") or "").strip():
        return False
    status = str(episode.get("status") or "").strip().lower()
    if status in {"aborted", "missing", "missing_trace", "corrupt", "corrupted_trace"}:
        return False
    reason = str(episode.get("termination_reason") or "").strip().lower()
    return reason != "aborted" and not reason.startswith("aborted:")


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def entropy_effective_count(counts: Counter[Any]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return 2.0**entropy


def is_alternating_turn_block(block: Sequence[str]) -> bool:
    return all(block[idx] != block[idx - 1] for idx in range(1, len(block)))


def simplify_babyai_turn_block(block: Sequence[str]) -> tuple[list[str], int]:
    """Remove turn-only noise from one maximal BabyAI turn block.

    The rule is intentionally conservative:
    - same-direction full rotations are reduced modulo four;
    - alternating left/right or right/left blocks are removed only when they contain
      at least two canceling pairs, i.e. length >= 4.
    """
    actions = list(block)
    if not actions:
        return [], 0

    if len(set(actions)) == 1:
        keep = len(actions) % 4
        return actions[:keep], len(actions) - keep

    if is_alternating_turn_block(actions) and len(actions) >= 4:
        keep = len(actions) % 2
        return (actions[-keep:] if keep else []), len(actions) - keep

    return actions, 0


def prune_babyai_turn_noise_once(actions: Sequence[str]) -> tuple[list[str], int]:
    output: list[str] = []
    removed = 0
    idx = 0
    actions = list(actions)
    while idx < len(actions):
        action = actions[idx]
        if action not in BABYAI_TURN_ACTIONS:
            output.append(action)
            idx += 1
            continue

        end = idx + 1
        while end < len(actions) and actions[end] in BABYAI_TURN_ACTIONS:
            end += 1
        simplified, block_removed = simplify_babyai_turn_block(actions[idx:end])
        output.extend(simplified)
        removed += block_removed
        idx = end
    return output, removed


def prune_babyai_turn_noise(actions: Sequence[str]) -> tuple[tuple[str, ...], int]:
    current = list(actions)
    total_removed = 0
    while True:
        current, removed = prune_babyai_turn_noise_once(current)
        total_removed += removed
        if removed == 0:
            return tuple(current), total_removed


def _csv_bool(value: Any) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def _csv_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_terminal_or_reward_row(row: dict[str, str]) -> bool:
    return (
        _csv_float(row.get("reward")) > 0.0
        or _csv_bool(row.get("terminated")) is True
        or _csv_bool(row.get("truncated")) is True
        or _csv_bool(row.get("done")) is True
    )


def prune_babaisai_noop_loop_noise(
    rows: Sequence[dict[str, str]],
    *,
    action_column: str = "action_executed",
) -> tuple[tuple[str, ...], int]:
    """Remove Baba no-op steps and exact two-step inverse state loops.

    This filter intentionally uses environment observations, not action names alone:
    - non-terminal rows with ``obs_changed=False`` are removed as no-ops;
    - inverse movement pairs are canceled only when the second step returns to the
      exact textual observation before the first step.
    """
    stack: list[dict[str, str]] = []
    removed = 0
    for row in rows:
        action = (row.get(action_column) or "").strip()
        if not action:
            continue
        if _csv_bool(row.get("obs_changed")) is False and not _is_terminal_or_reward_row(row):
            removed += 1
            continue
        if stack:
            previous = stack[-1]
            previous_action = (previous.get(action_column) or "").strip()
            returned_to_previous_pre = (
                (previous.get("observation_pre") or "")
                and previous.get("observation_pre") == row.get("observation_post")
            )
            if (
                BABAISAI_INVERSE_ACTION.get(previous_action) == action
                and returned_to_previous_pre
                and not _is_terminal_or_reward_row(previous)
                and not _is_terminal_or_reward_row(row)
            ):
                stack.pop()
                removed += 2
                continue
        stack.append(row)
    return tuple((row.get(action_column) or "").strip() for row in stack), removed


def effective_strategy_key(
    actions: Sequence[str],
    *,
    filter_name: str = "babyai_turn_noise",
) -> tuple[str, ...]:
    if filter_name in {"none", "babaisai_raw"}:
        return tuple(actions)
    if filter_name == "babyai_turn_noise":
        return prune_babyai_turn_noise(actions)[0]
    if filter_name == "babaisai_noop_loop_prune":
        raise ValueError("babaisai_noop_loop_prune requires run-dir CSV observation rows")
    raise ValueError(f"unsupported effective diversity filter: {filter_name}")


def read_episode_json_records(run_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(run_dir.rglob("*_run_*.json")):
        if path.name.endswith("_llm_trace.json") or path.name.endswith("_summary.json"):
            continue
        records.append((path.stem, json.loads(path.read_text(encoding="utf-8"))))
    return records


def read_action_sequences(
    run_dir: Path,
    *,
    action_column: str = "action_executed",
) -> dict[str, tuple[str, ...]]:
    sequences: dict[str, tuple[str, ...]] = {}
    for path in sorted(run_dir.rglob("*_run_*.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            actions = tuple(
                action
                for action in ((row.get(action_column) or "").strip() for row in reader)
                if action
            )
        sequences[path.stem] = actions
    return sequences


def read_action_rows(run_dir: Path) -> dict[str, list[dict[str, str]]]:
    rows_by_stem: dict[str, list[dict[str, str]]] = {}
    for path in sorted(run_dir.rglob("*_run_*.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows_by_stem[path.stem] = list(csv.DictReader(handle))
    return rows_by_stem


def aggregate_effective_diversity(
    *,
    success_stems: Iterable[str],
    action_sequences: dict[str, tuple[str, ...]],
    budget: int,
    filter_name: str = "babyai_turn_noise",
    episode_count: int | None = None,
    valid_episode_count: int | None = None,
) -> EffectiveDiversityStats:
    success_stem_set = set(success_stems)
    raw_counter = Counter(
        sequence for stem, sequence in action_sequences.items() if stem in success_stem_set
    )

    effective_counter: Counter[tuple[str, ...]] = Counter()
    changed_success_trajectories = 0
    removed_turn_actions = 0
    for stem, actions in action_sequences.items():
        if stem not in success_stem_set:
            continue
        if filter_name == "babyai_turn_noise":
            key, removed = prune_babyai_turn_noise(actions)
        elif filter_name in {"none", "babaisai_raw"}:
            key, removed = tuple(actions), 0
        elif filter_name == "babaisai_noop_loop_prune":
            raise ValueError("babaisai_noop_loop_prune requires run-dir CSV observation rows")
        else:
            raise ValueError(f"unsupported effective diversity filter: {filter_name}")
        effective_counter[key] += 1
        if removed:
            changed_success_trajectories += 1
            removed_turn_actions += removed

    total_count = int(episode_count if episode_count is not None else len(action_sequences))
    valid_count = int(
        valid_episode_count
        if valid_episode_count is not None
        else episode_count
        if episode_count is not None
        else budget
    )
    return _stats_from_counters(
        episode_count=total_count,
        valid_episode_count=valid_count,
        success_count=len(success_stem_set),
        raw_counter=raw_counter,
        effective_counter=effective_counter,
        changed_success_trajectories=changed_success_trajectories,
        removed_turn_actions=removed_turn_actions,
    )


def _stats_from_counters(
    *,
    episode_count: int,
    valid_episode_count: int,
    success_count: int,
    raw_counter: Counter[Any],
    effective_counter: Counter[Any],
    changed_success_trajectories: int,
    removed_turn_actions: int,
) -> EffectiveDiversityStats:
    raw_unique = len(raw_counter)
    effective_unique = len(effective_counter)
    raw_entropy = entropy_effective_count(raw_counter)
    effective_entropy = entropy_effective_count(effective_counter)
    valid_denominator = max(1, valid_episode_count)
    success_denominator = max(1, success_count)
    return EffectiveDiversityStats(
        episode_count=episode_count,
        valid_episode_count=valid_episode_count,
        success_count=success_count,
        raw_unique_success=raw_unique,
        effective_unique_success=effective_unique,
        raw_entropy_effective_success=raw_entropy,
        effective_entropy_effective_success=effective_entropy,
        raw_esd=raw_unique / valid_denominator,
        raw_h_esd=raw_entropy / valid_denominator,
        raw_wsd=raw_unique / success_denominator,
        raw_h_wsd=raw_entropy / success_denominator,
        effective_esd=effective_unique / valid_denominator,
        effective_h_esd=effective_entropy / valid_denominator,
        effective_wsd=effective_unique / success_denominator,
        effective_h_wsd=effective_entropy / success_denominator,
        changed_success_trajectories=changed_success_trajectories,
        removed_turn_actions=removed_turn_actions,
    )


def _trace_aborted(payload: dict[str, Any]) -> bool:
    calls = payload.get("calls")
    if not isinstance(calls, list):
        return True
    for call in calls:
        if not isinstance(call, dict):
            continue
        reason = str(call.get("termination_reason") or "").strip().lower()
        extras = call.get("extras") if isinstance(call.get("extras"), dict) else {}
        if reason == "aborted" or reason.startswith("aborted:") or extras.get("abort_reason"):
            return True
    return False


def _webshop_trace_success(payload: dict[str, Any], success_threshold: float = 0.9) -> bool:
    calls = payload.get("calls") if isinstance(payload.get("calls"), list) else []
    if not calls or not isinstance(calls[-1], dict):
        return False
    terminal = calls[-1]
    reward = _optional_float(terminal.get("reward")) or 0.0
    progression = _optional_float(terminal.get("progression")) or 0.0
    return max(reward, progression) >= success_threshold


def aggregate_webshop_run_dir(
    run_dir: Path,
    *,
    success_threshold: float = 0.9,
    trajectory_class: str = "purchase_action_type_trace",
) -> EffectiveDiversityStats:
    from ddo.metrics.webshop.summarize import (
        DIVERSITY_PURCHASE_ACTION_TYPE_TRACE,
        MissingPurchaseSignatureError,
        SUPPORTED_DIVERSITY_KEYS,
        _purchase_signature,
        _strategy_key,
    )

    if trajectory_class not in SUPPORTED_DIVERSITY_KEYS:
        raise ValueError(f"Unsupported WebShop trajectory class: {trajectory_class}")

    payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(run_dir.rglob("*_llm_trace.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("env_name") or payload.get("benchmark") or "").lower() != "webshop":
            continue
        payloads.append((path, payload))

    success_payloads = [
        record
        for record in payloads
        if not _trace_aborted(record[1])
        and _webshop_trace_success(record[1], success_threshold)
    ]
    missing_success_paths = [
        path for path, payload in success_payloads if not _purchase_signature(payload)
    ]
    if trajectory_class == DIVERSITY_PURCHASE_ACTION_TYPE_TRACE and missing_success_paths:
        examples = ", ".join(str(path) for path in missing_success_paths[:5])
        raise MissingPurchaseSignatureError(
            "purchase_action_type_trace cannot classify successful rollouts: "
            f"missing_purchase_signature_count={len(missing_success_paths)}; traces={examples}"
        )

    raw_counter: Counter[tuple[str, ...]] = Counter()
    effective_counter: Counter[str] = Counter()
    for _path, payload in success_payloads:
        calls = payload.get("calls") if isinstance(payload.get("calls"), list) else []
        actions = tuple(
            str(call.get("action") or "").strip()
            for call in calls
            if isinstance(call, dict) and str(call.get("action") or "").strip()
        )
        raw_counter[actions] += 1
        effective_counter[_strategy_key(payload, trajectory_class)] += 1
    return _stats_from_counters(
        episode_count=len(payloads),
        valid_episode_count=len(payloads),
        success_count=len(success_payloads),
        raw_counter=raw_counter,
        effective_counter=effective_counter,
        changed_success_trajectories=len(success_payloads),
        removed_turn_actions=0,
    )


def aggregate_run_dir_effective_diversity(
    run_dir: Path,
    *,
    budget: int,
    action_column: str = "action_executed",
    filter_name: str = "babyai_turn_noise",
    webshop_success_threshold: float = 0.9,
    webshop_trajectory_class: str = "purchase_action_type_trace",
) -> EffectiveDiversityStats:
    if filter_name == "webshop_strategy":
        return aggregate_webshop_run_dir(
            run_dir,
            success_threshold=webshop_success_threshold,
            trajectory_class=webshop_trajectory_class,
        )

    episode_records = read_episode_json_records(run_dir)
    valid_records = [record for record in episode_records if is_valid_episode(record[1])]
    success_stems = [stem for stem, episode in valid_records if is_successful_episode(episode)]
    if filter_name == "babaisai_noop_loop_prune":
        success_stem_set = set(success_stems)
        action_rows = read_action_rows(run_dir)
        raw_counter = Counter(
            tuple(
                action
                for action in ((row.get(action_column) or "").strip() for row in rows)
                if action
            )
            for stem, rows in action_rows.items()
            if stem in success_stem_set
        )
        effective_counter: Counter[tuple[str, ...]] = Counter()
        changed_success_trajectories = 0
        removed_actions = 0
        for stem, rows in action_rows.items():
            if stem not in success_stem_set:
                continue
            raw_key = tuple(
                action
                for action in ((row.get(action_column) or "").strip() for row in rows)
                if action
            )
            effective_key, removed = prune_babaisai_noop_loop_noise(
                rows,
                action_column=action_column,
            )
            effective_counter[effective_key] += 1
            if effective_key != raw_key:
                changed_success_trajectories += 1
            removed_actions += removed
        success_count = len(success_stem_set)
        return _stats_from_counters(
            episode_count=len(episode_records),
            valid_episode_count=len(valid_records),
            success_count=success_count,
            raw_counter=raw_counter,
            effective_counter=effective_counter,
            changed_success_trajectories=changed_success_trajectories,
            removed_turn_actions=removed_actions,
        )
    action_sequences = read_action_sequences(run_dir, action_column=action_column)
    return aggregate_effective_diversity(
        success_stems=success_stems,
        action_sequences=action_sequences,
        budget=budget,
        filter_name=filter_name,
        episode_count=len(episode_records),
        valid_episode_count=len(valid_records),
    )


def mean_effective_diversity(stats: Sequence[EffectiveDiversityStats]) -> dict[str, float]:
    if not stats:
        raise ValueError("cannot average empty effective diversity stats")
    return {
        "success_count": sum(row.success_count for row in stats) / len(stats),
        "valid_episode_count": sum(row.valid_episode_count for row in stats) / len(stats),
        "raw_unique_success": sum(row.raw_unique_success for row in stats) / len(stats),
        "effective_unique_success": sum(row.effective_unique_success for row in stats) / len(stats),
        "raw_entropy_effective_success": sum(row.raw_entropy_effective_success for row in stats)
        / len(stats),
        "effective_entropy_effective_success": sum(
            row.effective_entropy_effective_success for row in stats
        )
        / len(stats),
        "raw_esd": sum(row.raw_esd for row in stats) / len(stats),
        "raw_h_esd": sum(row.raw_h_esd for row in stats) / len(stats),
        "raw_wsd": sum(row.raw_wsd for row in stats) / len(stats),
        "raw_h_wsd": sum(row.raw_h_wsd for row in stats) / len(stats),
        "effective_esd": sum(row.effective_esd for row in stats) / len(stats),
        "effective_h_esd": sum(row.effective_h_esd for row in stats) / len(stats),
        "effective_wsd": sum(row.effective_wsd for row in stats) / len(stats),
        "effective_h_wsd": sum(row.effective_h_wsd for row in stats) / len(stats),
        "changed_success_trajectories": sum(row.changed_success_trajectories for row in stats),
        "removed_turn_actions": sum(row.removed_turn_actions for row in stats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute raw and effective success-conditioned diversity for one eval run directory."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--action-column", default="action_executed")
    parser.add_argument(
        "--filter",
        default="babyai_turn_noise",
        choices=(
            "babyai_turn_noise",
            "babaisai_raw",
            "babaisai_noop_loop_prune",
            "webshop_strategy",
            "none",
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of key=value lines.")
    args = parser.parse_args()

    stats = aggregate_run_dir_effective_diversity(
        args.run_dir,
        budget=args.budget,
        action_column=args.action_column,
        filter_name=args.filter,
    )
    payload = asdict(stats)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        for key, value in payload.items():
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
