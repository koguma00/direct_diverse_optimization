#!/usr/bin/env python3
"""Build legacy DPO/DDO pair JSONLs with explicit win filtering from a DTC run."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


USABLE_TRAJ_BASE_ALT = "base-alt"
USABLE_TRAJ_BASE_ALT_ALT = "base-alt-alt"
CRITERION_TERMINAL_PROGRESSION = "terminal_progression"
CRITERION_TEXTWORLD_TASK_PROGRESSION_V1 = "textworld_task_progression_v1"
CRITERION_WEBSHOP_REWARD_THRESHOLD = "webshop_reward_threshold"
TEXTWORLD_COOKING_WIN_THRESHOLD = 10.0 / 17.0
WEBSHOP_SUCCESS_THRESHOLD = 0.9
QUALITY_SCORE_MAX_SAME_OBS_ACTION_RUN = "max_same_obs_action_run"
PAIR_TYPE_WIN_LOSE = "win_lose"
PAIR_TYPE_WIN_WIN = "win_win"
PAIR_SEMANTICS_OBSERVED_WIN_LOSE = "observed_win_lose"
PAIR_SEMANTICS_OBSERVED_WIN_WIN = "observed_win_win"
PAIR_SEMANTICS_PSEUDO_BASE_ALT_WIN_LENGTH = "pseudo_base_alt_win_length"
RANKING_SCORE_POST_DIVERGENCE_CALL_COUNT = "post_divergence_call_count"


@dataclass(frozen=True)
class Candidate:
    family_id: str
    trajectory_id: str
    trajectory_kind: str
    divergence_step: int
    state_key: str
    task_id: str
    seed: int
    prompt_messages: list[dict[str, str]]
    response_text: str
    action_text: str
    terminal_progression: float
    terminal_done: bool
    source_trace: str
    quality_score: int
    is_win: bool
    post_divergence_call_count: int


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


def _find_latest_run_dir(dtc_root: Path) -> Path:
    run_dirs = sorted(path for path in dtc_root.iterdir() if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No DTC run directories found under: {dtc_root}")
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


def _build_response_text(call: dict[str, Any]) -> str:
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


def _flatten_messages(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts).strip()


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
                raise ValueError(f"{path}:{line_idx} invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_idx} row must be an object")
            rows.append(row)
    return rows


def _collect_base_trace_paths(base_run_dir: Path) -> list[Path]:
    trace_paths = sorted(base_run_dir.glob("**/*_llm_trace.json"))
    return [
        path
        for path in trace_paths
        if "_dtc" not in path.parts and "__dtc_" not in path.name
    ]


def _episode_id_from_trace(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episode_id = str(payload.get("episode_id") or path.stem.replace("_llm_trace", "")).strip()
    if not episode_id:
        raise ValueError(f"Missing episode_id in trace: {path}")
    return episode_id


def _resolve_trace_path(dtc_run_dir: Path, candidate: str, trajectory_id: str) -> Path:
    candidate_path = Path(candidate)
    if not candidate_path.is_absolute():
        candidate_path = dtc_run_dir / candidate_path
    if candidate_path.exists():
        return candidate_path

    matches = sorted(dtc_run_dir.glob(f"**/{trajectory_id}_llm_trace.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Trace not found for trajectory: {trajectory_id}")
    raise ValueError(f"Multiple matching traces found for trajectory: {trajectory_id}")


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("calls"), list) or not payload["calls"]:
        raise ValueError(f"Trace has no calls: {path}")
    return payload


def _terminal_progression(payload: dict[str, Any]) -> float:
    return _as_float(payload["calls"][-1].get("progression"), default=0.0)


def _terminal_reward(payload: dict[str, Any]) -> float:
    return _as_float(payload["calls"][-1].get("reward"), default=0.0)


def _terminal_score(payload: dict[str, Any]) -> float:
    return max(_terminal_progression(payload), _terminal_reward(payload))


def _terminal_done(payload: dict[str, Any]) -> bool:
    return bool(payload["calls"][-1].get("done"))


def _payload_task_slug(payload: dict[str, Any]) -> str:
    task_id = str(payload.get("task_or_env_params") or "").strip()
    if not task_id:
        return ""
    return task_id.rsplit("/", maxsplit=1)[-1]


def _is_win(payload: dict[str, Any], criterion: str, *, success_threshold: float = WEBSHOP_SUCCESS_THRESHOLD) -> bool:
    progression = _terminal_progression(payload)
    if criterion == CRITERION_TERMINAL_PROGRESSION:
        return progression >= 1.0
    if criterion == CRITERION_WEBSHOP_REWARD_THRESHOLD:
        return _terminal_score(payload) >= float(success_threshold)
    if criterion == CRITERION_TEXTWORLD_TASK_PROGRESSION_V1:
        task_slug = _payload_task_slug(payload)
        if task_slug == "the_cooking_game":
            return progression >= TEXTWORLD_COOKING_WIN_THRESHOLD
        if task_slug in {"treasure_hunter", "coin_collector"}:
            return progression >= 1.0
        raise ValueError(
            "textworld_task_progression_v1 only supports TextWorld tasks; "
            f"got task={task_slug or '<missing>'}"
        )
    raise ValueError(f"Unsupported criterion: {criterion}")


def compute_max_same_obs_action_run(payload: dict[str, Any]) -> int:
    calls = payload.get("calls") or []
    if not isinstance(calls, list) or not calls:
        return 0

    max_run = 0
    current_run = 0
    previous_key: tuple[str, str] | None = None
    for call in calls:
        if not isinstance(call, dict):
            previous_key = None
            current_run = 0
            continue
        key = (
            str(call.get("observation") or ""),
            str(call.get("action") or ""),
        )
        if key == previous_key:
            current_run += 1
        else:
            current_run = 1
            previous_key = key
        if current_run > max_run:
            max_run = current_run
    return max_run


def _compute_quality_score(payload: dict[str, Any], quality_score: str) -> int:
    if quality_score != QUALITY_SCORE_MAX_SAME_OBS_ACTION_RUN:
        raise ValueError(f"Unsupported quality score: {quality_score}")
    return compute_max_same_obs_action_run(payload)


def _candidate_from_payload(
    *,
    payload: dict[str, Any],
    trace_path: Path,
    family_id: str,
    trajectory_id: str,
    trajectory_kind: str,
    divergence_step: int,
    state_key: str,
    prompt_override: list[dict[str, str]] | None,
    criterion: str,
    quality_score: str,
    success_threshold: float = WEBSHOP_SUCCESS_THRESHOLD,
) -> Candidate:
    calls = payload["calls"]
    if divergence_step < 0 or divergence_step >= len(calls):
        raise ValueError(f"divergence_step {divergence_step} is out of range for {trace_path}")

    call = calls[divergence_step]
    prompt_messages = list(prompt_override) if prompt_override is not None else _prompt_messages_from_call(call)
    response_text = _build_response_text(call)
    if not prompt_messages:
        raise ValueError(f"Missing prompt messages at divergence step for {trace_path}")
    if not response_text:
        raise ValueError(f"Missing response text at divergence step for {trace_path}")

    return Candidate(
        family_id=family_id,
        trajectory_id=trajectory_id,
        trajectory_kind=trajectory_kind,
        divergence_step=divergence_step,
        state_key=state_key,
        task_id=str(payload.get("task_or_env_params") or "").strip(),
        seed=_as_int(payload.get("seed"), default=-1),
        prompt_messages=prompt_messages,
        response_text=response_text,
        action_text=str(call.get("action") or "").strip(),
        terminal_progression=_terminal_progression(payload),
        terminal_done=_terminal_done(payload),
        source_trace=str(trace_path),
        quality_score=_compute_quality_score(payload, quality_score),
        is_win=_is_win(payload, criterion, success_threshold=success_threshold),
        post_divergence_call_count=len(calls) - divergence_step,
    )


def _task_slug(task_id: str) -> str:
    task_id = str(task_id).strip()
    if "/" in task_id:
        return task_id.rsplit("/", 1)[-1]
    return task_id or "unknown"


def _pair_origin(candidate_a: Candidate, candidate_b: Candidate) -> str:
    if "base" in {candidate_a.trajectory_kind, candidate_b.trajectory_kind}:
        return "base-alt"
    return "alt-alt"


def _is_allowed_pair(candidate_a: Candidate, candidate_b: Candidate, usable_traj: str) -> bool:
    kinds = {candidate_a.trajectory_kind, candidate_b.trajectory_kind}
    if kinds == {"base", "alt"}:
        return True
    if kinds == {"alt"} and usable_traj == USABLE_TRAJ_BASE_ALT_ALT:
        return True
    return False


def _ordered_win_win(candidate_a: Candidate, candidate_b: Candidate) -> tuple[Candidate, Candidate]:
    if candidate_a.trajectory_kind == "base" and candidate_b.trajectory_kind != "base":
        return candidate_a, candidate_b
    if candidate_b.trajectory_kind == "base" and candidate_a.trajectory_kind != "base":
        return candidate_b, candidate_a
    if candidate_a.trajectory_id <= candidate_b.trajectory_id:
        return candidate_a, candidate_b
    return candidate_b, candidate_a


def _ordered_by_post_divergence_call_count(
    candidate_a: Candidate,
    candidate_b: Candidate,
) -> tuple[Candidate, Candidate]:
    ranked = sorted(
        (candidate_a, candidate_b),
        key=lambda candidate: (
            candidate.post_divergence_call_count,
            0 if candidate.trajectory_kind == "base" else 1,
            candidate.trajectory_id,
        ),
    )
    return ranked[0], ranked[1]


def _legacy_pair_row(
    *,
    pair_type: str,
    chosen: Candidate,
    rejected: Candidate,
    quality_score_name: str,
    pair_semantics: str | None = None,
    ranking_score_name: str | None = None,
    chosen_ranking_score: int | float | None = None,
    rejected_ranking_score: int | float | None = None,
) -> dict[str, Any]:
    return {
        "pair_type": pair_type,
        "pair_semantics": pair_semantics,
        "task_id": chosen.task_id,
        "family_id": chosen.family_id,
        "divergence_step": chosen.divergence_step,
        "state_key": chosen.state_key,
        "pair_origin": _pair_origin(chosen, rejected),
        "chosen_seed": chosen.seed,
        "rejected_seed": rejected.seed,
        "chosen_trajectory_id": chosen.trajectory_id,
        "rejected_trajectory_id": rejected.trajectory_id,
        "chosen_trajectory_kind": chosen.trajectory_kind,
        "rejected_trajectory_kind": rejected.trajectory_kind,
        "chosen_source_trace": chosen.source_trace,
        "rejected_source_trace": rejected.source_trace,
        "quality_score_name": quality_score_name,
        "chosen_quality_score": chosen.quality_score,
        "rejected_quality_score": rejected.quality_score,
        "ranking_score_name": ranking_score_name,
        "chosen_ranking_score": chosen_ranking_score,
        "rejected_ranking_score": rejected_ranking_score,
        "prompt": _flatten_messages(chosen.prompt_messages),
        "chosen": {
            "action_text": chosen.response_text,
            "success": chosen.is_win,
            "progress": chosen.terminal_progression,
        },
        "rejected": {
            "action_text": rejected.response_text,
            "success": rejected.is_win,
            "progress": rejected.terminal_progression,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_filtered_legacy_pairs(
    *,
    dtc_run_dir: Path,
    base_run_dir: Path,
    usable_traj: str,
    criterion: str,
    quality_score: str,
    max_same_obs_action_run_threshold: int,
    include_base_alt_win_pseudo_pairs: bool = False,
    max_families: int = 0,
    success_threshold: float = WEBSHOP_SUCCESS_THRESHOLD,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any], list[dict[str, Any]]]:
    if max_same_obs_action_run_threshold < 1:
        raise ValueError("max_same_obs_action_run_threshold must be >= 1")

    branch_index_path = dtc_run_dir / "_dtc" / "branch_index.jsonl"
    if not branch_index_path.exists():
        raise FileNotFoundError(f"Missing DTC branch index: {branch_index_path}")

    base_trace_by_id: dict[str, Path] = {}
    for trace_path in _collect_base_trace_paths(base_run_dir):
        base_trace_by_id[_episode_id_from_trace(trace_path)] = trace_path

    allowed_families = sorted(base_trace_by_id)
    if max_families > 0:
        allowed_families = allowed_families[:max_families]
    allowed_family_set = set(allowed_families)

    branch_records = _read_jsonl(branch_index_path)
    grouped_records: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for record in branch_records:
        if record.get("status") != "generated":
            continue
        if not bool(record.get("replay_success")):
            continue
        family_id = str(record.get("base_traj_id") or "").strip()
        if family_id not in allowed_family_set:
            continue
        divergence_step = _as_int(record.get("divergence_step"), default=-1)
        state_key = str(record.get("state_key") or "").strip()
        if divergence_step < 0 or not state_key:
            continue
        grouped_records.setdefault((family_id, divergence_step, state_key), []).append(record)

    win_lose_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    win_win_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "task_id": "",
            "task_slug": "",
            "state_group_count": 0,
            "raw_win_candidates": 0,
            "quality_filtered_win_candidates": 0,
            "support_filtered_win_candidates": 0,
            "lose_candidates": 0,
            "win_lose_rows": 0,
            "observed_win_lose_rows": 0,
            "pseudo_win_lose_rows": 0,
            "win_win_rows": 0,
            "quality_filter_threshold": max_same_obs_action_run_threshold,
            "quality_score": quality_score,
            "win_trajectory_ids": set(),
            "lose_trajectory_ids": set(),
            "rejected_trajectory_ids": set(),
        }
    )

    global_stats = {
        "run_dir": str(dtc_run_dir),
        "run_id": dtc_run_dir.name,
        "usable_traj": usable_traj,
        "criterion": criterion,
        "success_threshold": float(success_threshold),
        "quality_score": quality_score,
        "max_same_obs_action_run_threshold": max_same_obs_action_run_threshold,
        "include_base_alt_win_pseudo_pairs": include_base_alt_win_pseudo_pairs,
        "family_count": len(allowed_families),
        "state_group_count": len(grouped_records),
        "raw_win_candidates": 0,
        "quality_filtered_win_candidates": 0,
        "support_filtered_win_candidates": 0,
        "lose_candidates": 0,
        "win_lose_rows": 0,
        "observed_win_lose_rows": 0,
        "pseudo_win_lose_rows": 0,
        "win_win_rows": 0,
    }

    for (family_id, divergence_step, state_key), records in tqdm(
        grouped_records.items(),
        total=len(grouped_records),
        desc="Filtered Preference States",
        unit="state",
    ):
        base_trace_path = base_trace_by_id.get(family_id)
        if base_trace_path is None:
            raise FileNotFoundError(f"Missing base trace for family: {family_id}")

        base_payload = _load_payload(base_trace_path)
        base_prompt = _prompt_messages_from_call(base_payload["calls"][divergence_step])
        base_candidate = _candidate_from_payload(
            payload=base_payload,
            trace_path=base_trace_path,
            family_id=family_id,
            trajectory_id=family_id,
            trajectory_kind="base",
            divergence_step=divergence_step,
            state_key=state_key,
            prompt_override=base_prompt,
            criterion=criterion,
            quality_score=quality_score,
            success_threshold=success_threshold,
        )

        task_slug = _task_slug(base_candidate.task_id)
        task_stat = task_stats[task_slug]
        task_stat["task_id"] = base_candidate.task_id
        task_stat["task_slug"] = task_slug
        task_stat["state_group_count"] += 1

        candidates: list[Candidate] = [base_candidate]
        for record in records:
            alt_id = str(record.get("alt_traj_id") or "").strip()
            if not alt_id:
                continue
            alt_trace_path = _resolve_trace_path(
                dtc_run_dir,
                str(record.get("output_trace") or "").strip(),
                alt_id,
            )
            alt_payload = _load_payload(alt_trace_path)
            candidates.append(
                _candidate_from_payload(
                    payload=alt_payload,
                    trace_path=alt_trace_path,
                    family_id=family_id,
                    trajectory_id=alt_id,
                    trajectory_kind="alt",
                    divergence_step=divergence_step,
                    state_key=state_key,
                    prompt_override=base_prompt,
                    criterion=criterion,
                    quality_score=quality_score,
                    success_threshold=success_threshold,
                )
            )

        win_candidates = [candidate for candidate in candidates if candidate.is_win]
        lose_candidates = [candidate for candidate in candidates if not candidate.is_win]
        quality_filtered_wins = [
            candidate
            for candidate in win_candidates
            if candidate.quality_score <= max_same_obs_action_run_threshold
        ]
        support_filtered_wins = [
            candidate
            for candidate in quality_filtered_wins
            if any(_is_allowed_pair(candidate, lose_candidate, usable_traj) for lose_candidate in lose_candidates)
        ]

        global_stats["raw_win_candidates"] += len(win_candidates)
        global_stats["quality_filtered_win_candidates"] += len(quality_filtered_wins)
        global_stats["support_filtered_win_candidates"] += len(support_filtered_wins)
        global_stats["lose_candidates"] += len(lose_candidates)
        task_stat["raw_win_candidates"] += len(win_candidates)
        task_stat["quality_filtered_win_candidates"] += len(quality_filtered_wins)
        task_stat["support_filtered_win_candidates"] += len(support_filtered_wins)
        task_stat["lose_candidates"] += len(lose_candidates)

        for chosen in support_filtered_wins:
            connected_loses = [
                rejected
                for rejected in lose_candidates
                if _is_allowed_pair(chosen, rejected, usable_traj)
            ]
            for rejected in connected_loses:
                row = _legacy_pair_row(
                    pair_type=PAIR_TYPE_WIN_LOSE,
                    chosen=chosen,
                    rejected=rejected,
                    quality_score_name=quality_score,
                    pair_semantics=PAIR_SEMANTICS_OBSERVED_WIN_LOSE,
                )
                win_lose_by_task[task_slug].append(row)
                global_stats["win_lose_rows"] += 1
                global_stats["observed_win_lose_rows"] += 1
                task_stat["win_lose_rows"] += 1
                task_stat["observed_win_lose_rows"] += 1
                task_stat["win_trajectory_ids"].add(chosen.trajectory_id)
                task_stat["lose_trajectory_ids"].add(rejected.trajectory_id)
                task_stat["rejected_trajectory_ids"].add(rejected.trajectory_id)

        if include_base_alt_win_pseudo_pairs:
            base_quality_filtered_win = next(
                (
                    candidate
                    for candidate in quality_filtered_wins
                    if candidate.trajectory_kind == "base"
                ),
                None,
            )
            if base_quality_filtered_win is not None:
                for alt_quality_filtered_win in quality_filtered_wins:
                    if alt_quality_filtered_win.trajectory_kind != "alt":
                        continue
                    if not _is_allowed_pair(base_quality_filtered_win, alt_quality_filtered_win, usable_traj):
                        continue
                    chosen, rejected = _ordered_by_post_divergence_call_count(
                        base_quality_filtered_win,
                        alt_quality_filtered_win,
                    )
                    row = _legacy_pair_row(
                        pair_type=PAIR_TYPE_WIN_LOSE,
                        chosen=chosen,
                        rejected=rejected,
                        quality_score_name=quality_score,
                        pair_semantics=PAIR_SEMANTICS_PSEUDO_BASE_ALT_WIN_LENGTH,
                        ranking_score_name=RANKING_SCORE_POST_DIVERGENCE_CALL_COUNT,
                        chosen_ranking_score=chosen.post_divergence_call_count,
                        rejected_ranking_score=rejected.post_divergence_call_count,
                    )
                    win_lose_by_task[task_slug].append(row)
                    global_stats["win_lose_rows"] += 1
                    global_stats["pseudo_win_lose_rows"] += 1
                    task_stat["win_lose_rows"] += 1
                    task_stat["pseudo_win_lose_rows"] += 1
                    task_stat["win_trajectory_ids"].add(chosen.trajectory_id)
                    task_stat["win_trajectory_ids"].add(rejected.trajectory_id)
                    task_stat["rejected_trajectory_ids"].add(rejected.trajectory_id)

        ordered_wins = sorted(
            support_filtered_wins,
            key=lambda candidate: (
                0 if candidate.trajectory_kind == "base" else 1,
                candidate.trajectory_id,
            ),
        )
        for candidate_a, candidate_b in itertools.combinations(ordered_wins, 2):
            if not _is_allowed_pair(candidate_a, candidate_b, usable_traj):
                continue
            chosen, rejected = _ordered_win_win(candidate_a, candidate_b)
            row = _legacy_pair_row(
                pair_type=PAIR_TYPE_WIN_WIN,
                chosen=chosen,
                rejected=rejected,
                quality_score_name=quality_score,
                pair_semantics=PAIR_SEMANTICS_OBSERVED_WIN_WIN,
            )
            win_win_by_task[task_slug].append(row)
            global_stats["win_win_rows"] += 1
            task_stat["win_win_rows"] += 1
            task_stat["win_trajectory_ids"].add(chosen.trajectory_id)
            task_stat["win_trajectory_ids"].add(rejected.trajectory_id)

    task_rows: list[dict[str, Any]] = []
    for task_slug in sorted(task_stats):
        task_stat = dict(task_stats[task_slug])
        task_stat["unique_win_trajectory_count"] = len(task_stat.pop("win_trajectory_ids"))
        task_stat["unique_lose_trajectory_count"] = len(task_stat.pop("lose_trajectory_ids"))
        task_stat["unique_rejected_trajectory_count"] = len(task_stat.pop("rejected_trajectory_ids"))
        task_rows.append(task_stat)

    summary = {
        **global_stats,
        "tasks": task_rows,
    }
    return dict(win_lose_by_task), dict(win_win_by_task), summary, task_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build filtered legacy DPO/DDO pair JSONLs from a DTC run while preserving same-state semantics."
    )
    parser.add_argument(
        "--dtc-root",
        default="data/raw/trajectories/balrog/babyai/divergence_tree",
        help="Root directory containing per-run DTC directories",
    )
    parser.add_argument(
        "--dtc-run-dir",
        default=None,
        help="Specific DTC run directory. If omitted, the latest run under --dtc-root is used.",
    )
    parser.add_argument(
        "--base-run-dir",
        required=True,
        help="Directory containing the separately stored base *_llm_trace.json files.",
    )
    parser.add_argument(
        "--usable-traj",
        required=True,
        choices=(USABLE_TRAJ_BASE_ALT, USABLE_TRAJ_BASE_ALT_ALT),
        help="Which trajectory combinations to consider within each same-state group",
    )
    parser.add_argument(
        "--criterion",
        required=True,
        choices=(
            CRITERION_TERMINAL_PROGRESSION,
            CRITERION_TEXTWORLD_TASK_PROGRESSION_V1,
            CRITERION_WEBSHOP_REWARD_THRESHOLD,
        ),
        help="Win/lose criterion for each trajectory",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=WEBSHOP_SUCCESS_THRESHOLD,
        help="Reward/progression threshold used by criterion=webshop_reward_threshold.",
    )
    parser.add_argument(
        "--quality-score",
        required=True,
        choices=(QUALITY_SCORE_MAX_SAME_OBS_ACTION_RUN,),
        help="Win-trajectory quality score used for filtering",
    )
    parser.add_argument(
        "--max-same-obs-action-run-threshold",
        type=int,
        required=True,
        help="Keep only win trajectories whose max_same_obs_action_run is <= this threshold",
    )
    parser.add_argument(
        "--max-families",
        type=int,
        default=0,
        help="Optional cap for number of episode families to process (0 means all)",
    )
    parser.add_argument(
        "--include-base-alt-win-pseudo-pairs",
        action="store_true",
        help=(
            "Add pseudo win_lose rows for quality-filtered base/alt successful pairs. "
            "The shorter post-divergence trajectory is preferred, and ties favor base."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory for per-task legacy pair JSONLs "
            "(default: data/dataset/preferences_filtered/<run_id>/<usable_traj>/<criterion>/<quality>_leq_<threshold>)"
        ),
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Output JSON path for summary stats (default: <out-dir>/conversion_stats.json)",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Output CSV path for per-task summary rows (default: <out-dir>/task_pair_counts.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dtc_root = Path(args.dtc_root)
    dtc_run_dir = Path(args.dtc_run_dir) if args.dtc_run_dir else _find_latest_run_dir(dtc_root)
    base_run_dir = Path(args.base_run_dir)
    run_id = dtc_run_dir.name

    quality_suffix = f"{args.quality_score}_leq_{int(args.max_same_obs_action_run_threshold)}"
    if args.include_base_alt_win_pseudo_pairs:
        quality_suffix = f"{quality_suffix}__base_alt_win_pseudo"
    criterion_suffix = args.criterion
    if args.criterion == CRITERION_WEBSHOP_REWARD_THRESHOLD:
        criterion_suffix = f"{criterion_suffix}_geq_{str(args.success_threshold).replace('.', 'p')}"
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("data/dataset/preferences_filtered") / run_id / args.usable_traj / criterion_suffix / quality_suffix
    )
    summary_out = Path(args.summary_out) if args.summary_out else out_dir / "conversion_stats.json"
    csv_out = Path(args.csv_out) if args.csv_out else out_dir / "task_pair_counts.csv"

    win_lose_by_task, win_win_by_task, summary, task_rows = build_filtered_legacy_pairs(
        dtc_run_dir=dtc_run_dir,
        base_run_dir=base_run_dir,
        usable_traj=args.usable_traj,
        criterion=args.criterion,
        quality_score=args.quality_score,
        max_same_obs_action_run_threshold=args.max_same_obs_action_run_threshold,
        include_base_alt_win_pseudo_pairs=args.include_base_alt_win_pseudo_pairs,
        max_families=args.max_families,
        success_threshold=args.success_threshold,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    task_slugs = sorted(set(win_lose_by_task) | set(win_win_by_task))
    for task_slug in task_slugs:
        _write_jsonl(out_dir / f"{task_slug}_win_lose.jsonl", win_lose_by_task.get(task_slug, []))
        _write_jsonl(out_dir / f"{task_slug}_win_win.jsonl", win_win_by_task.get(task_slug, []))

    _write_json(summary_out, summary)
    _write_csv(csv_out, task_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
