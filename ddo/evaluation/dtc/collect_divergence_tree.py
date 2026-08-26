#!/usr/bin/env python3
"""Divergence Tree Collection (DTC) for BALROG thought-action runs.

This collector:
- reads complete base trajectories from an existing BALROG run directory,
- reconstructs divergence states by replaying executed actions,
- branches with alternative executed actions at selected divergence steps,
- rolls out alternative trajectories with the BALROG agent stack when needed,
- mirrors base trajectory artifacts under a new divergence-tree output root,
- and writes DTC-specific branch metadata under a separate `_dtc/` subtree.

The implementation intentionally leaves the legacy DFS collector untouched.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import multiprocessing
import numpy as np
import random
import re
import shutil
import sys
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BALROG_ROOT = REPO_ROOT / "benchmarks" / "BALROG"
if str(BALROG_ROOT) not in sys.path:
    sys.path.insert(0, str(BALROG_ROOT))

from omegaconf import OmegaConf
from tqdm import tqdm

from ddo.evaluation.action import run_agent_action
from ddo.evaluation.balrog.actions import (
    extract_fixed_action_thought,
    extract_single_action,
    extract_thought_action,
    format_thought_action_completion,
)
from ddo.evaluation.balrog.agent import create_agent
from ddo.evaluation.balrog.env import (
    BABAISAI_ACTION_SPACE,
    BABYAI_ACTION_SPACE,
    make_env,
)
from ddo.evaluation.balrog.run_config import (
    DEFAULT_BALROG_CONFIG_PATH,
    resolve_run_config_path,
)


logger = logging.getLogger(__name__)
DTC_STAGE_TOTAL = 4
CSV_COLUMNS = [
    "step",
    "instruction",
    "observation_pre",
    "thought",
    "action_model",
    "action_executed",
    "raw_output",
    "feedback",
    "observation_post",
    "obs_changed",
    "reward",
    "progression",
    "terminated",
    "truncated",
    "done",
    "termination_reason",
    "action_defaulted",
    "input_tokens",
    "output_tokens",
    "step_wall_time_sec",
    "won",
    "lost",
]

INVALID_ACTION_NOTICE_PATTERN = re.compile(
    r"^\s*(Your previous output did not contain a valid action\. Defaulted to action:\s*.*?\n\nObservation:\n)",
    flags=re.DOTALL,
)

INVALID_ACTION_NOTICE_TEMPLATE = (
    "\n\nYour previous output did not contain a valid action. "
    "Defaulted to action: {executed_action}\n\nObservation:\n"
)
TEXTWORLD_COMMANDS_HEADER = "Available commands (choose exactly one as your final Action):"
SUPPORTED_DTC_ENVS = {"babyai", "babaisai", "textworld"}
RETRYABLE_JOB_FAILURE_STATUSES = {"worker_exception"}
RESUME_INCOMPLETE_STATUSES = {"retry_exhausted", "worker_exception"}
ALT_MODE_RANDOM = "random"
ALT_MODE_REQUEST = "request"
ALT_MODE_RANDOM_TEACHER = "random_teacher"

_DTC_PROGRESS_STREAM = None


def _get_progress_stream():
    return _DTC_PROGRESS_STREAM or sys.stderr


def _write_progress_message(message: str) -> None:
    tqdm.write(message, file=_get_progress_stream())


def _write_stage_message(stage_idx: int, label: str) -> None:
    _write_progress_message(f"[{int(stage_idx)}/{DTC_STAGE_TOTAL}] {label}")


def _current_worker_identity() -> tuple[int, ...]:
    identity = getattr(multiprocessing.current_process(), "_identity", ())
    if not identity:
        return ()
    return tuple(int(part) for part in identity)


def _current_worker_position() -> int:
    identity = _current_worker_identity()
    if identity:
        return max(1, int(identity[0]))
    return 1


def _current_worker_label() -> str:
    identity = _current_worker_identity()
    if identity:
        return f"W{int(identity[0]):02d}"
    return "W00"


def _build_worker_progress_desc(*, artifact: "TrajectoryArtifact", divergence_step: int, alt_index: int) -> str:
    return (
        f"{_current_worker_label()} "
        f"{artifact.run_stem} "
        f"d{int(divergence_step):02d} "
        f"a{int(alt_index):02d}"
    )


@contextmanager
def _redirect_collection_output(log_path: Path | None):
    global _DTC_PROGRESS_STREAM

    if _DTC_PROGRESS_STREAM is None:
        _DTC_PROGRESS_STREAM = sys.stderr
    if log_path is None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            stream=sys.stderr,
            force=True,
        )
        logging.captureWarnings(True)
        yield
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    logging.captureWarnings(True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            yield


@dataclass(frozen=True)
class TrajectoryArtifact:
    task_id: str
    env_name: str
    episode_id: str
    episode_idx: int
    run_stem: str
    seed: int
    relative_dir: Path
    csv_path: Path
    json_path: Path
    trace_path: Path
    csv_rows: list[dict[str, str]]
    episode_log: dict[str, Any]
    trace_payload: dict[str, Any]
    success: bool

    @property
    def trace_calls(self) -> list[dict[str, Any]]:
        calls = self.trace_payload.get("calls") or []
        return calls if isinstance(calls, list) else []

    @property
    def instruction_prompt(self) -> str:
        calls = self.trace_calls
        if calls:
            return str(calls[0].get("instruction") or "").strip()
        return ""

    @property
    def executed_actions(self) -> list[str]:
        return [str(call.get("action") or "").strip() for call in self.trace_calls]

    @property
    def total_steps(self) -> int:
        return len(self.trace_calls)


@dataclass(frozen=True)
class BranchJob:
    task_id: str
    base_traj_id: str
    base_run_dir: Path
    trace_path: Path
    csv_path: Path
    json_path: Path
    relative_dir: Path
    config_path: Path
    output_root: Path
    divergence_step: int
    alt_mode: str
    alt_plan: "SelectedAltActionPlan"
    alt_index: int
    alt_budget: int
    alt_budget_used: int
    rollout_max_steps: int | None
    rollout_extra_steps: int
    client_max_tokens_override: int | None = None
    client_timeout_override: float | None = None
    client_max_retries_override: int | None = None


@dataclass(frozen=True)
class RequestDivergenceJob:
    task_id: str
    base_traj_id: str
    base_run_dir: Path
    trace_path: Path
    csv_path: Path
    json_path: Path
    relative_dir: Path
    config_path: Path
    output_root: Path
    divergence_step: int
    alt_mode: str
    target_alt_count: int
    existing_records: tuple[dict[str, Any], ...]
    rollout_max_steps: int | None
    rollout_extra_steps: int
    exclude_alt_actions: tuple[str, ...] = ()
    client_max_tokens_override: int | None = None
    client_timeout_override: float | None = None
    client_max_retries_override: int | None = None


@dataclass(frozen=True)
class SelectedAltActionPlan:
    raw_action_text: str
    executed_action_text: str
    action_defaulted: bool
    request_count: int
    raw_output: str
    thought: str
    interaction: dict[str, Any]
    selection_attempts: int
    generation_wall_time_sec: float = 0.0


def _build_alt_traj_id(base_traj_id: str, divergence_step: int, alt_index: int) -> str:
    return f"{base_traj_id}__dtc_d{divergence_step:02d}_a{alt_index:02d}"


def _build_alt_stem(run_stem: str, divergence_step: int, alt_index: int) -> str:
    return f"{run_stem}__dtc_d{divergence_step:02d}_a{alt_index:02d}"


def _request_progress_path(
    *,
    output_root: Path,
    task_id: str,
    base_traj_id: str,
    divergence_step: int,
) -> Path:
    inflight_root = output_root / "_dtc" / "inflight" / _safe_task_slug(task_id)
    filename = f"{_safe_task_slug(base_traj_id)}__d{int(divergence_step):02d}.jsonl"
    return inflight_root / filename


def _parse_alt_index(alt_traj_id: str) -> int | None:
    match = re.search(r"__dtc_d\d+_a(\d+)$", str(alt_traj_id or "").strip())
    if match is None:
        return None
    return _as_int(match.group(1), default=-1)


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


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().split())


def _normalize_action_list(actions: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for action in actions:
        text = _normalize_text(action)
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _extract_textworld_admissible_commands(observation_text: str) -> list[str]:
    raw = str(observation_text or "")
    if TEXTWORLD_COMMANDS_HEADER not in raw:
        return []
    suffix = raw.split(TEXTWORLD_COMMANDS_HEADER, 1)[1]
    commands: list[str] = []
    for line in suffix.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("- "):
            break
        commands.append(stripped[2:].strip())
    return _normalize_action_list(commands)


def _safe_task_slug(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", str(task_id or "").strip()).strip("_") or "task"


def _guess_env_slug(base_run_dir: Path) -> str:
    parts = {part.strip() for part in base_run_dir.parts}
    for env_name in ("babyai", "textworld", "crafter", "babaisai", "nle", "minihack"):
        if env_name in parts:
            return env_name
    return "unknown_env"


def _sha1_lines(parts: list[str]) -> str:
    payload = "\n<SEP>\n".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _seed_process_rng(seed: int | None) -> None:
    if seed is None:
        return
    resolved_seed = _as_int(seed, default=-1)
    if resolved_seed < 0:
        return
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)


def _extract_long_term_observation(obs: dict[str, Any]) -> str:
    if isinstance(obs, dict):
        text = obs.get("text", {})
        if isinstance(text, dict):
            return str(text.get("long_term_context", "") or "")
    return str(obs) if obs is not None else ""


def _serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for msg in messages:
        serialized.append(
            {
                "role": getattr(msg, "role", ""),
                "content": getattr(msg, "content", ""),
                "has_attachment": bool(getattr(msg, "attachment", None)),
            }
        )
    return serialized


def _get_last_raw_completion(agent, fallback: str = "") -> str:
    if hasattr(agent, "get_last_llm_interaction"):
        interaction = agent.get_last_llm_interaction()
        if interaction:
            response_info = interaction.get("response", {})
            raw_completion = response_info.get("raw_completion")
            if raw_completion is not None:
                return str(raw_completion)
    return fallback if fallback is not None else ""


def _append_extra_user_text(obs: dict[str, Any], extra_user_text: str) -> dict[str, Any]:
    obs_for_agent = copy.deepcopy(obs)
    if not extra_user_text:
        return obs_for_agent
    obs_text = obs_for_agent.get("text", {}) if isinstance(obs_for_agent, dict) else {}
    long_term = str(obs_text.get("long_term_context", "") or "")
    obs_text["long_term_context"] = long_term + "\n\n" + extra_user_text
    obs_for_agent["text"] = obs_text
    return obs_for_agent


def _available_actions_for_call(*, env_name: str, observation_text: str) -> list[str]:
    if env_name == "babyai":
        return list(BABYAI_ACTION_SPACE)
    if env_name == "babaisai":
        return list(BABAISAI_ACTION_SPACE)
    if env_name == "textworld":
        return _extract_textworld_admissible_commands(observation_text)
    return []


def _available_actions_for_divergence(
    *,
    artifact: "TrajectoryArtifact",
    divergence_step: int,
) -> list[str]:
    observation_text = str(artifact.trace_calls[divergence_step].get("observation") or "")
    return _available_actions_for_call(
        env_name=artifact.env_name,
        observation_text=observation_text,
    )


def _normalize_excluded_alt_actions(actions: Sequence[str] | None) -> tuple[str, ...]:
    if not actions:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for action in actions:
        text = str(action or "").strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return tuple(normalized)


def _filter_excluded_alt_actions(
    valid_actions: Sequence[str],
    excluded_alt_actions: Sequence[str] | None,
) -> list[str]:
    excluded = set(_normalize_excluded_alt_actions(excluded_alt_actions))
    if not excluded:
        return list(valid_actions)
    return [action for action in valid_actions if action not in excluded]


def _build_alt_request_instruction(
    *,
    env_name: str,
    base_action: str,
    excluded_executed_actions: list[str],
    valid_actions: list[str],
) -> str:
    excluded = [action for action in excluded_executed_actions if action]
    excluded_text = ", ".join(excluded) if excluded else base_action
    valid_actions_text = ", ".join(valid_actions)
    if env_name == "textworld":
        return (
            "Your final Action must be exactly one valid TextWorld command from the current admissible set.\n"
            f"Valid actions: {valid_actions_text}\n"
            f"Your executed action must be different from: {excluded_text}\n"
            "Do not repeat any excluded action. Reply with one exact admissible command."
        )
    if env_name == "babaisai":
        return (
            "Your final Action must be exactly one valid BabaIsAI action.\n"
            f"Valid actions: {valid_actions_text}\n"
            f"Your executed action must be different from: {excluded_text}\n"
            "Do not repeat any excluded action."
        )
    return (
        "Your final Action must be exactly one valid BabyAI action.\n"
        f"Valid actions: {valid_actions_text}\n"
        f"Your executed action must be different from: {excluded_text}\n"
        "Do not repeat any excluded action."
    )


def _build_prompt_obs_from_text(observation_text: str) -> dict[str, Any]:
    return {
        "text": {
            "long_term_context": str(observation_text or ""),
            "short_term_context": "",
        },
        "image": None,
    }


def _rebuild_prompt_state_from_trace_calls(
    *,
    agent,
    trace_calls: list[dict[str, Any]],
    divergence_step: int,
    instruction_prompt: str,
) -> dict[str, Any]:
    agent.reset()
    if hasattr(agent, "clear_llm_interactions"):
        agent.clear_llm_interactions()
    agent.prompt_builder.update_instruction_prompt(str(instruction_prompt or ""))

    prev_executed_action: str | None = None
    for step_idx in range(divergence_step):
        if prev_executed_action:
            agent.prompt_builder.update_action(prev_executed_action)
        agent.prompt_builder.update_observation(
            _build_prompt_obs_from_text(str(trace_calls[step_idx].get("observation") or ""))
        )
        prev_executed_action = str(trace_calls[step_idx].get("action") or "").strip()

    current_obs = _build_prompt_obs_from_text(
        str(trace_calls[divergence_step].get("observation") or "")
    )
    return {
        "prompt_builder": copy.deepcopy(agent.prompt_builder),
        "obs": current_obs,
        "prev_executed_action": prev_executed_action,
    }


def _resolve_target_alt_count(
    *,
    valid_actions: list[str],
    base_action: str,
    alt_budget: int,
) -> int:
    available_count = len([action for action in valid_actions if action != base_action])
    if available_count <= 0:
        return 0
    if alt_budget <= 0 or alt_budget >= available_count:
        return available_count
    return alt_budget


def _random_alt_actions(
    *,
    valid_actions: list[str],
    base_action: str,
    alt_budget: int,
    sampling_key: str,
) -> list[str]:
    available = [action for action in valid_actions if action != base_action]
    target_count = _resolve_target_alt_count(
        valid_actions=valid_actions,
        base_action=base_action,
        alt_budget=alt_budget,
    )
    if target_count <= 0:
        return []
    rng = random.Random(
        _stable_sampling_seed(
            [
                "dtc_random_alt_actions",
                sampling_key,
                base_action,
                str(target_count),
            ]
        )
    )
    return rng.sample(available, k=target_count)


def _split_observation_notice(observation_text: str) -> tuple[str, str, str]:
    raw = str(observation_text or "")
    match = INVALID_ACTION_NOTICE_PATTERN.match(raw)
    if not match:
        return raw, "", ""
    notice = match.group(1).strip()
    clean = raw[match.end() :]
    return raw, clean, notice


def _inject_invalid_action_notice(obs: dict[str, Any], executed_action: str) -> None:
    if not isinstance(obs, dict):
        return
    text = obs.get("text", {})
    if not isinstance(text, dict):
        return
    existing = str(text.get("long_term_context", "") or "")
    text["long_term_context"] = INVALID_ACTION_NOTICE_TEMPLATE.format(
        executed_action=executed_action
    ) + existing
    obs["text"] = text


def _get_trace_model_action(call: dict[str, Any]) -> str:
    extras = call.get("extras", {}) if isinstance(call, dict) else {}
    balrog_raw = extras.get("balrog_raw", {}) if isinstance(extras, dict) else {}
    response = balrog_raw.get("response", {}) if isinstance(balrog_raw, dict) else {}
    parsed = str(response.get("parsed_action") or "").strip()
    if parsed:
        return parsed

    raw_output = str(call.get("raw_output") or "").strip()
    if raw_output:
        _thought, action = extract_thought_action(raw_output)
        if action:
            return action
        return extract_single_action(raw_output)
    return ""


def _trace_call_was_fallback(call: dict[str, Any]) -> bool:
    executed = str(call.get("action") or "").strip()
    model_action = _get_trace_model_action(call)
    return bool(executed and model_action and executed != model_action)


def _trace_call_token_usage(call: dict[str, Any]) -> dict[str, int]:
    token_usage = call.get("token_usage", {}) if isinstance(call, dict) else {}
    if not isinstance(token_usage, dict):
        token_usage = {}
    return {
        "input": _as_int(token_usage.get("input"), default=0),
        "output": _as_int(token_usage.get("output"), default=0),
        "total": _as_int(token_usage.get("total"), default=0),
    }


def _trace_last_call(trace_payload: dict[str, Any]) -> dict[str, Any]:
    calls = trace_payload.get("calls") or []
    if isinstance(calls, list) and calls:
        return calls[-1]
    return {}


def _is_successful_trajectory(episode_log: dict[str, Any], trace_payload: dict[str, Any]) -> bool:
    last_call = _trace_last_call(trace_payload)
    won = bool(last_call.get("won", False))
    progression = _as_float(
        episode_log.get("progression", last_call.get("progression")),
        default=0.0,
    )
    episode_return = _as_float(
        episode_log.get("episode_return", last_call.get("reward")),
        default=0.0,
    )
    done = bool(episode_log.get("done", last_call.get("done", False)))
    return won or progression >= 1.0 or (done and episode_return > 0.0)


def _get_termination_reason(
    terminated: bool,
    truncated: bool,
    reward: float,
    progression: float,
    won: bool,
    lost: bool,
) -> str:
    if won:
        return "won"
    if lost:
        return "lost"
    if terminated and (reward > 0 or progression >= 1.0):
        return "success"
    if truncated:
        return "truncated"
    if terminated:
        return "terminated"
    return ""


def _build_state_key(task_id: str, seed: int, prefix_actions: list[str]) -> str:
    return _sha1_lines(
        [
            str(task_id or "").strip(),
            str(seed),
            json.dumps(prefix_actions, ensure_ascii=True),
        ]
    )


def _uniform_divergence_steps(total_steps: int, divergence_count: int) -> list[int]:
    if total_steps <= 1:
        return []

    available = list(range(1, total_steps))
    if not available:
        return []

    if divergence_count <= 0:
        return available

    if len(available) == 1:
        return available

    if divergence_count == 1:
        return [available[-1]]

    if divergence_count >= len(available):
        return available

    selected = {available[0], available[-1]}
    remaining = divergence_count - len(selected)
    if remaining <= 0:
        return sorted(selected)

    span = len(available) - 1
    for slot in range(1, remaining + 1):
        position = round((slot * span) / (remaining + 1))
        selected.add(available[position])
    return sorted(selected)


def _stable_sampling_seed(parts: list[str]) -> int:
    digest = hashlib.sha1("\n<SEP>\n".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _random_divergence_steps(
    total_steps: int,
    divergence_count: int,
    *,
    sampling_key: str,
) -> list[int]:
    if total_steps <= 1:
        return []

    available = list(range(1, total_steps))
    if not available:
        return []

    if divergence_count <= 0 or divergence_count >= len(available):
        return available

    rng = random.Random(
        _stable_sampling_seed(
            [
                "dtc_random_divergence_steps",
                sampling_key,
                str(total_steps),
                str(divergence_count),
            ]
        )
    )
    return sorted(rng.sample(available, k=divergence_count))


def _select_divergence_steps(
    total_steps: int,
    divergence_count: int,
    *,
    step_sampling_mode: str,
    sampling_key: str,
) -> list[int]:
    if step_sampling_mode == "uniform":
        return _uniform_divergence_steps(total_steps, divergence_count)
    if step_sampling_mode == "random":
        return _random_divergence_steps(
            total_steps,
            divergence_count,
            sampling_key=sampling_key,
        )
    raise ValueError(f"Unsupported step_sampling_mode: {step_sampling_mode}")


def _agent_act_with_prompt_builder(
    *,
    agent,
    prompt_builder,
    obs: dict[str, Any],
    prev_action: str | None,
    extra_user_text: str = "",
    validate_action=None,
    agent_method_name: str = "act",
    agent_method_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original_prompt_builder = agent.prompt_builder
    try:
        agent.prompt_builder = copy.deepcopy(prompt_builder)
        act_result = run_agent_action(
            agent=agent,
            obs=obs,
            prev_action=prev_action,
            validate_action=validate_action,
            extra_user_text=extra_user_text,
            agent_method_name=agent_method_name,
            agent_method_kwargs=agent_method_kwargs,
        )
        if not act_result["ok"]:
            return {
                "ok": False,
                "response": act_result.get("response"),
                "interaction": act_result.get("interaction", {}),
                "raw_output": str(act_result.get("raw_output", "") or ""),
                "action": str(act_result.get("model_action", "") or ""),
                "request_count": int(act_result.get("request_count", 0) or 0),
                "thought": str(act_result.get("thought", "") or ""),
                "abort_reason": act_result.get("abort_reason"),
                "abort_stop_reason": act_result.get("abort_stop_reason", ""),
                "abort_leaf_history": act_result.get("abort_leaf_history", []),
                "abort_leaf_counts": act_result.get("abort_leaf_counts", {}),
                "abort_leaf_last": act_result.get("abort_leaf_last", ""),
                "retry_attempts": act_result.get("retry_attempts", []),
                "prompt_builder": copy.deepcopy(act_result.get("prompt_builder")),
            }
        return {
            "ok": True,
            "action": str(act_result["model_action"] or "").strip(),
            "executed_action": str(act_result["executed_action"] or "").strip(),
            "raw_output": str(act_result["raw_output"] or ""),
            "response": act_result["response"],
            "interaction": act_result.get("interaction", {}),
            "request_count": int(act_result.get("request_count", 0) or 0),
            "thought": str(act_result.get("thought", "") or ""),
            "retry_attempts": act_result.get("retry_attempts", []),
            "prompt_builder": copy.deepcopy(act_result["prompt_builder"]),
        }
    finally:
        agent.prompt_builder = original_prompt_builder


def _request_fixed_action_teacher_completion(
    *,
    agent,
    prompt_builder,
    obs: dict[str, Any],
    prev_action: str | None,
    fixed_action: str,
    alt_mode: str,
) -> dict[str, Any]:
    fixed_action = str(fixed_action or "").strip()
    if not fixed_action:
        raise ValueError("fixed_action must be non-empty")

    started = time.perf_counter()
    act_result = _agent_act_with_prompt_builder(
        agent=agent,
        prompt_builder=prompt_builder,
        obs=obs,
        prev_action=prev_action,
        agent_method_name="generate_fixed_action_thought",
        agent_method_kwargs={"fixed_action": fixed_action},
    )
    generation_wall_time_sec = time.perf_counter() - started

    interaction = copy.deepcopy(act_result.get("interaction") or {})
    response_info = interaction.setdefault("response", {})
    interaction_meta = interaction.setdefault("meta", {})
    raw_thought_output = str(act_result.get("raw_output", "") or "")
    parsed_thought = extract_fixed_action_thought(raw_thought_output)
    if not parsed_thought:
        parsed_thought = str(act_result.get("thought", "") or "").strip()

    interaction_meta["alt_mode"] = alt_mode
    interaction_meta["fixed_action"] = fixed_action
    interaction_meta["fixed_action_thought_raw_output"] = raw_thought_output
    interaction_meta["retry_attempts"] = act_result.get("retry_attempts", [])

    if act_result.get("ok") and parsed_thought:
        full_completion = format_thought_action_completion(parsed_thought, fixed_action)
        interaction_meta["source"] = "dtc.random_teacher_thought"
        interaction_meta["fixed_action_thought_status"] = "ok"
        response_info["raw_completion"] = full_completion
        response_info["reasoning"] = parsed_thought
        response_info["parsed_action"] = fixed_action
        return {
            "ok": True,
            "thought": parsed_thought,
            "raw_output": full_completion,
            "request_count": int(act_result.get("request_count", 0) or 0),
            "interaction": interaction,
            "generation_wall_time_sec": float(generation_wall_time_sec),
        }

    interaction_meta["source"] = "dtc.random_teacher_thought"
    interaction_meta["fixed_action_thought_status"] = "fallback_action_only"
    if not act_result.get("ok"):
        interaction_meta["fixed_action_thought_abort_reason"] = str(
            act_result.get("abort_reason") or "retry_exhausted"
        )
    response_info["raw_completion"] = fixed_action
    response_info["reasoning"] = ""
    response_info["parsed_action"] = fixed_action
    return {
        "ok": False,
        "thought": "",
        "raw_output": fixed_action,
        "request_count": int(act_result.get("request_count", 0) or 0),
        "interaction": interaction,
        "generation_wall_time_sec": float(generation_wall_time_sec),
    }


def _resolve_rollout_step_cap(
    *,
    base_total_steps: int,
    env_max_steps: int | None,
    rollout_max_steps: int | None,
    rollout_extra_steps: int,
) -> int | None:
    cap_candidates: list[int] = []

    if env_max_steps is not None and int(env_max_steps) > 0:
        cap_candidates.append(int(env_max_steps))

    if base_total_steps > 0:
        cap_candidates.append(int(base_total_steps) + max(0, int(rollout_extra_steps)))

    if rollout_max_steps is not None:
        cap_candidates.append(int(rollout_max_steps))

    if not cap_candidates:
        return None
    return min(cap_candidates)


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_artifact_from_paths(
    *,
    base_run_dir: Path,
    trace_path: Path,
    csv_path: Path | None = None,
    json_path: Path | None = None,
    require_success: bool = False,
) -> TrajectoryArtifact | None:
    run_stem = trace_path.name[: -len("_llm_trace.json")]
    csv_path = csv_path or trace_path.with_name(f"{run_stem}.csv")
    json_path = json_path or trace_path.with_name(f"{run_stem}.json")
    if not csv_path.exists() or not json_path.exists():
        return None

    trace_payload = _load_json(trace_path)
    env_name = str(trace_payload.get("env_name") or "").strip()
    if env_name not in SUPPORTED_DTC_ENVS:
        return None

    task_id = str(trace_payload.get("task_or_env_params") or "").strip()
    if not task_id:
        return None

    episode_log = _load_json(json_path)
    csv_rows = _load_csv_rows(csv_path)
    success = _is_successful_trajectory(episode_log, trace_payload)
    if require_success and not success:
        return None

    return TrajectoryArtifact(
        task_id=task_id,
        env_name=env_name,
        episode_id=str(trace_payload.get("episode_id") or run_stem),
        episode_idx=_as_int(trace_payload.get("episode_idx"), default=-1),
        run_stem=run_stem,
        seed=_as_int(trace_payload.get("seed"), default=-1),
        relative_dir=trace_path.parent.relative_to(base_run_dir),
        csv_path=csv_path,
        json_path=json_path,
        trace_path=trace_path,
        csv_rows=csv_rows,
        episode_log=episode_log,
        trace_payload=trace_payload,
        success=success,
    )


def _build_runtime_config_for_artifact(
    runtime_config,
    artifact: TrajectoryArtifact,
    *,
    client_max_tokens_override: int | None = None,
    client_timeout_override: float | None = None,
    client_max_retries_override: int | None = None,
):
    config = OmegaConf.create(OmegaConf.to_container(runtime_config, resolve=True))
    agent_snapshot = artifact.episode_log.get("agent")
    client_snapshot = artifact.episode_log.get("client")
    if isinstance(agent_snapshot, dict) and agent_snapshot:
        config.agent = OmegaConf.create(copy.deepcopy(agent_snapshot))
    if isinstance(client_snapshot, dict) and client_snapshot:
        config.client = OmegaConf.create(copy.deepcopy(client_snapshot))
    if client_max_tokens_override is not None:
        if config.client.get("generate_kwargs") is None:
            config.client.generate_kwargs = OmegaConf.create({})
        config.client.generate_kwargs.max_tokens = int(client_max_tokens_override)
    if client_timeout_override is not None:
        config.client.timeout = float(client_timeout_override)
    if client_max_retries_override is not None:
        config.client.max_retries = int(client_max_retries_override)
    config.envs.names = artifact.env_name
    return config


def _build_outcome_summary_from_logs(
    episode_log: dict[str, Any],
    trace_payload: dict[str, Any],
) -> dict[str, Any]:
    last_call = _trace_last_call(trace_payload)
    progression = _as_float(
        episode_log.get("progression", last_call.get("progression")),
        default=0.0,
    )
    episode_return = _as_float(
        episode_log.get("episode_return", last_call.get("reward")),
        default=0.0,
    )
    done = bool(episode_log.get("done", last_call.get("done", False)))
    success = _is_successful_trajectory(episode_log, trace_payload)
    return {
        "success": bool(success),
        "progression": progression,
        "reward": episode_return,
        "done": done,
        "num_steps": _as_int(episode_log.get("num_steps", trace_payload.get("num_calls")), default=0),
    }


def _make_env_for_artifact(config, artifact: TrajectoryArtifact):
    _seed_process_rng(artifact.seed)
    env_seed = artifact.seed if artifact.env_name in {"textworld", "babaisai"} else None
    return make_env(artifact.env_name, artifact.task_id, config, env_seed=env_seed)


def _select_task_artifacts_for_collection(
    task_artifacts: list[TrajectoryArtifact],
    *,
    success_only: bool,
    smoke_test: bool,
) -> list[TrajectoryArtifact]:
    eligible_artifacts = (
        [artifact for artifact in task_artifacts if artifact.success]
        if success_only
        else list(task_artifacts)
    )
    if not smoke_test or not eligible_artifacts:
        return eligible_artifacts

    branchable_artifacts = [artifact for artifact in eligible_artifacts if artifact.total_steps > 1]
    if branchable_artifacts:
        return [branchable_artifacts[0]]
    return [eligible_artifacts[0]]


def _branch_record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(record.get("task_id") or ""),
        _as_int(record.get("seed"), default=-1),
        str(record.get("base_traj_id") or ""),
        _as_int(record.get("divergence_step"), default=-1),
        str(record.get("alt_traj_id") or ""),
    )


def _collection_job_sort_key(job: BranchJob | RequestDivergenceJob) -> tuple[Any, ...]:
    if isinstance(job, RequestDivergenceJob):
        return (
            job.task_id,
            job.base_traj_id,
            int(job.divergence_step),
            -1,
            "request",
            str(job.trace_path),
        )
    return (
        job.task_id,
        job.base_traj_id,
        int(job.divergence_step),
        int(job.alt_index),
        job.alt_plan.executed_action_text,
        str(job.trace_path),
    )


def _branch_job_resume_key(job: BranchJob) -> tuple[Any, ...]:
    return (
        job.task_id,
        job.base_traj_id,
        int(job.divergence_step),
        _build_alt_traj_id(job.base_traj_id, job.divergence_step, job.alt_index),
    )


def _collection_job_key(job: BranchJob | RequestDivergenceJob) -> tuple[Any, ...]:
    if isinstance(job, RequestDivergenceJob):
        return (
            "request",
            job.task_id,
            job.base_traj_id,
            int(job.divergence_step),
            int(job.target_alt_count),
        )
    return ("branch",) + _branch_job_resume_key(job)


def _request_job_resume_keys(job: RequestDivergenceJob) -> list[tuple[Any, ...]]:
    return [
        (
            job.task_id,
            job.base_traj_id,
            int(job.divergence_step),
            _build_alt_traj_id(job.base_traj_id, job.divergence_step, alt_index),
        )
        for alt_index in range(int(job.target_alt_count))
    ]


def _branch_record_resume_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(record.get("task_id") or ""),
        str(record.get("base_traj_id") or ""),
        _as_int(record.get("divergence_step"), default=-1),
        str(record.get("alt_traj_id") or ""),
    )


def _resume_record_is_complete(record: dict[str, Any], output_root: Path) -> bool:
    status = str(record.get("status") or "").strip()
    if not status:
        return False

    if status in RESUME_INCOMPLETE_STATUSES:
        return False

    if status != "generated":
        return True

    for field in ("output_csv", "output_json", "output_trace"):
        value = str(record.get(field) or "").strip()
        if not value:
            return False
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = output_root / candidate
        if not candidate.exists():
            return False
    return True


def _load_jsonl_records(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    records_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    if not path.exists():
        return records_by_key
    dtc_root = next((parent.parent for parent in path.parents if parent.name == "_dtc"), path.parent)
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not _resume_record_is_complete(record, dtc_root):
                continue
            records_by_key[_branch_record_resume_key(record)] = record
    return records_by_key


def _load_resume_records(
    branch_index_path: Path,
    *,
    inflight_root: Path | None = None,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    records_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    if inflight_root is not None and inflight_root.exists():
        for inflight_path in sorted(inflight_root.glob("**/*.jsonl")):
            records_by_key.update(_load_jsonl_records(inflight_path))
    records_by_key.update(_load_jsonl_records(branch_index_path))
    return records_by_key


def _partition_collection_jobs_for_resume(
    collection_jobs: list[BranchJob | RequestDivergenceJob],
    existing_records_by_key: dict[tuple[Any, ...], dict[str, Any]],
) -> list[BranchJob | RequestDivergenceJob]:
    remaining_jobs: list[BranchJob | RequestDivergenceJob] = []
    for job in collection_jobs:
        if isinstance(job, RequestDivergenceJob):
            resume_keys = _request_job_resume_keys(job)
            if resume_keys and all(key in existing_records_by_key for key in resume_keys):
                continue
            remaining_jobs.append(job)
            continue

        record = existing_records_by_key.get(_branch_job_resume_key(job))
        if record is None:
            remaining_jobs.append(job)
    return remaining_jobs


def _result_records_are_retryable_failures(result_records: list[dict[str, Any]]) -> bool:
    if not result_records:
        return False
    statuses = {str(record.get("status") or "").strip() for record in result_records}
    if not statuses:
        return False
    return statuses.issubset(RETRYABLE_JOB_FAILURE_STATUSES)


def _annotate_result_records(
    result_records: list[dict[str, Any]],
    *,
    job_attempt: int,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for record in result_records:
        updated = dict(record)
        updated["job_attempt"] = int(job_attempt)
        annotated.append(updated)
    return annotated


def _resolve_main_progress_counts(
    total_jobs: int,
    *,
    resume: bool,
    resume_skipped_jobs: int,
) -> tuple[int, int]:
    if not resume:
        return total_jobs, 0

    skipped_jobs = max(0, min(resume_skipped_jobs, total_jobs))
    return total_jobs, skipped_jobs


def _normalize_run_request_for_resume_check(run_request: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(run_request)
    comparable["resume"] = False
    comparable.pop("num_workers", None)
    comparable.pop("client_max_tokens_override", None)
    comparable.pop("client_timeout_override", None)
    comparable.pop("client_max_retries_override", None)
    comparable.pop("external_retry_attempts", None)
    return comparable


def _write_json_file(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_terminal_summary_lines(summary: dict[str, Any]) -> list[str]:
    output_root = str(summary.get("output_root") or "")
    summary_path = Path(output_root) / "_dtc" / "summary.json" if output_root else None
    branch_index_path = Path(output_root) / "_dtc" / "branch_index.jsonl" if output_root else None
    tasks = summary.get("tasks", {}) if isinstance(summary.get("tasks"), dict) else {}

    total_selected_bases = 0
    total_generated = 0
    total_failed = 0
    for task_summary in tasks.values():
        if not isinstance(task_summary, dict):
            continue
        total_selected_bases += _as_int(task_summary.get("selected_bases"), default=0)
        total_generated += _as_int(task_summary.get("generated_branches"), default=0)
        total_failed += _as_int(task_summary.get("failed_branches"), default=0)

    lines = [
        "DTC completed.",
        f"output_root: {output_root}",
        f"summary_json: {summary_path}" if summary_path is not None else "summary_json:",
        (
            f"branch_index_jsonl: {branch_index_path}"
            if branch_index_path is not None
            else "branch_index_jsonl:"
        ),
        (
            f"selected_bases={total_selected_bases} "
            f"generated_branches={total_generated} "
            f"failed_branches={total_failed}"
        ),
    ]
    for task_id in sorted(tasks):
        task_summary = tasks.get(task_id) or {}
        if not isinstance(task_summary, dict):
            continue
        lines.append(
            f"{task_id}: "
            f"selected_bases={_as_int(task_summary.get('selected_bases'), default=0)} "
            f"generated={_as_int(task_summary.get('generated_branches'), default=0)} "
            f"failed={_as_int(task_summary.get('failed_branches'), default=0)}"
        )
    return lines


def _summary_branch_totals(summary: dict[str, Any]) -> tuple[int, int, int]:
    tasks = summary.get("tasks", {}) if isinstance(summary.get("tasks"), dict) else {}
    selected = generated = failed = 0
    for task_summary in tasks.values():
        if not isinstance(task_summary, dict):
            continue
        selected += _as_int(task_summary.get("selected_bases"), default=0)
        generated += _as_int(task_summary.get("generated_branches"), default=0)
        failed += _as_int(task_summary.get("failed_branches"), default=0)
    return selected, generated, failed


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _ensure_divergence_step_summary(
    task_summary: dict[str, Any],
    *,
    base_traj_id: str,
    divergence_step: int,
) -> dict[str, Any]:
    key = f"{base_traj_id}@{divergence_step}"
    return task_summary["divergence_steps"].setdefault(
        key,
        {
            "base_traj_id": base_traj_id,
            "divergence_step": divergence_step,
            "attempts": 0,
            "successful_branches": 0,
            "failed_branches": 0,
        },
    )


def _populate_task_summaries(
    tasks_summary: dict[str, dict[str, Any]],
    branch_records: list[dict[str, Any]],
) -> None:
    for record in branch_records:
        task_summary = tasks_summary.get(str(record.get("task_id") or ""))
        if task_summary is None:
            continue
        step_summary = _ensure_divergence_step_summary(
            task_summary,
            base_traj_id=str(record.get("base_traj_id") or ""),
            divergence_step=_as_int(record.get("divergence_step"), default=-1),
        )
        if record.get("status") == "generated":
            step_summary["successful_branches"] += 1
            task_summary["generated_branches"] += 1
        else:
            step_summary["failed_branches"] += 1
            task_summary["failed_branches"] += 1


class DivergenceTreeCollector:
    """Collector for divergence-tree alternative trajectories."""

    def __init__(
        self,
        *,
        base_run_dir: Path,
        divergence_count: int,
        alt_budget: int,
        alt_mode: str,
        num_workers: int = 16,
        resume: bool = False,
        success_only: bool = False,
        smoke_test: bool = False,
        step_sampling_mode: str,
        task_filter: str | None,
        config_path_override: Path | None = None,
        output_root: Path | None = None,
        rollout_max_steps: int | None = None,
        rollout_extra_steps: int = 10,
        client_max_tokens_override: int | None = None,
        client_timeout_override: float | None = None,
        client_max_retries_override: int | None = None,
        external_retry_attempts: int = 0,
        exclude_alt_actions: Sequence[str] | None = None,
    ) -> None:
        self.base_run_dir = base_run_dir.resolve()
        self.divergence_count = int(divergence_count)
        self.alt_budget = max(0, int(alt_budget))
        self.alt_mode = str(alt_mode)
        self.num_workers = max(1, int(num_workers))
        self.resume = bool(resume)
        self.success_only = bool(success_only)
        self.smoke_test = bool(smoke_test)
        self.step_sampling_mode = str(step_sampling_mode)
        self.task_filter = task_filter
        env_slug = _guess_env_slug(self.base_run_dir)
        default_output_root = (
            REPO_ROOT
            / "data"
            / "raw"
            / "trajectories"
            / "balrog"
            / env_slug
            / "divergence_tree"
            / self.base_run_dir.name
        )
        self.output_root = (output_root.resolve() if output_root else default_output_root)
        self.rollout_max_steps = rollout_max_steps
        self.rollout_extra_steps = max(0, int(rollout_extra_steps))
        self.client_max_tokens_override = (
            None if client_max_tokens_override is None else int(client_max_tokens_override)
        )
        self.client_timeout_override = (
            None if client_timeout_override is None else float(client_timeout_override)
        )
        self.client_max_retries_override = (
            None if client_max_retries_override is None else int(client_max_retries_override)
        )
        self.external_retry_attempts = max(0, int(external_retry_attempts))
        self.exclude_alt_actions = _normalize_excluded_alt_actions(exclude_alt_actions)
        self.config_path, self.config_source = resolve_run_config_path(
            self.base_run_dir,
            config_override=config_path_override,
        )
        self.runtime_config = OmegaConf.load(self.config_path)

    def collect_divergence_tree(self) -> dict[str, Any]:
        if not self.base_run_dir.exists():
            raise FileNotFoundError(f"base_run_dir does not exist: {self.base_run_dir}")

        _write_stage_message(1, "main process discover_base_artifacts")
        artifacts_by_task = self._discover_base_artifacts()
        if not artifacts_by_task:
            raise RuntimeError(f"No compatible BALROG trajectories found under: {self.base_run_dir}")
        total_artifacts = sum(len(task_artifacts) for task_artifacts in artifacts_by_task.values())
        _write_progress_message(
            f"[1/{DTC_STAGE_TOTAL}] discovered {total_artifacts} base trajectories across {len(artifacts_by_task)} tasks."
        )

        self.output_root.mkdir(parents=True, exist_ok=True)
        dtc_root = self.output_root / "_dtc"
        dtc_root.mkdir(parents=True, exist_ok=True)
        inflight_root = dtc_root / "inflight"
        if not self.resume:
            shutil.rmtree(inflight_root, ignore_errors=True)

        summary: dict[str, Any] = {
            "collector": "divergence_tree",
            "base_run_dir": str(self.base_run_dir),
            "config_path": str(self.config_path),
            "config_source": self.config_source,
            "output_root": str(self.output_root),
            "divergence_count": self.divergence_count,
            "alt_budget": self.alt_budget,
            "alt_mode": self.alt_mode,
            "num_workers": self.num_workers,
            "resume": self.resume,
            "success_only": self.success_only,
            "smoke_test": self.smoke_test,
            "step_sampling_mode": self.step_sampling_mode,
            "rollout_max_steps": self.rollout_max_steps,
            "rollout_extra_steps": self.rollout_extra_steps,
            "client_max_tokens_override": self.client_max_tokens_override,
            "client_timeout_override": self.client_timeout_override,
            "client_max_retries_override": self.client_max_retries_override,
            "external_retry_attempts": self.external_retry_attempts,
            "exclude_alt_actions": list(self.exclude_alt_actions),
            "tasks": {},
        }
        run_request = {
            "collector": "divergence_tree",
            "base_run_dir": str(self.base_run_dir),
            "config_path": str(self.config_path),
            "config_source": self.config_source,
            "output_root": str(self.output_root),
            "divergence_count": self.divergence_count,
            "alt_budget": self.alt_budget,
            "alt_mode": self.alt_mode,
            "num_workers": self.num_workers,
            "resume": self.resume,
            "success_only": self.success_only,
            "smoke_test": self.smoke_test,
            "step_sampling_mode": self.step_sampling_mode,
            "task_filter": self.task_filter,
            "rollout_max_steps": self.rollout_max_steps,
            "rollout_extra_steps": self.rollout_extra_steps,
            "client_max_tokens_override": self.client_max_tokens_override,
            "client_timeout_override": self.client_timeout_override,
            "client_max_retries_override": self.client_max_retries_override,
            "external_retry_attempts": self.external_retry_attempts,
            "exclude_alt_actions": list(self.exclude_alt_actions),
        }
        run_request_path = dtc_root / "run_request.json"
        branch_index_path = dtc_root / "branch_index.jsonl"
        if self.resume and run_request_path.exists():
            existing_request = _load_json(run_request_path)
            comparable_existing = _normalize_run_request_for_resume_check(existing_request)
            comparable_current = _normalize_run_request_for_resume_check(run_request)
            if comparable_existing != comparable_current:
                raise RuntimeError(
                    "DTC resume request does not match the existing run_request.json. "
                    "Use the same collector options or a different output_root."
                )
        _write_json_file(run_request_path, run_request)
        existing_records_by_key = (
            _load_resume_records(branch_index_path, inflight_root=inflight_root)
            if self.resume
            else {}
        )

        _write_stage_message(2, "main process prepare_branch_jobs")
        selected_bases_manifest, branch_jobs, first_selected_base = self._prepare_branch_jobs(
            artifacts_by_task,
            summary["tasks"],
            existing_records_by_key=existing_records_by_key,
        )
        _write_progress_message(
            f"[2/{DTC_STAGE_TOTAL}] prepared {len(selected_bases_manifest)} selected bases and {len(branch_jobs)} worker jobs."
        )
        _write_json_file(dtc_root / "selected_bases.json", selected_bases_manifest)

        _write_stage_message(3, "worker pool collect trajectories")
        all_branch_records, resume_stats = self._run_branch_jobs(
            branch_jobs,
            branch_index_path=branch_index_path,
            existing_records_by_key=existing_records_by_key,
        )
        _write_progress_message(
            f"[3/{DTC_STAGE_TOTAL}] completed branch collection with {len(all_branch_records)} branch records."
        )
        _populate_task_summaries(summary["tasks"], all_branch_records)
        first_successful_branch = next(
            (record for record in all_branch_records if record.get("status") == "generated"),
            None,
        )
        summary.update(resume_stats)

        _write_stage_message(4, "main process finalize outputs")
        compatibility_report = self._build_compatibility_report(
            base_artifact=first_selected_base,
            successful_branch_record=first_successful_branch,
        )
        _write_json_file(dtc_root / "compatibility_report.json", compatibility_report)
        summary["compatibility_report"] = compatibility_report

        _write_json_file(dtc_root / "summary.json", summary)
        shutil.rmtree(inflight_root, ignore_errors=True)
        _write_progress_message(f"[4/{DTC_STAGE_TOTAL}] wrote final DTC manifests and summary.")
        return summary

    def _discover_base_artifacts(self) -> dict[str, list[TrajectoryArtifact]]:
        trace_paths = sorted(self.base_run_dir.glob("**/*_llm_trace.json"))
        artifacts_by_task: dict[str, list[TrajectoryArtifact]] = defaultdict(list)

        for trace_path in trace_paths:
            artifact = _load_artifact_from_paths(
                base_run_dir=self.base_run_dir,
                trace_path=trace_path,
                require_success=False,
            )
            if artifact is None:
                continue
            if self.task_filter and artifact.task_id != self.task_filter:
                continue
            artifacts_by_task[artifact.task_id].append(artifact)

        for task_id in list(artifacts_by_task):
            artifacts_by_task[task_id] = sorted(
                artifacts_by_task[task_id],
                key=lambda artifact: (artifact.episode_idx, artifact.run_stem, str(artifact.trace_path)),
            )

        return dict(artifacts_by_task)

    def _prepare_branch_jobs(
        self,
        artifacts_by_task: dict[str, list[TrajectoryArtifact]],
        tasks_summary: dict[str, dict[str, Any]],
        *,
        existing_records_by_key: dict[tuple[Any, ...], dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[BranchJob | RequestDivergenceJob], TrajectoryArtifact | None]:
        selected_bases_manifest: list[dict[str, Any]] = []
        collection_jobs: list[BranchJob | RequestDivergenceJob] = []
        first_selected_base: TrajectoryArtifact | None = None

        selected_bases_by_task: dict[str, list[TrajectoryArtifact]] = {}
        divergence_steps_by_artifact: dict[tuple[str, str], list[int]] = {}
        total_selected_bases = 0
        total_divergence_targets = 0

        for task_id in sorted(artifacts_by_task):
            task_artifacts = artifacts_by_task[task_id]
            selected_bases = _select_task_artifacts_for_collection(
                task_artifacts,
                success_only=self.success_only,
                smoke_test=self.smoke_test,
            )
            if not selected_bases:
                continue

            selected_bases_by_task[task_id] = selected_bases
            total_selected_bases += len(selected_bases)
            for artifact in selected_bases:
                divergence_steps = _select_divergence_steps(
                    artifact.total_steps,
                    self.divergence_count,
                    step_sampling_mode=self.step_sampling_mode,
                    sampling_key=f"{artifact.task_id}::{artifact.episode_id}",
                )
                divergence_steps_by_artifact[(artifact.task_id, artifact.episode_id)] = divergence_steps
                total_divergence_targets += len(divergence_steps)

        prepare_bases_bar = None
        prepare_divergence_bar = None
        if total_selected_bases > 0:
            prepare_bases_bar = tqdm(
                total=total_selected_bases,
                desc="DTC Prepare Bases",
                unit="base",
                position=0,
                leave=False,
                dynamic_ncols=True,
                file=_get_progress_stream(),
            )
        if total_divergence_targets > 0:
            divergence_desc = "DTC Prepare Divergences"
            prepare_divergence_bar = tqdm(
                total=total_divergence_targets,
                desc=divergence_desc,
                unit="divergence",
                position=1,
                leave=False,
                dynamic_ncols=True,
                file=_get_progress_stream(),
            )

        try:
            for task_id in sorted(selected_bases_by_task):
                task_artifacts = artifacts_by_task[task_id]
                selected_bases = selected_bases_by_task[task_id]

                tasks_summary[task_id] = {
                    "task_id": task_id,
                    "available_base_trajectories": len(task_artifacts),
                    "available_successful_trajectories": sum(
                        1 for artifact in task_artifacts if artifact.success
                    ),
                    "eligible_base_trajectories": len(
                        [artifact for artifact in task_artifacts if (artifact.success or not self.success_only)]
                    ),
                    "selected_bases": len(selected_bases),
                    "selected_base_ids": [artifact.episode_id for artifact in selected_bases],
                    "alt_mode": self.alt_mode,
                    "success_only": self.success_only,
                    "smoke_test": self.smoke_test,
                    "step_sampling_mode": self.step_sampling_mode,
                    "divergence_steps": {},
                    "generated_branches": 0,
                    "failed_branches": 0,
                }

                for artifact in selected_bases:
                    if first_selected_base is None:
                        first_selected_base = artifact
                    selected_bases_manifest.append(self._build_base_manifest_record(artifact))

                    divergence_steps = divergence_steps_by_artifact.get(
                        (artifact.task_id, artifact.episode_id),
                        [],
                    )
                    for divergence_step in divergence_steps:
                        base_action = artifact.executed_actions[divergence_step]
                        valid_actions = _available_actions_for_divergence(
                            artifact=artifact,
                            divergence_step=divergence_step,
                        )
                        valid_actions = _filter_excluded_alt_actions(
                            valid_actions,
                            self.exclude_alt_actions,
                        )
                        target_alt_count = _resolve_target_alt_count(
                            valid_actions=valid_actions,
                            base_action=base_action,
                            alt_budget=self.alt_budget,
                        )
                        if target_alt_count <= 0:
                            continue
                        step_summary = _ensure_divergence_step_summary(
                            tasks_summary[task_id],
                            base_traj_id=artifact.episode_id,
                            divergence_step=divergence_step,
                        )
                        step_summary["attempts"] += int(target_alt_count)

                        if prepare_divergence_bar is not None:
                            prepare_divergence_bar.set_postfix_str(
                                f"{artifact.run_stem} d{int(divergence_step):02d}"
                            )
                        if self.alt_mode == ALT_MODE_REQUEST:
                            existing_records = tuple(
                                copy.deepcopy(existing_records_by_key[key])
                                for key in _request_job_resume_keys(
                                    RequestDivergenceJob(
                                        task_id=artifact.task_id,
                                        base_traj_id=artifact.episode_id,
                                        base_run_dir=self.base_run_dir,
                                        trace_path=artifact.trace_path,
                                        csv_path=artifact.csv_path,
                                        json_path=artifact.json_path,
                                        relative_dir=artifact.relative_dir,
                                        config_path=self.config_path,
                                        output_root=self.output_root,
                                        divergence_step=divergence_step,
                                        alt_mode=self.alt_mode,
                                        target_alt_count=target_alt_count,
                                        existing_records=tuple(),
                                        rollout_max_steps=self.rollout_max_steps,
                                        rollout_extra_steps=self.rollout_extra_steps,
                                        exclude_alt_actions=self.exclude_alt_actions,
                                        client_max_tokens_override=self.client_max_tokens_override,
                                        client_timeout_override=self.client_timeout_override,
                                        client_max_retries_override=self.client_max_retries_override,
                                    )
                                )
                                if key in existing_records_by_key
                            )
                            collection_jobs.append(
                                RequestDivergenceJob(
                                    task_id=artifact.task_id,
                                    base_traj_id=artifact.episode_id,
                                    base_run_dir=self.base_run_dir,
                                    trace_path=artifact.trace_path,
                                    csv_path=artifact.csv_path,
                                    json_path=artifact.json_path,
                                    relative_dir=artifact.relative_dir,
                                    config_path=self.config_path,
                                    output_root=self.output_root,
                                    divergence_step=divergence_step,
                                    alt_mode=self.alt_mode,
                                    target_alt_count=target_alt_count,
                                    existing_records=existing_records,
                                    rollout_max_steps=self.rollout_max_steps,
                                    rollout_extra_steps=self.rollout_extra_steps,
                                    exclude_alt_actions=self.exclude_alt_actions,
                                    client_max_tokens_override=self.client_max_tokens_override,
                                    client_timeout_override=self.client_timeout_override,
                                    client_max_retries_override=self.client_max_retries_override,
                                )
                            )
                        else:
                            alt_plans = self._select_alt_plans_for_divergence(
                                artifact,
                                divergence_step=divergence_step,
                                existing_records_by_key=existing_records_by_key,
                            )
                            for candidate_idx, alt_plan in enumerate(alt_plans):
                                collection_jobs.append(
                                    BranchJob(
                                        task_id=artifact.task_id,
                                        base_traj_id=artifact.episode_id,
                                        base_run_dir=self.base_run_dir,
                                        trace_path=artifact.trace_path,
                                        csv_path=artifact.csv_path,
                                        json_path=artifact.json_path,
                                        relative_dir=artifact.relative_dir,
                                        config_path=self.config_path,
                                        output_root=self.output_root,
                                        divergence_step=divergence_step,
                                        alt_mode=self.alt_mode,
                                        alt_plan=alt_plan,
                                        alt_index=candidate_idx,
                                        alt_budget=self.alt_budget,
                                        alt_budget_used=len(alt_plans),
                                        rollout_max_steps=self.rollout_max_steps,
                                        rollout_extra_steps=self.rollout_extra_steps,
                                        client_max_tokens_override=self.client_max_tokens_override,
                                        client_timeout_override=self.client_timeout_override,
                                        client_max_retries_override=self.client_max_retries_override,
                                    )
                                )
                        if prepare_divergence_bar is not None:
                            prepare_divergence_bar.update(1)

                    if prepare_bases_bar is not None:
                        prepare_bases_bar.set_postfix_str(artifact.run_stem)
                        prepare_bases_bar.update(1)
        finally:
            if prepare_divergence_bar is not None:
                prepare_divergence_bar.close()
            if prepare_bases_bar is not None:
                prepare_bases_bar.close()

        collection_jobs.sort(key=_collection_job_sort_key)
        return selected_bases_manifest, collection_jobs, first_selected_base

    def _run_branch_jobs(
        self,
        branch_jobs: list[BranchJob | RequestDivergenceJob],
        *,
        branch_index_path: Path,
        existing_records_by_key: dict[tuple[Any, ...], dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        if not branch_jobs:
            if not self.resume:
                branch_index_path.write_text("", encoding="utf-8")
            return [], {
                "resume_loaded_records": 0,
                "resume_remaining_jobs": 0,
                "resume_skipped_jobs": 0,
            }

        existing_records_by_key = existing_records_by_key or {}
        remaining_jobs = _partition_collection_jobs_for_resume(
            branch_jobs,
            existing_records_by_key,
        )
        resume_stats = {
            "resume_loaded_records": len(existing_records_by_key),
            "resume_remaining_jobs": len(remaining_jobs),
            "resume_skipped_jobs": len(branch_jobs) - len(remaining_jobs),
        }

        if not self.resume:
            branch_index_path.write_text("", encoding="utf-8")

        progress_desc = "DTC Divergences" if self.alt_mode == ALT_MODE_REQUEST else "DTC Branches"
        records: list[dict[str, Any]] = list(existing_records_by_key.values())
        if not remaining_jobs:
            records.sort(key=_branch_record_sort_key)
            with branch_index_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return records, resume_stats

        open_mode = "a" if self.resume else "w"
        progress_prefix = "Resuming" if self.resume and existing_records_by_key else "Running"
        progress_total, progress_initial = _resolve_main_progress_counts(
            len(branch_jobs),
            resume=self.resume and bool(existing_records_by_key),
            resume_skipped_jobs=resume_stats["resume_skipped_jobs"],
        )
        remaining_job_count = len(remaining_jobs)
        completed_job_count = progress_initial
        retry_budget = max(0, int(self.external_retry_attempts))
        retry_stats = {
            "external_retry_attempts": retry_budget,
            "external_retries_scheduled": 0,
            "external_retry_exhausted_jobs": 0,
        }

        def finalize_result_records(
            *,
            result_records: list[dict[str, Any]],
            job_attempt: int,
            progress_handle,
            pbar,
        ) -> None:
            annotated_records = _annotate_result_records(
                result_records,
                job_attempt=job_attempt,
            )
            records.extend(annotated_records)
            for record in annotated_records:
                progress_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress_handle.flush()
            pbar.update(1)

        def job_retry_label(job: BranchJob | RequestDivergenceJob) -> str:
            if isinstance(job, RequestDivergenceJob):
                return (
                    f"task={job.task_id} base={job.base_traj_id} "
                    f"divergence={int(job.divergence_step)} mode=request"
                )
            return (
                f"task={job.task_id} base={job.base_traj_id} "
                f"divergence={int(job.divergence_step)} alt_index={int(job.alt_index)}"
            )

        def maybe_retry_job(
            *,
            job: BranchJob | RequestDivergenceJob,
            result_records: list[dict[str, Any]],
            attempt: int,
            next_pending_jobs: list[BranchJob | RequestDivergenceJob],
        ) -> bool:
            if not _result_records_are_retryable_failures(result_records):
                return False
            if attempt <= retry_budget:
                retry_stats["external_retries_scheduled"] += 1
                _write_progress_message(
                    f"[DTC] retrying {job_retry_label(job)} "
                    f"after retryable worker failure ({attempt}/{retry_budget})"
                )
                next_pending_jobs.append(job)
                return True
            retry_stats["external_retry_exhausted_jobs"] += 1
            logger.error(
                "DTC retry budget exhausted for %s after %d failed attempts",
                job_retry_label(job),
                attempt,
            )
            return False

        if self.num_workers <= 1 or len(remaining_jobs) == 1:
            if completed_job_count:
                progress_message = (
                    f"[DTC] {progress_prefix} {remaining_job_count} remaining worker jobs sequentially "
                    f"({completed_job_count}/{progress_total} already completed)."
                )
            else:
                progress_message = f"[DTC] {progress_prefix} {remaining_job_count} worker jobs sequentially."
            _write_progress_message(
                progress_message
            )
            with branch_index_path.open(open_mode, encoding="utf-8") as progress_handle:
                with tqdm(
                    total=progress_total,
                    initial=progress_initial,
                    desc=progress_desc,
                    unit="job",
                    position=0,
                    leave=True,
                    dynamic_ncols=True,
                    file=_get_progress_stream(),
                ) as pbar:
                    pending_jobs = list(remaining_jobs)
                    job_attempts: dict[tuple[Any, ...], int] = defaultdict(int)
                    while pending_jobs:
                        next_pending_jobs: list[BranchJob | RequestDivergenceJob] = []
                        for job in pending_jobs:
                            job_key = _collection_job_key(job)
                            job_attempts[job_key] += 1
                            attempt = int(job_attempts[job_key])
                            result_records = _run_collection_job(job)
                            if maybe_retry_job(
                                job=job,
                                result_records=result_records,
                                attempt=attempt,
                                next_pending_jobs=next_pending_jobs,
                            ):
                                continue
                            finalize_result_records(
                                result_records=result_records,
                                job_attempt=attempt,
                                progress_handle=progress_handle,
                                pbar=pbar,
                            )
                        pending_jobs = sorted(next_pending_jobs, key=_collection_job_sort_key)
            records.sort(key=_branch_record_sort_key)
            with branch_index_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            resume_stats.update(retry_stats)
            return records, resume_stats

        worker_count = min(self.num_workers, len(remaining_jobs))
        if completed_job_count:
            progress_message = (
                f"[DTC] {progress_prefix} {remaining_job_count} remaining worker jobs with {worker_count} workers "
                f"({completed_job_count}/{progress_total} already completed)."
            )
        else:
            progress_message = f"[DTC] {progress_prefix} {remaining_job_count} worker jobs with {worker_count} workers."
        _write_progress_message(
            progress_message
        )
        ctx = multiprocessing.get_context("fork")
        with branch_index_path.open(open_mode, encoding="utf-8") as progress_handle:
            with ctx.Pool(processes=worker_count) as pool:
                with tqdm(
                    total=progress_total,
                    initial=progress_initial,
                    desc=progress_desc,
                    unit="job",
                    position=0,
                    leave=True,
                    dynamic_ncols=True,
                    file=_get_progress_stream(),
                ) as pbar:
                    pending_jobs = list(remaining_jobs)
                    job_attempts: dict[tuple[Any, ...], int] = defaultdict(int)
                    while pending_jobs:
                        current_jobs_by_key = {
                            _collection_job_key(job): job
                            for job in pending_jobs
                        }
                        next_pending_jobs: list[BranchJob | RequestDivergenceJob] = []
                        for job_key, result_records in pool.imap_unordered(
                            _run_collection_job_enveloped,
                            pending_jobs,
                            chunksize=1,
                        ):
                            job = current_jobs_by_key[job_key]
                            job_attempts[job_key] += 1
                            attempt = int(job_attempts[job_key])
                            if maybe_retry_job(
                                job=job,
                                result_records=result_records,
                                attempt=attempt,
                                next_pending_jobs=next_pending_jobs,
                            ):
                                continue
                            finalize_result_records(
                                result_records=result_records,
                                job_attempt=attempt,
                                progress_handle=progress_handle,
                                pbar=pbar,
                            )
                        pending_jobs = sorted(next_pending_jobs, key=_collection_job_sort_key)

        records.sort(key=_branch_record_sort_key)
        with branch_index_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        resume_stats.update(retry_stats)
        return records, resume_stats

    def _build_base_manifest_record(self, artifact: TrajectoryArtifact) -> dict[str, Any]:
        return {
            "task_id": artifact.task_id,
            "episode_id": artifact.episode_id,
            "run_stem": artifact.run_stem,
            "seed": artifact.seed,
            "csv_path": str(artifact.csv_path),
            "json_path": str(artifact.json_path),
            "trace_path": str(artifact.trace_path),
            "total_steps": artifact.total_steps,
            "success": artifact.success,
            "episode_return": _as_float(artifact.episode_log.get("episode_return"), default=0.0),
            "progression": _as_float(artifact.episode_log.get("progression"), default=0.0),
        }

    def _select_alt_plans_for_divergence(
        self,
        artifact: TrajectoryArtifact,
        *,
        divergence_step: int,
        existing_records_by_key: dict[tuple[Any, ...], dict[str, Any]],
    ) -> list[SelectedAltActionPlan]:
        base_action = artifact.executed_actions[divergence_step]
        valid_actions = _available_actions_for_divergence(
            artifact=artifact,
            divergence_step=divergence_step,
        )
        valid_actions = _filter_excluded_alt_actions(valid_actions, self.exclude_alt_actions)
        target_count = _resolve_target_alt_count(
            valid_actions=valid_actions,
            base_action=base_action,
            alt_budget=self.alt_budget,
        )
        if target_count <= 0:
            return []

        existing_records_by_index: dict[int, dict[str, Any]] = {}
        for alt_index in range(target_count):
            key = (artifact.task_id, artifact.episode_id, divergence_step, _build_alt_traj_id(artifact.episode_id, divergence_step, alt_index))
            record = existing_records_by_key.get(key)
            if record is not None:
                existing_records_by_index[alt_index] = record

        if self.alt_mode in {ALT_MODE_RANDOM, ALT_MODE_RANDOM_TEACHER}:
            random_actions = _random_alt_actions(
                valid_actions=valid_actions,
                base_action=base_action,
                alt_budget=self.alt_budget,
                sampling_key=f"{artifact.task_id}::{artifact.episode_id}::{divergence_step}",
            )
            return [
                SelectedAltActionPlan(
                    raw_action_text=action,
                    executed_action_text=action,
                    action_defaulted=False,
                    request_count=0,
                    raw_output=action,
                    thought="",
                    interaction={
                        "messages": [],
                        "response": {
                            "raw_completion": action,
                            "reasoning": "",
                            "parsed_action": action,
                            "stop_reason": "dtc_random_action",
                            "input_tokens": 0,
                            "output_tokens": 0,
                        },
                        "meta": {
                            "source": "dtc.random_action",
                            "request_kwargs": {},
                            "llm_options_applied": {},
                        },
                    },
                    selection_attempts=1,
                )
                for action in random_actions
            ]

        if self.alt_mode != ALT_MODE_REQUEST:
            raise ValueError(f"Unsupported alt_mode: {self.alt_mode}")

        alt_plans: dict[int, SelectedAltActionPlan] = {}
        excluded_executed_actions = {base_action}
        for alt_index, record in sorted(existing_records_by_index.items()):
            executed_action_text = str(record.get("alt_action_text") or "").strip()
            if executed_action_text:
                excluded_executed_actions.add(executed_action_text)
            raw_action_text = str(record.get("alt_raw_action_text") or executed_action_text)
            alt_plans[alt_index] = SelectedAltActionPlan(
                raw_action_text=raw_action_text,
                executed_action_text=executed_action_text,
                action_defaulted=bool(record.get("alt_action_was_fallback", False)),
                request_count=_as_int(record.get("alt_request_count"), default=1),
                raw_output=str(record.get("alt_raw_output") or raw_action_text),
                thought=str(record.get("alt_thought") or ""),
                interaction={},
                selection_attempts=_as_int(record.get("branch_generation_attempts"), default=max(1, alt_index + 1)),
            )

        remaining_indices = [alt_index for alt_index in range(target_count) if alt_index not in alt_plans]
        if remaining_indices:
            requested_plans, _request_meta = self._request_alt_plans_for_divergence(
                artifact,
                divergence_step=divergence_step,
                target_count=len(remaining_indices),
                excluded_executed_actions=sorted(excluded_executed_actions),
                exclude_alt_actions=self.exclude_alt_actions,
            )
            for alt_index, plan in zip(remaining_indices, requested_plans):
                alt_plans[alt_index] = plan

        return [alt_plans[alt_index] for alt_index in sorted(alt_plans)]

    def _request_alt_plans_for_divergence(
        self,
        artifact: TrajectoryArtifact,
        *,
        divergence_step: int,
        target_count: int,
        excluded_executed_actions: list[str],
        exclude_alt_actions: Sequence[str] | None = None,
        selection_bar=None,
    ) -> tuple[list[SelectedAltActionPlan], dict[str, Any]]:
        if target_count <= 0:
            return [], {"ok": True, "attempts": 0, "message": "", "status": "ok"}

        logger.info(
            "DTC request-alt selection start: task=%s episode=%s divergence_step=%d target_count=%d excluded=%s",
            artifact.task_id,
            artifact.episode_id,
            divergence_step,
            target_count,
            excluded_executed_actions,
        )

        config = _build_runtime_config_for_artifact(
            self.runtime_config,
            artifact,
            client_max_tokens_override=self.client_max_tokens_override,
            client_timeout_override=self.client_timeout_override,
            client_max_retries_override=self.client_max_retries_override,
        )
        env = _make_env_for_artifact(config, artifact)
        agent = create_agent(config)
        if hasattr(agent, "clear_llm_interactions"):
            agent.clear_llm_interactions()
        agent.reset()
        agent.prompt_builder.update_instruction_prompt(artifact.instruction_prompt)

        try:
            replay_state = self._replay_base_prefix(
                artifact=artifact,
                env=env,
                prompt_builder=agent.prompt_builder,
                divergence_step=divergence_step,
                feedback_on_invalid_action=bool(config.eval.feedback_on_invalid_action),
            )
            if not replay_state["ok"]:
                logger.warning(
                    "DTC request-alt selection replay failed: task=%s episode=%s divergence_step=%d status=%s message=%s",
                    artifact.task_id,
                    artifact.episode_id,
                    divergence_step,
                    replay_state["status"],
                    replay_state["message"],
                )
                return [], {
                    "ok": False,
                    "attempts": 0,
                    "message": replay_state["message"],
                    "status": replay_state["status"],
                }

            prefix_prompt_builder = copy.deepcopy(replay_state["prompt_builder"])
            current_obs = replay_state["obs"]
            prev_executed_action = replay_state["prev_executed_action"]
            base_action = artifact.executed_actions[divergence_step]
            valid_actions = _available_actions_for_divergence(
                artifact=artifact,
                divergence_step=divergence_step,
            )
            valid_actions = _filter_excluded_alt_actions(valid_actions, exclude_alt_actions)

            selected_plans: list[SelectedAltActionPlan] = []
            excluded_actions = {action for action in excluded_executed_actions if action}
            excluded_actions.add(base_action)
            max_attempts = max(target_count * 4, max(1, len(valid_actions)) * 2)
            attempts = 0

            while len(selected_plans) < target_count and attempts < max_attempts:
                attempts += 1
                if selection_bar is not None:
                    selection_bar.set_postfix_str(
                        f"{artifact.run_stem} d{int(divergence_step):02d} attempt={attempts} selected={len(selected_plans)}/{target_count}"
                    )
                instruction = _build_alt_request_instruction(
                    env_name=artifact.env_name,
                    base_action=base_action,
                    excluded_executed_actions=sorted(excluded_actions),
                    valid_actions=valid_actions,
                )
                act_result = _agent_act_with_prompt_builder(
                    agent=agent,
                    prompt_builder=prefix_prompt_builder,
                    obs=current_obs,
                    prev_action=prev_executed_action,
                    extra_user_text=instruction,
                    validate_action=env.check_action_validity,
                )
                if not act_result["ok"]:
                    return selected_plans, {
                        "ok": False,
                        "attempts": int(act_result.get("request_count", attempts) or attempts),
                        "message": json.dumps(
                            {
                                "abort_reason": act_result.get("abort_reason"),
                                "abort_leaf_history": act_result.get("abort_leaf_history", []),
                                "abort_leaf_counts": act_result.get("abort_leaf_counts", {}),
                                "abort_leaf_last": act_result.get("abort_leaf_last", ""),
                            },
                            ensure_ascii=False,
                        ),
                        "status": str(act_result.get("abort_reason") or "retry_exhausted"),
                    }
                response = act_result["response"]
                interaction = copy.deepcopy(act_result.get("interaction") or {})
                interaction_meta = interaction.setdefault("meta", {})
                interaction_meta["source"] = "dtc.request_action"
                interaction_meta["excluded_executed_actions"] = sorted(excluded_actions)
                interaction_meta["alt_mode"] = "request"
                interaction_meta["retry_attempts"] = act_result.get("retry_attempts", [])

                raw_action_text = str(act_result["action"] or "").strip()
                executed_action_text = str(act_result.get("executed_action") or raw_action_text)
                if executed_action_text in excluded_actions:
                    continue

                selected_plans.append(
                    SelectedAltActionPlan(
                        raw_action_text=raw_action_text,
                        executed_action_text=executed_action_text,
                        action_defaulted=False,
                        request_count=int(act_result.get("request_count", 1) or 1),
                        raw_output=str(act_result["raw_output"] or raw_action_text),
                        thought=str(act_result.get("thought") or getattr(response, "reasoning", "") or ""),
                        interaction=interaction,
                        selection_attempts=attempts,
                    )
                )
                excluded_actions.add(executed_action_text)
                if selection_bar is not None:
                    selection_bar.update(1)

            logger.info(
                "DTC request-alt selection done: task=%s episode=%s divergence_step=%d selected=%d/%d attempts=%d",
                artifact.task_id,
                artifact.episode_id,
                divergence_step,
                len(selected_plans),
                target_count,
                attempts,
            )
            return selected_plans, {
                "ok": True,
                "attempts": attempts,
                "message": "",
                "status": "ok",
            }
        finally:
            env.close()

    def _build_request_alt_unavailable_record(
        self,
        *,
        artifact: TrajectoryArtifact,
        divergence_step: int,
        alt_index: int,
        base_outcome: dict[str, Any],
        state_key: str,
        base_action: str,
        replay_success: bool,
        replay_error: str | None,
        alt_budget_used: int,
        selection_attempts: int,
    ) -> dict[str, Any]:
        alt_traj_id = _build_alt_traj_id(artifact.episode_id, divergence_step, alt_index)
        return {
            "status": "request_alt_unavailable",
            "task_id": artifact.task_id,
            "seed": artifact.seed,
            "base_traj_id": artifact.episode_id,
            "alt_traj_id": alt_traj_id,
            "divergence_step": divergence_step,
            "base_action_text": base_action,
            "alt_action_text": "",
            "alt_raw_action_text": "",
            "alt_action_was_fallback": False,
            "alt_request_count": 0,
            "alt_thought": "",
            "alt_raw_output": "",
            "alt_mode": "request",
            "base_outcome": base_outcome,
            "alt_outcome": None,
            "state_key": state_key,
            "replay_success": bool(replay_success),
            "replay_error": replay_error,
            "branch_generation_attempts": int(selection_attempts),
            "alt_budget_used": int(alt_budget_used),
        }

    def _collect_request_divergence(
        self,
        *,
        artifact: TrajectoryArtifact,
        config,
        divergence_step: int,
        target_alt_count: int,
        existing_records: tuple[dict[str, Any], ...],
        output_root: Path,
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        if target_alt_count <= 0:
            return []

        base_action = artifact.executed_actions[divergence_step]
        base_outcome = self._build_outcome_summary(artifact.episode_log, artifact.trace_payload)
        prefix_actions = artifact.executed_actions[:divergence_step]
        state_key = _build_state_key(artifact.task_id, artifact.seed, prefix_actions)
        records: list[dict[str, Any]] = []

        existing_records_by_index: dict[int, dict[str, Any]] = {}
        excluded_executed_actions = {base_action}
        for record in existing_records:
            alt_index = _parse_alt_index(str(record.get("alt_traj_id") or ""))
            if alt_index is None or alt_index < 0:
                continue
            existing_records_by_index[int(alt_index)] = copy.deepcopy(record)
            executed_action = str(record.get("alt_action_text") or "").strip()
            if executed_action:
                excluded_executed_actions.add(executed_action)

        missing_indices = [alt_index for alt_index in range(target_alt_count) if alt_index not in existing_records_by_index]
        progress_path = _request_progress_path(
            output_root=output_root,
            task_id=artifact.task_id,
            base_traj_id=artifact.episode_id,
            divergence_step=divergence_step,
        )

        selection_bar = None
        if missing_indices:
            selection_bar = tqdm(
                total=target_alt_count,
                initial=len(existing_records_by_index),
                desc=f"{_current_worker_label()} {artifact.run_stem} d{int(divergence_step):02d} select",
                unit="alt",
                position=_current_worker_position(),
                leave=False,
                dynamic_ncols=True,
                file=_get_progress_stream(),
            )

        try:
            selected_plans: list[SelectedAltActionPlan] = []
            replay_success = True
            replay_error = None
            selection_attempts = 0
            if missing_indices:
                selected_plans, request_meta = self._request_alt_plans_for_divergence(
                    artifact,
                    divergence_step=divergence_step,
                    target_count=len(missing_indices),
                    excluded_executed_actions=sorted(excluded_executed_actions),
                    exclude_alt_actions=self.exclude_alt_actions,
                    selection_bar=selection_bar,
                )
                replay_success = bool(request_meta.get("ok", True))
                replay_error = str(request_meta.get("message") or "") or None
                selection_attempts = _as_int(request_meta.get("attempts"), default=0)
            alt_budget_used = len(existing_records_by_index) + len(selected_plans)
            for alt_index, alt_plan in zip(missing_indices, selected_plans):
                branch_record = self._collect_single_branch(
                    artifact=artifact,
                    config=config,
                    divergence_step=divergence_step,
                    alt_mode="request",
                    alt_plan=alt_plan,
                    alt_index=alt_index,
                    alt_budget_used=alt_budget_used,
                    output_dir=output_dir,
                )
                records.append(branch_record)
                _append_jsonl_record(progress_path, branch_record)

            selected_count = len(selected_plans)
            for alt_index in missing_indices[selected_count:]:
                failure_record = self._build_request_alt_unavailable_record(
                    artifact=artifact,
                    divergence_step=divergence_step,
                    alt_index=alt_index,
                    base_outcome=base_outcome,
                    state_key=state_key,
                    base_action=base_action,
                    replay_success=replay_success,
                    replay_error=replay_error,
                    alt_budget_used=alt_budget_used,
                    selection_attempts=selection_attempts,
                )
                records.append(failure_record)
                _append_jsonl_record(progress_path, failure_record)

            return records
        finally:
            if selection_bar is not None:
                selection_bar.close()

    def _collect_single_branch(
        self,
        *,
        artifact: TrajectoryArtifact,
        config,
        divergence_step: int,
        alt_mode: str,
        alt_plan: SelectedAltActionPlan,
        alt_index: int,
        alt_budget_used: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        base_outcome = self._build_outcome_summary(artifact.episode_log, artifact.trace_payload)
        prefix_actions = artifact.executed_actions[:divergence_step]
        state_key = _build_state_key(artifact.task_id, artifact.seed, prefix_actions)
        base_action = artifact.executed_actions[divergence_step]
        raw_alt_action = alt_plan.raw_action_text
        executed_alt_action = alt_plan.executed_action_text
        alt_traj_id = _build_alt_traj_id(artifact.episode_id, divergence_step, alt_index)
        alt_stem = _build_alt_stem(artifact.run_stem, divergence_step, alt_index)

        env = _make_env_for_artifact(config, artifact)
        agent = create_agent(config)
        if hasattr(agent, "clear_llm_interactions"):
            agent.clear_llm_interactions()
        agent.reset()
        agent.prompt_builder.update_instruction_prompt(artifact.instruction_prompt)

        try:
            max_steps = _resolve_rollout_step_cap(
                base_total_steps=artifact.total_steps,
                env_max_steps=None,
                rollout_max_steps=self.rollout_max_steps,
                rollout_extra_steps=self.rollout_extra_steps,
            )
            if max_steps is not None and max_steps < (divergence_step + 1):
                max_steps = divergence_step + 1
            progress_total = (
                int(max_steps)
                if max_steps is not None
                else max(divergence_step + 1, artifact.total_steps + max(0, int(self.rollout_extra_steps)))
            )
            progress_desc = _build_worker_progress_desc(
                artifact=artifact,
                divergence_step=divergence_step,
                alt_index=alt_index,
            )
            with tqdm(
                total=progress_total,
                desc=progress_desc,
                unit="step",
                position=_current_worker_position(),
                leave=False,
                dynamic_ncols=True,
                file=_get_progress_stream(),
            ) as branch_pbar:
                replay_state = self._replay_base_prefix(
                    artifact=artifact,
                    env=env,
                    prompt_builder=agent.prompt_builder,
                    divergence_step=divergence_step,
                    feedback_on_invalid_action=bool(config.eval.feedback_on_invalid_action),
                    progress_bar=branch_pbar,
                )
                if not replay_state["ok"]:
                    return {
                        "status": replay_state["status"],
                        "task_id": artifact.task_id,
                        "seed": artifact.seed,
                        "base_traj_id": artifact.episode_id,
                        "alt_traj_id": alt_traj_id,
                        "divergence_step": divergence_step,
                        "base_action_text": base_action,
                        "alt_action_text": executed_alt_action,
                        "alt_raw_action_text": raw_alt_action,
                        "alt_action_was_fallback": bool(alt_plan.action_defaulted),
                        "alt_request_count": int(alt_plan.request_count),
                        "alt_thought": str(alt_plan.thought or ""),
                        "alt_raw_output": str(alt_plan.raw_output or raw_alt_action),
                        "alt_mode": alt_mode,
                        "base_outcome": base_outcome,
                        "alt_outcome": None,
                        "state_key": state_key,
                        "replay_success": False,
                        "replay_error": replay_state["message"],
                        "branch_generation_attempts": int(alt_plan.selection_attempts),
                        "alt_budget_used": int(alt_budget_used),
                    }

                current_obs = replay_state["obs"]
                prev_executed_action = replay_state["prev_executed_action"]
                prefix_reward = replay_state["prefix_reward"]
                env_max_steps = int(env.max_steps) if getattr(env, "max_steps", None) is not None else None
                max_steps = _resolve_rollout_step_cap(
                    base_total_steps=artifact.total_steps,
                    env_max_steps=env_max_steps,
                    rollout_max_steps=self.rollout_max_steps,
                    rollout_extra_steps=self.rollout_extra_steps,
                )
                if max_steps is not None and max_steps < (divergence_step + 1):
                    max_steps = divergence_step + 1
                if max_steps is not None and branch_pbar.total != max_steps:
                    branch_pbar.total = max(max_steps, branch_pbar.n)
                    branch_pbar.refresh()

                prompt_builder_at_divergence = copy.deepcopy(replay_state["prompt_builder"])
                if alt_mode == ALT_MODE_RANDOM_TEACHER:
                    teacher_completion = _request_fixed_action_teacher_completion(
                        agent=agent,
                        prompt_builder=prompt_builder_at_divergence,
                        obs=current_obs,
                        prev_action=prev_executed_action,
                        fixed_action=executed_alt_action,
                        alt_mode=alt_mode,
                    )
                    alt_plan = replace(
                        alt_plan,
                        request_count=int(teacher_completion["request_count"]),
                        raw_output=str(teacher_completion["raw_output"] or raw_alt_action),
                        thought=str(teacher_completion["thought"] or ""),
                        interaction=copy.deepcopy(teacher_completion["interaction"]),
                        generation_wall_time_sec=float(
                            teacher_completion["generation_wall_time_sec"]
                        ),
                    )

                # Materialize the prompt state for the divergence step without mutating the base trace.
                agent.prompt_builder = copy.deepcopy(prompt_builder_at_divergence)
                if prev_executed_action:
                    agent.prompt_builder.update_action(prev_executed_action)
                agent.prompt_builder.update_observation(current_obs)

                prefix_trace_calls = copy.deepcopy(artifact.trace_calls[:divergence_step])
                prefix_csv_rows = artifact.csv_rows[:divergence_step]

                generated_calls: list[dict[str, Any]] = []
                generated_csv_rows: list[list[Any]] = []

                action_defaulted = bool(alt_plan.action_defaulted)
                executed_alt = executed_alt_action
                if executed_alt == base_action:
                    return {
                        "status": "rejected_same_executed_action",
                        "task_id": artifact.task_id,
                        "seed": artifact.seed,
                        "base_traj_id": artifact.episode_id,
                        "alt_traj_id": alt_traj_id,
                        "divergence_step": divergence_step,
                        "base_action_text": base_action,
                        "alt_action_text": executed_alt,
                        "alt_raw_action_text": raw_alt_action,
                        "alt_action_was_fallback": action_defaulted,
                        "alt_request_count": int(alt_plan.request_count),
                        "alt_thought": str(alt_plan.thought or ""),
                        "alt_raw_output": str(alt_plan.raw_output or raw_alt_action),
                        "alt_mode": alt_mode,
                        "base_outcome": base_outcome,
                        "alt_outcome": None,
                        "state_key": state_key,
                        "replay_success": True,
                        "replay_error": None,
                        "branch_generation_attempts": int(alt_plan.selection_attempts),
                        "alt_budget_used": int(alt_budget_used),
                    }

                divergence_record = self._execute_branch_step(
                    env=env,
                    instruction_prompt=artifact.instruction_prompt,
                    step_index=divergence_step,
                    obs=current_obs,
                    raw_action_text=raw_alt_action,
                    executed_action_text=executed_alt,
                    action_defaulted=action_defaulted,
                    request_count=int(alt_plan.request_count),
                    raw_output=str(alt_plan.raw_output or raw_alt_action),
                    thought=str(alt_plan.thought or ""),
                    interaction=copy.deepcopy(alt_plan.interaction),
                    request_wall_time_sec=float(alt_plan.generation_wall_time_sec or 0.0),
                    feedback_on_invalid_action=bool(config.eval.feedback_on_invalid_action),
                )
                generated_calls.append(divergence_record["trace_call"])
                generated_csv_rows.append(divergence_record["csv_row"])
                branch_pbar.update(1)

                total_reward = prefix_reward + divergence_record["reward"]
                done = divergence_record["done"]
                prev_executed_action = divergence_record["executed_action"]
                current_obs = divergence_record["obs"]
                current_info = divergence_record["info"]
                current_stats = divergence_record["stats"]

                step_index = divergence_step + 1
                rollout_capped = False

                while not done and (max_steps is None or step_index < max_steps):
                    rollout_step = self._execute_agent_step(
                        env=env,
                        agent=agent,
                        instruction_prompt=artifact.instruction_prompt,
                        step_index=step_index,
                        obs=current_obs,
                        prev_executed_action=prev_executed_action,
                        feedback_on_invalid_action=bool(config.eval.feedback_on_invalid_action),
                    )
                    if not rollout_step.get("ok", False):
                        return {
                            "status": str(rollout_step.get("status") or "retry_exhausted"),
                            "task_id": artifact.task_id,
                            "seed": artifact.seed,
                            "base_traj_id": artifact.episode_id,
                            "alt_traj_id": alt_traj_id,
                            "divergence_step": divergence_step,
                            "base_action_text": base_action,
                            "alt_action_text": executed_alt,
                            "alt_raw_action_text": raw_alt_action,
                            "alt_action_was_fallback": action_defaulted,
                            "alt_request_count": int(alt_plan.request_count),
                            "alt_thought": str(alt_plan.thought or ""),
                            "alt_raw_output": str(alt_plan.raw_output or raw_alt_action),
                            "alt_mode": alt_mode,
                            "base_outcome": base_outcome,
                            "alt_outcome": None,
                            "state_key": state_key,
                            "replay_success": True,
                            "replay_error": None,
                            "branch_generation_attempts": int(alt_plan.selection_attempts),
                            "alt_budget_used": int(alt_budget_used),
                            "abort_step": int(rollout_step.get("abort_step", step_index)),
                            "abort_reason": str(rollout_step.get("abort_reason") or "retry_exhausted"),
                            "abort_leaf_history": rollout_step.get("abort_leaf_history", []),
                            "abort_leaf_counts": rollout_step.get("abort_leaf_counts", {}),
                            "abort_leaf_last": rollout_step.get("abort_leaf_last", ""),
                            "abort_stop_reason": rollout_step.get("abort_stop_reason", ""),
                            "retry_attempts": rollout_step.get("retry_attempts", []),
                        }
                    generated_calls.append(rollout_step["trace_call"])
                    generated_csv_rows.append(rollout_step["csv_row"])
                    total_reward += rollout_step["reward"]
                    done = rollout_step["done"]
                    prev_executed_action = rollout_step["executed_action"]
                    current_obs = rollout_step["obs"]
                    current_info = rollout_step["info"]
                    current_stats = rollout_step["stats"]
                    step_index += 1
                    branch_pbar.update(1)

                if not done and max_steps is not None and step_index >= max_steps:
                    rollout_capped = True

                full_trace_calls = prefix_trace_calls + generated_calls
                full_csv_rows = [
                    [row.get(column, "") for column in CSV_COLUMNS]
                    for row in prefix_csv_rows
                ] + generated_csv_rows

                output_dir.mkdir(parents=True, exist_ok=True)
                csv_path = output_dir / f"{alt_stem}.csv"
                json_path = output_dir / f"{alt_stem}.json"
                trace_path = output_dir / f"{alt_stem}_llm_trace.json"

                self._write_csv(csv_path, full_csv_rows)
                episode_log = self._build_alt_episode_log(
                    artifact=artifact,
                    trace_calls=full_trace_calls,
                    env_stats=current_stats,
                    done=done,
                    episode_return=total_reward,
                    failed_candidates=env.failed_candidates,
                    rollout_capped=rollout_capped,
                    base_action=base_action,
                    alt_action=executed_alt,
                    alt_mode=alt_mode,
                    divergence_step=divergence_step,
                    alt_traj_id=alt_traj_id,
                    state_key=state_key,
                    rollout_step_cap=max_steps,
                )
                json_path.write_text(json.dumps(episode_log, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
                trace_payload = {
                    "schema_version": "thought_action_v1",
                    "benchmark": "balrog",
                    "env_name": artifact.env_name,
                    "task_or_env_params": artifact.task_id,
                    "episode_id": alt_traj_id,
                    "episode_idx": int(artifact.episode_idx),
                    "seed": int(artifact.seed),
                    "num_calls": len(full_trace_calls),
                    "calls": full_trace_calls,
                    "extras": {
                        "process_num": artifact.episode_log.get("process_num"),
                        "search_method": "divergence_tree",
                        "base_traj_id": artifact.episode_id,
                        "divergence_step": divergence_step,
                        "base_action_text": base_action,
                        "alt_action_text": executed_alt,
                        "alt_mode": alt_mode,
                        "state_key": state_key,
                        "call_alignment": {
                            "step_records": len(full_trace_calls),
                            "llm_interactions": len(full_trace_calls),
                        },
                    },
                }
                trace_path.write_text(json.dumps(trace_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

                alt_outcome = self._build_outcome_summary(episode_log, trace_payload)
                return {
                    "status": "generated",
                    "task_id": artifact.task_id,
                    "seed": artifact.seed,
                    "base_traj_id": artifact.episode_id,
                    "alt_traj_id": alt_traj_id,
                    "divergence_step": divergence_step,
                    "base_action_text": base_action,
                    "alt_action_text": executed_alt,
                    "alt_raw_action_text": raw_alt_action,
                    "alt_action_was_fallback": action_defaulted,
                    "alt_request_count": int(alt_plan.request_count),
                    "alt_thought": str(alt_plan.thought or ""),
                    "alt_raw_output": str(alt_plan.raw_output or raw_alt_action),
                    "alt_mode": alt_mode,
                    "base_outcome": base_outcome,
                    "alt_outcome": alt_outcome,
                    "state_key": state_key,
                    "replay_success": True,
                    "replay_error": None,
                    "branch_generation_attempts": int(alt_plan.selection_attempts),
                    "alt_budget_used": int(alt_budget_used),
                    "rollout_capped": rollout_capped,
                    "rollout_step_cap": max_steps,
                    "output_csv": str(csv_path),
                    "output_json": str(json_path),
                    "output_trace": str(trace_path),
                }
        finally:
            env.close()

    def _replay_base_prefix(
        self,
        *,
        artifact: TrajectoryArtifact,
        env,
        prompt_builder,
        divergence_step: int,
        feedback_on_invalid_action: bool,
        progress_bar=None,
    ) -> dict[str, Any]:
        _seed_process_rng(artifact.seed)
        obs, _info = env.reset(seed=artifact.seed)
        prefix_reward = 0.0
        prev_executed_action: str | None = None

        for step_idx in range(divergence_step):
            expected_observation = str(artifact.trace_calls[step_idx].get("observation") or "")
            actual_observation = _extract_long_term_observation(obs)
            if _normalize_text(actual_observation) != _normalize_text(expected_observation):
                return {
                    "ok": False,
                    "status": "replay_mismatch",
                    "message": f"prefix step {step_idx} observation mismatch",
                }

            if prev_executed_action:
                prompt_builder.update_action(prev_executed_action)
            prompt_builder.update_observation(obs)

            base_call = artifact.trace_calls[step_idx]
            executed_action = str(base_call.get("action") or "").strip()
            obs, reward, terminated, truncated, _info = env.step(executed_action)
            prefix_reward += reward
            if progress_bar is not None:
                progress_bar.update(1)
            if _trace_call_was_fallback(base_call) and feedback_on_invalid_action:
                _inject_invalid_action_notice(obs, executed_action)
            prev_executed_action = executed_action

            if (terminated or truncated) and step_idx < (divergence_step - 1):
                return {
                    "ok": False,
                    "status": "prefix_terminated_early",
                    "message": f"prefix terminated at step {step_idx}",
                }

        expected_divergence_observation = str(
            artifact.trace_calls[divergence_step].get("observation") or ""
        )
        actual_divergence_observation = _extract_long_term_observation(obs)
        if _normalize_text(actual_divergence_observation) != _normalize_text(expected_divergence_observation):
            return {
                "ok": False,
                "status": "replay_mismatch",
                "message": f"divergence observation mismatch at step {divergence_step}",
            }

        return {
            "ok": True,
            "status": "ok",
            "message": "",
            "obs": obs,
            "prev_executed_action": prev_executed_action,
            "prefix_reward": prefix_reward,
            "prompt_builder": copy.deepcopy(prompt_builder),
        }

    def _execute_branch_step(
        self,
        *,
        env,
        instruction_prompt: str,
        step_index: int,
        obs: dict[str, Any],
        raw_action_text: str,
        executed_action_text: str,
        action_defaulted: bool,
        request_count: int,
        raw_output: str,
        thought: str,
        interaction: dict[str, Any] | None,
        request_wall_time_sec: float = 0.0,
        feedback_on_invalid_action: bool,
    ) -> dict[str, Any]:
        step_started = time.perf_counter()
        observation_raw = _extract_long_term_observation(obs)
        observation_raw, observation_clean, invalid_action_notice = _split_observation_notice(observation_raw)

        obs_after, reward, terminated, truncated, info = env.step(executed_action_text)
        done = terminated or truncated
        step_wall_time_sec = time.perf_counter() - step_started + max(
            0.0,
            float(request_wall_time_sec or 0.0),
        )

        observation_post_raw = _extract_long_term_observation(obs_after)
        observation_post_raw, observation_post_clean, _ = _split_observation_notice(observation_post_raw)
        observation_pre_for_compare = observation_clean or observation_raw
        observation_post_for_compare = observation_post_clean or observation_post_raw
        obs_changed = observation_pre_for_compare != observation_post_for_compare

        if action_defaulted and feedback_on_invalid_action:
            _inject_invalid_action_notice(obs_after, executed_action_text)

        current_stats = env.get_stats()
        current_progression = _as_float(current_stats.get("progression"), default=0.0)
        feedback = str(info.get("feedback", "") or "") if isinstance(info, dict) else ""
        won = bool(info.get("won", False)) if isinstance(info, dict) else False
        lost = bool(info.get("lost", False)) if isinstance(info, dict) else False
        termination_reason = _get_termination_reason(
            terminated=bool(terminated),
            truncated=bool(truncated),
            reward=float(reward),
            progression=current_progression,
            won=won,
            lost=lost,
        )

        response_info = {}
        interaction_meta = {}
        serialized_messages: list[dict[str, Any]] = []
        if interaction:
            response_info = interaction.get("response", {}) if isinstance(interaction, dict) else {}
            interaction_meta = interaction.get("meta", {}) if isinstance(interaction, dict) else {}
            serialized_messages = interaction.get("messages", []) if isinstance(interaction, dict) else []
        input_tokens = int(response_info.get("input_tokens", 0) or 0)
        output_tokens = int(response_info.get("output_tokens", 0) or 0)

        trace_call = {
            "call_idx": step_index,
            "global_step": step_index,
            "episode_step": step_index,
            "instruction": instruction_prompt,
            "observation": observation_raw,
            "thought": thought,
            "action": executed_action_text,
            "raw_output": raw_output,
            "feedback": feedback,
            "reward": float(reward),
            "progression": current_progression,
            "done": bool(done),
            "won": won,
            "lost": lost,
            "request_count": int(request_count),
            "step_wall_time_sec": float(step_wall_time_sec),
            "token_usage": {
                "total": input_tokens + output_tokens,
                "input": input_tokens,
                "output": output_tokens,
            },
            "extras": {
                "observation_clean": observation_clean,
                "observation_post_clean": observation_post_clean,
                "invalid_action_notice": invalid_action_notice,
                "raw_action_text": raw_action_text,
                "executed_action_text": executed_action_text,
                "was_fallback": bool(action_defaulted),
                "llm_options_applied": interaction_meta.get("llm_options_applied", {}),
                "request_kwargs": interaction_meta.get("request_kwargs", {}),
                "balrog_raw": {
                    "messages": serialized_messages,
                    "response": {
                        "raw_completion": response_info.get("raw_completion", raw_output),
                        "reasoning": response_info.get("reasoning", thought),
                        "parsed_action": response_info.get("parsed_action", raw_action_text),
                        "stop_reason": response_info.get("stop_reason", ""),
                        "input_tokens": int(response_info.get("input_tokens", input_tokens) or 0),
                        "output_tokens": int(response_info.get("output_tokens", output_tokens) or 0),
                    },
                    "meta": {
                        "source": interaction_meta.get("source", "dtc.synthetic"),
                        "llm_options_applied": interaction_meta.get("llm_options_applied", {}),
                        "request_kwargs": interaction_meta.get("request_kwargs", {}),
                    },
                },
            },
        }

        csv_row = [
            step_index,
            instruction_prompt,
            observation_raw,
            thought,
            raw_action_text,
            executed_action_text,
            raw_output,
            feedback,
            observation_post_raw,
            obs_changed,
            reward,
            current_progression,
            terminated,
            truncated,
            done,
            termination_reason,
            action_defaulted,
            input_tokens,
            output_tokens,
            step_wall_time_sec,
            won,
            lost,
        ]

        return {
            "trace_call": trace_call,
            "csv_row": csv_row,
            "reward": float(reward),
            "done": bool(done),
            "executed_action": executed_action_text,
            "obs": obs_after,
            "info": info,
            "stats": current_stats,
        }

    def _execute_agent_step(
        self,
        *,
        env,
        agent,
        instruction_prompt: str,
        step_index: int,
        obs: dict[str, Any],
        prev_executed_action: str,
        feedback_on_invalid_action: bool,
    ) -> dict[str, Any]:
        step_started = time.perf_counter()
        observation_raw = _extract_long_term_observation(obs)
        observation_raw, observation_clean, invalid_action_notice = _split_observation_notice(observation_raw)

        act_result = run_agent_action(
            agent=agent,
            obs=obs,
            prev_action=prev_executed_action,
            validate_action=env.check_action_validity,
        )
        interaction = act_result.get("interaction", {})
        raw_output = str(act_result.get("raw_output", "") or "")
        raw_action_text = str(act_result.get("model_action", "") or "").strip()
        executed_action_text = str(act_result.get("executed_action", "") or "").strip()
        action_defaulted = bool(act_result.get("action_defaulted", False))
        thought = str(act_result.get("thought", "") or "")
        input_tokens = int(act_result.get("input_tokens", 0) or 0)
        output_tokens = int(act_result.get("output_tokens", 0) or 0)

        if not act_result["ok"]:
            step_wall_time_sec = time.perf_counter() - step_started
            return {
                "ok": False,
                "status": str(act_result.get("abort_reason") or "retry_exhausted"),
                "abort_reason": str(act_result.get("abort_reason") or "retry_exhausted"),
                "abort_step": int(step_index),
                "abort_leaf_history": act_result.get("abort_leaf_history", []),
                "abort_leaf_counts": act_result.get("abort_leaf_counts", {}),
                "abort_leaf_last": act_result.get("abort_leaf_last", ""),
                "abort_stop_reason": act_result.get("abort_stop_reason", ""),
                "retry_attempts": act_result.get("retry_attempts", []),
                "request_count": int(act_result.get("request_count", 0) or 0),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "raw_output": raw_output,
                "raw_action_text": raw_action_text,
                "thought": thought,
                "step_wall_time_sec": float(step_wall_time_sec),
            }

        executed_action = executed_action_text

        if hasattr(agent, "update_env_action"):
            agent.update_env_action(executed_action)

        obs_after, reward, terminated, truncated, info = env.step(executed_action)
        done = terminated or truncated
        step_wall_time_sec = time.perf_counter() - step_started

        observation_post_raw = _extract_long_term_observation(obs_after)
        observation_post_raw, observation_post_clean, _ = _split_observation_notice(observation_post_raw)
        observation_pre_for_compare = observation_clean or observation_raw
        observation_post_for_compare = observation_post_clean or observation_post_raw
        obs_changed = observation_pre_for_compare != observation_post_for_compare

        if action_defaulted and feedback_on_invalid_action:
            _inject_invalid_action_notice(obs_after, executed_action_text)

        current_stats = env.get_stats()
        current_progression = _as_float(current_stats.get("progression"), default=0.0)
        feedback = str(info.get("feedback", "") or "") if isinstance(info, dict) else ""
        won = bool(info.get("won", False)) if isinstance(info, dict) else False
        lost = bool(info.get("lost", False)) if isinstance(info, dict) else False
        termination_reason = _get_termination_reason(
            terminated=bool(terminated),
            truncated=bool(truncated),
            reward=float(reward),
            progression=current_progression,
            won=won,
            lost=lost,
        )

        interaction_meta = interaction.get("meta", {}) if isinstance(interaction, dict) else {}
        if not isinstance(interaction_meta, dict):
            interaction_meta = {}
        serialized_messages = interaction.get("messages", []) if isinstance(interaction, dict) else []
        response_info = interaction.get("response", {}) if isinstance(interaction, dict) else {}
        if not isinstance(response_info, dict):
            response_info = {}

        trace_call = {
            "call_idx": step_index,
            "global_step": step_index,
            "episode_step": step_index,
            "instruction": instruction_prompt,
            "observation": observation_raw,
            "thought": thought,
            "action": executed_action_text,
            "raw_output": raw_output,
            "feedback": feedback,
            "reward": float(reward),
            "progression": current_progression,
            "done": bool(done),
            "won": won,
            "lost": lost,
            "request_count": int(act_result.get("request_count", 1) or 1),
            "step_wall_time_sec": float(step_wall_time_sec),
            "token_usage": {
                "total": input_tokens + output_tokens,
                "input": input_tokens,
                "output": output_tokens,
            },
            "extras": {
                "observation_clean": observation_clean,
                "observation_post_clean": observation_post_clean,
                "invalid_action_notice": invalid_action_notice,
                "raw_action_text": raw_action_text,
                "executed_action_text": executed_action_text,
                "was_fallback": bool(action_defaulted),
                "llm_options_applied": interaction_meta.get("llm_options_applied", {}),
                "request_kwargs": interaction_meta.get("request_kwargs", {}),
                "retry_attempts": act_result.get("retry_attempts", []),
                "balrog_raw": {
                    "messages": serialized_messages,
                    "response": {
                        "raw_completion": response_info.get("raw_completion", raw_output),
                        "reasoning": response_info.get("reasoning", thought),
                        "parsed_action": response_info.get("parsed_action", raw_action_text),
                        "stop_reason": response_info.get("stop_reason", ""),
                        "input_tokens": int(response_info.get("input_tokens", input_tokens) or 0),
                        "output_tokens": int(response_info.get("output_tokens", output_tokens) or 0),
                    },
                    "meta": {
                        "source": interaction_meta.get("source", "agent.act"),
                        "llm_options_applied": interaction_meta.get("llm_options_applied", {}),
                        "request_kwargs": interaction_meta.get("request_kwargs", {}),
                    },
                },
            },
        }

        csv_row = [
            step_index,
            instruction_prompt,
            observation_raw,
            thought,
            raw_action_text,
            executed_action_text,
            raw_output,
            feedback,
            observation_post_raw,
            obs_changed,
            reward,
            current_progression,
            terminated,
            truncated,
            done,
            termination_reason,
            action_defaulted,
            input_tokens,
            output_tokens,
            step_wall_time_sec,
            won,
            lost,
        ]

        return {
            "ok": True,
            "trace_call": trace_call,
            "csv_row": csv_row,
            "reward": float(reward),
            "done": bool(done),
            "executed_action": executed_action_text,
            "obs": obs_after,
            "info": info,
            "stats": current_stats,
        }

    def _build_alt_episode_log(
        self,
        *,
        artifact: TrajectoryArtifact,
        trace_calls: list[dict[str, Any]],
        env_stats: dict[str, Any],
        done: bool,
        episode_return: float,
        failed_candidates: list[Any],
        rollout_capped: bool,
        base_action: str,
        alt_action: str,
        alt_mode: str,
        divergence_step: int,
        alt_traj_id: str,
        state_key: str,
        rollout_step_cap: int | None,
    ) -> dict[str, Any]:
        input_tokens = sum(_trace_call_token_usage(call)["input"] for call in trace_calls)
        output_tokens = sum(_trace_call_token_usage(call)["output"] for call in trace_calls)
        request_count = sum(_as_int(call.get("request_count"), default=0) for call in trace_calls)
        action_frequency: dict[str, int] = defaultdict(int)
        duration_seconds = 0.0
        for call in trace_calls:
            action_frequency[str(call.get("action") or "")] += 1
            duration_seconds += _as_float(call.get("step_wall_time_sec"), default=0.0)

        duration_seconds = max(duration_seconds, 1e-9)
        duration_minutes = duration_seconds / 60.0

        payload = {
            "task": artifact.task_id,
            "action_frequency": dict(action_frequency),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "request_count": int(request_count),
            "seed_mode": artifact.episode_log.get("seed_mode"),
            "base_seed": artifact.episode_log.get("base_seed"),
            "done": bool(done),
            "episode_return": float(episode_return),
            "num_steps": len(trace_calls),
            "failed_candidates": failed_candidates,
            "process_num": artifact.episode_log.get("process_num"),
            "seed": artifact.seed,
            "agent": copy.deepcopy(artifact.episode_log.get("agent", {})),
            "client": copy.deepcopy(artifact.episode_log.get("client", {})),
            "duration_seconds": duration_seconds,
            "rpm": request_count / duration_minutes,
            "tpm": (input_tokens + output_tokens) / duration_minutes,
            "input_tpm": input_tokens / duration_minutes,
            "output_tpm": output_tokens / duration_minutes,
            "collection_method": "divergence_tree",
            "dtc": {
                "base_traj_id": artifact.episode_id,
                "alt_traj_id": alt_traj_id,
                "divergence_step": divergence_step,
                "base_action_text": base_action,
                "alt_action_text": alt_action,
                "alt_mode": alt_mode,
                "state_key": state_key,
                "rollout_capped": bool(rollout_capped),
                "rollout_step_cap": rollout_step_cap,
                "rollout_extra_steps": int(self.rollout_extra_steps),
            },
        }
        if isinstance(env_stats, dict):
            payload.update(copy.deepcopy(env_stats))
        return payload

    def _build_outcome_summary(
        self,
        episode_log: dict[str, Any],
        trace_payload: dict[str, Any],
    ) -> dict[str, Any]:
        return _build_outcome_summary_from_logs(episode_log, trace_payload)

    def _build_compatibility_report(
        self,
        *,
        base_artifact: TrajectoryArtifact | None,
        successful_branch_record: dict[str, Any] | None,
    ) -> dict[str, Any]:
        report = {
            "compatible": False,
            "checked": False,
            "intentional_deviations": [
                "Alternative filename stems append `__dtc_dXX_aYY` to preserve the base family while avoiding collisions.",
                "In `alt_mode=random`, the divergence-step trace row is synthetic and uses `request_count=0` with DTC-specific `balrog_raw` metadata.",
                "In `alt_mode=random_teacher`, the divergence-step executed action is sampled first, then an auxiliary fixed-action teacher request may fill the `Thought:` text before the branch is written.",
                "In `alt_mode=request`, the divergence-step trace row comes from an actual LLM request made only for that divergence state with a different-action constraint.",
                "DTC-specific branch metadata is written under `_dtc/` instead of overloading base `.csv/.json/_llm_trace.json` meanings.",
            ],
        }
        if base_artifact is None or successful_branch_record is None:
            report["reason"] = "No successful generated branch was available for compatibility comparison."
            return report

        base_csv_header = list(base_artifact.csv_rows[0].keys()) if base_artifact.csv_rows else []
        generated_csv_path = Path(successful_branch_record["output_csv"])
        generated_json_path = Path(successful_branch_record["output_json"])
        generated_trace_path = Path(successful_branch_record["output_trace"])

        if not generated_csv_path.exists() or not generated_json_path.exists() or not generated_trace_path.exists():
            report["reason"] = "Generated branch artifact family is incomplete."
            return report

        with generated_csv_path.open("r", encoding="utf-8") as handle:
            generated_header = next(csv.reader(handle))
        generated_json = _load_json(generated_json_path)
        generated_trace = _load_json(generated_trace_path)

        base_json_keys = sorted(base_artifact.episode_log.keys())
        generated_json_keys = sorted(generated_json.keys())
        base_trace_keys = sorted(base_artifact.trace_payload.keys())
        generated_trace_keys = sorted(generated_trace.keys())
        base_call_keys = sorted((base_artifact.trace_calls[0] if base_artifact.trace_calls else {}).keys())
        generated_call_keys = sorted((generated_trace.get("calls") or [{}])[0].keys())

        report.update(
            {
                "checked": True,
                "base_example": str(base_artifact.trace_path),
                "generated_example": str(generated_trace_path),
                "required_files_present": True,
                "csv_header_match": generated_header == CSV_COLUMNS and base_csv_header == CSV_COLUMNS,
                "base_json_keys": base_json_keys,
                "generated_json_keys": generated_json_keys,
                "base_trace_keys": base_trace_keys,
                "generated_trace_keys": generated_trace_keys,
                "base_call_keys": base_call_keys,
                "generated_call_keys": generated_call_keys,
                "json_extra_keys": sorted(set(generated_json_keys) - set(base_json_keys)),
                "json_missing_keys": sorted(set(base_json_keys) - set(generated_json_keys)),
                "trace_extra_keys": sorted(set(generated_trace_keys) - set(base_trace_keys)),
                "trace_missing_keys": sorted(set(base_trace_keys) - set(generated_trace_keys)),
                "call_extra_keys": sorted(set(generated_call_keys) - set(base_call_keys)),
                "call_missing_keys": sorted(set(base_call_keys) - set(generated_call_keys)),
            }
        )
        report["compatible"] = (
            report["required_files_present"]
            and report["csv_header_match"]
            and not report["trace_missing_keys"]
            and not report["call_missing_keys"]
        )
        return report

    @staticmethod
    def _write_csv(path: Path, rows: list[list[Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, escapechar="˘", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(rows)


class _BranchCollectorRuntime:
    _collect_request_divergence = DivergenceTreeCollector._collect_request_divergence
    _collect_single_branch = DivergenceTreeCollector._collect_single_branch
    _replay_base_prefix = DivergenceTreeCollector._replay_base_prefix
    _execute_branch_step = DivergenceTreeCollector._execute_branch_step
    _execute_agent_step = DivergenceTreeCollector._execute_agent_step
    _build_alt_episode_log = DivergenceTreeCollector._build_alt_episode_log
    _build_outcome_summary = DivergenceTreeCollector._build_outcome_summary
    _build_request_alt_unavailable_record = DivergenceTreeCollector._build_request_alt_unavailable_record
    _request_alt_plans_for_divergence = DivergenceTreeCollector._request_alt_plans_for_divergence
    _write_csv = staticmethod(DivergenceTreeCollector._write_csv)

    def __init__(
        self,
        *,
        runtime_config,
        alt_budget: int,
        rollout_max_steps: int | None,
        rollout_extra_steps: int,
        exclude_alt_actions: Sequence[str] | None = None,
    ) -> None:
        self.runtime_config = runtime_config
        self.alt_budget = alt_budget
        self.rollout_max_steps = rollout_max_steps
        self.rollout_extra_steps = rollout_extra_steps
        self.exclude_alt_actions = _normalize_excluded_alt_actions(exclude_alt_actions)


@lru_cache(maxsize=32)
def _load_cached_runtime_config(config_path: str):
    return OmegaConf.load(config_path)


@lru_cache(maxsize=256)
def _load_cached_base_artifact(base_run_dir: str, trace_path: str) -> TrajectoryArtifact:
    artifact = _load_artifact_from_paths(
        base_run_dir=Path(base_run_dir),
        trace_path=Path(trace_path),
        require_success=False,
    )
    if artifact is None:
        raise RuntimeError(f"Failed to load base artifact for DTC branch job: {trace_path}")
    return artifact


def _portable_branch_record(record: dict[str, Any], output_root: Path) -> dict[str, Any]:
    portable = dict(record)
    for field in ("output_csv", "output_json", "output_trace"):
        value = str(portable.get(field) or "").strip()
        if not value:
            continue
        try:
            portable[field] = str(Path(value).resolve().relative_to(output_root.resolve()))
        except ValueError:
            portable[field] = value
    return portable


def _run_branch_job(job: BranchJob) -> dict[str, Any]:
    artifact = _load_cached_base_artifact(
        str(job.base_run_dir),
        str(job.trace_path),
    )
    runtime_config = _load_cached_runtime_config(str(job.config_path))
    config = _build_runtime_config_for_artifact(
        runtime_config,
        artifact,
        client_max_tokens_override=job.client_max_tokens_override,
        client_timeout_override=job.client_timeout_override,
        client_max_retries_override=job.client_max_retries_override,
    )
    output_dir = job.output_root / job.relative_dir
    runner = _BranchCollectorRuntime(
        runtime_config=runtime_config,
        alt_budget=job.alt_budget,
        rollout_max_steps=job.rollout_max_steps,
        rollout_extra_steps=job.rollout_extra_steps,
    )
    record = runner._collect_single_branch(
        artifact=artifact,
        config=config,
        divergence_step=job.divergence_step,
        alt_mode=job.alt_mode,
        alt_plan=job.alt_plan,
        alt_index=job.alt_index,
        alt_budget_used=job.alt_budget_used,
        output_dir=output_dir,
    )

    return _portable_branch_record(record, job.output_root)


def _run_request_divergence_job(job: RequestDivergenceJob) -> list[dict[str, Any]]:
    artifact = _load_cached_base_artifact(
        str(job.base_run_dir),
        str(job.trace_path),
    )
    runtime_config = _load_cached_runtime_config(str(job.config_path))
    config = _build_runtime_config_for_artifact(
        runtime_config,
        artifact,
        client_max_tokens_override=job.client_max_tokens_override,
        client_timeout_override=job.client_timeout_override,
        client_max_retries_override=job.client_max_retries_override,
    )
    output_dir = job.output_root / job.relative_dir
    runner = _BranchCollectorRuntime(
        runtime_config=runtime_config,
        alt_budget=job.target_alt_count,
        rollout_max_steps=job.rollout_max_steps,
        rollout_extra_steps=job.rollout_extra_steps,
        exclude_alt_actions=job.exclude_alt_actions,
    )
    records = runner._collect_request_divergence(
        artifact=artifact,
        config=config,
        divergence_step=job.divergence_step,
        target_alt_count=job.target_alt_count,
        existing_records=job.existing_records,
        output_root=job.output_root,
        output_dir=output_dir,
    )

    return [_portable_branch_record(record, job.output_root) for record in records]


def _run_collection_job(job: BranchJob | RequestDivergenceJob) -> list[dict[str, Any]]:
    try:
        if isinstance(job, RequestDivergenceJob):
            return _run_request_divergence_job(job)
        return [_run_branch_job(job)]
    except Exception as exc:
        return _build_worker_exception_records(job, exc)


def _run_collection_job_enveloped(
    job: BranchJob | RequestDivergenceJob,
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    return _collection_job_key(job), _run_collection_job(job)


def _build_worker_exception_records(
    job: BranchJob | RequestDivergenceJob,
    exc: Exception,
) -> list[dict[str, Any]]:
    worker_traceback = traceback.format_exc()
    worker_error = f"{type(exc).__name__}: {exc}"
    logger.exception(
        "DTC worker job failed: task=%s base_traj=%s divergence_step=%s",
        getattr(job, "task_id", ""),
        getattr(job, "base_traj_id", ""),
        getattr(job, "divergence_step", -1),
    )

    artifact: TrajectoryArtifact | None = None
    try:
        artifact = _load_cached_base_artifact(
            str(job.base_run_dir),
            str(job.trace_path),
        )
    except Exception:
        logger.exception(
            "Failed to reload base artifact while converting worker exception to fallback record: %s",
            getattr(job, "trace_path", ""),
        )

    task_id = str(getattr(job, "task_id", "") or "")
    seed = -1
    base_action = ""
    state_key = ""
    base_outcome = {
        "episode_return": 0.0,
        "progression": 0.0,
        "success": False,
        "won": False,
        "lost": False,
        "total_steps": 0,
    }
    if artifact is not None:
        task_id = artifact.task_id
        seed = int(artifact.seed)
        if 0 <= int(job.divergence_step) < len(artifact.executed_actions):
            base_action = artifact.executed_actions[int(job.divergence_step)]
        state_key = _build_state_key(
            artifact.task_id,
            artifact.seed,
            artifact.executed_actions[: int(job.divergence_step)],
        )
        base_outcome = _build_outcome_summary_from_logs(artifact.episode_log, artifact.trace_payload)

    common = {
        "status": "worker_exception",
        "task_id": task_id,
        "seed": seed,
        "base_traj_id": str(getattr(job, "base_traj_id", "") or ""),
        "divergence_step": int(getattr(job, "divergence_step", -1)),
        "base_action_text": base_action,
        "base_outcome": base_outcome,
        "state_key": state_key,
        "replay_success": False,
        "replay_error": worker_error,
        "worker_error": worker_error,
        "worker_traceback": worker_traceback,
    }

    if isinstance(job, BranchJob):
        alt_plan = job.alt_plan
        return [
            {
                **common,
                "alt_traj_id": _build_alt_traj_id(job.base_traj_id, job.divergence_step, job.alt_index),
                "alt_action_text": str(alt_plan.executed_action_text or ""),
                "alt_raw_action_text": str(alt_plan.raw_action_text or ""),
                "alt_action_was_fallback": bool(alt_plan.action_defaulted),
                "alt_request_count": int(alt_plan.request_count),
                "alt_thought": str(alt_plan.thought or ""),
                "alt_raw_output": str(alt_plan.raw_output or alt_plan.raw_action_text or ""),
                "alt_mode": str(job.alt_mode),
                "alt_outcome": None,
                "branch_generation_attempts": int(alt_plan.selection_attempts),
                "alt_budget_used": int(job.alt_budget_used),
            }
        ]

    existing_records_by_index: dict[int, dict[str, Any]] = {}
    for record in job.existing_records:
        alt_index = _parse_alt_index(str(record.get("alt_traj_id") or ""))
        if alt_index is None or alt_index < 0:
            continue
        existing_records_by_index[int(alt_index)] = record
    missing_indices = [
        alt_index
        for alt_index in range(int(job.target_alt_count))
        if alt_index not in existing_records_by_index
    ]
    return [
        {
            **common,
            "alt_traj_id": _build_alt_traj_id(job.base_traj_id, job.divergence_step, alt_index),
            "alt_action_text": "",
            "alt_raw_action_text": "",
            "alt_action_was_fallback": False,
            "alt_request_count": 0,
            "alt_thought": "",
            "alt_raw_output": "",
            "alt_mode": str(job.alt_mode),
            "alt_outcome": None,
            "branch_generation_attempts": 0,
            "alt_budget_used": int(job.target_alt_count),
        }
        for alt_index in missing_indices
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect BALROG divergence-tree trajectories")
    parser.add_argument("--base-run-dir", required=True, help="Existing BALROG base run directory")
    parser.add_argument(
        "--divergence-count",
        type=int,
        required=True,
        help="Selected divergence steps per base trajectory",
    )
    parser.add_argument(
        "--alt-budget",
        type=int,
        required=True,
        help="Alternative branch attempts per divergence step",
    )
    parser.add_argument(
        "--alt-mode",
        choices=(ALT_MODE_RANDOM, ALT_MODE_REQUEST, ALT_MODE_RANDOM_TEACHER),
        required=True,
        help="How to choose alternative actions at divergence steps.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of branch worker processes to use within a single DTC run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous DTC run from the existing _dtc/branch_index.jsonl under output_root.",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Restrict collection to successful base trajectories only. Default: use all complete trajectories.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a deterministic smoke subset by selecting at most one branchable trajectory per task.",
    )
    parser.add_argument(
        "--step-sampling-mode",
        choices=("uniform", "random"),
        required=True,
        help="How to choose divergence steps within each trajectory.",
    )
    parser.add_argument(
        "--task-filter",
        default=None,
        help="Optional exact task filter. Without this, tasks are processed separately and never mixed.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help=(
            "Optional config override. By default DTC uses <base_run_dir>/resolved_config.yaml "
            f"when present, otherwise falls back to {DEFAULT_BALROG_CONFIG_PATH}"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional explicit divergence-tree output root",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Optional collection log path outside the portable DTC artifact directory.",
    )
    parser.add_argument(
        "--rollout-max-steps",
        type=int,
        default=None,
        help="Optional absolute hard cap for total alternative trajectory steps",
    )
    parser.add_argument(
        "--rollout-extra-steps",
        type=int,
        default=10,
        help="Additional total steps allowed beyond the base trajectory length for each alternative rollout",
    )
    parser.add_argument(
        "--client-max-tokens-override",
        type=int,
        default=None,
        help=(
            "Optional client.generate_kwargs.max_tokens override applied at DTC runtime "
            "without mutating the base run artifacts."
        ),
    )
    parser.add_argument(
        "--client-timeout-override",
        type=float,
        default=None,
        help="Optional client timeout override applied at DTC runtime.",
    )
    parser.add_argument(
        "--client-max-retries-override",
        type=int,
        default=None,
        help="Optional client max_retries override applied at DTC runtime.",
    )
    parser.add_argument(
        "--external-retry-attempts",
        type=int,
        default=0,
        help="Additional per-job retries for retryable external worker failures such as API disconnects.",
    )
    parser.add_argument(
        "--exclude-alt-actions",
        nargs="*",
        default=(),
        help="Action strings to remove from divergence alternative-action candidates.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    collector = DivergenceTreeCollector(
        base_run_dir=Path(args.base_run_dir),
        divergence_count=args.divergence_count,
        alt_budget=args.alt_budget,
        alt_mode=args.alt_mode,
        num_workers=args.num_workers,
        resume=args.resume,
        success_only=args.success_only,
        smoke_test=args.smoke_test,
        step_sampling_mode=args.step_sampling_mode,
        task_filter=args.task_filter,
        config_path_override=Path(args.config_path).resolve() if args.config_path else None,
        output_root=Path(args.output_root).resolve() if args.output_root else None,
        rollout_max_steps=args.rollout_max_steps,
        rollout_extra_steps=args.rollout_extra_steps,
        client_max_tokens_override=args.client_max_tokens_override,
        client_timeout_override=args.client_timeout_override,
        client_max_retries_override=args.client_max_retries_override,
        external_retry_attempts=args.external_retry_attempts,
        exclude_alt_actions=args.exclude_alt_actions,
    )
    log_path = Path(args.log_path) if args.log_path else None
    failure_message = None
    summary = None
    with _redirect_collection_output(log_path):
        logger.info("Starting DTC collection")
        logger.info("DTC output_root: %s", collector.output_root)
        if log_path is not None:
            logger.info("DTC log_path: %s", log_path)
        try:
            summary = collector.collect_divergence_tree()
        except Exception:
            logger.exception("DTC collection failed")
            failure_message = (
                f"DTC collection failed. See {log_path}"
                if log_path is not None
                else "DTC collection failed."
            )
    if failure_message is not None:
        raise SystemExit(failure_message)
    if summary is None:
        message = (
            f"DTC collection did not produce a summary. See {log_path}"
            if log_path is not None
            else "DTC collection did not produce a summary."
        )
        raise SystemExit(message)
    selected, generated, failed = _summary_branch_totals(summary)
    if selected > 0 and generated == 0:
        message = (
            "DTC collection selected base trajectories but generated no branches "
            f"(selected_bases={selected}, failed_branches={failed})."
        )
        if log_path is not None:
            message += f" See {log_path}"
        raise SystemExit(message)
    for line in _build_terminal_summary_lines(summary):
        print(line, flush=True)


if __name__ == "__main__":
    main()
