#!/usr/bin/env python3
"""Collect WebShop divergence-tree trajectories from a base WebShop run."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from concurrent.futures import ProcessPoolExecutor
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddo.evaluation.webshop.common import (
    ALT_MODE_RANDOM,
    ALT_MODE_RANDOM_TEACHER,
    ALT_MODE_REQUEST,
    ALT_MODE_STRUCTURED,
    CSV_COLUMNS,
    SUPPORTED_ALT_MODES,
    WEBSHOP_INVALID_ACTION_RETRY_NOTICE,
    WEBSHOP_SUCCESS_THRESHOLD,
    WEBSHOP_SYSTEM_PROMPT,
    WebShopArtifact,
    action_type_signature,
    available_actions,
    create_agent,
    default_run_id,
    deterministic_alt_actions,
    discover_artifacts,
    instruction_text,
    make_prompt_obs,
    make_webshop_env,
    normalize_action,
    normalize_observation,
    purchase_signature_from_snapshot,
    reset_env,
    resolve_episode_seed,
    session_snapshot,
    shard_items,
    state_key,
    structured_alt_actions,
    step_success,
    step_env,
    trace_success,
    validate_action_for_env,
    write_json,
    write_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
from ddo.evaluation.action import run_agent_action


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect WebShop divergence-tree trajectories.")
    parser.add_argument("--base-run-dir", required=True, help="Base WebShop direct run directory")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--task-id", default="webshop")
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--success-threshold", type=float, default=WEBSHOP_SUCCESS_THRESHOLD)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument(
        "--parallel-workers",
        action="store_true",
        help="Run all selected base artifacts in a local process pool.",
    )
    parser.add_argument("--divergence-count", type=int, default=5)
    parser.add_argument("--step-sampling-mode", choices=("uniform", "random"), default="uniform")
    parser.add_argument("--alt-mode", choices=SUPPORTED_ALT_MODES, default=ALT_MODE_RANDOM)
    parser.add_argument("--alt-budget", type=int, default=3)
    parser.add_argument("--rollout-max-steps", type=int, default=30)
    parser.add_argument("--max-families", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260305)
    parser.add_argument("--seed-mode", choices=("fixed", "per_episode"), default="per_episode")
    parser.add_argument("--num-products", type=int, default=1000)
    parser.add_argument("--human-goals", type=int, default=0)
    parser.add_argument("--show-attrs", action="store_true")
    parser.add_argument("--file-path", default=None)
    parser.add_argument("--max-search-queries", type=int, default=6)
    parser.add_argument("--client-name", default="vllm")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--client-timeout", type=float, default=60.0)
    parser.add_argument("--client-max-retries", type=int, default=1)
    parser.add_argument("--client-delay", type=float, default=1.0)
    parser.add_argument("--llm-seed", type=int, default=None)
    parser.add_argument("--vllm-disable-thinking", action="store_true")
    parser.add_argument("--max-text-history", type=int, default=16)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, escapechar="\\", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def _relative_dir(base_run_dir: Path, trace_path: Path) -> Path:
    try:
        return trace_path.parent.relative_to(base_run_dir)
    except ValueError:
        return Path("webshop")



def _selected_steps(artifact: WebShopArtifact, *, divergence_count: int, sampling_mode: str) -> list[int]:
    candidate_steps = [
        idx
        for idx, call in enumerate(artifact.calls)
        if normalize_action(str(call.get("action") or ""))
    ]
    if not candidate_steps:
        return []
    if divergence_count <= 0 or divergence_count >= len(candidate_steps):
        return candidate_steps
    if sampling_mode == "random":
        rng = random.Random(artifact.seed + artifact.episode_idx * 1009)
        return sorted(rng.sample(candidate_steps, k=int(divergence_count)))
    if sampling_mode != "uniform":
        raise ValueError(f"Unsupported step_sampling_mode: {sampling_mode}")
    if divergence_count == 1:
        return [candidate_steps[len(candidate_steps) // 2]]
    positions = [
        round(idx * (len(candidate_steps) - 1) / (divergence_count - 1))
        for idx in range(divergence_count)
    ]
    return sorted({candidate_steps[int(position)] for position in positions})


def _make_env(args: argparse.Namespace):
    return make_webshop_env(
        num_products=args.num_products,
        human_goals=args.human_goals,
        show_attrs=args.show_attrs,
        file_path=args.file_path,
    )


def _replay_prefix(
    args: argparse.Namespace,
    artifact: WebShopArtifact,
    prefix_actions: list[str],
) -> tuple[Any, str, str, bool, str]:
    random.seed(int(artifact.seed))
    np.random.seed(int(artifact.seed))
    env = _make_env(args)
    observation = reset_env(env, artifact.session_id)
    instruction = instruction_text(env)
    done = False
    for action in prefix_actions:
        if done:
            return env, observation, instruction, False, "prefix_reached_terminal"
        observation, _reward, done, _info = step_env(env, action)
    return env, observation, instruction, True, ""


def _seed_for_branch(args: argparse.Namespace, artifact: WebShopArtifact, alt_index: int) -> int:
    seed = resolve_episode_seed(
        seed_mode=args.seed_mode,
        base_seed=args.seed,
        episode_idx=artifact.episode_idx,
    )
    return int(seed) + int(alt_index)


def _csv_row_from_call(call: dict[str, Any]) -> list[object]:
    tokens = call.get("token_usage") if isinstance(call.get("token_usage"), dict) else {}
    return [
        call.get("episode_step", call.get("call_idx", 0)),
        call.get("instruction", ""),
        call.get("observation", ""),
        call.get("thought", ""),
        call.get("action_model", call.get("action", "")),
        call.get("action_executed", call.get("action", "")),
        call.get("raw_output", ""),
        call.get("feedback", ""),
        call.get("observation_post", ""),
        bool(call.get("obs_changed", False)),
        float(call.get("reward", 0.0) or 0.0),
        float(call.get("progression", call.get("reward", 0.0)) or 0.0),
        bool(call.get("terminated", call.get("done", False))),
        bool(call.get("truncated", False)),
        bool(call.get("done", False)),
        call.get("termination_reason", ""),
        bool(call.get("action_defaulted", False)),
        int(tokens.get("input", call.get("input_tokens", 0)) or 0),
        int(tokens.get("output", call.get("output_tokens", 0)) or 0),
        float(call.get("step_wall_time_sec", 0.0) or 0.0),
        bool(call.get("won", False)),
        bool(call.get("lost", False)),
    ]


def _build_call(
    *,
    step: int,
    instruction: str,
    observation_pre: str,
    thought: str,
    model_action: str,
    executed_action: str,
    raw_output: str,
    observation_post: str,
    reward: float,
    done: bool,
    termination_reason: str,
    action_defaulted: bool,
    input_tokens: int,
    output_tokens: int,
    step_wall_time_sec: float,
    request_count: int,
    retry_attempts: list[dict[str, Any]],
    snapshot: dict[str, Any],
    available: list[str],
    balrog_raw: dict[str, Any] | None = None,
    success_threshold: float = WEBSHOP_SUCCESS_THRESHOLD,
) -> dict[str, Any]:
    progression = float(reward)
    won = bool(
        done
        and step_success(
            reward=reward,
            progression=progression,
            success_threshold=success_threshold,
        )
    )
    lost = bool(done and not won)
    extras: dict[str, Any] = {
        "webshop": {
            "available_actions": list(available),
            "session_snapshot": snapshot,
            "purchase_signature": purchase_signature_from_snapshot(snapshot),
        },
        "retry_attempts": retry_attempts,
    }
    if balrog_raw:
        extras["balrog_raw"] = balrog_raw
    return {
        "call_idx": step,
        "global_step": step,
        "episode_step": step,
        "instruction": instruction,
        "observation": observation_pre,
        "thought": thought,
        "action": executed_action,
        "action_model": model_action,
        "action_executed": executed_action,
        "raw_output": raw_output,
        "feedback": "",
        "observation_post": observation_post,
        "obs_changed": observation_pre != observation_post,
        "reward": float(reward),
        "progression": float(reward),
        "terminated": bool(done),
        "truncated": False,
        "done": bool(done),
        "termination_reason": termination_reason,
        "action_defaulted": bool(action_defaulted),
        "won": won,
        "lost": lost,
        "request_count": int(request_count),
        "step_wall_time_sec": float(step_wall_time_sec),
        "token_usage": {
            "input": int(input_tokens),
            "output": int(output_tokens),
            "total": int(input_tokens) + int(output_tokens),
        },
        "extras": extras,
    }


def _base_prompt_builder_from_prefix(args: argparse.Namespace, base_calls: list[dict[str, Any]], divergence_step: int):
    agent = create_agent(args)
    agent.reset()
    agent.clear_llm_interactions()
    agent.prompt_builder.update_instruction_prompt(WEBSHOP_SYSTEM_PROMPT)
    for call in base_calls[:divergence_step]:
        agent.prompt_builder.update_observation(make_prompt_obs(str(call.get("observation") or "")))
        agent.prompt_builder.update_action(normalize_action(str(call.get("action") or "")))
    return agent


def _validate_from_pool(pool: list[str]) -> Callable[[str], str]:
    allowed = set(pool)

    def _validate(action: str) -> str:
        normalized = normalize_action(action)
        return normalized if normalized in allowed else ""

    return _validate


def _request_alt_actions(
    args: argparse.Namespace,
    *,
    artifact: WebShopArtifact,
    divergence_step: int,
    observation: str,
    base_action: str,
    candidate_pool: list[str],
) -> list[str]:
    selected: list[str] = []
    for _attempt in range(max(1, int(args.alt_budget))):
        pool = [action for action in candidate_pool if action not in selected and action != base_action]
        if not pool:
            break
        agent = _base_prompt_builder_from_prefix(args, artifact.calls, divergence_step)
        extra_user_text = (
            "Choose a valid alternative action for this exact WebShop state. "
            f"Do not choose the base action `{base_action}`.\n"
            "Candidate alternative actions:\n"
            + "\n".join(f"- {action}" for action in pool)
        )
        result = run_agent_action(
            agent=agent,
            obs=make_prompt_obs(observation, pool),
            prev_action=None,
            validate_action=_validate_from_pool(pool),
            extra_user_text=extra_user_text,
            invalid_action_retry_notice=WEBSHOP_INVALID_ACTION_RETRY_NOTICE,
        )
        action = normalize_action(str(result.get("executed_action") or ""))
        if action and action in pool and action not in selected:
            selected.append(action)
    return selected


def _branch_trace_paths(output_root: Path, rel_dir: Path, base_episode_id: str, divergence_step: int, alt_index: int):
    alt_id = f"{base_episode_id}__dtc_d{int(divergence_step):02d}_a{int(alt_index):02d}"
    task_dir = output_root / rel_dir
    return (
        alt_id,
        task_dir / f"{alt_id}.csv",
        task_dir / f"{alt_id}.json",
        task_dir / f"{alt_id}_llm_trace.json",
    )


def _rollout_branch(
    args: argparse.Namespace,
    *,
    artifact: WebShopArtifact,
    rel_dir: Path,
    output_root: Path,
    divergence_step: int,
    alt_index: int,
    alt_action: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = _seed_for_branch(args, artifact, alt_index)
    random.seed(seed)
    np.random.seed(seed)

    prefix_actions = [
        normalize_action(str(call.get("action") or ""))
        for call in artifact.calls[:divergence_step]
    ]
    env, observation, instruction, replay_ok, replay_error = _replay_prefix(args, artifact, prefix_actions)
    expected_observation = str(artifact.calls[divergence_step].get("observation") or "")
    if not replay_ok or normalize_observation(observation) != normalize_observation(expected_observation):
        return (
            {
                "status": "replay_failed",
                "replay_success": False,
                "replay_error": replay_error or "observation_mismatch",
                "expected_observation": expected_observation,
                "actual_observation": str(observation),
            },
            {},
        )

    alt_id, csv_path, json_path, trace_path = _branch_trace_paths(
        output_root,
        rel_dir,
        artifact.episode_id,
        divergence_step,
        alt_index,
    )
    trace_calls: list[dict[str, Any]] = [copy.deepcopy(call) for call in artifact.calls[:divergence_step]]
    observation_pre = str(observation)
    available = available_actions(env, instruction, max_search_queries=args.max_search_queries)
    teacher_result: dict[str, Any] | None = None
    if args.alt_mode == ALT_MODE_RANDOM_TEACHER:
        teacher_agent = _base_prompt_builder_from_prefix(args, artifact.calls, divergence_step)
        teacher_result = run_agent_action(
            agent=teacher_agent,
            obs=make_prompt_obs(observation_pre, [alt_action]),
            prev_action=None,
            validate_action=_validate_from_pool([alt_action]),
            extra_user_text=(
                "The alternative executable action is already fixed. "
                f"Explain it briefly and output exactly this action: {alt_action}"
            ),
            invalid_action_retry_notice=WEBSHOP_INVALID_ACTION_RETRY_NOTICE,
        )
    step_started = time.perf_counter()
    observation, reward, done, _info = step_env(env, alt_action)
    snapshot = session_snapshot(env)
    if teacher_result is not None:
        thought = str(teacher_result.get("thought") or "")
        model_action = str(teacher_result.get("model_action") or "")
        raw_output = str(teacher_result.get("raw_output") or f"Action: {alt_action}")
        request_count = int(teacher_result.get("request_count") or 0)
        input_tokens = int(teacher_result.get("input_tokens") or 0)
        output_tokens = int(teacher_result.get("output_tokens") or 0)
        interaction = (
            teacher_result.get("interaction")
            if isinstance(teacher_result.get("interaction"), dict)
            else None
        )
        action_defaulted = normalize_action(str(teacher_result.get("executed_action") or "")) != alt_action
    else:
        thought = (
            "Random alternative action."
            if args.alt_mode == ALT_MODE_RANDOM
            else "Structured alternative action."
            if args.alt_mode == ALT_MODE_STRUCTURED
            else "Requested alternative action."
        )
        model_action = alt_action
        raw_output = f"Action: {alt_action}"
        request_count = 0
        input_tokens = 0
        output_tokens = 0
        interaction = None
        action_defaulted = False
    trace_calls.append(
        _build_call(
            step=divergence_step,
            instruction=instruction,
            observation_pre=observation_pre,
            thought=thought,
            model_action=model_action,
            executed_action=alt_action,
            raw_output=raw_output,
            observation_post=str(observation),
            reward=reward,
            done=done,
            termination_reason=(
                "success"
                if done
                and step_success(
                    reward=reward,
                    progression=reward,
                    success_threshold=args.success_threshold,
                )
                else "terminated" if done else ""
            ),
            action_defaulted=action_defaulted,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            step_wall_time_sec=time.perf_counter() - step_started,
            request_count=request_count,
            retry_attempts=[],
            snapshot=snapshot,
            available=available,
            balrog_raw=interaction or {
                "messages": [
                    {"role": "user", "content": WEBSHOP_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Current Observation:\n{observation_pre}",
                    },
                ],
                "response": {
                    "raw_completion": raw_output,
                    "reasoning": "",
                    "parsed_action": alt_action,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            },
            success_threshold=args.success_threshold,
        )
    )

    agent = _base_prompt_builder_from_prefix(args, artifact.calls, divergence_step)
    prev_action = alt_action
    aborted = False
    abort_reason = ""
    max_steps = max(divergence_step + 1, int(args.rollout_max_steps))
    for step in range(divergence_step + 1, max_steps):
        if done:
            break
        step_started = time.perf_counter()
        observation_pre = str(observation)
        available = available_actions(env, instruction, max_search_queries=args.max_search_queries)
        act_result = run_agent_action(
            agent=agent,
            obs=make_prompt_obs(observation_pre, available),
            prev_action=prev_action,
            validate_action=validate_action_for_env(
                env,
                instruction,
                max_search_queries=args.max_search_queries,
            ),
            invalid_action_retry_notice=WEBSHOP_INVALID_ACTION_RETRY_NOTICE,
        )
        if act_result.get("abort_reason"):
            snapshot = session_snapshot(env)
            call = _build_call(
                step=step,
                instruction=instruction,
                observation_pre=observation_pre,
                thought=str(act_result.get("thought") or ""),
                model_action=str(act_result.get("model_action") or ""),
                executed_action="",
                raw_output=str(act_result.get("raw_output") or ""),
                observation_post=observation_pre,
                reward=0.0,
                done=False,
                termination_reason="aborted",
                action_defaulted=bool(act_result.get("action_defaulted")),
                input_tokens=int(act_result.get("input_tokens") or 0),
                output_tokens=int(act_result.get("output_tokens") or 0),
                step_wall_time_sec=time.perf_counter() - step_started,
                request_count=int(act_result.get("request_count") or 0),
                retry_attempts=list(act_result.get("retry_attempts") or []),
                snapshot=snapshot,
                available=available,
                balrog_raw=act_result.get("interaction") if isinstance(act_result.get("interaction"), dict) else None,
                success_threshold=args.success_threshold,
            )
            call["extras"]["abort_reason"] = act_result.get("abort_reason")
            trace_calls.append(call)
            aborted = True
            abort_reason = str(act_result.get("abort_reason") or "")
            break

        executed_action = normalize_action(str(act_result["executed_action"]))
        observation, reward, done, _info = step_env(env, executed_action)
        snapshot = session_snapshot(env)
        trace_calls.append(
            _build_call(
                step=step,
                instruction=instruction,
                observation_pre=observation_pre,
                thought=str(act_result.get("thought") or ""),
                model_action=str(act_result.get("model_action") or ""),
                executed_action=executed_action,
                raw_output=str(act_result.get("raw_output") or ""),
                observation_post=str(observation),
                reward=reward,
                done=done,
                termination_reason=(
                    "success"
                    if done
                    and step_success(
                        reward=reward,
                        progression=reward,
                        success_threshold=args.success_threshold,
                    )
                    else "terminated" if done else ""
                ),
                action_defaulted=bool(act_result.get("action_defaulted")),
                input_tokens=int(act_result.get("input_tokens") or 0),
                output_tokens=int(act_result.get("output_tokens") or 0),
                step_wall_time_sec=time.perf_counter() - step_started,
                request_count=int(act_result.get("request_count") or 0),
                retry_attempts=list(act_result.get("retry_attempts") or []),
                snapshot=snapshot,
                available=available,
                balrog_raw=act_result.get("interaction") if isinstance(act_result.get("interaction"), dict) else None,
                success_threshold=args.success_threshold,
            )
        )
        prev_action = executed_action

    final_snapshot = session_snapshot(env)
    trace_payload = {
        "schema_version": "thought_action_v1",
        "benchmark": "webshop",
        "env_name": "webshop",
        "task_or_env_params": artifact.task_id,
        "seed": seed,
        "base_seed": args.seed,
        "seed_mode": args.seed_mode,
        "episode_idx": artifact.episode_idx,
        "episode_id": alt_id,
        "session_id": artifact.session_id,
        "trajectory_kind": "alt",
        "base_episode_id": artifact.episode_id,
        "divergence_step": int(divergence_step),
        "alt_index": int(alt_index),
        "alt_mode": args.alt_mode,
        "num_calls": len(trace_calls),
        "success_threshold": float(args.success_threshold),
        "calls": trace_calls,
        "extras": {
            "instruction": instruction,
            "session_snapshot": final_snapshot,
            "purchase_signature": purchase_signature_from_snapshot(final_snapshot),
            "action_type_signature": action_type_signature([call.get("action", "") for call in trace_calls]),
            "dtc": {
                "trajectory_kind": "alt",
                "base_episode_id": artifact.episode_id,
                "divergence_step": int(divergence_step),
                "alt_action": alt_action,
            },
        },
    }
    success = trace_success(trace_payload, args.success_threshold)
    episode_log = {
        "status": "finished",
        "benchmark": "webshop",
        "env_name": "webshop",
        "task": artifact.task_id,
        "seed": seed,
        "base_seed": args.seed,
        "seed_mode": args.seed_mode,
        "episode_idx": artifact.episode_idx,
        "episode_id": alt_id,
        "session_id": artifact.session_id,
        "base_episode_id": artifact.episode_id,
        "divergence_step": int(divergence_step),
        "alt_index": int(alt_index),
        "alt_mode": args.alt_mode,
        "success_threshold": float(args.success_threshold),
        "episode_return": float(sum(float(call.get("reward", 0.0) or 0.0) for call in trace_calls)),
        "progression": float(trace_calls[-1].get("progression", 0.0) if trace_calls else 0.0),
        "done": bool(trace_calls and trace_calls[-1].get("done")),
        "aborted": bool(aborted),
        "abort_reason": abort_reason,
        "num_steps": len(trace_calls),
        "input_tokens": sum(int(call.get("token_usage", {}).get("input", 0)) for call in trace_calls),
        "output_tokens": sum(int(call.get("token_usage", {}).get("output", 0)) for call in trace_calls),
        "request_count": sum(int(call.get("request_count", 0)) for call in trace_calls),
        "success": success,
        "purchase_signature": purchase_signature_from_snapshot(final_snapshot),
    }

    _write_csv(csv_path, [_csv_row_from_call(call) for call in trace_calls])
    write_json(json_path, episode_log)
    write_json(trace_path, trace_payload)
    return (
        {
            "status": "generated",
            "replay_success": True,
            "output_csv": str(csv_path.relative_to(output_root)),
            "output_json": str(json_path.relative_to(output_root)),
            "output_trace": str(trace_path.relative_to(output_root)),
        },
        trace_payload,
    )


def _base_outcome(artifact: WebShopArtifact) -> dict[str, Any]:
    snapshot = artifact.trace_payload.get("extras", {}).get("session_snapshot", {})
    return {
        "success": artifact.success,
        "reward": float(artifact.calls[-1].get("reward", 0.0) if artifact.calls else 0.0),
        "progression": float(artifact.calls[-1].get("progression", 0.0) if artifact.calls else 0.0),
        "purchase_signature": purchase_signature_from_snapshot(snapshot) if isinstance(snapshot, dict) else "",
    }


def _alt_outcome(trace_payload: dict[str, Any], success_threshold: float) -> dict[str, Any]:
    calls = trace_payload.get("calls") or []
    snapshot = trace_payload.get("extras", {}).get("session_snapshot", {})
    last_call = calls[-1] if calls else {}
    return {
        "success": trace_success(trace_payload, success_threshold),
        "reward": float(last_call.get("reward", 0.0) or 0.0),
        "progression": float(last_call.get("progression", 0.0) or 0.0),
        "purchase_signature": purchase_signature_from_snapshot(snapshot) if isinstance(snapshot, dict) else "",
    }


def _branch_records_for_artifact(
    args: argparse.Namespace,
    *,
    artifact: WebShopArtifact,
    base_run_dir: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    rel_dir = _relative_dir(base_run_dir, artifact.trace_path)
    branch_records: list[dict[str, Any]] = []

    for divergence_step in _selected_steps(
        artifact,
        divergence_count=args.divergence_count,
        sampling_mode=args.step_sampling_mode,
    ):
        prefix_actions = [
            normalize_action(str(call.get("action") or ""))
            for call in artifact.calls[:divergence_step]
        ]
        base_call = artifact.calls[divergence_step]
        base_action = normalize_action(str(base_call.get("action") or ""))
        env, observation, instruction, replay_ok, replay_error = _replay_prefix(args, artifact, prefix_actions)
        expected_observation = str(base_call.get("observation") or "")
        replay_success = bool(
            replay_ok and normalize_observation(observation) == normalize_observation(expected_observation)
        )
        state_hash = state_key(artifact.task_id, artifact.session_id, prefix_actions)

        base_record = {
            "task_id": artifact.task_id,
            "seed": artifact.seed,
            "session_id": artifact.session_id,
            "base_traj_id": artifact.episode_id,
            "divergence_step": int(divergence_step),
            "state_key": state_hash,
            "base_action_text": base_action,
            "alt_mode": args.alt_mode,
            "base_outcome": _base_outcome(artifact),
        }
        if not replay_success:
            branch_records.append(
                {
                    **base_record,
                    "status": "replay_failed",
                    "replay_success": False,
                    "replay_error": replay_error or "observation_mismatch",
                    "expected_observation": expected_observation,
                    "actual_observation": str(observation),
                    "alt_traj_id": "",
                    "alt_action_text": "",
                    "branch_generation_attempts": 0,
                    "alt_budget_used": 0,
                }
            )
            continue

        action_pool = available_actions(
            env,
            instruction,
            max_search_queries=args.max_search_queries,
        )
        candidate_pool = [action for action in action_pool if action != base_action]
        if args.alt_mode in {ALT_MODE_RANDOM, ALT_MODE_RANDOM_TEACHER}:
            alt_actions = deterministic_alt_actions(
                actions=action_pool,
                base_action=base_action,
                alt_budget=args.alt_budget,
                sampling_key=f"{artifact.episode_id}:{divergence_step}:{state_hash}",
            )
            branch_generation_attempts = 0
        elif args.alt_mode == ALT_MODE_STRUCTURED:
            alt_actions = structured_alt_actions(
                actions=action_pool,
                base_action=base_action,
                alt_budget=args.alt_budget,
                sampling_key=f"{artifact.episode_id}:{divergence_step}:{state_hash}",
            )
            branch_generation_attempts = 0
        elif args.alt_mode == ALT_MODE_REQUEST:
            alt_actions = _request_alt_actions(
                args,
                artifact=artifact,
                divergence_step=divergence_step,
                observation=str(observation),
                base_action=base_action,
                candidate_pool=candidate_pool,
            )
            branch_generation_attempts = len(alt_actions)
        else:
            raise ValueError(f"Unsupported alt_mode: {args.alt_mode}")

        if not alt_actions:
            branch_records.append(
                {
                    **base_record,
                    "status": "no_alt_actions",
                    "replay_success": True,
                    "alt_traj_id": "",
                    "alt_action_text": "",
                    "branch_generation_attempts": branch_generation_attempts,
                    "alt_budget_used": 0,
                    "available_actions": action_pool,
                }
            )
            continue

        for alt_index, alt_action in enumerate(alt_actions):
            branch_result, alt_payload = _rollout_branch(
                args,
                artifact=artifact,
                rel_dir=rel_dir,
                output_root=output_root,
                divergence_step=divergence_step,
                alt_index=alt_index,
                alt_action=alt_action,
            )
            alt_id = f"{artifact.episode_id}__dtc_d{int(divergence_step):02d}_a{int(alt_index):02d}"
            record = {
                **base_record,
                **branch_result,
                "alt_traj_id": alt_id if branch_result.get("status") == "generated" else "",
                "alt_action_text": alt_action,
                "branch_generation_attempts": branch_generation_attempts,
                "alt_budget_used": len(alt_actions),
                "available_actions": action_pool,
            }
            if branch_result.get("status") == "generated":
                record["alt_outcome"] = _alt_outcome(alt_payload, args.success_threshold)
            branch_records.append(record)
    return branch_records


def _branch_records_job(job):
    args, artifact, base_run_dir, output_root = job
    return _branch_records_for_artifact(
        args, artifact=artifact, base_run_dir=base_run_dir, output_root=output_root
    )


def main() -> None:
    args = _parse_args()
    if int(args.num_workers) > 1 and not args.parallel_workers and not args.run_id:
        raise ValueError("--run-id is required when --num-workers > 1")
    base_run_dir = Path(args.base_run_dir)
    if not base_run_dir.exists():
        raise FileNotFoundError(f"Missing base WebShop run directory: {base_run_dir}")

    run_id = args.run_id or default_run_id("webshop_dtc", args.model_id)
    output_root = (
        Path(args.output_root)
        if args.output_root
        else Path("data/raw/trajectories/webshop/divergence_tree") / f"{base_run_dir.name}__{run_id}"
    )
    artifacts = discover_artifacts(base_run_dir, success_threshold=args.success_threshold)
    if args.success_only:
        artifacts = [artifact for artifact in artifacts if artifact.success]
    if args.max_families > 0:
        artifacts = artifacts[: int(args.max_families)]
    total_artifact_count = len(artifacts)
    if args.parallel_workers:
        selected_artifacts = artifacts
        jobs = [
            (args, artifact, base_run_dir, output_root)
            for artifact in selected_artifacts
        ]
        max_workers = min(int(args.num_workers), len(jobs)) if jobs else 1
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            record_groups = list(executor.map(_branch_records_job, jobs))
        all_branch_records = [record for records in record_groups for record in records]
    else:
        selected_artifacts = shard_items(
            artifacts,
            num_workers=args.num_workers,
            worker_index=args.worker_index,
        )
        all_branch_records = []
        for artifact in selected_artifacts:
            all_branch_records.extend(
                _branch_records_for_artifact(
                    args,
                    artifact=artifact,
                    base_run_dir=base_run_dir,
                    output_root=output_root,
                )
            )
    for record in all_branch_records:
        print(json.dumps(record, ensure_ascii=True))

    if args.parallel_workers or int(args.num_workers) == 1:
        branch_index_path = output_root / "_dtc" / "branch_index.jsonl"
        summary_path = output_root / "_dtc" / "summary.json"
    else:
        worker_dir = output_root / "_dtc" / "_workers"
        branch_index_path = worker_dir / f"branch_index_worker_{int(args.worker_index):02d}.jsonl"
        summary_path = worker_dir / f"summary_worker_{int(args.worker_index):02d}.json"
    write_jsonl(branch_index_path, all_branch_records)
    write_json(
        summary_path,
        {
            "base_run_dir": str(base_run_dir),
            "output_root": str(output_root),
            "run_id": run_id,
            "num_workers": int(args.num_workers),
            "worker_index": int(args.worker_index),
            "success_threshold": float(args.success_threshold),
            "alt_mode": args.alt_mode,
            "alt_budget": int(args.alt_budget),
            "divergence_count": int(args.divergence_count),
            "total_artifact_count": total_artifact_count,
            "artifact_count": len(selected_artifacts),
            "branch_record_count": len(all_branch_records),
            "generated_count": sum(1 for record in all_branch_records if record.get("status") == "generated"),
            "replay_failed_count": sum(1 for record in all_branch_records if record.get("status") == "replay_failed"),
            "no_alt_actions_count": sum(1 for record in all_branch_records if record.get("status") == "no_alt_actions"),
        },
    )
    print(f"wrote WebShop DTC run: {output_root}")


if __name__ == "__main__":
    main()
