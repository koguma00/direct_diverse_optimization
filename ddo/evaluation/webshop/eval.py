#!/usr/bin/env python3
"""Collect base WebShop thought-action trajectories."""

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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddo.evaluation.webshop.common import (
    CSV_COLUMNS,
    WEBSHOP_INVALID_ACTION_RETRY_NOTICE,
    WEBSHOP_SUCCESS_THRESHOLD,
    WEBSHOP_SYSTEM_PROMPT,
    action_type_signature,
    available_actions,
    create_agent,
    default_run_id,
    episode_paths,
    instruction_text,
    make_prompt_obs,
    make_webshop_env,
    normalize_action,
    purchase_signature_from_snapshot,
    reset_env,
    resolve_episode_seed,
    session_snapshot,
    shard_bounds,
    step_success,
    step_env,
    trace_success,
    validate_action_for_env,
    write_json,
)

from ddo.evaluation.action import run_agent_action


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WebShop thought-action evaluation.")
    parser.add_argument("--output-root", default="data/raw/trajectories/webshop/direct")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--task-id", default="webshop")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=2)
    parser.add_argument("--session-start", type=int, default=0)
    parser.add_argument(
        "--session-mode",
        choices=("per_episode", "fixed"),
        default="per_episode",
        help="Use consecutive sessions or repeat one fixed session.",
    )
    parser.add_argument("--seed", type=int, default=20260305)
    parser.add_argument("--seed-mode", choices=("fixed", "per_episode"), default="per_episode")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--success-threshold", type=float, default=WEBSHOP_SUCCESS_THRESHOLD)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument(
        "--parallel-workers",
        action="store_true",
        help="Run the complete episode range in a local process pool.",
    )
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
    parser.add_argument(
        "--llm-seed-mode",
        choices=("none", "fixed", "per_episode"),
        default="none",
    )
    parser.add_argument("--vllm-disable-thinking", action="store_true")
    parser.add_argument("--max-text-history", type=int, default=16)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, escapechar="˘", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def _step_record(
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
    retry_attempts: list[dict],
    snapshot: dict,
    available: list[str],
    success_threshold: float,
    balrog_raw: dict | None = None,
) -> tuple[list[object], dict]:
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
    csv_row = [
        step,
        instruction,
        observation_pre,
        thought,
        model_action,
        executed_action,
        raw_output,
        "",
        observation_post,
        observation_pre != observation_post,
        reward,
        reward,
        done,
        False,
        done,
        termination_reason,
        action_defaulted,
        input_tokens,
        output_tokens,
        step_wall_time_sec,
        won,
        lost,
    ]
    extras = {
        "webshop": {
            "available_actions": list(available),
            "session_snapshot": snapshot,
            "purchase_signature": purchase_signature_from_snapshot(snapshot),
        },
        "retry_attempts": retry_attempts,
    }
    if balrog_raw:
        extras["balrog_raw"] = balrog_raw

    trace_call = {
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
    return csv_row, trace_call


def resolve_session_id(*, session_start: int, session_mode: str, episode_idx: int) -> int:
    if session_mode == "fixed":
        return int(session_start)
    if session_mode == "per_episode":
        return int(session_start) + int(episode_idx)
    raise ValueError(f"unsupported session mode: {session_mode}")


def resolve_llm_seed(
    *,
    llm_seed: int | None,
    llm_seed_mode: str,
    episode_idx: int,
) -> int | None:
    if llm_seed_mode == "none":
        return None
    if llm_seed is None:
        raise ValueError("--llm-seed is required unless --llm-seed-mode=none")
    if llm_seed_mode == "fixed":
        return int(llm_seed)
    if llm_seed_mode == "per_episode":
        return int(llm_seed) + int(episode_idx)
    raise ValueError(f"unsupported LLM seed mode: {llm_seed_mode}")


def run_episode(
    args: argparse.Namespace,
    *,
    run_id: str,
    episode_idx: int,
    env=None,
    agent=None,
) -> dict:
    session_id = resolve_session_id(
        session_start=args.session_start,
        session_mode=args.session_mode,
        episode_idx=episode_idx,
    )
    seed = resolve_episode_seed(seed_mode=args.seed_mode, base_seed=args.seed, episode_idx=episode_idx)
    resolved_llm_seed = resolve_llm_seed(
        llm_seed=args.llm_seed,
        llm_seed_mode=args.llm_seed_mode,
        episode_idx=episode_idx,
    )
    random.seed(seed)
    np.random.seed(seed)

    if env is None:
        env = make_webshop_env(
            num_products=args.num_products,
            human_goals=args.human_goals,
            show_attrs=args.show_attrs,
            file_path=args.file_path,
        )
    observation = reset_env(env, session_id)
    instruction = instruction_text(env)
    if agent is None:
        episode_args = copy.copy(args)
        episode_args.llm_seed = resolved_llm_seed
        agent = create_agent(episode_args)
    agent.reset()
    agent.clear_llm_interactions()
    agent.prompt_builder.update_instruction_prompt(WEBSHOP_SYSTEM_PROMPT)

    paths = episode_paths(Path(args.output_root), run_id, args.task_id, episode_idx)
    csv_rows: list[list[object]] = []
    trace_calls: list[dict] = []
    prev_action = None
    episode_return = 0.0
    episode_started = time.time()
    aborted = False
    abort_reason = ""

    for step in range(int(args.max_steps)):
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
            csv_row, trace_call = _step_record(
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
                success_threshold=args.success_threshold,
                balrog_raw=act_result.get("interaction")
                if isinstance(act_result.get("interaction"), dict)
                else None,
            )
            trace_call["extras"]["abort_reason"] = act_result.get("abort_reason")
            csv_rows.append(csv_row)
            trace_calls.append(trace_call)
            aborted = True
            abort_reason = str(act_result.get("abort_reason") or "")
            break

        executed_action = normalize_action(str(act_result["executed_action"]))
        observation, reward, done, _info = step_env(env, executed_action)
        episode_return += reward
        snapshot = session_snapshot(env)
        termination_reason = (
            "success"
            if done
            and step_success(
                reward=reward,
                progression=reward,
                success_threshold=args.success_threshold,
            )
            else "terminated" if done else ""
        )
        csv_row, trace_call = _step_record(
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
            termination_reason=termination_reason,
            action_defaulted=bool(act_result.get("action_defaulted")),
            input_tokens=int(act_result.get("input_tokens") or 0),
            output_tokens=int(act_result.get("output_tokens") or 0),
            step_wall_time_sec=time.perf_counter() - step_started,
            request_count=int(act_result.get("request_count") or 0),
            retry_attempts=list(act_result.get("retry_attempts") or []),
            snapshot=snapshot,
            available=available,
            success_threshold=args.success_threshold,
            balrog_raw=act_result.get("interaction")
            if isinstance(act_result.get("interaction"), dict)
            else None,
        )
        csv_rows.append(csv_row)
        trace_calls.append(trace_call)
        prev_action = executed_action
        if done:
            break

    final_snapshot = session_snapshot(env)
    trace_payload = {
        "schema_version": "thought_action_v1",
        "benchmark": "webshop",
        "env_name": "webshop",
        "task_or_env_params": args.task_id,
        "seed": seed,
        "base_seed": args.seed,
        "seed_mode": args.seed_mode,
        "llm_seed": resolved_llm_seed,
        "llm_seed_mode": args.llm_seed_mode,
        "episode_idx": episode_idx,
        "episode_id": paths.run_stem,
        "session_id": session_id,
        "num_calls": len(trace_calls),
        "success_threshold": float(args.success_threshold),
        "calls": trace_calls,
        "extras": {
            "instruction": instruction,
            "session_snapshot": final_snapshot,
            "purchase_signature": purchase_signature_from_snapshot(final_snapshot),
            "action_type_signature": action_type_signature([call.get("action", "") for call in trace_calls]),
        },
    }
    success = trace_success(trace_payload, args.success_threshold)
    episode_log = {
        "status": "finished",
        "benchmark": "webshop",
        "env_name": "webshop",
        "task": args.task_id,
        "seed": seed,
        "base_seed": args.seed,
        "seed_mode": args.seed_mode,
        "llm_seed": resolved_llm_seed,
        "llm_seed_mode": args.llm_seed_mode,
        "episode_idx": episode_idx,
        "episode_id": paths.run_stem,
        "session_id": session_id,
        "instruction": instruction,
        "success_threshold": float(args.success_threshold),
        "session_mode": args.session_mode,
        "llm_seed_mode": args.llm_seed_mode,
        "episode_return": float(episode_return),
        "progression": float(episode_return),
        "done": bool(trace_calls and trace_calls[-1].get("done")),
        "aborted": bool(aborted),
        "abort_reason": abort_reason,
        "num_steps": len(trace_calls),
        "duration_seconds": time.time() - episode_started,
        "input_tokens": sum(int(call.get("token_usage", {}).get("input", 0)) for call in trace_calls),
        "output_tokens": sum(int(call.get("token_usage", {}).get("output", 0)) for call in trace_calls),
        "request_count": sum(int(call.get("request_count", 0)) for call in trace_calls),
        "success": success,
        "purchase_signature": purchase_signature_from_snapshot(final_snapshot),
    }

    _write_csv(paths.csv_path, csv_rows)
    write_json(paths.json_path, episode_log)
    write_json(paths.trace_path, trace_payload)
    return episode_log


def _run_episode_shard(
    job: tuple[argparse.Namespace, str, int, int],
) -> list[dict]:
    args, run_id, shard_start, shard_end = job
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    env = make_webshop_env(
        num_products=args.num_products,
        human_goals=args.human_goals,
        show_attrs=args.show_attrs,
        file_path=args.file_path,
    )
    agent = (
        create_agent(args) if args.llm_seed_mode in {"none", "fixed"} else None
    )
    return [
        run_episode(args, run_id=run_id, episode_idx=episode_idx, env=env, agent=agent)
        for episode_idx in range(shard_start, shard_end)
    ]


def main() -> None:
    args = _parse_args()
    if int(args.num_workers) > 1 and not args.run_id:
        raise ValueError("--run-id is required when --num-workers > 1")
    run_id = args.run_id or default_run_id("webshop", args.model_id)
    if args.parallel_workers:
        shard_start, shard_end = int(args.episode_start), int(args.episode_end)
        episode_count = max(0, shard_end - shard_start)
        max_workers = min(int(args.num_workers), episode_count) if episode_count else 1
        jobs = []
        for worker_index in range(max_workers):
            worker_start, worker_end = shard_bounds(
                start=shard_start,
                end=shard_end,
                num_workers=max_workers,
                worker_index=worker_index,
            )
            jobs.append((args, run_id, worker_start, worker_end))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            shard_results = list(executor.map(_run_episode_shard, jobs))
        results = [result for worker_results in shard_results for result in worker_results]
    else:
        shard_start, shard_end = shard_bounds(
            start=args.episode_start,
            end=args.episode_end,
            num_workers=args.num_workers,
            worker_index=args.worker_index,
        )
        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        env = make_webshop_env(
            num_products=args.num_products,
            human_goals=args.human_goals,
            show_attrs=args.show_attrs,
            file_path=args.file_path,
        )
        agent = (
            create_agent(args) if args.llm_seed_mode in {"none", "fixed"} else None
        )
        results = [
            run_episode(args, run_id=run_id, episode_idx=episode_idx, env=env, agent=agent)
            for episode_idx in range(shard_start, shard_end)
        ]
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    run_dir = Path(args.output_root) / run_id
    summary = {
        "run_id": run_id,
        "episode_start": int(args.episode_start),
        "episode_end": int(args.episode_end),
        "shard_start": int(shard_start),
        "shard_end": int(shard_end),
        "num_workers": int(args.num_workers),
        "worker_index": int(args.worker_index),
        "success_threshold": float(args.success_threshold),
        "episode_count": len(results),
        "success_count": sum(1 for row in results if row.get("success")),
        "episodes": results,
    }
    summary_path = (
        run_dir / "summary.json"
        if args.parallel_workers or int(args.num_workers) == 1
        else run_dir / "_workers" / f"summary_worker_{int(args.worker_index):02d}.json"
    )
    write_json(summary_path, summary)
    print(f"wrote WebShop run: {run_dir}")


if __name__ == "__main__":
    main()
