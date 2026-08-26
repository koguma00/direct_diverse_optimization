#!/usr/bin/env python3
"""Evaluate BabyAI and BabaIsAI through the DDO-owned BALROG adapter."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
BALROG_ROOT = REPO_ROOT / "benchmarks" / "BALROG"
if str(BALROG_ROOT) not in sys.path:
    sys.path.insert(0, str(BALROG_ROOT))

from ddo.evaluation.action import run_agent_action
from ddo.evaluation.balrog.agent import create_agent
from ddo.evaluation.balrog.env import make_env
from ddo.evaluation.balrog.run_config import (
    DEFAULT_BALROG_CONFIG_PATH,
    write_run_config_snapshot,
)


CSV_COLUMNS = [
    "step", "instruction", "observation_pre", "thought", "action_model",
    "action_executed", "raw_output", "feedback", "observation_post",
    "obs_changed", "reward", "progression", "terminated", "truncated",
    "done", "termination_reason", "action_defaulted", "input_tokens",
    "output_tokens", "step_wall_time_sec", "won", "lost",
]


def _observation_text(obs: Any) -> str:
    if isinstance(obs, dict):
        text = obs.get("text") or {}
        if isinstance(text, dict):
            return str(text.get("long_term_context", "") or "")
    return str(obs or "")


def _termination_reason(
    *, terminated: bool, truncated: bool, reward: float, progression: float,
    won: bool, lost: bool,
) -> str:
    if won:
        return "won"
    if lost:
        return "lost"
    if terminated and (reward > 0 or progression >= 1.0):
        return "success"
    if truncated:
        return "truncated"
    return "terminated" if terminated else ""


def _task_output_dir(output_dir: Path, env_name: str, task: str) -> Path:
    path = output_dir / env_name / task
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_episode(
    *, config: Any, env_name: str, task: str, output_dir: Path,
    episode_idx: int, seed: int, llm_seed: int | None, seed_mode: str, max_steps: int | None,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    env = make_env(env_name, task, config, env_seed=seed)
    agent = create_agent(config)
    agent.reset()
    agent.clear_llm_interactions()
    started = time.time()

    obs, _ = env.reset(seed=seed)
    mission = obs.get("mission") if env_name == "babyai" and isinstance(obs, dict) else None
    instruction = env.get_instruction_prompt(mission)
    agent.prompt_builder.update_instruction_prompt(instruction)
    step_cap = int(max_steps if max_steps is not None else env.max_steps)
    previous_action: str | None = None
    episode_return = 0.0
    rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    action_frequency: Counter[str] = Counter()
    failed = False

    for step in range(step_cap):
        step_started = time.perf_counter()
        observation_pre = _observation_text(obs)
        result = run_agent_action(
            agent=agent,
            obs=obs,
            prev_action=previous_action,
            validate_action=env.check_action_validity,
        )
        model_action = str(result["model_action"] or "")
        executed_action = str(result["executed_action"] or "")
        if not executed_action:
            failed = True
            break
        action_frequency[executed_action] += 1
        obs, reward, terminated, truncated, info = env.step(executed_action)
        info = info if isinstance(info, dict) else {}
        if bool(result["action_defaulted"]) and bool(config.eval.feedback_on_invalid_action):
            context = str(obs.get("text", {}).get("long_term_context", "") or "")
            obs["text"]["long_term_context"] = (
                "Your previous output did not contain a valid action. "
                f"Defaulted to action: {executed_action}\n\nObservation:\n{context}"
            )
        episode_return += float(reward)
        stats = env.get_stats()
        progression = float(stats.get("progression", 0.0) or 0.0)
        won = bool(info.get("won", False))
        lost = bool(info.get("lost", False))
        done = bool(terminated or truncated)
        reason = _termination_reason(
            terminated=bool(terminated), truncated=bool(truncated),
            reward=float(reward), progression=progression, won=won, lost=lost,
        )
        observation_post = _observation_text(obs)
        elapsed = time.perf_counter() - step_started
        row = {
            "step": step,
            "instruction": instruction,
            "observation_pre": observation_pre,
            "thought": result["thought"],
            "action_model": model_action,
            "action_executed": executed_action,
            "raw_output": result["raw_output"],
            "feedback": str(info.get("feedback", "") or ""),
            "observation_post": observation_post,
            "obs_changed": observation_pre != observation_post,
            "reward": float(reward),
            "progression": progression,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done": done,
            "termination_reason": reason,
            "action_defaulted": bool(result["action_defaulted"]),
            "input_tokens": int(result["input_tokens"]),
            "output_tokens": int(result["output_tokens"]),
            "step_wall_time_sec": elapsed,
            "won": won,
            "lost": lost,
        }
        rows.append(row)
        calls.append(
            {
                "call_idx": step,
                "global_step": step,
                "episode_step": step,
                "instruction": instruction,
                "observation": observation_pre,
                "thought": row["thought"],
                "action": executed_action,
                "raw_output": row["raw_output"],
                "feedback": row["feedback"],
                "reward": row["reward"],
                "progression": progression,
                "done": done,
                "won": won,
                "lost": lost,
                "request_count": 1,
                "step_wall_time_sec": elapsed,
                "token_usage": {
                    "input": row["input_tokens"],
                    "output": row["output_tokens"],
                    "total": row["input_tokens"] + row["output_tokens"],
                },
                "extras": {
                    "action_model": model_action,
                    "action_defaulted": row["action_defaulted"],
                    "observation_post": observation_post,
                    "llm_interaction": result["interaction"],
                },
            }
        )
        previous_action = executed_action
        if done:
            break

    final_stats = env.get_stats()
    final_progression = float(final_stats.get("progression", 0.0) or 0.0)
    episode_id = f"{task}_run_{episode_idx:02d}"
    episode_log = {
        "task": task,
        "episode_id": episode_id,
        "episode_idx": episode_idx,
        "seed": seed,
        "llm_seed": llm_seed,
        "seed_mode": seed_mode,
        "episode_return": episode_return,
        "progression": final_progression,
        "num_steps": len(rows),
        "done": bool(rows and rows[-1]["done"]),
        "aborted": failed,
        "failed_candidates": list(env.failed_candidates),
        "action_frequency": dict(action_frequency),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "request_count": len(calls),
        "duration_seconds": time.time() - started,
        **final_stats,
    }
    trace = {
        "schema_version": "thought_action_v1",
        "benchmark": "balrog",
        "env_name": env_name,
        "task_or_env_params": task,
        "episode_id": episode_id,
        "episode_idx": episode_idx,
        "seed": seed,
        "llm_seed": llm_seed,
        "num_calls": len(calls),
        "calls": calls,
        "extras": {"adapter": "ddo.evaluation.balrog", "search_method": "none"},
    }
    task_dir = _task_output_dir(output_dir, env_name, task)
    stem = f"{task}_run_{episode_idx:02d}"
    csv_path = task_dir / f"{stem}.csv"
    json_path = task_dir / f"{stem}.json"
    trace_path = task_dir / f"{stem}_llm_trace.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(episode_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    trace_path.write_text(
        json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if hasattr(env, "close"):
        env.close()
    return episode_log


def _run_episode_job(job: dict[str, Any]) -> dict[str, Any]:
    config = OmegaConf.create(job["config"])
    config.client.generate_kwargs.seed = job["llm_seed"]
    return _run_episode(
        config=config,
        env_name=job["env_name"],
        task=job["task"],
        output_dir=Path(job["output_dir"]),
        episode_idx=job["episode_idx"],
        seed=job["seed"],
        llm_seed=job["llm_seed"],
        seed_mode=job["seed_mode"],
        max_steps=job["max_steps"],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("babyai", "babaisai"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260305)
    parser.add_argument("--seed-mode", choices=("fixed", "per_episode"), default="per_episode")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-text-history", type=int, default=16)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--client-timeout", type=float, default=60.0)
    parser.add_argument("--client-max-retries", type=int, default=1)
    parser.add_argument("--llm-seed-base", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be positive")
    if args.num_episodes < 1:
        raise ValueError("--num-episodes must be positive")
    if not BALROG_ROOT.is_dir():
        raise FileNotFoundError(
            f"official BALROG checkout not found at {BALROG_ROOT}; follow README setup"
        )
    config = OmegaConf.load(DEFAULT_BALROG_CONFIG_PATH)
    config.envs.names = args.benchmark
    config.envs.env_kwargs.seed = args.seed
    config.envs.env_kwargs.seed_mode = args.seed_mode
    config.agent.max_text_history = args.max_text_history
    config.client.model_id = args.model_id
    config.client.base_url = args.base_url
    config.client.timeout = args.client_timeout
    config.client.max_retries = args.client_max_retries
    config.client.generate_kwargs.temperature = args.temperature
    config.client.generate_kwargs.top_p = args.top_p
    config.client.generate_kwargs.max_tokens = args.max_tokens
    config.client.generate_kwargs.seed = args.llm_seed_base
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config_snapshot(args.output_dir, config)

    config_payload = OmegaConf.to_container(config, resolve=True)
    jobs = []
    for episode_idx in range(args.num_episodes):
        seed = args.seed if args.seed_mode == "fixed" else args.seed + episode_idx
        llm_seed = (
            None if args.llm_seed_base is None else args.llm_seed_base + episode_idx
        )
        jobs.append(
            {
                "config": config_payload,
                "env_name": args.benchmark,
                "task": args.task,
                "output_dir": str(args.output_dir),
                "episode_idx": episode_idx,
                "seed": seed,
                "llm_seed": llm_seed,
                "seed_mode": args.seed_mode,
                "max_steps": args.max_steps,
            }
        )
    if args.num_workers == 1:
        results = [_run_episode_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(executor.map(_run_episode_job, jobs))
    summary = {
        "benchmark": args.benchmark,
        "task": args.task,
        "episodes": len(results),
        "successes": sum(
            bool(item.get("progression", 0.0) >= 1.0)
            or bool(item.get("done") and item.get("episode_return", 0.0) > 0.0)
            for item in results
        ),
    }
    summary["success_rate"] = summary["successes"] / summary["episodes"]
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
