#!/usr/bin/env python3
"""Build DivPO-style legacy DPO pair JSONLs from a DTC run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tqdm.auto import tqdm


USABLE_TRAJ_BASE_ALT = "base-alt"
USABLE_TRAJ_BASE_ALT_ALT = "base-alt-alt"
CRITERION_TERMINAL_PROGRESSION = "terminal_progression"
CRITERION_TEXTWORLD_TASK_PROGRESSION_V1 = "textworld_task_progression_v1"
CRITERION_WEBSHOP_REWARD_THRESHOLD = "webshop_reward_threshold"
TEXTWORLD_COOKING_WIN_THRESHOLD = 10.0 / 17.0
WEBSHOP_SUCCESS_THRESHOLD = 0.9
PAIR_TYPE_WIN_LOSE = "win_lose"
DIVPO_DIVERSITY_PROB = "prob"
DIVPO_DIVERSITY_ACTION_FREQ = "action_freq"
ACTION_FREQ_SCOPE_TASK = "task"
ACTION_FREQ_SCOPE_STATE = "state"
PROB_RESPONSE_SCOPE_FULL = "full"
PROB_RESPONSE_SCOPE_ACTION_ONLY = "action_only"
PAIR_SEMANTICS_OBSERVED_WIN_LOSE = "observed_win_lose"
PAIR_SEMANTICS_PSEUDO_BASE_ALT_WIN_LENGTH = "pseudo_base_alt_win_length"
RANKING_SCORE_POST_DIVERGENCE_CALL_COUNT = "post_divergence_call_count"
ACTION_LINE_MARKER = "Action:"


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
    is_win: bool
    post_divergence_call_count: int


@dataclass(frozen=True)
class StateGroup:
    family_id: str
    divergence_step: int
    state_key: str
    task_id: str
    task_slug: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class PairCandidate:
    chosen: Candidate
    rejected: Candidate
    pair_semantics: str
    ranking_score_name: str | None = None
    chosen_ranking_score: int | float | None = None
    rejected_ranking_score: int | float | None = None


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


def _response_text_for_prob_scope(response_text: str, response_scope: str) -> str:
    normalized = str(response_text).strip()
    if response_scope == PROB_RESPONSE_SCOPE_FULL:
        if not normalized:
            raise ValueError("Probability scoring requires a non-empty response_text")
        return normalized
    if response_scope == PROB_RESPONSE_SCOPE_ACTION_ONLY:
        action_idx = normalized.rfind(ACTION_LINE_MARKER)
        if action_idx < 0:
            raise ValueError(
                f"Probability scoring with response_scope={response_scope!r} requires {ACTION_LINE_MARKER!r}"
            )
        action_only = normalized[action_idx:].strip()
        if not action_only:
            raise ValueError("Probability scoring action-only response became empty")
        return action_only
    raise ValueError(f"Unsupported probability response scope: {response_scope}")


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


def _task_slug(task_id: str) -> str:
    task_id = str(task_id).strip()
    if "/" in task_id:
        return task_id.rsplit("/", 1)[-1]
    return task_id or "unknown"


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
        is_win=_is_win(payload, criterion, success_threshold=success_threshold),
        post_divergence_call_count=len(calls) - divergence_step,
    )


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


def _candidate_key(candidate: Candidate) -> tuple[str, int, str]:
    return (candidate.trajectory_id, candidate.divergence_step, candidate.response_text)


def _candidate_tie_key(candidate: Candidate) -> tuple[int, str]:
    return (0 if candidate.trajectory_kind == "base" else 1, candidate.trajectory_id)


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


def _build_action_frequency_tables(state_groups: list[StateGroup]) -> dict[str, Counter[str]]:
    counts_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for state_group in state_groups:
        for candidate in state_group.candidates:
            counts_by_task[state_group.task_slug][candidate.action_text] += 1
    return counts_by_task


def _build_prob_diversity_scorer(
    *,
    model_name_or_path: str,
    adapter_path: str | None,
    device: str,
    torch_dtype: str,
    max_prompt_length: int,
    max_length: int,
    normalize: str,
    response_scope: str,
) -> Callable[[Candidate], float]:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Probability-based DivPO scoring requires `torch` and `transformers`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype_map = {
        "auto": None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    resolved_dtype = dtype_map[torch_dtype]
    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    if adapter_path:
        try:
            from peft import AutoPeftModelForCausalLM
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Probability-based DivPO scoring with --prob-adapter-path requires `peft`."
            ) from exc
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_path,
            torch_dtype=resolved_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=resolved_dtype,
        )

    model.to(resolved_device)
    model.eval()

    def _messages_to_prompt_text(messages: list[dict[str, str]]) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except Exception:
                pass
        return _flatten_messages(messages)

    def _append_eos(completion_ids: list[int]) -> list[int]:
        if max_length <= 0:
            raise ValueError("max_length must be positive for probability scoring")
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            return completion_ids
        if not completion_ids:
            return [eos_token_id]
        return completion_ids + [eos_token_id]

    def _score(candidate: Candidate) -> float:
        prompt_ids = tokenizer(
            _messages_to_prompt_text(candidate.prompt_messages),
            add_special_tokens=False,
        )["input_ids"]
        prompt_ids = prompt_ids[-max_prompt_length:]

        completion_text = _response_text_for_prob_scope(candidate.response_text, response_scope)
        completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
        completion_ids = _append_eos(completion_ids)
        if not completion_ids:
            raise ValueError(f"Probability scoring found empty completion for {candidate.source_trace}")

        max_completion_length = max_length - len(prompt_ids)
        if max_completion_length <= 0:
            prompt_ids = prompt_ids[-(max_length - 1):]
            max_completion_length = max_length - len(prompt_ids)
        completion_ids = completion_ids[:max_completion_length]
        if not completion_ids:
            raise ValueError(f"Probability scoring truncated completion to zero tokens for {candidate.source_trace}")

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=resolved_device)
        label_tensor = torch.tensor([labels], dtype=torch.long, device=resolved_device)
        attention_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            logits = model(input_ids=input_tensor, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :].float()
        shift_labels = label_tensor[:, 1:]
        loss_mask = shift_labels != -100
        safe_labels = shift_labels.masked_fill(~loss_mask, 0)
        token_logps = F.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        masked_logps = token_logps * loss_mask
        logprob_sum = float(masked_logps.sum().item())
        token_count = int(loss_mask.sum().item())
        if token_count <= 0:
            raise ValueError(f"Probability scoring produced zero target tokens for {candidate.source_trace}")
        if normalize == "sum":
            return -logprob_sum
        return -(logprob_sum / token_count)

    return _score


def build_divpo_legacy_pairs(
    *,
    dtc_run_dir: Path,
    base_run_dir: Path,
    usable_traj: str,
    criterion: str,
    diversity_criterion: str,
    action_freq_scope: str,
    success_threshold: float = WEBSHOP_SUCCESS_THRESHOLD,
    include_base_alt_win_pseudo_pairs: bool = False,
    prob_response_scope: str = PROB_RESPONSE_SCOPE_FULL,
    prob_scorer: Callable[[Candidate], float] | None = None,
    max_families: int = 0,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[dict[str, Any]]]:
    if diversity_criterion == DIVPO_DIVERSITY_PROB and prob_scorer is None:
        raise ValueError("diversity_criterion='prob' requires prob_scorer")

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

    state_groups: list[StateGroup] = []
    for (family_id, divergence_step, state_key), records in tqdm(
        grouped_records.items(),
        total=len(grouped_records),
        desc="DivPO Candidate States",
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
            success_threshold=success_threshold,
        )

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
                    success_threshold=success_threshold,
                )
            )

        state_groups.append(
            StateGroup(
                family_id=family_id,
                divergence_step=divergence_step,
                state_key=state_key,
                task_id=base_candidate.task_id,
                task_slug=_task_slug(base_candidate.task_id),
                candidates=tuple(candidates),
            )
        )

    action_counts_by_task = _build_action_frequency_tables(state_groups)
    diversity_score_cache: dict[tuple[str, int, str], float] = {}

    if diversity_criterion == DIVPO_DIVERSITY_PROB:
        unique_candidates: dict[tuple[str, int, str], Candidate] = {}
        for state_group in state_groups:
            for candidate in state_group.candidates:
                unique_candidates[_candidate_key(candidate)] = candidate
        for candidate in tqdm(
            unique_candidates.values(),
            total=len(unique_candidates),
            desc="DivPO Prob Scores",
            unit="candidate",
        ):
            diversity_score_cache[_candidate_key(candidate)] = float(prob_scorer(candidate))

    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "task_id": "",
            "task_slug": "",
            "state_group_count": 0,
            "eligible_state_group_count": 0,
            "raw_win_candidates": 0,
            "raw_lose_candidates": 0,
            "observed_pair_candidate_count": 0,
            "pseudo_pair_candidate_count": 0,
            "selected_observed_row_count": 0,
            "selected_pseudo_row_count": 0,
            "output_rows": 0,
            "unique_chosen_trajectory_ids": set(),
            "unique_rejected_trajectory_ids": set(),
        }
    )
    summary = {
        "run_dir": str(dtc_run_dir),
        "run_id": dtc_run_dir.name,
        "usable_traj": usable_traj,
        "criterion": criterion,
        "success_threshold": float(success_threshold),
        "diversity_criterion": diversity_criterion,
        "action_freq_scope": action_freq_scope,
        "prob_response_scope": prob_response_scope if diversity_criterion == DIVPO_DIVERSITY_PROB else None,
        "include_base_alt_win_pseudo_pairs": include_base_alt_win_pseudo_pairs,
        "family_count": len(allowed_families),
        "state_group_count": len(state_groups),
        "eligible_state_group_count": 0,
        "raw_win_candidates": 0,
        "raw_lose_candidates": 0,
        "observed_pair_candidate_count": 0,
        "pseudo_pair_candidate_count": 0,
        "selected_observed_row_count": 0,
        "selected_pseudo_row_count": 0,
        "output_rows": 0,
    }

    def candidate_diversity_score(candidate: Candidate, state_group: StateGroup) -> float:
        key = _candidate_key(candidate)
        if key in diversity_score_cache:
            return diversity_score_cache[key]

        if diversity_criterion != DIVPO_DIVERSITY_ACTION_FREQ:
            raise ValueError(f"Unsupported diversity criterion: {diversity_criterion}")

        if action_freq_scope == ACTION_FREQ_SCOPE_TASK:
            action_count = action_counts_by_task[state_group.task_slug][candidate.action_text]
        elif action_freq_scope == ACTION_FREQ_SCOPE_STATE:
            action_count = sum(1 for peer in state_group.candidates if peer.action_text == candidate.action_text)
        else:
            raise ValueError(f"Unsupported action_freq_scope: {action_freq_scope}")

        score = 1.0 / float(max(1, action_count))
        diversity_score_cache[key] = score
        return score

    for state_group in tqdm(
        state_groups,
        total=len(state_groups),
        desc="DivPO Legacy Pairs",
        unit="state",
    ):
        wins = [candidate for candidate in state_group.candidates if candidate.is_win]
        loses = [candidate for candidate in state_group.candidates if not candidate.is_win]

        task_stat = task_stats[state_group.task_slug]
        task_stat["task_id"] = state_group.task_id
        task_stat["task_slug"] = state_group.task_slug
        task_stat["state_group_count"] += 1
        task_stat["raw_win_candidates"] += len(wins)
        task_stat["raw_lose_candidates"] += len(loses)
        summary["raw_win_candidates"] += len(wins)
        summary["raw_lose_candidates"] += len(loses)

        observed_pairs: list[PairCandidate] = [
            PairCandidate(
                chosen=win,
                rejected=lose,
                pair_semantics=PAIR_SEMANTICS_OBSERVED_WIN_LOSE,
            )
            for win in wins
            for lose in loses
            if _is_allowed_pair(win, lose, usable_traj)
        ]
        pseudo_pairs: list[PairCandidate] = []
        if include_base_alt_win_pseudo_pairs:
            base_win = next(
                (
                    candidate
                    for candidate in state_group.candidates
                    if candidate.trajectory_kind == "base" and candidate.is_win
                ),
                None,
            )
            if base_win is not None:
                for alt_win in state_group.candidates:
                    if alt_win.trajectory_kind != "alt" or not alt_win.is_win:
                        continue
                    if not _is_allowed_pair(base_win, alt_win, usable_traj):
                        continue
                    chosen, rejected = _ordered_by_post_divergence_call_count(base_win, alt_win)
                    pseudo_pairs.append(
                        PairCandidate(
                            chosen=chosen,
                            rejected=rejected,
                            pair_semantics=PAIR_SEMANTICS_PSEUDO_BASE_ALT_WIN_LENGTH,
                            ranking_score_name=RANKING_SCORE_POST_DIVERGENCE_CALL_COUNT,
                            chosen_ranking_score=chosen.post_divergence_call_count,
                            rejected_ranking_score=rejected.post_divergence_call_count,
                        )
                    )

        summary["observed_pair_candidate_count"] += len(observed_pairs)
        summary["pseudo_pair_candidate_count"] += len(pseudo_pairs)
        task_stat["observed_pair_candidate_count"] += len(observed_pairs)
        task_stat["pseudo_pair_candidate_count"] += len(pseudo_pairs)

        allowed_pairs = observed_pairs + pseudo_pairs
        if not allowed_pairs:
            continue

        task_stat["eligible_state_group_count"] += 1
        summary["eligible_state_group_count"] += 1

        def _pair_rank(pair: PairCandidate) -> tuple[float, float, tuple[int, str], tuple[int, str]]:
            chosen, rejected = pair.chosen, pair.rejected
            return (
                -candidate_diversity_score(chosen, state_group),
                candidate_diversity_score(rejected, state_group),
                _candidate_tie_key(chosen),
                _candidate_tie_key(rejected),
            )

        selected_pair = min(allowed_pairs, key=_pair_rank)
        chosen, rejected = selected_pair.chosen, selected_pair.rejected
        chosen_score = candidate_diversity_score(chosen, state_group)
        rejected_score = candidate_diversity_score(rejected, state_group)

        row = {
            "pair_type": PAIR_TYPE_WIN_LOSE,
            "pair_semantics": selected_pair.pair_semantics,
            "task_id": chosen.task_id,
            "family_id": chosen.family_id,
            "divergence_step": chosen.divergence_step,
            "state_key": chosen.state_key,
            "pair_origin": _pair_origin(chosen, rejected),
            "selection_method": "DivPO",
            "diversity_criterion": diversity_criterion,
            "action_freq_scope": action_freq_scope if diversity_criterion == DIVPO_DIVERSITY_ACTION_FREQ else None,
            "prob_response_scope": prob_response_scope if diversity_criterion == DIVPO_DIVERSITY_PROB else None,
            "chosen_seed": chosen.seed,
            "rejected_seed": rejected.seed,
            "chosen_trajectory_id": chosen.trajectory_id,
            "rejected_trajectory_id": rejected.trajectory_id,
            "chosen_trajectory_kind": chosen.trajectory_kind,
            "rejected_trajectory_kind": rejected.trajectory_kind,
            "chosen_source_trace": chosen.source_trace,
            "rejected_source_trace": rejected.source_trace,
            "chosen_quality_score": chosen.terminal_progression,
            "rejected_quality_score": rejected.terminal_progression,
            "ranking_score_name": selected_pair.ranking_score_name,
            "chosen_ranking_score": selected_pair.chosen_ranking_score,
            "rejected_ranking_score": selected_pair.rejected_ranking_score,
            "chosen_diversity_score": chosen_score,
            "rejected_diversity_score": rejected_score,
            "state_candidate_count": len(state_group.candidates),
            "state_win_candidate_count": len(wins),
            "state_lose_candidate_count": len(loses),
            "state_observed_pair_candidate_count": len(observed_pairs),
            "state_pseudo_pair_candidate_count": len(pseudo_pairs),
            "state_pair_candidate_count": len(allowed_pairs),
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
        rows_by_task[state_group.task_slug].append(row)
        if selected_pair.pair_semantics == PAIR_SEMANTICS_PSEUDO_BASE_ALT_WIN_LENGTH:
            summary["selected_pseudo_row_count"] += 1
            task_stat["selected_pseudo_row_count"] += 1
        else:
            summary["selected_observed_row_count"] += 1
            task_stat["selected_observed_row_count"] += 1
        summary["output_rows"] += 1
        task_stat["output_rows"] += 1
        task_stat["unique_chosen_trajectory_ids"].add(chosen.trajectory_id)
        task_stat["unique_rejected_trajectory_ids"].add(rejected.trajectory_id)

    task_rows: list[dict[str, Any]] = []
    for task_slug in sorted(task_stats):
        task_stat = dict(task_stats[task_slug])
        task_stat["unique_chosen_trajectory_count"] = len(task_stat.pop("unique_chosen_trajectory_ids"))
        task_stat["unique_rejected_trajectory_count"] = len(task_stat.pop("unique_rejected_trajectory_ids"))
        task_rows.append(task_stat)

    summary["tasks"] = task_rows
    return dict(rows_by_task), summary, task_rows


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DivPO-style trainer-ready legacy win_lose JSONLs from a DTC run."
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
        "--diversity-criterion",
        required=True,
        choices=(DIVPO_DIVERSITY_PROB, DIVPO_DIVERSITY_ACTION_FREQ),
        help="DivPO diversity criterion used to pick one chosen/rejected pair per state",
    )
    parser.add_argument(
        "--action-freq-scope",
        default=ACTION_FREQ_SCOPE_TASK,
        choices=(ACTION_FREQ_SCOPE_TASK, ACTION_FREQ_SCOPE_STATE),
        help="For action_freq, count action support within the task or within each state group.",
    )
    parser.add_argument(
        "--include-base-alt-win-pseudo-pairs",
        action="store_true",
        help=(
            "Add pseudo base/alt win-win candidates before DivPO top-1 selection. "
            "The shorter post-divergence trajectory is preferred, and ties favor base."
        ),
    )
    parser.add_argument(
        "--prob-model-name-or-path",
        default=None,
        help="Required for diversity-criterion=prob. Model or tokenizer source used for logprob scoring.",
    )
    parser.add_argument(
        "--prob-adapter-path",
        default=None,
        help="Optional PEFT adapter path loaded on top of --prob-model-name-or-path for prob scoring.",
    )
    parser.add_argument(
        "--prob-device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device used for probability scoring.",
    )
    parser.add_argument(
        "--prob-torch-dtype",
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
        help="Torch dtype used for probability scoring.",
    )
    parser.add_argument(
        "--prob-max-prompt-length",
        type=int,
        default=2048,
        help="Prompt truncation length for probability scoring.",
    )
    parser.add_argument(
        "--prob-max-length",
        type=int,
        default=2304,
        help="Total prompt+response length cap for probability scoring.",
    )
    parser.add_argument(
        "--prob-normalize",
        default="mean",
        choices=("mean", "sum"),
        help="Probability scoring normalization over completion tokens.",
    )
    parser.add_argument(
        "--prob-response-scope",
        default=PROB_RESPONSE_SCOPE_FULL,
        choices=(PROB_RESPONSE_SCOPE_FULL, PROB_RESPONSE_SCOPE_ACTION_ONLY),
        help=(
            "Which part of the response to score for diversity-criterion=prob. "
            "'full' uses the whole completion; 'action_only' scores only the final Action line."
        ),
    )
    parser.add_argument(
        "--max-families",
        type=int,
        default=0,
        help="Optional cap for number of episode families to process (0 means all)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory for DivPO files "
            "(default: data/dataset/DivPO/<run_id>/<usable_traj>/<criterion>/<diversity>)"
        ),
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Output JSON path for summary stats (default: <out-dir>/DivPO_conversion_stats.json)",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Output CSV path for per-task summary rows (default: <out-dir>/DivPO_task_pair_counts.csv)",
    )
    parser.add_argument(
        "--aggregate-out",
        default=None,
        help="Output JSONL path for all-task DivPO win_lose rows (default: <out-dir>/train_pairs_DivPO.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dtc_root = Path(args.dtc_root)
    dtc_run_dir = Path(args.dtc_run_dir) if args.dtc_run_dir else _find_latest_run_dir(dtc_root)
    base_run_dir = Path(args.base_run_dir)
    run_id = dtc_run_dir.name

    diversity_suffix = args.diversity_criterion
    if args.diversity_criterion == DIVPO_DIVERSITY_ACTION_FREQ:
        diversity_suffix = f"{args.diversity_criterion}_{args.action_freq_scope}"
    elif (
        args.diversity_criterion == DIVPO_DIVERSITY_PROB
        and args.prob_response_scope == PROB_RESPONSE_SCOPE_ACTION_ONLY
    ):
        diversity_suffix = "prob_actiononly"
    if args.include_base_alt_win_pseudo_pairs:
        diversity_suffix = f"{diversity_suffix}__base_alt_win_pseudo"
    criterion_suffix = args.criterion
    if args.criterion == CRITERION_WEBSHOP_REWARD_THRESHOLD:
        criterion_suffix = f"{criterion_suffix}_geq_{str(args.success_threshold).replace('.', 'p')}"

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("data/dataset/DivPO") / run_id / args.usable_traj / criterion_suffix / diversity_suffix
    )
    summary_out = Path(args.summary_out) if args.summary_out else out_dir / "DivPO_conversion_stats.json"
    csv_out = Path(args.csv_out) if args.csv_out else out_dir / "DivPO_task_pair_counts.csv"
    aggregate_out = Path(args.aggregate_out) if args.aggregate_out else out_dir / "train_pairs_DivPO.jsonl"

    prob_scorer = None
    if args.diversity_criterion == DIVPO_DIVERSITY_PROB:
        if not args.prob_model_name_or_path:
            raise ValueError("--prob-model-name-or-path is required for --diversity-criterion prob")
        prob_scorer = _build_prob_diversity_scorer(
            model_name_or_path=args.prob_model_name_or_path,
            adapter_path=args.prob_adapter_path,
            device=args.prob_device,
            torch_dtype=args.prob_torch_dtype,
            max_prompt_length=args.prob_max_prompt_length,
            max_length=args.prob_max_length,
            normalize=args.prob_normalize,
            response_scope=args.prob_response_scope,
        )

    rows_by_task, summary, task_rows = build_divpo_legacy_pairs(
        dtc_run_dir=dtc_run_dir,
        base_run_dir=base_run_dir,
        usable_traj=args.usable_traj,
        criterion=args.criterion,
        diversity_criterion=args.diversity_criterion,
        action_freq_scope=args.action_freq_scope,
        success_threshold=args.success_threshold,
        include_base_alt_win_pseudo_pairs=args.include_base_alt_win_pseudo_pairs,
        prob_response_scope=args.prob_response_scope,
        prob_scorer=prob_scorer,
        max_families=args.max_families,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    task_slugs = sorted(rows_by_task)
    aggregate_rows: list[dict[str, Any]] = []
    for task_slug in task_slugs:
        rows = rows_by_task.get(task_slug, [])
        aggregate_rows.extend(rows)
        _write_jsonl(out_dir / f"{task_slug}_DivPO_win_lose.jsonl", rows)

    summary["out_dir"] = str(out_dir)
    summary["aggregate_out"] = str(aggregate_out)
    _write_jsonl(aggregate_out, aggregate_rows)
    _write_json(summary_out, summary)
    _write_csv(csv_out, task_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
