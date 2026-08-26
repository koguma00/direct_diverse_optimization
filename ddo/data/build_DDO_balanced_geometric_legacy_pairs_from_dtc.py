#!/usr/bin/env python3
"""Build DDO balanced-geometric legacy pair JSONLs from a DTC run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from tqdm.auto import tqdm

MIN_SOFTMAX_EXP = sys.float_info.min

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddo.data.build_filtered_legacy_pairs_from_dtc import (
    CRITERION_TERMINAL_PROGRESSION,
    CRITERION_TEXTWORLD_TASK_PROGRESSION_V1,
    CRITERION_WEBSHOP_REWARD_THRESHOLD,
    QUALITY_SCORE_MAX_SAME_OBS_ACTION_RUN,
    USABLE_TRAJ_BASE_ALT,
    USABLE_TRAJ_BASE_ALT_ALT,
    WEBSHOP_SUCCESS_THRESHOLD,
    Candidate,
    _candidate_from_payload,
    _collect_base_trace_paths,
    _episode_id_from_trace,
    _find_latest_run_dir,
    _flatten_messages,
    _is_allowed_pair,
    _legacy_pair_row,
    _load_payload,
    _ordered_win_win,
    _prompt_messages_from_call,
    _read_jsonl,
    _resolve_trace_path,
    _task_slug,
    _write_csv,
    _as_int,
    _write_json,
    _write_jsonl,
)


ROW_FIELD_PAIR_WEIGHT = "pair_weight"
ROW_FIELD_CHOSEN_TARGET_PROB = "chosen_target_prob"
ROW_FIELD_TARGET_ALPHA = "target_alpha"
ROW_FIELD_TARGET_ETA = "target_eta"
ROW_FIELD_TARGET_MODE = "target_mode"
ROW_FIELD_TARGET_BETA = "target_beta"
ROW_FIELD_TARGET_MARGIN = "target_margin"
ROW_FIELD_STATE_SUCCESS_COUNT = "state_success_count"
ROW_FIELD_INTERNAL_STATE_GROUP_ID = "__state_group_id"
PAIR_TYPE_POSITIVE_WIN = "positive_win"
PAIR_SEMANTICS_POSITIVE_SUPPORT = "positive_support"
REF_RESPONSE_SCOPE_FULL = "full"
REF_RESPONSE_SCOPE_ACTION_ONLY = "action_only"
REF_NORMALIZE_SUM = "sum"
REF_NORMALIZE_AVG = "avg"
TARGET_MODE_PAIR_PROB = "pair_prob"
TARGET_MODE_REFERENCE_RELATIVE = "reference_relative"
ACTION_LINE_MARKER = "Action:"

DEFAULT_REF_DEVICE = "auto"
DEFAULT_REF_TORCH_DTYPE = "auto"
DEFAULT_MAX_PROMPT_LENGTH = 2048
DEFAULT_MAX_LENGTH = 2304


def _candidate_key(candidate: Candidate) -> str:
    return candidate.trajectory_id


def _task_ids_match(task_id: str, task_filter: str) -> bool:
    return (
        task_id == task_filter
        or task_id.rsplit("/", 1)[-1] == task_filter.rsplit("/", 1)[-1]
    )


def _state_group_id(*, family_id: str, divergence_step: int, state_key: str) -> str:
    return f"{family_id}::{divergence_step}::{state_key}"


def _validate_eta(eta: float) -> None:
    if not math.isfinite(eta) or eta < 0.0 or eta > 1.0:
        raise ValueError("eta must be a finite float in [0, 1]")


def _alpha_from_eta(eta: float) -> float:
    _validate_eta(eta)
    return 1.0 - eta


def _validate_target_config(target_mode: str, target_beta: float | None) -> None:
    if target_mode not in {TARGET_MODE_PAIR_PROB, TARGET_MODE_REFERENCE_RELATIVE}:
        raise ValueError(
            f"target_mode must be one of {TARGET_MODE_PAIR_PROB!r}, {TARGET_MODE_REFERENCE_RELATIVE!r}"
        )
    if target_mode == TARGET_MODE_REFERENCE_RELATIVE:
        if target_beta is None or not math.isfinite(target_beta) or target_beta <= 0.0:
            raise ValueError("target_beta must be a finite float > 0 for reference_relative targets")


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inv = math.exp(-value)
        return 1.0 / (1.0 + inv)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _compute_state_target_probs(
    successful_candidates: list[Candidate],
    *,
    ref_logprob_by_candidate: dict[str, float],
    eta: float,
) -> dict[str, float]:
    _validate_eta(eta)
    if not successful_candidates:
        return {}

    exponent = 1.0 - eta
    if exponent == 0.0:
        uniform_prob = 1.0 / len(successful_candidates)
        return {_candidate_key(candidate): uniform_prob for candidate in successful_candidates}

    scaled_scores = []
    for candidate in successful_candidates:
        score = ref_logprob_by_candidate[_candidate_key(candidate)]
        scaled_scores.append(exponent * score)

    max_score = max(scaled_scores)
    exp_scores = [max(math.exp(score - max_score), MIN_SOFTMAX_EXP) for score in scaled_scores]
    normalizer = sum(exp_scores)
    if normalizer <= 0.0 or not math.isfinite(normalizer):
        raise ValueError("state target normalization failed for geometric target")

    probs: dict[str, float] = {}
    for candidate, exp_score in zip(successful_candidates, exp_scores, strict=True):
        probs[_candidate_key(candidate)] = exp_score / normalizer
    return probs


def _pair_target_prob(
    *,
    chosen: Candidate,
    rejected: Candidate,
    state_target_probs: dict[str, float],
) -> float:
    chosen_prob = state_target_probs[_candidate_key(chosen)]
    rejected_prob = state_target_probs[_candidate_key(rejected)]
    denom = chosen_prob + rejected_prob
    if denom <= 0.0 or not math.isfinite(denom):
        raise ValueError("pair target probability denominator must be positive")
    return chosen_prob / denom


def _reference_relative_pair_target(
    *,
    chosen: Candidate,
    rejected: Candidate,
    state_target_probs: dict[str, float],
    ref_logprob_by_candidate: dict[str, float],
    target_beta: float,
) -> tuple[float, float]:
    chosen_key = _candidate_key(chosen)
    rejected_key = _candidate_key(rejected)
    chosen_prob = state_target_probs[chosen_key]
    rejected_prob = state_target_probs[rejected_key]
    if chosen_prob <= 0.0 or rejected_prob <= 0.0:
        raise ValueError("state target probabilities must be positive for reference-relative targets")
    target_log_odds = math.log(chosen_prob) - math.log(rejected_prob)
    reference_log_odds = ref_logprob_by_candidate[chosen_key] - ref_logprob_by_candidate[rejected_key]
    target_margin = target_log_odds - reference_log_odds
    return _sigmoid(target_beta * target_margin), target_margin


def _attach_split_local_weights(rows_by_task: dict[str, list[dict[str, Any]]]) -> tuple[int, int]:
    all_rows = [row for rows in rows_by_task.values() for row in rows]
    if not all_rows:
        return 0, 0

    rows_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        rows_by_state[str(row[ROW_FIELD_INTERNAL_STATE_GROUP_ID])].append(row)

    total_row_count = len(all_rows)
    state_count = len(rows_by_state)
    for state_rows in rows_by_state.values():
        row_weight = total_row_count / (state_count * len(state_rows))
        for row in state_rows:
            row[ROW_FIELD_PAIR_WEIGHT] = float(row_weight)

    for row in all_rows:
        row.pop(ROW_FIELD_INTERNAL_STATE_GROUP_ID, None)
    return total_row_count, state_count


def _response_text_for_ref_scope(response_text: str, response_scope: str) -> str:
    normalized = str(response_text).strip()
    if response_scope == REF_RESPONSE_SCOPE_FULL:
        if not normalized:
            raise ValueError("Reference scoring requires a non-empty response_text")
        return normalized
    if response_scope == REF_RESPONSE_SCOPE_ACTION_ONLY:
        action_idx = normalized.rfind(ACTION_LINE_MARKER)
        if action_idx < 0:
            raise ValueError(
                f"Reference scoring with response_scope={response_scope!r} requires {ACTION_LINE_MARKER!r}"
            )
        action_only = normalized[action_idx:].strip()
        if not action_only:
            raise ValueError("Reference scoring action-only response became empty")
        return action_only
    raise ValueError(f"Unsupported reference response scope: {response_scope}")


def _positive_row(
    *,
    chosen: Candidate,
    quality_score_name: str,
) -> dict[str, Any]:
    return {
        "pair_type": PAIR_TYPE_POSITIVE_WIN,
        "pair_semantics": PAIR_SEMANTICS_POSITIVE_SUPPORT,
        "task_id": chosen.task_id,
        "family_id": chosen.family_id,
        "divergence_step": chosen.divergence_step,
        "state_key": chosen.state_key,
        "trajectory_origin": chosen.trajectory_kind,
        "trajectory_id": chosen.trajectory_id,
        "chosen_trajectory_id": chosen.trajectory_id,
        "chosen_trajectory_kind": chosen.trajectory_kind,
        "chosen_source_trace": chosen.source_trace,
        "chosen_seed": chosen.seed,
        "quality_score_name": quality_score_name,
        "chosen_quality_score": chosen.quality_score,
        "prompt": _flatten_messages(chosen.prompt_messages),
        "chosen": {
            "action_text": chosen.response_text,
            "success": chosen.is_win,
            "progress": chosen.terminal_progression,
        },
    }


def _build_ref_logprob_scorer(
    *,
    model_name_or_path: str,
    adapter_path: str | None,
    device: str,
    torch_dtype: str,
    max_prompt_length: int,
    max_length: int,
    response_scope: str,
    normalize: str,
) -> Callable[[Candidate], float]:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Balanced-geometric reference scoring requires `torch` and `transformers`."
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
                "Balanced-geometric reference scoring with --ref-adapter-path requires `peft`."
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
            raise ValueError("max_length must be positive for reference scoring")
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

        completion_text = _response_text_for_ref_scope(candidate.response_text, response_scope)
        completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
        completion_ids = _append_eos(completion_ids)
        if not completion_ids:
            raise ValueError(f"Reference scoring found empty completion for {candidate.source_trace}")

        max_completion_length = max_length - len(prompt_ids)
        if max_completion_length <= 0:
            prompt_ids = prompt_ids[-(max_length - 1):]
            max_completion_length = max_length - len(prompt_ids)
        completion_ids = completion_ids[:max_completion_length]
        if not completion_ids:
            raise ValueError(
                f"Reference scoring truncated completion to zero tokens for {candidate.source_trace}"
            )

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
        masked_logps = token_logps.masked_fill(~loss_mask, 0.0)
        logprob_sum = float(masked_logps.sum().item())
        token_count = int(loss_mask.sum().item())
        if token_count <= 0:
            raise ValueError(f"Reference scoring produced zero target tokens for {candidate.source_trace}")
        if normalize == REF_NORMALIZE_SUM:
            return logprob_sum
        return logprob_sum / token_count

    return _score


def build_balanced_geometric_legacy_pairs(
    *,
    dtc_run_dir: Path,
    base_run_dir: Path,
    usable_traj: str,
    criterion: str,
    quality_score: str,
    max_same_obs_action_run_threshold: int,
    eta: float,
    ref_logprob_scorer: Callable[[Candidate], float],
    target_mode: str = TARGET_MODE_PAIR_PROB,
    target_beta: float | None = None,
    max_families: int = 0,
    success_threshold: float = WEBSHOP_SUCCESS_THRESHOLD,
    task_filter: str | None = None,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    if max_same_obs_action_run_threshold < 1:
        raise ValueError("max_same_obs_action_run_threshold must be >= 1")
    _validate_eta(eta)
    _validate_target_config(target_mode, target_beta)

    branch_index_path = dtc_run_dir / "_dtc" / "branch_index.jsonl"
    if not branch_index_path.exists():
        raise FileNotFoundError(f"Missing DTC branch index: {branch_index_path}")

    base_trace_by_id: dict[str, Path] = {}
    for trace_path in _collect_base_trace_paths(base_run_dir):
        base_trace_by_id[_episode_id_from_trace(trace_path)] = trace_path

    normalized_task_filter = str(task_filter or "").strip()
    if normalized_task_filter:
        base_trace_by_id = {
            family_id: trace_path
            for family_id, trace_path in base_trace_by_id.items()
            if _task_ids_match(
                str(_load_payload(trace_path).get("task_or_env_params") or "").strip(),
                normalized_task_filter,
            )
        }
        if not base_trace_by_id:
            raise ValueError(
                f"--task-filter {normalized_task_filter!r} matched zero base trajectories"
            )

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
    positive_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "task_id": "",
            "task_slug": "",
            "state_group_count": 0,
            "raw_win_candidates": 0,
            "quality_filtered_win_candidates": 0,
            "lose_candidates": 0,
            "win_lose_eligible_wins": 0,
            "win_win_eligible_wins": 0,
            "win_lose_rows": 0,
            "win_win_rows": 0,
            "positive_rows": 0,
            "win_lose_state_count": 0,
            "win_win_state_count": 0,
            "positive_state_count": 0,
            "positive_eligible_wins": 0,
            "quality_filter_threshold": max_same_obs_action_run_threshold,
            "quality_score": quality_score,
            "alpha": _alpha_from_eta(eta),
            "eta": eta,
            "target_mode": target_mode,
            "target_beta": target_beta,
            "win_trajectory_ids": set(),
            "lose_trajectory_ids": set(),
            "positive_trajectory_ids": set(),
        }
    )

    summary = {
        "run_dir": str(dtc_run_dir),
        "run_id": dtc_run_dir.name,
        "usable_traj": usable_traj,
        "criterion": criterion,
        "success_threshold": float(success_threshold),
        "quality_score": quality_score,
        "max_same_obs_action_run_threshold": max_same_obs_action_run_threshold,
        "alpha": _alpha_from_eta(eta),
        "eta": eta,
        "target_mode": target_mode,
        "target_beta": target_beta,
        "task_filter": normalized_task_filter or None,
        "family_count": len(allowed_families),
        "state_group_count": len(grouped_records),
        "raw_win_candidates": 0,
        "quality_filtered_win_candidates": 0,
        "lose_candidates": 0,
        "win_lose_eligible_wins": 0,
        "win_win_eligible_wins": 0,
        "win_lose_rows": 0,
        "win_win_rows": 0,
        "positive_rows": 0,
        "win_lose_state_count": 0,
        "win_win_state_count": 0,
        "positive_state_count": 0,
        "positive_eligible_wins": 0,
    }

    for (family_id, divergence_step, state_key), records in tqdm(
        grouped_records.items(),
        total=len(grouped_records),
        desc="Balanced Geometric States",
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

        win_lose_eligible_wins = [
            candidate
            for candidate in quality_filtered_wins
            if any(_is_allowed_pair(candidate, lose_candidate, usable_traj) for lose_candidate in lose_candidates)
        ]

        win_win_state_candidates = sorted(
            quality_filtered_wins,
            key=lambda candidate: (
                0 if candidate.trajectory_kind == "base" else 1,
                candidate.trajectory_id,
            ),
        )

        summary["raw_win_candidates"] += len(win_candidates)
        summary["quality_filtered_win_candidates"] += len(quality_filtered_wins)
        summary["lose_candidates"] += len(lose_candidates)
        summary["win_lose_eligible_wins"] += len(win_lose_eligible_wins)
        task_stat["raw_win_candidates"] += len(win_candidates)
        task_stat["quality_filtered_win_candidates"] += len(quality_filtered_wins)
        task_stat["lose_candidates"] += len(lose_candidates)
        task_stat["win_lose_eligible_wins"] += len(win_lose_eligible_wins)

        state_group = _state_group_id(
            family_id=family_id,
            divergence_step=divergence_step,
            state_key=state_key,
        )
        state_contributed_win_lose = False
        for chosen in win_lose_eligible_wins:
            connected_loses = [
                rejected
                for rejected in lose_candidates
                if _is_allowed_pair(chosen, rejected, usable_traj)
            ]
            for rejected in connected_loses:
                row = _legacy_pair_row(
                    pair_type="win_lose",
                    chosen=chosen,
                    rejected=rejected,
                    quality_score_name=quality_score,
                )
                row[ROW_FIELD_CHOSEN_TARGET_PROB] = 1.0
                row[ROW_FIELD_PAIR_WEIGHT] = 0.0
                row[ROW_FIELD_TARGET_ALPHA] = _alpha_from_eta(eta)
                row[ROW_FIELD_TARGET_ETA] = eta
                row[ROW_FIELD_TARGET_MODE] = target_mode
                if target_beta is not None:
                    row[ROW_FIELD_TARGET_BETA] = target_beta
                row[ROW_FIELD_STATE_SUCCESS_COUNT] = len(win_win_state_candidates)
                row[ROW_FIELD_INTERNAL_STATE_GROUP_ID] = state_group
                win_lose_by_task[task_slug].append(row)
                summary["win_lose_rows"] += 1
                task_stat["win_lose_rows"] += 1
                task_stat["win_trajectory_ids"].add(chosen.trajectory_id)
                task_stat["lose_trajectory_ids"].add(rejected.trajectory_id)
                state_contributed_win_lose = True

        if state_contributed_win_lose:
            summary["win_lose_state_count"] += 1
            task_stat["win_lose_state_count"] += 1

        if win_win_state_candidates:
            summary["positive_eligible_wins"] += len(win_win_state_candidates)
            task_stat["positive_eligible_wins"] += len(win_win_state_candidates)

        state_contributed_positive = False
        for chosen in win_win_state_candidates:
            row = _positive_row(
                chosen=chosen,
                quality_score_name=quality_score,
            )
            row[ROW_FIELD_PAIR_WEIGHT] = 0.0
            row[ROW_FIELD_TARGET_ALPHA] = _alpha_from_eta(eta)
            row[ROW_FIELD_TARGET_ETA] = eta
            row[ROW_FIELD_TARGET_MODE] = target_mode
            if target_beta is not None:
                row[ROW_FIELD_TARGET_BETA] = target_beta
            row[ROW_FIELD_STATE_SUCCESS_COUNT] = len(win_win_state_candidates)
            row[ROW_FIELD_INTERNAL_STATE_GROUP_ID] = state_group
            positive_by_task[task_slug].append(row)
            summary["positive_rows"] += 1
            task_stat["positive_rows"] += 1
            task_stat["win_trajectory_ids"].add(chosen.trajectory_id)
            task_stat["positive_trajectory_ids"].add(chosen.trajectory_id)
            state_contributed_positive = True

        if state_contributed_positive:
            summary["positive_state_count"] += 1
            task_stat["positive_state_count"] += 1

        if len(win_win_state_candidates) < 2:
            continue

        allowed_win_win_pairs: list[tuple[Candidate, Candidate]] = []
        for idx, candidate_a in enumerate(win_win_state_candidates):
            for candidate_b in win_win_state_candidates[idx + 1:]:
                if _is_allowed_pair(candidate_a, candidate_b, usable_traj):
                    allowed_win_win_pairs.append(_ordered_win_win(candidate_a, candidate_b))
        if not allowed_win_win_pairs:
            continue

        task_stat["win_win_eligible_wins"] += len(win_win_state_candidates)
        summary["win_win_eligible_wins"] += len(win_win_state_candidates)
        ref_logprob_by_candidate = {
            _candidate_key(candidate): float(ref_logprob_scorer(candidate))
            for candidate in win_win_state_candidates
        }
        state_target_probs = _compute_state_target_probs(
            win_win_state_candidates,
            ref_logprob_by_candidate=ref_logprob_by_candidate,
            eta=eta,
        )

        state_contributed_win_win = False
        for chosen, rejected in allowed_win_win_pairs:
            row = _legacy_pair_row(
                pair_type="win_win",
                chosen=chosen,
                rejected=rejected,
                quality_score_name=quality_score,
            )
            if target_mode == TARGET_MODE_REFERENCE_RELATIVE:
                assert target_beta is not None
                target_prob, target_margin = _reference_relative_pair_target(
                    chosen=chosen,
                    rejected=rejected,
                    state_target_probs=state_target_probs,
                    ref_logprob_by_candidate=ref_logprob_by_candidate,
                    target_beta=target_beta,
                )
                row[ROW_FIELD_CHOSEN_TARGET_PROB] = target_prob
                row[ROW_FIELD_TARGET_MARGIN] = target_margin
                row[ROW_FIELD_TARGET_BETA] = target_beta
            else:
                row[ROW_FIELD_CHOSEN_TARGET_PROB] = _pair_target_prob(
                    chosen=chosen,
                    rejected=rejected,
                    state_target_probs=state_target_probs,
                )
            row[ROW_FIELD_PAIR_WEIGHT] = 0.0
            row[ROW_FIELD_TARGET_ALPHA] = _alpha_from_eta(eta)
            row[ROW_FIELD_TARGET_ETA] = eta
            row[ROW_FIELD_TARGET_MODE] = target_mode
            row[ROW_FIELD_STATE_SUCCESS_COUNT] = len(win_win_state_candidates)
            row[ROW_FIELD_INTERNAL_STATE_GROUP_ID] = state_group
            win_win_by_task[task_slug].append(row)
            summary["win_win_rows"] += 1
            task_stat["win_win_rows"] += 1
            task_stat["win_trajectory_ids"].add(chosen.trajectory_id)
            task_stat["win_trajectory_ids"].add(rejected.trajectory_id)
            state_contributed_win_win = True

        if state_contributed_win_win:
            summary["win_win_state_count"] += 1
            task_stat["win_win_state_count"] += 1

    win_lose_row_count, win_lose_weighted_state_count = _attach_split_local_weights(win_lose_by_task)
    win_win_row_count, win_win_weighted_state_count = _attach_split_local_weights(win_win_by_task)
    positive_row_count, positive_weighted_state_count = _attach_split_local_weights(positive_by_task)
    summary["weighted_win_lose_row_count"] = win_lose_row_count
    summary["weighted_win_win_row_count"] = win_win_row_count
    summary["weighted_positive_row_count"] = positive_row_count
    summary["weighted_win_lose_state_count"] = win_lose_weighted_state_count
    summary["weighted_win_win_state_count"] = win_win_weighted_state_count
    summary["weighted_positive_state_count"] = positive_weighted_state_count

    task_rows: list[dict[str, Any]] = []
    for task_slug in sorted(task_stats):
        task_stat = dict(task_stats[task_slug])
        task_stat["unique_win_trajectory_count"] = len(task_stat.pop("win_trajectory_ids"))
        task_stat["unique_lose_trajectory_count"] = len(task_stat.pop("lose_trajectory_ids"))
        task_stat["unique_positive_trajectory_count"] = len(task_stat.pop("positive_trajectory_ids"))
        task_rows.append(task_stat)

    summary["tasks"] = task_rows
    return dict(win_lose_by_task), dict(win_win_by_task), dict(positive_by_task), summary, task_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DDO balanced-geometric legacy pair JSONLs from a DTC run."
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
        "--task-filter",
        default=None,
        help="Optional canonical task id or final task-id component to include.",
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
    target_exponent = parser.add_mutually_exclusive_group(required=True)
    target_exponent.add_argument(
        "--alpha",
        type=float,
        help="Paper target exponent in q_target(j) proportional to q_ref(j)^alpha",
    )
    target_exponent.add_argument(
        "--eta",
        type=float,
        help="Deprecated internal flattening strength; equivalent to alpha=1-eta",
    )
    parser.add_argument(
        "--target-mode",
        choices=(TARGET_MODE_PAIR_PROB, TARGET_MODE_REFERENCE_RELATIVE),
        default=TARGET_MODE_PAIR_PROB,
        help=(
            "How to convert the flattened successful-set target into win-win soft labels. "
            "Use reference_relative for DDO-v3."
        ),
    )
    parser.add_argument(
        "--target-beta",
        type=float,
        default=None,
        help="Beta used to convert reference-relative DDO-v3 target margins into soft labels.",
    )
    parser.add_argument(
        "--ref-model-name-or-path",
        required=True,
        help="Reference model used to score successful branches offline",
    )
    parser.add_argument(
        "--ref-adapter-path",
        default=None,
        help="Optional PEFT adapter path used to score the reference branch probabilities",
    )
    parser.add_argument(
        "--ref-device",
        default=DEFAULT_REF_DEVICE,
        help="Reference scoring device. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument(
        "--ref-torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default=DEFAULT_REF_TORCH_DTYPE,
        help="Reference scoring dtype.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=DEFAULT_MAX_PROMPT_LENGTH,
        help="Prompt truncation length for offline reference scoring.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Prompt+completion length cap for offline reference scoring.",
    )
    parser.add_argument(
        "--ref-response-scope",
        choices=(REF_RESPONSE_SCOPE_FULL, REF_RESPONSE_SCOPE_ACTION_ONLY),
        default=REF_RESPONSE_SCOPE_FULL,
        help="Which part of the candidate completion to score with the reference model.",
    )
    parser.add_argument(
        "--ref-normalize",
        choices=(REF_NORMALIZE_SUM, REF_NORMALIZE_AVG),
        default=REF_NORMALIZE_SUM,
        help="Whether to use summed or token-averaged reference log-prob scores.",
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
            "Output directory for per-task balanced-geometric pair JSONLs "
            "(default: data/dataset/DDO_balanced_geometric/<run_id>/<usable_traj>/<criterion>/<quality>_leq_<threshold>/alpha_<alpha>)"
        ),
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Output JSON path for summary stats (default: <out-dir>/DDO_balanced_geometric_conversion_stats.json)",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Output CSV path for per-task summary rows (default: <out-dir>/DDO_balanced_geometric_task_pair_counts.csv)",
    )
    parser.add_argument(
        "--aggregate-win-lose-out",
        default=None,
        help="Output JSONL path for aggregate win_lose rows (default: <out-dir>/train_pairs_DDO_balanced_geometric_win_lose.jsonl)",
    )
    parser.add_argument(
        "--aggregate-win-win-out",
        default=None,
        help="Output JSONL path for aggregate win_win rows (default: <out-dir>/train_pairs_DDO_balanced_geometric_win_win.jsonl)",
    )
    parser.add_argument(
        "--aggregate-positive-out",
        default=None,
        help="Output JSONL path for aggregate positive rows (default: <out-dir>/train_pairs_DDO_balanced_geometric_positive.jsonl)",
    )
    return parser.parse_args()


def _default_output_layout(
    *,
    run_id: str,
    usable_traj: str,
    criterion_suffix: str,
    quality_suffix: str,
    eta_suffix: str,
    target_mode: str,
    target_beta: float | None,
) -> tuple[str, Path]:
    if target_mode == TARGET_MODE_REFERENCE_RELATIVE:
        if target_beta is None:
            raise ValueError("--target-beta is required when --target-mode reference_relative")
        dataset_prefix = "DDO_v3"
        target_suffix = f"beta_{str(target_beta).replace('.', 'p')}"
        return (
            dataset_prefix,
            Path("data/dataset/DDO_v3")
            / run_id
            / usable_traj
            / criterion_suffix
            / quality_suffix
            / eta_suffix
            / target_suffix,
        )

    dataset_prefix = "DDO_balanced_geometric"
    return (
        dataset_prefix,
        Path("data/dataset/DDO_balanced_geometric")
        / run_id
        / usable_traj
        / criterion_suffix
        / quality_suffix
        / eta_suffix,
    )


def main() -> None:
    args = _parse_args()
    if args.alpha is not None:
        if not math.isfinite(args.alpha) or not 0.0 <= args.alpha <= 1.0:
            raise ValueError("alpha must be a finite float in [0, 1]")
        alpha = args.alpha
        eta = 1.0 - alpha
    else:
        _validate_eta(args.eta)
        eta = args.eta
        alpha = _alpha_from_eta(eta)
    dtc_root = Path(args.dtc_root)
    dtc_run_dir = Path(args.dtc_run_dir) if args.dtc_run_dir else _find_latest_run_dir(dtc_root)
    base_run_dir = Path(args.base_run_dir)
    run_id = dtc_run_dir.name

    quality_suffix = f"{args.quality_score}_leq_{int(args.max_same_obs_action_run_threshold)}"
    criterion_suffix = args.criterion
    if args.criterion == CRITERION_WEBSHOP_REWARD_THRESHOLD:
        criterion_suffix = f"{criterion_suffix}_geq_{str(args.success_threshold).replace('.', 'p')}"
    eta_suffix = f"alpha_{str(alpha).replace('.', 'p')}"
    dataset_prefix, default_out_dir = _default_output_layout(
        run_id=run_id,
        usable_traj=args.usable_traj,
        criterion_suffix=criterion_suffix,
        quality_suffix=quality_suffix,
        eta_suffix=eta_suffix,
        target_mode=args.target_mode,
        target_beta=args.target_beta,
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else default_out_dir
    )
    summary_out = (
        Path(args.summary_out)
        if args.summary_out
        else out_dir / f"{dataset_prefix}_conversion_stats.json"
    )
    csv_out = (
        Path(args.csv_out)
        if args.csv_out
        else out_dir / f"{dataset_prefix}_task_pair_counts.csv"
    )
    aggregate_win_lose_out = (
        Path(args.aggregate_win_lose_out)
        if args.aggregate_win_lose_out
        else out_dir / f"train_pairs_{dataset_prefix}_win_lose.jsonl"
    )
    aggregate_win_win_out = (
        Path(args.aggregate_win_win_out)
        if args.aggregate_win_win_out
        else out_dir / f"train_pairs_{dataset_prefix}_win_win.jsonl"
    )
    aggregate_positive_out = (
        Path(args.aggregate_positive_out)
        if args.aggregate_positive_out
        else out_dir / f"train_pairs_{dataset_prefix}_positive.jsonl"
    )

    ref_logprob_scorer = _build_ref_logprob_scorer(
        model_name_or_path=args.ref_model_name_or_path,
        adapter_path=args.ref_adapter_path,
        device=args.ref_device,
        torch_dtype=args.ref_torch_dtype,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        response_scope=args.ref_response_scope,
        normalize=args.ref_normalize,
    )

    win_lose_by_task, win_win_by_task, positive_by_task, summary, task_rows = build_balanced_geometric_legacy_pairs(
        dtc_run_dir=dtc_run_dir,
        base_run_dir=base_run_dir,
        usable_traj=args.usable_traj,
        criterion=args.criterion,
        quality_score=args.quality_score,
        max_same_obs_action_run_threshold=args.max_same_obs_action_run_threshold,
        eta=eta,
        ref_logprob_scorer=ref_logprob_scorer,
        target_mode=args.target_mode,
        target_beta=args.target_beta,
        max_families=args.max_families,
        success_threshold=args.success_threshold,
        task_filter=args.task_filter,
    )

    summary["ref_model_name_or_path"] = args.ref_model_name_or_path
    summary["ref_adapter_path"] = args.ref_adapter_path
    summary["ref_device"] = args.ref_device
    summary["ref_torch_dtype"] = args.ref_torch_dtype
    summary["max_prompt_length"] = args.max_prompt_length
    summary["max_length"] = args.max_length
    summary["ref_response_scope"] = args.ref_response_scope
    summary["ref_normalize"] = args.ref_normalize
    summary["alpha"] = alpha
    summary["dataset_prefix"] = dataset_prefix

    out_dir.mkdir(parents=True, exist_ok=True)
    task_slugs = sorted(set(win_lose_by_task) | set(win_win_by_task) | set(positive_by_task))
    aggregate_win_lose_rows: list[dict[str, Any]] = []
    aggregate_win_win_rows: list[dict[str, Any]] = []
    aggregate_positive_rows: list[dict[str, Any]] = []
    for task_slug in task_slugs:
        task_win_lose_rows = win_lose_by_task.get(task_slug, [])
        task_win_win_rows = win_win_by_task.get(task_slug, [])
        task_positive_rows = positive_by_task.get(task_slug, [])
        _write_jsonl(out_dir / f"{task_slug}_{dataset_prefix}_win_lose.jsonl", task_win_lose_rows)
        _write_jsonl(out_dir / f"{task_slug}_{dataset_prefix}_win_win.jsonl", task_win_win_rows)
        _write_jsonl(out_dir / f"{task_slug}_{dataset_prefix}_positive.jsonl", task_positive_rows)
        aggregate_win_lose_rows.extend(task_win_lose_rows)
        aggregate_win_win_rows.extend(task_win_win_rows)
        aggregate_positive_rows.extend(task_positive_rows)

    _write_jsonl(aggregate_win_lose_out, aggregate_win_lose_rows)
    _write_jsonl(aggregate_win_win_out, aggregate_win_win_rows)
    _write_jsonl(aggregate_positive_out, aggregate_positive_rows)
    _write_json(summary_out, summary)
    _write_csv(csv_out, task_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
