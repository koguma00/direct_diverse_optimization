"""BALROG-compatible agent action execution with paper-faithful retries."""

from __future__ import annotations

import copy
import time
from collections import Counter
from typing import Any

from ddo.evaluation.balrog.client import (
    classify_llm_response_abort_reason,
    classify_retryable_exception_leaf,
)


DEFAULT_INVALID_ACTION_RETRY_NOTICE = (
    "Your previous output did not contain a valid action for the current step. "
    "Retry the same step and output a single valid action in the required format."
)


def _reset_agent(agent: Any, prompt_builder: Any, interactions: Any) -> None:
    agent.prompt_builder = copy.deepcopy(prompt_builder)
    if hasattr(agent, "_llm_interactions") and interactions is not None:
        agent._llm_interactions = copy.deepcopy(interactions)


def _prompt_obs(obs: dict[str, Any], extra_user_text: str) -> dict[str, Any]:
    result = copy.deepcopy(obs)
    if extra_user_text:
        text = result.setdefault("text", {})
        context = str(text.get("long_term_context", "") or "")
        text["long_term_context"] = (
            f"{context}\n\n{extra_user_text}" if context else extra_user_text
        )
    return result


def _last_interaction(agent: Any) -> dict[str, Any]:
    if not hasattr(agent, "get_last_llm_interaction"):
        return {}
    value = agent.get_last_llm_interaction()
    return value if isinstance(value, dict) else {}


def _attempt_record(
    attempt_idx: int,
    leaf_error: str,
    wait_seconds: float,
    *,
    response: Any = None,
    raw_output: str = "",
    model_action: str = "",
    executed_action: str = "",
    action_defaulted: bool = False,
    exception: Exception | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt_idx": attempt_idx,
        "leaf_error": leaf_error,
        "raw_stop_reason": str(getattr(response, "stop_reason", "") or ""),
        "raw_output": raw_output,
        "parsed_action": model_action,
        "executed_action": executed_action,
        "action_defaulted": action_defaulted,
        "wait_seconds": wait_seconds,
        "input_tokens": int(getattr(response, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(response, "output_tokens", 0) or 0),
    }
    if exception is not None:
        record["exception_type"] = (
            f"{exception.__class__.__module__}.{exception.__class__.__name__}"
        )
        record["exception_message"] = str(exception)
    return record


def _abort_fields(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    history = [str(attempt.get("leaf_error") or "") for attempt in attempts]
    return {
        "abort_reason": "retry_exhausted",
        "abort_leaf_history": history,
        "abort_leaf_counts": dict(Counter(history)),
        "abort_leaf_last": history[-1] if history else "",
    }


def run_agent_action(
    *,
    agent: Any,
    obs: dict[str, Any],
    prev_action: str | None,
    validate_action=None,
    extra_user_text: str = "",
    invalid_action_retry_notice: str = DEFAULT_INVALID_ACTION_RETRY_NOTICE,
    agent_method_name: str = "act",
    agent_method_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate and validate an action with the paper evaluator's retry policy.

    client.max_retries is the number of additional same-step attempts. A value
    of one permits two model generations before retry_exhausted.
    """

    client = getattr(agent, "client", None)
    max_retries = int(getattr(client, "max_retries", 0) or 0)
    delay = float(getattr(client, "delay", 0) or 0)
    prompt_snapshot = copy.deepcopy(agent.prompt_builder)
    interactions_snapshot = (
        agent.get_llm_interactions() if hasattr(agent, "get_llm_interactions") else None
    )
    attempts: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    request_count = 0
    retry_notice = ""
    method_kwargs = dict(agent_method_kwargs or {})

    for attempt_idx in range(1, max_retries + 2):
        _reset_agent(agent, prompt_snapshot, interactions_snapshot)
        extra = str(extra_user_text or "").strip()
        if retry_notice:
            extra = f"{extra}\n\n{retry_notice}" if extra else retry_notice

        try:
            response = getattr(agent, agent_method_name)(
                _prompt_obs(obs, extra),
                prev_action=prev_action,
                **method_kwargs,
            )
        except Exception as exc:
            leaf_error = classify_retryable_exception_leaf(exc)
            if leaf_error is None:
                raise
            request_count += 1
            exhausted = attempt_idx > max_retries
            raw_output = str(getattr(exc, "partial_output", "") or "")
            attempts.append(
                _attempt_record(
                    attempt_idx,
                    leaf_error,
                    0.0 if exhausted else delay,
                    raw_output=raw_output,
                    exception=exc,
                )
            )
            if exhausted:
                _reset_agent(agent, prompt_snapshot, interactions_snapshot)
                payload = {
                    "ok": False,
                    "response": None,
                    "interaction": {},
                    "raw_output": raw_output,
                    "model_action": "",
                    "executed_action": "",
                    "action_defaulted": False,
                    "thought": "",
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "request_count": request_count,
                    "retry_attempts": attempts,
                    "abort_stop_reason": "",
                    "prompt_builder": copy.deepcopy(agent.prompt_builder),
                }
                payload.update(_abort_fields(attempts))
                return payload
            if delay:
                time.sleep(delay)
            continue

        request_count += 1
        interaction = _last_interaction(agent)
        response_info = interaction.get("response", {}) if interaction else {}
        model_action = str(getattr(response, "completion", "") or "").strip()
        raw_output = str(
            response_info.get("raw_completion")
            if response_info.get("raw_completion") is not None
            else getattr(response, "completion", "") or ""
        )
        thought = str(getattr(response, "reasoning", "") or "")
        input_tokens = int(getattr(response, "input_tokens", 0) or 0)
        output_tokens = int(getattr(response, "output_tokens", 0) or 0)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        leaf_error = classify_llm_response_abort_reason(response)
        executed_action = model_action
        action_defaulted = False
        # Even an otherwise aborted response can contain malformed action text.
        # Keep the environment boundary safe by applying the same validation
        # and default-action policy before returning an exhausted retry.
        if validate_action is not None:
            executed_action = str(validate_action(model_action) or "")
            action_defaulted = executed_action != model_action
            if leaf_error is None and action_defaulted:
                leaf_error = "invalid_action"

        if leaf_error is None:
            return {
                "ok": True,
                "response": response,
                "interaction": interaction,
                "raw_output": raw_output,
                "model_action": model_action,
                "executed_action": executed_action,
                "action_defaulted": action_defaulted,
                "thought": thought,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "request_count": request_count,
                "retry_attempts": attempts,
                "abort_reason": None,
                "abort_stop_reason": "",
                "prompt_builder": copy.deepcopy(agent.prompt_builder),
            }

        exhausted = attempt_idx > max_retries
        attempts.append(
            _attempt_record(
                attempt_idx,
                leaf_error,
                0.0 if exhausted else delay,
                response=response,
                raw_output=raw_output,
                model_action=model_action,
                executed_action=executed_action,
                action_defaulted=action_defaulted,
            )
        )
        if leaf_error == "invalid_action":
            retry_notice = invalid_action_retry_notice
        if exhausted:
            payload = {
                "ok": False,
                "response": response,
                "interaction": interaction,
                "raw_output": raw_output,
                "model_action": model_action,
                "executed_action": executed_action,
                "action_defaulted": action_defaulted,
                "thought": thought,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "request_count": request_count,
                "retry_attempts": attempts,
                "abort_stop_reason": str(getattr(response, "stop_reason", "") or ""),
                "prompt_builder": copy.deepcopy(prompt_snapshot),
            }
            _reset_agent(agent, prompt_snapshot, interactions_snapshot)
            payload.update(_abort_fields(attempts))
            return payload
        if delay:
            time.sleep(delay)

    raise RuntimeError("unreachable retry loop state")
