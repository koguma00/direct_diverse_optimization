"""Thought/action parsing used by the paper's BALROG adapter."""

from __future__ import annotations

import re
from typing import Any


THOUGHT_ACTION_INSTRUCTION = """
Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>
""".strip()

BABA_STRICT_THOUGHT_ACTION_INSTRUCTION = """
Your response should use the following format:

Thought: <one short thought>
Action: <exactly one of: idle, up, right, down, left>

Rules:
- The Action line must be exactly one of: idle, up, right, down, left.
- Do not add any other words, punctuation, or explanation on the Action line.
- Output exactly one Thought line and exactly one Action line.
- Do not output anything after the Action line.
""".strip()

BABA_SYSTEM_PROMPT_PREFIX = "Baba Is You is a puzzle game"


def _normalize_candidate(candidate: str) -> str:
    text = (candidate or "").strip()
    if not text:
        return ""
    text = text.splitlines()[0].strip()
    text = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", text)
    text = text.strip().strip("`\"'")
    text = re.sub(r"\s*\([^()]*\)\s*$", "", text)
    text = re.split(r"\s+(?:--|//|#)\s*", text, maxsplit=1)[0].strip()
    return re.sub(r"\s+", " ", text).strip()


def extract_single_action(completion: str) -> str:
    """Extract one executable action from a free-form completion."""

    raw = (completion or "").strip()
    if not raw:
        return ""

    candidates: list[str] = []
    for match in re.findall(
        r"<\|ACTION\|>(.*?)<\|END\|>", raw, flags=re.IGNORECASE | re.DOTALL
    ):
        normalized = _normalize_candidate(match)
        if normalized:
            candidates.append(normalized)

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        action_match = re.search(r"(?i)action\s*:\s*(.+)$", stripped)
        if action_match:
            normalized = _normalize_candidate(action_match.group(1))
            if normalized:
                candidates.append(normalized)
            continue
        step_match = re.match(r"(?i)^step\s*\d+\s*:\s*(.+)$", stripped)
        if step_match:
            normalized = _normalize_candidate(step_match.group(1))
            if normalized:
                candidates.append(normalized)

    if candidates:
        return candidates[-1]

    non_empty_lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(non_empty_lines):
        normalized = _normalize_candidate(line)
        if normalized and len(normalized) <= 100 and not re.search(r"[.!?]", normalized):
            return normalized
    for line in non_empty_lines:
        normalized = _normalize_candidate(line)
        if normalized:
            return normalized
    return _normalize_candidate(raw)


def extract_thought_action(completion: str) -> tuple[str, str]:
    """Extract the final ``Thought:`` and ``Action:`` pair."""

    raw = (completion or "").strip()
    if not raw:
        return "", ""

    thought = ""
    matches = re.findall(
        r"(?is)thought\s*:\s*(.*?)(?=\n\s*action\s*:|\n\s*step\s*\d+\s*:|\n\s*<\|ACTION\|>|\Z)",
        raw,
    )
    if matches:
        thought = re.sub(
            r"<\|ACTION\|>.*?<\|END\|>",
            "",
            matches[-1].strip(),
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    actions: list[str] = []
    for line in raw.splitlines():
        match = re.search(r"(?i)^\s*action\s*:\s*(.+)$", line.strip())
        if match:
            normalized = _normalize_candidate(match.group(1))
            if normalized:
                actions.append(normalized)
    if actions:
        return thought, actions[-1]

    tagged = [
        _normalize_candidate(match)
        for match in re.findall(
            r"<\|ACTION\|>(.*?)<\|END\|>", raw, flags=re.IGNORECASE | re.DOTALL
        )
    ]
    tagged = [action for action in tagged if action]
    if tagged:
        return thought, tagged[-1]
    return thought, extract_single_action(raw)


def instruction_for_messages(messages: list[Any]) -> str:
    if messages:
        first = str(getattr(messages[0], "content", "") or "")
        if first.startswith(BABA_SYSTEM_PROMPT_PREFIX):
            return BABA_STRICT_THOUGHT_ACTION_INSTRUCTION
    return THOUGHT_ACTION_INSTRUCTION


def fixed_action_thought_instruction_for_messages(messages: list[Any], fixed_action: str) -> str:
    fixed_action = str(fixed_action or "").strip()
    if not fixed_action:
        raise ValueError("fixed_action must be a non-empty string")
    is_baba = bool(
        messages
        and str(getattr(messages[0], "content", "") or "").startswith(
            BABA_SYSTEM_PROMPT_PREFIX
        )
    )
    thought_hint = "<one short thought>" if is_baba else "<your thoughts>"
    return f"""
Your response should use the following format:

Thought: {thought_hint}

The current Action is already fixed:
Action: {fixed_action}

Rules:
- The Action is already fixed. Do not choose a different action.
- Output exactly one Thought line.
- Do not output an Action line.
- Do not output anything after the Thought line.
""".strip()


def extract_fixed_action_thought(raw_output: str) -> str:
    raw = str(raw_output or "").strip()
    if not raw:
        return ""
    thought, _ = extract_thought_action(raw)
    if thought:
        return thought.strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return ""
    match = re.match(r"(?i)^thought\s*:\s*(.*)$", lines[0])
    return match.group(1).strip() if match else lines[0]


def format_thought_action_completion(thought: str, action: str) -> str:
    thought = str(thought or "").strip()
    action = str(action or "").strip()
    if not action:
        raise ValueError("action must be a non-empty string")
    return f"Thought: {thought}\n\nAction: {action}" if thought else action
