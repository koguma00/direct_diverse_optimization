"""Thought/action agent from the original DDO experiments."""

from __future__ import annotations

import copy
from typing import Any

from .actions import (
    extract_fixed_action_thought,
    extract_thought_action,
    fixed_action_thought_instruction_for_messages,
    instruction_for_messages,
)
from .client import create_client
from .prompt import HistoryPromptBuilder


class ThoughtActionAgent:
    def __init__(self, client_factory, prompt_builder: HistoryPromptBuilder) -> None:
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self._llm_interactions: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.prompt_builder.reset()

    def clear_llm_interactions(self) -> None:
        self._llm_interactions = []

    def get_llm_interactions(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._llm_interactions)

    def get_last_llm_interaction(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._llm_interactions[-1]) if self._llm_interactions else None

    def _record(self, messages, response, *, parsed_action: str, source: str) -> None:
        self._llm_interactions.append(
            {
                "messages": [
                    {
                        "role": getattr(message, "role", ""),
                        "content": getattr(message, "content", ""),
                        "has_attachment": bool(getattr(message, "attachment", None)),
                    }
                    for message in messages
                ],
                "response": {
                    "raw_completion": getattr(response, "completion", ""),
                    "reasoning": getattr(response, "reasoning", "") or "",
                    "parsed_action": parsed_action,
                    "stop_reason": getattr(response, "stop_reason", ""),
                    "input_tokens": int(getattr(response, "input_tokens", 0) or 0),
                    "output_tokens": int(getattr(response, "output_tokens", 0) or 0),
                },
                "meta": {
                    "source": source,
                    "llm_options_applied": self.client.get_last_applied_options(),
                    "request_kwargs": self.client.get_last_request_kwargs(),
                },
            }
        )

    def act(self, obs: dict[str, Any], prev_action: str | None = None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + instruction_for_messages(messages)
        response = self.client.generate(messages)
        thought, action = extract_thought_action(response.completion)
        self._record(messages, response, parsed_action=action, source="agent.act")
        return response._replace(completion=action, reasoning=thought)

    def generate_fixed_action_thought(
        self,
        obs: dict[str, Any],
        prev_action: str | None = None,
        *,
        fixed_action: str,
    ):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + fixed_action_thought_instruction_for_messages(
                messages, fixed_action
            )
        response = self.client.generate(messages)
        thought = extract_fixed_action_thought(response.completion)
        self._record(
            messages,
            response,
            parsed_action=fixed_action,
            source="agent.fixed_action_thought",
        )
        return response._replace(completion=fixed_action, reasoning=thought)


def create_agent(config: Any) -> ThoughtActionAgent:
    agent = config.agent
    if str(agent.type) != "thought_action":
        raise ValueError(f"DDO BALROG adapter supports agent.type=thought_action, got {agent.type}")
    prompt = HistoryPromptBuilder(
        max_text_history=int(agent.max_text_history),
        max_image_history=int(getattr(agent, "max_image_history", 0)),
        max_cot_history=int(getattr(agent, "max_cot_history", 1)),
    )
    return ThoughtActionAgent(create_client(config.client), prompt)
