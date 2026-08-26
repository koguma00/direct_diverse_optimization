"""Prompt history used for the paper's BALROG trajectories."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class Message:
    role: str
    content: str
    attachment: Any | None = None


class HistoryPromptBuilder:
    def __init__(
        self,
        max_text_history: int = 16,
        max_image_history: int = 0,
        system_prompt: str | None = None,
        max_cot_history: int = 1,
    ) -> None:
        self.max_text_history = max_text_history
        self.max_image_history = max_image_history
        self.max_history = max(max_text_history, max_image_history, 1)
        self.system_prompt = system_prompt
        self._events: deque[dict[str, Any]] = deque(maxlen=self.max_history * 2)
        self._last_short_term_obs: str | None = None
        self.previous_reasoning: str | None = None
        self.max_cot_history = max_cot_history

    def update_instruction_prompt(self, instruction: str) -> None:
        self.system_prompt = instruction

    def update_observation(self, obs: dict[str, Any]) -> None:
        text = obs.get("text") or {}
        self._last_short_term_obs = str(text.get("short_term_context", "") or "")
        self._events.append(
            {
                "type": "observation",
                "text": str(text.get("long_term_context", "") or ""),
                "image": obs.get("image"),
            }
        )

    def update_action(self, action: str) -> None:
        self._events.append(
            {"type": "action", "action": action, "reasoning": self.previous_reasoning}
        )

    def update_reasoning(self, reasoning: str) -> None:
        self.previous_reasoning = reasoning

    def reset(self) -> None:
        self._events.clear()
        self._last_short_term_obs = None
        self.previous_reasoning = None

    def get_prompt(self, icl_episodes: bool = False) -> list[Message]:
        messages: list[Message] = []
        if self.system_prompt and not icl_episodes:
            messages.append(Message(role="user", content=self.system_prompt))

        events = list(self._events)
        observations = [i for i, event in enumerate(events) if event["type"] == "observation"]
        if not observations:
            return messages
        selected = observations[-self.max_text_history :] if self.max_text_history > 0 else []
        image_observations = [i for i in observations if events[i].get("image") is not None]
        selected_images = set(
            image_observations[-self.max_image_history :]
            if self.max_image_history > 0
            else []
        )
        reasoning_actions = [
            i
            for i, event in enumerate(events)
            if event["type"] == "action" and event.get("reasoning") is not None
        ]
        selected_reasoning = set(
            reasoning_actions[-self.max_cot_history :] if self.max_cot_history > 0 else []
        )
        last_observation = observations[-1]

        for index in selected:
            if index > 0 and events[index - 1]["type"] == "action":
                action = events[index - 1]
                content = (
                    "Previous plan:\n" + str(action["reasoning"])
                    if index - 1 in selected_reasoning
                    else str(action["action"])
                )
                messages.append(Message(role="assistant", content=content))

            observation = events[index]
            parts = ["Current Observation:" if index == last_observation else "Observation:"]
            if index == last_observation and self._last_short_term_obs:
                parts.append(self._last_short_term_obs)
            parts.append(str(observation.get("text", "")))
            attachment = observation.get("image") if index in selected_images else None
            if attachment is not None:
                parts.append("Image observation provided.")
            messages.append(
                Message(role="user", content="\n".join(parts), attachment=attachment)
            )
        return messages
