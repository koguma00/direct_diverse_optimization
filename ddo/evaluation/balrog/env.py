"""BabyAI/BabaIsAI environment adapter used by DDO."""

from __future__ import annotations

from collections import defaultdict
from typing import Any
import warnings

import gym

# BALROG imports MiniGrid registration modules in every worker. Gymnasium emits
# one warning per already-registered environment, which can overwhelm the actual
# collection progress without indicating a functional problem.
warnings.filterwarnings(
    "ignore", message=r".*Overriding environment .* already in registry.*"
)


BABYAI_ACTION_SPACE = [
    "turn left",
    "turn right",
    "go forward",
    "pick up",
    "drop",
    "toggle",
]
BABAISAI_ACTION_SPACE = ["idle", "up", "right", "down", "left"]
BABAISAI_LOSS_NOTICE = "The active YOU rule was broken. Episode lost."
BABAISAI_LOSS_OBSERVATION = "No controllable YOU object remains on the map."


class DDOEnvWrapper:
    """Small stable surface over the official BALROG environment wrappers."""

    def __init__(self, env: Any, env_name: str, task_name: str) -> None:
        self.env = env
        self.env_name = env_name
        self.task_name = task_name
        self.failed_candidates: list[Any] = []

    @property
    def max_steps(self):
        return self.env.max_steps

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        return result if isinstance(result, tuple) else (result, {})

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            return result
        obs, reward, done, info = result
        return obs, reward, done, False, info

    def check_action_validity(self, candidate_action):
        candidate = str(candidate_action or "").strip()
        if candidate in self.env.language_action_space:
            return candidate
        self.failed_candidates.append(candidate_action)
        return self.env.default_action

    def get_stats(self):
        return self.env.get_stats()

    def close(self) -> None:
        self.env.close()

    def get_instruction_prompt(self, instructions: str | None = None) -> str:
        return instruction_prompt(self.env_name, mission=instructions)


def instruction_prompt(env_name: str, *, mission: str | None = None) -> str:
    if env_name == "babyai":
        actions = {
            "turn left": "turn to the left",
            "turn right": "turn to the right",
            "go forward": "take one step forward",
            "pick up": "pick up the object directly in front of you (1 step forward)",
            "drop": "place the object you are carrying on an empty tile directly in front of you (1 step forward)",
            "toggle": "interact with the object directly in front of you; effects depend on object and state (1 step forward)",
        }
        action_text = ",\n".join(f"{action}: {description}" for action, description in actions.items())
        return f"""
You are an agent playing a simple navigation game. Your goal is to {mission}. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_text}.

In a moment I will present you an observation.

Tips:
- Use 'pick up' to collect carryable items directly in front of you (1 step forward).
- Use 'toggle' to interact with the object directly in front of you.
- Use 'drop' to place the carried object on an empty tile directly in front of you (1 step forward).
- Some actions may have no effect if their preconditions are not met (for example, blocked front tile or no carried object).
- Avoid repeating the same action over and over if the observation doesn't change.

PLAY!
""".strip()

    action_text = ",\n".join(
        [
            "idle: wait for one step",
            "up: take one step up",
            "right: take one step to the right",
            "down: take one step down",
            "left: take one step to the left",
        ]
    )
    return f"""
Baba Is You is a puzzle game where you can manipulate the rules of each level. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_text}.

Tips:
- Examine the level carefully, noting all objects and text blocks present.
- Identify the current rules, which are formed by text blocks in the format "[Subject] IS [Property]" (e.g. "BABA IS YOU").
- Consider how you can change or create new rules by moving text blocks around.
- Remember that you can only move objects or text that are not defined as "STOP" or similar immovable properties.
- Your goal is usually to reach an object defined as "WIN", but this can be changed.
- Think creatively about how changing rules can alter the properties and behaviors of objects in unexpected ways.
- If stuck, try breaking apart existing rules or forming completely new ones.
- Sometimes the solution involves making yourself a different object or changing what counts as the win condition.

PLAY!
""".strip()


def make_env(env_name: str, task: str, config: Any, *, env_seed: int | None = None):
    if env_name == "babyai":
        from balrog.environments.babyai_text.babyai_env import make_babyai_env

        base = make_babyai_env(env_name, task, config)
    elif env_name == "babaisai":
        from baba import make

        kwargs = dict(config.envs.babaisai_kwargs)
        base = BabaIsAIWrapper(make(task, **kwargs), add_ruleset=kwargs.get("add_ruleset", True))
        if env_seed is not None:
            base.seed(env_seed)
    else:
        raise ValueError(f"unsupported DDO BALROG environment: {env_name}")
    return DDOEnvWrapper(base, env_name, task)


class BabaIsAIWrapper(gym.Wrapper):
    """Paper behavior: losing the active YOU rule terminates instead of resetting."""

    def __init__(self, env: Any, add_ruleset: bool = True, vlm: bool = False) -> None:
        del vlm
        super().__init__(env)
        self.add_ruleset = add_ruleset
        self.language_action_space = BABAISAI_ACTION_SPACE[:]
        self.progression = 0.0
        self.target_plan = None
        self._pending_seed: int | None = None

    @property
    def default_action(self):
        return BABAISAI_ACTION_SPACE[0]

    def seed(self, seed=None):
        self._pending_seed = seed
        return [seed]

    def get_ruleset(self) -> str:
        from baba.world_object import name_mapping

        rules = []
        for rule in self.env.grid._ruleset["_rule_"]:
            if "object" not in rule:
                continue
            rules.append(
                f"{rule['object'].removeprefix('f')} is {name_mapping[rule['property']]}"
            )
        return "\n".join(rules)

    def get_text_observation(self) -> tuple[str, bool]:
        import numpy as np
        from baba.world_object import name_mapping

        def find_objects(objects):
            found = []
            for y in range(self.env.height):
                for x in range(self.env.width):
                    cell = self.env.grid.get(x, y)
                    if cell is None or cell.type not in objects:
                        continue
                    if cell.type == "rule_object":
                        name = f"rule `{cell.name}`"
                    elif cell.type == "rule_is":
                        name = f"rule `{name_mapping[cell.name]}`"
                    elif cell.type == "rule_property":
                        name = f"rule `{name_mapping[cell.property]}`"
                    else:
                        name = cell.type
                    found.append(((x, y), name))
            return found

        you = None
        for rule in self.env.grid._ruleset["_rule_"]:
            if "property" in rule and name_mapping[rule["property"]] == "you":
                you = rule["object"]
        player = find_objects([you])
        if not player:
            return BABAISAI_LOSS_OBSERVATION, True
        others = find_objects(
            ["fball", "fwall", "fdoor", "fkey", "rule_object", "rule_is", "rule_property"]
        )
        origin = np.asarray(player[0][0])
        lines = []
        for position, name in others:
            x, y = np.asarray(position) - origin
            parts = []
            if x:
                parts.append(f"{abs(x)} {'step' if abs(x) == 1 else 'steps'} to the {'right' if x > 0 else 'left'}")
            if y:
                parts.append(f"{abs(y)} {'step' if abs(y) == 1 else 'steps'} {'down' if y > 0 else 'up'}")
            if parts:
                lines.append(f"{name.removeprefix('f')} " + " and ".join(parts))
        return "\n".join(lines), False

    def _process(self, obs):
        del obs
        from PIL import Image

        text, lost = self.get_text_observation()
        sections = [BABAISAI_LOSS_NOTICE] if lost else []
        if self.add_ruleset:
            sections.append(f"Active rules:\n{self.get_ruleset()}")
        sections.append(f"Objects on the map:\n{text}")
        result = defaultdict(lambda: None)
        result["text"] = {"long_term_context": "\n\n".join(sections), "short_term_context": ""}
        result["image"] = Image.fromarray(self.env.render(mode="rgb_array")).convert("RGB")
        return result, lost

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", None)
        if seed is None:
            seed = self._pending_seed
        self._pending_seed = None
        if seed is not None:
            kwargs["seed"] = seed
        obs = self.env.reset(**kwargs)
        self.target_plan = self.env.target_plan
        self.progression = 0.0
        return self._process(obs)[0]

    def step(self, action):
        action_int = self.language_action_space.index(action)
        obs, reward, done, info = self.env.step(action_int)
        processed, lost = self._process(obs)
        if lost:
            info = dict(info or {})
            info["lost"] = True
            info.setdefault("won", False)
            info.setdefault("feedback", BABAISAI_LOSS_NOTICE)
            self.progression = 0.0
            return processed, -1.0, True, info
        if done:
            self.progression = 1.0 if reward > 0 else 0.0
        return processed, reward, done, info

    def get_stats(self):
        return {"target_plan": self.target_plan, "progression": self.progression}
