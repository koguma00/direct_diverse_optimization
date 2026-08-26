#!/usr/bin/env python3
"""Shared WebShop pilot utilities.

The module keeps WebShop-specific logic independent from the BALROG collector
while preserving the same trace shape consumed by P1 dataset builders.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_WEBSHOP_ROOT = REPO_ROOT / "benchmarks" / "WebShop"
if str(UPSTREAM_WEBSHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_WEBSHOP_ROOT))

CSV_COLUMNS = [
    "step",
    "instruction",
    "observation_pre",
    "thought",
    "action_model",
    "action_executed",
    "raw_output",
    "feedback",
    "observation_post",
    "obs_changed",
    "reward",
    "progression",
    "terminated",
    "truncated",
    "done",
    "termination_reason",
    "action_defaulted",
    "input_tokens",
    "output_tokens",
    "step_wall_time_sec",
    "won",
    "lost",
]

WEBSHOP_SUCCESS_THRESHOLD = 0.9

THOUGHT_ACTION_INSTRUCTION = """
Output exactly two lines and nothing else:

Thought: <10 words or fewer>
Action: <one valid action>
""".strip()

WEBSHOP_SYSTEM_PROMPT = """
You are a sequential WebShop text-environment agent.
Your mission is to satisfy the shopping instruction by navigating pages,
selecting required options, and buying the best matching product.

Executable actions are environment actions. At each decision step, output
exactly one action and no ordinary prose beyond the required Thought/Action
format. The terminal successful artifact is the environment transition caused
by click[buy now]; do not submit a separate final answer.

Valid action forms:
- search[keywords]: search from the search page only.
- click[value]: click a visible button, product id, navigation control, option,
  or click[buy now].

Action protocol:
- If an Available actions list is shown and search[...] actions are present,
  choose one listed search[...] or write a concise custom search[...] from the
  shopping instruction.
- Otherwise choose exactly one listed click[...] action.
- Do not output bare labels like next, product IDs, option names, or buy now;
  wrap them exactly as click[value].

Tips:
- Search with a short product-type query first; do not paste the full
  instruction or include words like "instruction", "find", "price", "lower",
  "color", or "size" unless they are part of the product name.
- On result pages, click a plausible product id or click[next >]; use
  click[back to search] only when changing the query.
- On product pages, choose required options such as color, size, quantity, or
  pack before buying.
- Use click[buy now] only when the selected product and options satisfy the
  instruction and price constraint.
- Do not repeat the same search/back/search loop; choose a product, go next, or
  use a different valid query.
""".strip()

WEBSHOP_INVALID_ACTION_RETRY_NOTICE = (
    "Your previous output did not contain a valid WebShop action for the current page. "
    "Retry the same step with exactly two short lines. Choose only from the current "
    "Available actions list. If no search[...] action is shown, do not output search[...]; "
    "use click[back to search] first if you need a new search."
)

ALT_MODE_RANDOM = "random"
ALT_MODE_REQUEST = "request"
ALT_MODE_RANDOM_TEACHER = "random_teacher"
ALT_MODE_STRUCTURED = "structured"
SUPPORTED_ALT_MODES = (
    ALT_MODE_RANDOM,
    ALT_MODE_REQUEST,
    ALT_MODE_RANDOM_TEACHER,
    ALT_MODE_STRUCTURED,
)

SEED_MODE_FIXED = "fixed"
SEED_MODE_PER_EPISODE = "per_episode"
SUPPORTED_SEED_MODES = (SEED_MODE_FIXED, SEED_MODE_PER_EPISODE)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "find",
    "instruction",
    "lower",
    "than",
    "the",
    "to",
    "with",
    "dollar",
    "dollars",
    "price",
}


@dataclass(frozen=True)
class EpisodePaths:
    run_dir: Path
    task_dir: Path
    run_stem: str
    csv_path: Path
    json_path: Path
    trace_path: Path


@dataclass(frozen=True)
class WebShopArtifact:
    task_id: str
    episode_id: str
    episode_idx: int
    session_id: int
    seed: int
    csv_path: Path
    json_path: Path
    trace_path: Path
    episode_log: dict[str, Any]
    trace_payload: dict[str, Any]
    success_threshold: float

    @property
    def calls(self) -> list[dict[str, Any]]:
        calls = self.trace_payload.get("calls") or []
        return calls if isinstance(calls, list) else []

    @property
    def total_steps(self) -> int:
        return len(self.calls)

    @property
    def success(self) -> bool:
        return terminal_reward(self.trace_payload) >= self.success_threshold


def build_client_config(args: argparse.Namespace) -> SimpleNamespace:
    generate_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.llm_seed,
    }
    if getattr(args, "vllm_disable_thinking", False):
        generate_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    return SimpleNamespace(
        client_name=args.client_name,
        model_id=args.model_id,
        base_url=args.base_url,
        timeout=args.client_timeout,
        max_retries=args.client_max_retries,
        delay=args.client_delay,
        alternate_roles=False,
        generate_kwargs=generate_kwargs,
    )


def create_agent(args: argparse.Namespace) -> Any:
    from ddo.evaluation.balrog.actions import extract_thought_action
    from ddo.evaluation.balrog.agent import ThoughtActionAgent
    from ddo.evaluation.balrog.client import create_client
    from ddo.evaluation.balrog.prompt import HistoryPromptBuilder

    class WebShopThoughtActionAgent(ThoughtActionAgent):
        """Thought/action agent with the WebShop-specific output instruction."""

        def act(self, obs: dict[str, Any], prev_action: str | None = None):
            if prev_action:
                self.prompt_builder.update_action(prev_action)

            self.prompt_builder.update_observation(obs)
            messages = self.prompt_builder.get_prompt()
            if messages and messages[-1].role == "user":
                messages[-1].content += "\n\n" + THOUGHT_ACTION_INSTRUCTION

            response = self.client.generate(messages)
            thought, action = extract_thought_action(response.completion)
            final_response = response._replace(completion=action, reasoning=thought)
            self._record(messages, response, parsed_action=action, source="webshop.agent.act")
            return final_response

    prompt_builder = HistoryPromptBuilder(
        max_text_history=args.max_text_history,
        max_image_history=0,
        max_cot_history=1,
    )
    return WebShopThoughtActionAgent(create_client(build_client_config(args)), prompt_builder)


def make_prompt_obs(observation: str, available: list[str] | None = None) -> dict[str, Any]:
    context = str(observation or "")
    if available:
        available_set = set(available)
        context = (
            context
            + "\n\nAvailable actions for this state:\n"
            + "\n".join(f"- {action}" for action in available)
        )
        has_search = any(action.startswith("search[") for action in available_set)
        if has_search:
            context += (
                "\n\nThis is the search page. Use one listed search[...] or a concise custom "
                "search[...] from the shopping instruction. Do not include words like "
                "'instruction' or 'find' in the query."
            )
        else:
            context += "\n\nChoose exactly one click[...] action from this list. Do not output search[...]."
        if not has_search and "click[back to search]" in available_set:
            context += "\nSearch is not available on this page; choose click[back to search] before a new search."
    return {
        "text": {
            "long_term_context": context,
            "short_term_context": "",
        },
        "image": None,
    }


def normalize_observation(observation: str) -> str:
    return re.sub(r"\s+", " ", str(observation or "")).strip()


def normalize_action(action: str) -> str:
    text = re.sub(r"\s+", " ", str(action or "").strip()).strip("`\"'")
    click_match = re.match(r"^click\[(.*)\]$", text, flags=re.IGNORECASE)
    if click_match:
        return f"click[{click_match.group(1).strip().lower()}]"
    search_match = re.match(r"^search\[(.*)\]$", text, flags=re.IGNORECASE)
    if search_match:
        return f"search[{search_match.group(1).strip()}]"
    return text


def _dedupe_ordered(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_action(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


OPTION_FIELD_PATTERN = re.compile(
    r"(?:,?\s*(?:and\s+)?(?:with\s+)?)"
    r"(?:color|size|fit\s+type|special\s+size|item\s+shape|flavor\s+name|"
    r"scent|style|pattern|material\s+type|item\s+package\s+quantity|"
    r"number\s+of\s+items|quantity)\s*:\s*[^,]+",
    flags=re.IGNORECASE,
)


def _searchable_instruction_text(instruction: str) -> str:
    text = str(instruction or "").lower()
    text = re.sub(r"^\s*instruction\s*:\s*", "", text)
    text = re.sub(r"\bfind\s+me\b", " ", text)
    text = re.sub(r"\bprice\s+lower\s+than\b[^,]+", " ", text)
    text = re.sub(r"\blower\s+than\b[^,]+", " ", text)
    text = OPTION_FIELD_PATTERN.sub(" ", text)
    return text


def _instruction_terms(instruction: str) -> list[str]:
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", _searchable_instruction_text(instruction))
    terms: list[str] = []
    for term in text.split():
        if term in STOPWORDS:
            continue
        if term.isdigit():
            continue
        if len(term) > 2 or term == "x":
            terms.append(term)
    return terms


def generate_search_queries(instruction: str, *, max_queries: int = 6) -> list[str]:
    terms = _instruction_terms(instruction)
    if not terms:
        return []

    candidates = [
        " ".join(terms[:8]),
        " ".join(terms[-8:]),
        " ".join(terms[:5]),
        " ".join(terms[-5:]),
    ]
    if len(terms) >= 3:
        candidates.extend(" ".join(terms[idx : idx + 3]) for idx in range(0, len(terms) - 2, 2))
    return _dedupe_ordered([candidate for candidate in candidates if candidate])[:max_queries]


def available_actions(env: Any, instruction: str, *, max_search_queries: int = 6) -> list[str]:
    info = env.get_available_actions()
    actions: list[str] = []
    for clickable in info.get("clickables", []) or []:
        text = normalize_action(str(clickable).lower())
        if text and text != "search":
            actions.append(f"click[{text}]")
    if info.get("has_search_bar"):
        actions.extend(f"search[{query}]" for query in generate_search_queries(instruction, max_queries=max_search_queries))
    return _dedupe_ordered(actions)


def validate_action_for_env(env: Any, instruction: str, *, max_search_queries: int = 6) -> Callable[[str], str]:
    def _validate(action: str) -> str:
        normalized = normalize_action(action)
        info = env.get_available_actions()
        if info.get("has_search_bar") and re.match(r"^search\[[^\]]+\]$", normalized):
            return normalized
        return normalized if normalized in set(available_actions(env, instruction, max_search_queries=max_search_queries)) else ""

    return _validate


def stable_int(parts: list[str]) -> int:
    digest = hashlib.sha1("\n<SEP>\n".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def state_key(task_id: str, session_id: int, prefix_actions: list[str]) -> str:
    return hashlib.sha1(
        json.dumps(
            {
                "task_id": str(task_id),
                "session_id": int(session_id),
                "prefix_actions": list(prefix_actions),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def deterministic_alt_actions(
    *,
    actions: list[str],
    base_action: str,
    alt_budget: int,
    sampling_key: str,
) -> list[str]:
    candidates = [action for action in _dedupe_ordered(actions) if action != normalize_action(base_action)]
    if not candidates:
        return []
    target = len(candidates) if alt_budget <= 0 else min(int(alt_budget), len(candidates))
    rng_seed = stable_int(["webshop_random_alt", sampling_key, normalize_action(base_action), str(target)])
    import random

    rng = random.Random(rng_seed)
    return rng.sample(candidates, k=target)


def webshop_action_type(action: str) -> str:
    """Classify an executable WebShop action for structured branch sampling."""

    normalized = normalize_action(action)
    if normalized.startswith("search["):
        return "search"
    if not normalized.startswith("click[") or not normalized.endswith("]"):
        return "other"
    value = normalized[len("click[") : -1]
    if value == "buy now":
        return "buy_now"
    if value in {"next >", "< prev", "back to search"}:
        return "navigation"
    if value in {"description", "features", "reviews"}:
        return "info_tab"
    if value.isalnum() and len(value) == 10:
        return "product_click"
    return "option_click"


def structured_alt_actions(
    *,
    actions: list[str],
    base_action: str,
    alt_budget: int,
    sampling_key: str,
) -> list[str]:
    """Sample same-type alternatives first, then fill from other action types."""

    candidates = [
        action
        for action in _dedupe_ordered(actions)
        if action != normalize_action(base_action)
    ]
    if not candidates:
        return []
    target = len(candidates) if alt_budget <= 0 else min(int(alt_budget), len(candidates))
    base_type = webshop_action_type(base_action)
    type_order = [
        "search",
        "product_click",
        "option_click",
        "info_tab",
        "navigation",
        "buy_now",
        "other",
    ]
    if base_type == "buy_now":
        priority = ["option_click", "info_tab", "navigation", "product_click", "search", "other"]
    else:
        priority = [base_type, *(kind for kind in type_order if kind != base_type)]

    selected: list[str] = []
    import random

    for action_type in priority:
        group = [
            action
            for action in candidates
            if action not in selected and webshop_action_type(action) == action_type
        ]
        if not group:
            continue
        remaining = target - len(selected)
        if remaining <= 0:
            break
        rng = random.Random(
            stable_int(
                [
                    "webshop_structured_alt",
                    sampling_key,
                    normalize_action(base_action),
                    action_type,
                ]
            )
        )
        selected.extend(rng.sample(group, k=min(remaining, len(group))))
    return selected


def make_webshop_env(*, num_products: int | None, human_goals: int, show_attrs: bool, file_path: str | None):
    # WebShop is a CPU text environment. Its Pyserini/JVM import followed by
    # spaCy optional CuPy probe can crash the interpreter, even though CuPy is
    # never used here. Hide only the optional probe while importing the pinned
    # upstream environment, then restore the process module table.
    missing = object()
    optional_gpu_modules = {
        name: sys.modules.get(name, missing) for name in ("cupy", "cupyx")
    }
    for name in optional_gpu_modules:
        sys.modules[name] = None
    try:
        from web_agent_site.envs import WebAgentTextEnv
        from web_agent_site.engine import engine as webshop_engine
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "WebShop is not ready. Clone WebShop into benchmarks/WebShop and run "
            "the data and indexing commands in README.md "
            "before using DDO WebShop entrypoints."
        ) from exc
    finally:
        for name, module in optional_gpu_modules.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    resolved_file_path = file_path
    use_full_catalog = not file_path and (
        num_products is None or num_products <= 0 or num_products > 1000
    )
    if use_full_catalog:
        full_products = UPSTREAM_WEBSHOP_ROOT / "data" / "items_shuffle.json"
        full_attributes = UPSTREAM_WEBSHOP_ROOT / "data" / "items_ins_v2.json"
        missing_paths = [path for path in (full_products, full_attributes) if not path.is_file()]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Full WebShop catalog is required but missing: {missing}")
        resolved_file_path = str(full_products)
        webshop_engine.DEFAULT_ATTR_PATH = str(full_attributes)

    kwargs: dict[str, Any] = {
        "observation_mode": "text",
        "human_goals": int(human_goals),
        "show_attrs": bool(show_attrs),
    }
    if num_products is not None and num_products > 0:
        kwargs["num_products"] = int(num_products)
    if resolved_file_path:
        kwargs["file_path"] = str(resolved_file_path)
    return WebAgentTextEnv(**kwargs)


def reset_env(env: Any, session_id: int) -> str:
    reset_result = env.reset(session=int(session_id))
    if isinstance(reset_result, tuple):
        return str(reset_result[0])
    return str(reset_result)


def step_env(env: Any, action: str) -> tuple[str, float, bool, dict[str, Any]]:
    result = env.step(action)
    if len(result) == 4:
        observation, reward, done, info = result
        return str(observation), float(reward or 0.0), bool(done), dict(info or {})
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return str(observation), float(reward or 0.0), bool(terminated or truncated), dict(info or {})
    raise ValueError(f"Unsupported WebShop step result length: {len(result)}")


def instruction_text(env: Any, fallback: str = "") -> str:
    if hasattr(env, "instruction_text"):
        text = getattr(env, "instruction_text")
        if text:
            return str(text)
    if hasattr(env, "get_instruction_text"):
        try:
            text = env.get_instruction_text()
            if text:
                return str(text)
        except Exception:
            pass
    return str(fallback or "")


def session_snapshot(env: Any) -> dict[str, Any]:
    session_id = str(getattr(env, "session", "") or "")
    server = getattr(env, "server", None)
    user_sessions = getattr(server, "user_sessions", {}) if server is not None else {}
    session = user_sessions.get(session_id, {}) if isinstance(user_sessions, dict) else {}
    options = session.get("options", {}) if isinstance(session, dict) else {}
    actions = session.get("actions", {}) if isinstance(session, dict) else {}
    return {
        "session": session_id,
        "asin": str(session.get("asin") or "") if isinstance(session, dict) else "",
        "options": dict(options) if isinstance(options, dict) else {},
        "actions": dict(actions) if hasattr(actions, "items") else {},
        "reward": float(session.get("reward", 0.0) or 0.0) if isinstance(session, dict) else 0.0,
        "done": bool(session.get("done", False)) if isinstance(session, dict) else False,
        "verbose_info": copy.deepcopy(session.get("verbose_info")) if isinstance(session, dict) else None,
    }


def purchase_signature_from_snapshot(snapshot: dict[str, Any]) -> str:
    asin = str(snapshot.get("asin") or "").strip().upper()
    options = snapshot.get("options") if isinstance(snapshot.get("options"), dict) else {}
    if not asin:
        return ""
    return json.dumps({"asin": asin, "options": dict(sorted(options.items()))}, sort_keys=True)


def terminal_reward(trace_payload: dict[str, Any]) -> float:
    calls = trace_payload.get("calls") or []
    if not calls:
        return 0.0
    last_call = calls[-1]
    reward = float(last_call.get("reward", 0.0) or 0.0)
    progression = float(last_call.get("progression", 0.0) or 0.0)
    return max(reward, progression)


def trace_success(trace_payload: dict[str, Any], success_threshold: float) -> bool:
    return terminal_reward(trace_payload) >= float(success_threshold)


def step_success(*, reward: float, progression: float, success_threshold: float) -> bool:
    return max(float(reward or 0.0), float(progression or 0.0)) >= float(success_threshold)


def shard_bounds(*, start: int, end: int, num_workers: int, worker_index: int) -> tuple[int, int]:
    start = int(start)
    end = int(end)
    num_workers = int(num_workers)
    worker_index = int(worker_index)
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError("worker_index must satisfy 0 <= worker_index < num_workers")
    if end < start:
        raise ValueError("episode end must be >= episode start")

    total = end - start
    base = total // num_workers
    remainder = total % num_workers
    offset = worker_index * base + min(worker_index, remainder)
    count = base + (1 if worker_index < remainder else 0)
    return start + offset, start + offset + count


def shard_items(items: list[Any], *, num_workers: int, worker_index: int) -> list[Any]:
    shard_start, shard_end = shard_bounds(
        start=0,
        end=len(items),
        num_workers=num_workers,
        worker_index=worker_index,
    )
    return list(items[shard_start:shard_end])


def action_type_signature(actions: list[str]) -> str:
    types = []
    for action in actions:
        if action.startswith("search["):
            types.append("search")
        elif action.startswith("click["):
            value = action[len("click[") : -1] if action.endswith("]") else action
            if value in {"buy now", "next >", "< prev", "back to search"}:
                types.append(f"click:{value}")
            elif value.isalnum() and len(value) == 10:
                types.append("click:product")
            else:
                types.append("click:option")
        else:
            types.append("other")
    return " > ".join(types)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def discover_artifacts(run_dir: Path, *, success_threshold: float) -> list[WebShopArtifact]:
    artifacts: list[WebShopArtifact] = []
    for trace_path in sorted(Path(run_dir).glob("**/*_llm_trace.json")):
        if "_dtc" in trace_path.parts or "__dtc_" in trace_path.name:
            continue
        json_path = trace_path.with_name(trace_path.name.replace("_llm_trace.json", ".json"))
        csv_path = trace_path.with_name(trace_path.name.replace("_llm_trace.json", ".csv"))
        if not json_path.exists() or not csv_path.exists():
            continue
        trace_payload = read_json(trace_path)
        if str(trace_payload.get("env_name") or "") != "webshop":
            continue
        episode_log = read_json(json_path)
        artifacts.append(
            WebShopArtifact(
                task_id=str(trace_payload.get("task_or_env_params") or "webshop"),
                episode_id=str(trace_payload.get("episode_id") or trace_path.stem.replace("_llm_trace", "")),
                episode_idx=int(trace_payload.get("episode_idx", -1)),
                session_id=int(trace_payload.get("session_id", trace_payload.get("episode_idx", -1))),
                seed=int(trace_payload.get("seed", -1)),
                csv_path=csv_path,
                json_path=json_path,
                trace_path=trace_path,
                episode_log=episode_log,
                trace_payload=trace_payload,
                success_threshold=float(success_threshold),
            )
        )
    return artifacts


def resolve_episode_seed(*, seed_mode: str, base_seed: int, episode_idx: int) -> int:
    if seed_mode == SEED_MODE_FIXED:
        return int(base_seed)
    if seed_mode == SEED_MODE_PER_EPISODE:
        return int(base_seed) + int(episode_idx)
    raise ValueError(f"Unsupported seed_mode: {seed_mode}")


def episode_paths(output_root: Path, run_id: str, task_id: str, episode_idx: int) -> EpisodePaths:
    run_dir = Path(output_root) / str(run_id)
    task_dir = run_dir / str(task_id)
    run_stem = f"{task_id}_run_{int(episode_idx):02d}"
    return EpisodePaths(
        run_dir=run_dir,
        task_dir=task_dir,
        run_stem=run_stem,
        csv_path=task_dir / f"{run_stem}.csv",
        json_path=task_dir / f"{run_stem}.json",
        trace_path=task_dir / f"{run_stem}_llm_trace.json",
    )


def default_run_id(prefix: str, model_id: str) -> str:
    from datetime import datetime

    stamp = datetime.now().strftime("%m%d_%H%M%S")
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_id or "model")).strip("_")
    return f"{stamp}_{model_slug}_{prefix}"
