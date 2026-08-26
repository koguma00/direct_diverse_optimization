"""Small OpenAI-compatible client for the DDO BALROG adapter."""

from __future__ import annotations

from collections import namedtuple
from typing import Any

from omegaconf import OmegaConf


LLMResponse = namedtuple(
    "LLMResponse",
    ["model_id", "completion", "stop_reason", "input_tokens", "output_tokens", "reasoning"],
)

RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
RETRYABLE_EXCEPTION_NAMES = {
    "openai.APITimeoutError",
    "openai.APIConnectionError",
    "openai.RateLimitError",
    "openai.InternalServerError",
}


def _normalize_stop_reason(stop_reason: Any) -> str:
    return str(stop_reason or "").strip().lower().replace("-", "_").replace(" ", "_")


def classify_llm_response_abort_reason(response: Any) -> str | None:
    """Return the paper evaluator's recoverable response failure category."""

    stop_reason = _normalize_stop_reason(getattr(response, "stop_reason", ""))
    completion = str(getattr(response, "completion", "") or "").strip()
    if stop_reason == "error_max_retries":
        return "retry_exhausted"
    cap_hints = ("length", "max_tokens", "max_output_tokens", "max_completion_tokens")
    if any(hint in stop_reason for hint in cap_hints):
        return "cap_hit"
    if stop_reason == "empty_response" or not completion:
        return "invalid_action"
    return None


def classify_retryable_exception_leaf(exception: Exception) -> str | None:
    """Classify provider failures that may recover on a same-step retry."""

    full_name = f"{exception.__class__.__module__}.{exception.__class__.__name__}"
    status_code = getattr(exception, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exception, "response", None), "status_code", None)
    message = str(exception).lower()
    if (
        isinstance(exception, TimeoutError)
        or status_code == 408
        or "timeout" in full_name.lower()
        or "timed out" in message
        or "timeout" in message
    ):
        return "timeout"
    if full_name in RETRYABLE_EXCEPTION_NAMES or status_code in RETRYABLE_HTTP_STATUS_CODES:
        return "connection"
    return None


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _plain_value(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    plain = _plain_value(value)
    if isinstance(plain, dict):
        return plain
    try:
        return {str(key): _plain_value(value[key]) for key in value}
    except (TypeError, KeyError):
        return {}


class OpenAICompatibleClient:
    def __init__(self, config: Any) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "BALROG evaluation requires the `openai` package; install the paper extras"
            ) from exc

        self.model_id = str(_get(config, "model_id"))
        self.generate_kwargs = _plain_mapping(_get(config, "generate_kwargs", {}))
        base_url = str(_get(config, "base_url", "") or "").strip() or None
        timeout = float(_get(config, "timeout", 60.0))
        max_retries = int(_get(config, "max_retries", 1))
        self.max_retries = max_retries
        self.delay = float(_get(config, "delay", 0.0))
        self._client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._last_request_kwargs: dict[str, Any] = {}

    def generate(self, messages: list[Any]) -> LLMResponse:
        converted = [
            {"role": str(message.role), "content": str(message.content)} for message in messages
        ]
        request: dict[str, Any] = {
            "model": self.model_id,
            "messages": converted,
            "max_tokens": int(self.generate_kwargs.get("max_tokens", 8192)),
        }
        for key in ("temperature", "top_p", "seed", "stop"):
            value = self.generate_kwargs.get(key)
            if value is not None:
                request[key] = value
        extra_body = _plain_mapping(self.generate_kwargs.get("extra_body"))
        if "qwen3" in self.model_id.lower():
            chat_template = _plain_mapping(extra_body.get("chat_template_kwargs"))
            chat_template.setdefault("enable_thinking", False)
            extra_body["chat_template_kwargs"] = chat_template
        if extra_body:
            request["extra_body"] = extra_body
        self._last_request_kwargs = dict(request)

        response = self._client.chat.completions.create(**request)
        choice = response.choices[0]
        message = choice.message
        completion = str(message.content or "").strip()
        reasoning = str(
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or ""
        ).strip()
        usage = response.usage
        return LLMResponse(
            model_id=self.model_id,
            completion=completion,
            stop_reason=str(choice.finish_reason or ""),
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            reasoning=reasoning,
        )

    def get_last_request_kwargs(self) -> dict[str, Any]:
        return dict(self._last_request_kwargs)

    def get_last_applied_options(self) -> dict[str, Any]:
        return dict(self.generate_kwargs)


def create_client(config: Any):
    return lambda: OpenAICompatibleClient(config)
