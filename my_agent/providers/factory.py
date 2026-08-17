from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..provider import ProviderAdapter
from .deepseek import DeepSeekAdapter
from .openai import OpenAIAdapter


SUPPORTED_PROVIDER_IDS = ("deepseek", "openai")


@dataclass(frozen=True)
class ProviderConfig:
    provider_id: str
    api_key: str | None = None
    model_id: str | None = None
    base_url: str | None = None
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

def create_provider_adapter(config: ProviderConfig) -> ProviderAdapter:
    if config.provider_id == "deepseek":
        unknown = set(config.options) - {"thinking"}
        if unknown:
            raise ValueError(
                f"unsupported DeepSeek provider options: {', '.join(sorted(unknown))}"
            )
        return DeepSeekAdapter(
            api_key=config.api_key,
            model=config.model_id,
            base_url=config.base_url,
            thinking=config.options.get("thinking"),
            context_window_tokens=config.context_window_tokens,
            max_output_tokens=config.max_output_tokens,
        )
    if config.provider_id == "openai":
        unknown = set(config.options) - {"reasoning_effort", "reasoning_summary"}
        if unknown:
            raise ValueError(
                f"unsupported OpenAI provider options: {', '.join(sorted(unknown))}"
            )
        return OpenAIAdapter(
            api_key=config.api_key,
            model=config.model_id,
            base_url=config.base_url,
            reasoning_effort=config.options.get("reasoning_effort"),
            reasoning_summary=config.options.get("reasoning_summary"),
            context_window_tokens=config.context_window_tokens,
            max_output_tokens=config.max_output_tokens,
        )
    raise ValueError(f"unsupported provider: {config.provider_id}")
