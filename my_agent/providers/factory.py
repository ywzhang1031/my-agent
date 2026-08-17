from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..provider import ProviderAdapter
from .deepseek import DeepSeekAdapter


SUPPORTED_PROVIDER_IDS = ("deepseek",)


@dataclass(frozen=True)
class ModelCatalogEntry:
    model_id: str
    description: str


PROVIDER_MODELS = {
    "deepseek": (
        ModelCatalogEntry("deepseek-v4-flash", "DeepSeek V4 Flash"),
        ModelCatalogEntry("deepseek-v4-pro", "DeepSeek V4 Pro"),
    ),
}


@dataclass(frozen=True)
class ProviderConfig:
    provider_id: str
    model_id: str | None = None
    base_url: str | None = None
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


def model_catalog(provider_id: str) -> tuple[ModelCatalogEntry, ...]:
    return PROVIDER_MODELS.get(provider_id, ())


def create_provider_adapter(config: ProviderConfig) -> ProviderAdapter:
    if config.provider_id == "deepseek":
        unknown = set(config.options) - {"thinking"}
        if unknown:
            raise ValueError(
                f"unsupported DeepSeek provider options: {', '.join(sorted(unknown))}"
            )
        return DeepSeekAdapter(
            model=config.model_id,
            base_url=config.base_url,
            thinking=config.options.get("thinking"),
            context_window_tokens=config.context_window_tokens,
            max_output_tokens=config.max_output_tokens,
        )
    raise ValueError(f"unsupported provider: {config.provider_id}")
