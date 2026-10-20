from __future__ import annotations

import copy
import json
import os
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


BUILTIN_CONFIG: dict[str, Any] = {
    "default_model": "deepseek/deepseek-v4-flash",
    "providers": {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
            "default_model": "deepseek-v4-flash",
            "options": {"thinking": "enabled"},
            "models": {
                "deepseek-v4-flash": {
                    "description": "DeepSeek V4 Flash",
                    "context_window_tokens": 1_000_000,
                    "max_output_tokens": 64_000,
                },
                "deepseek-v4-pro": {
                    "description": "DeepSeek V4 Pro",
                    "context_window_tokens": 1_000_000,
                    "max_output_tokens": 64_000,
                },
            },
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "default_model": "gpt-5.6",
            "options": {
                "reasoning_effort": "medium",
                "reasoning_summary": "auto",
            },
            "models": {
                "gpt-5.6": {
                    "description": "OpenAI GPT-5.6",
                    "context_window_tokens": 1_050_000,
                    "max_output_tokens": 128_000,
                }
            },
        },
    },
}


@dataclass(frozen=True)
class ConfiguredModel:
    provider_id: str
    model_id: str
    description: str
    context_window_tokens: int
    max_output_tokens: int

    @property
    def ref(self) -> str:
        return f"{self.provider_id}/{self.model_id}"


@dataclass(frozen=True)
class ProviderSettings:
    provider_id: str
    base_url: str
    api_key_env: str
    default_model: str
    models: Mapping[str, ConfiguredModel]
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedModel:
    model: ConfiguredModel
    base_url: str
    api_key: str | None
    options: Mapping[str, Any]


@dataclass(frozen=True)
class RuntimeConfig:
    config_path: Path
    credentials_path: Path
    default_model: str
    providers: Mapping[str, ProviderSettings]
    credentials: Mapping[str, str]

    def models(self) -> tuple[ConfiguredModel, ...]:
        return tuple(
            model
            for provider in self.providers.values()
            for model in provider.models.values()
        )

    def resolve_model(
        self,
        *,
        provider_id: str | None = None,
        model_id: str | None = None,
        model_ref: str | None = None,
    ) -> ResolvedModel:
        if model_ref is not None:
            provider_id, model_id = split_model_ref(model_ref)
        elif model_id is not None and "/" in model_id:
            ref_provider_id, model_id = split_model_ref(model_id)
            if provider_id is not None and provider_id != ref_provider_id:
                raise ValueError(
                    f"provider {provider_id} conflicts with model reference "
                    f"{ref_provider_id}/{model_id}"
                )
            provider_id = ref_provider_id
        elif provider_id is None:
            provider_id, default_model_id = split_model_ref(self.default_model)
            model_id = model_id or default_model_id

        provider = self.providers.get(provider_id)
        if provider is None:
            raise ValueError(f"unknown provider in config: {provider_id}")
        selected_model_id = model_id or provider.default_model
        model = provider.models.get(selected_model_id)
        if model is None:
            raise ValueError(
                f"unknown model in config: {provider.provider_id}/{selected_model_id}"
            )
        api_key = os.environ.get(provider.api_key_env) or self.credentials.get(
            provider.provider_id
        )
        return ResolvedModel(
            model=model,
            base_url=provider.base_url,
            api_key=api_key,
            options=provider.options,
        )


def load_runtime_config(
    config_path: str | Path | None = None,
    credentials_path: str | Path | None = None,
) -> RuntimeConfig:
    resolved_config_path, resolved_credentials_path = config_paths(
        config_path,
        credentials_path,
    )
    raw_config = copy.deepcopy(BUILTIN_CONFIG)
    if resolved_config_path.exists():
        _merge_config(raw_config, _read_json_object(resolved_config_path))
    providers = _parse_providers(raw_config.get("providers"))
    default_model = _required_string(raw_config.get("default_model"), "default_model")
    default_provider_id, default_model_id = split_model_ref(default_model)
    default_provider = providers.get(default_provider_id)
    if default_provider is None or default_model_id not in default_provider.models:
        raise ValueError(f"default_model is not configured: {default_model}")
    credentials = _load_credentials(resolved_credentials_path)
    return RuntimeConfig(
        config_path=resolved_config_path,
        credentials_path=resolved_credentials_path,
        default_model=default_model,
        providers=providers,
        credentials=credentials,
    )


def save_provider_credential(
    config: RuntimeConfig,
    provider_id: str,
    api_key: str,
) -> RuntimeConfig:
    """Persist one provider API key and return a freshly loaded config."""
    if provider_id not in config.providers:
        raise ValueError(f"unknown provider in config: {provider_id}")
    key = _required_string(api_key, f"credential {provider_id} key")
    if len(key) > 16_384:
        raise ValueError(f"credential {provider_id} key is too long")

    credentials = {
        name: {"type": "api", "key": value}
        for name, value in config.credentials.items()
    }
    credentials[provider_id] = {"type": "api", "key": key}
    _write_secure_json(config.credentials_path, credentials)
    return load_runtime_config(config.config_path, config.credentials_path)


def config_paths(
    config_path: str | Path | None = None,
    credentials_path: str | Path | None = None,
) -> tuple[Path, Path]:
    explicit_config = config_path or os.environ.get("MY_AGENT_CONFIG")
    if explicit_config:
        resolved_config = Path(explicit_config).expanduser().resolve()
        default_root = resolved_config.parent
    else:
        config_home = Path(
            os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
        ).expanduser()
        default_root = config_home / "my-agent"
        resolved_config = (default_root / "config.json").resolve()
    explicit_credentials = credentials_path or os.environ.get("MY_AGENT_CREDENTIALS")
    resolved_credentials = (
        Path(explicit_credentials).expanduser().resolve()
        if explicit_credentials
        else (default_root / "credentials.json").resolve()
    )
    return resolved_config, resolved_credentials


def split_model_ref(value: str) -> tuple[str, str]:
    provider_id, separator, model_id = value.partition("/")
    if not separator or not provider_id or not model_id:
        raise ValueError(f"model reference must use provider/model: {value}")
    return provider_id, model_id


def _merge_config(target: dict[str, Any], override: dict[str, Any]) -> None:
    unknown = set(override) - {"default_model", "providers"}
    if unknown:
        raise ValueError(f"unknown config fields: {', '.join(sorted(unknown))}")
    if "default_model" in override:
        target["default_model"] = override["default_model"]
    raw_providers = override.get("providers", {})
    if not isinstance(raw_providers, dict):
        raise ValueError("providers must be a JSON object")
    for provider_id, raw_provider in raw_providers.items():
        if not isinstance(raw_provider, dict):
            raise ValueError(f"provider {provider_id} must be a JSON object")
        existing = target["providers"].setdefault(provider_id, {})
        unknown_provider_fields = set(raw_provider) - {
            "base_url",
            "api_key_env",
            "default_model",
            "models",
            "options",
        }
        if unknown_provider_fields:
            raise ValueError(
                f"unknown fields for provider {provider_id}: "
                f"{', '.join(sorted(unknown_provider_fields))}"
            )
        for field_name in ("base_url", "api_key_env", "default_model"):
            if field_name in raw_provider:
                existing[field_name] = raw_provider[field_name]
        if "options" in raw_provider:
            raw_options = raw_provider["options"]
            if not isinstance(raw_options, dict):
                raise ValueError(f"provider {provider_id} options must be an object")
            existing.setdefault("options", {}).update(raw_options)
        if "models" in raw_provider:
            raw_models = raw_provider["models"]
            if not isinstance(raw_models, dict):
                raise ValueError(f"provider {provider_id} models must be an object")
            model_target = existing.setdefault("models", {})
            for model_id, raw_model in raw_models.items():
                if not isinstance(raw_model, dict):
                    raise ValueError(
                        f"model {provider_id}/{model_id} must be a JSON object"
                    )
                model_target.setdefault(model_id, {}).update(raw_model)


def _parse_providers(raw: Any) -> dict[str, ProviderSettings]:
    if not isinstance(raw, dict):
        raise ValueError("providers must be a JSON object")
    providers: dict[str, ProviderSettings] = {}
    for provider_id, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"provider {provider_id} must be a JSON object")
        models = _parse_models(provider_id, value.get("models"))
        default_model = _required_string(
            value.get("default_model"),
            f"provider {provider_id} default_model",
        )
        if default_model not in models:
            raise ValueError(
                f"provider {provider_id} default_model is not configured: {default_model}"
            )
        options = value.get("options", {})
        if not isinstance(options, dict):
            raise ValueError(f"provider {provider_id} options must be an object")
        providers[provider_id] = ProviderSettings(
            provider_id=provider_id,
            base_url=_required_string(
                value.get("base_url"), f"provider {provider_id} base_url"
            ).rstrip("/"),
            api_key_env=_required_string(
                value.get("api_key_env"), f"provider {provider_id} api_key_env"
            ),
            default_model=default_model,
            models=models,
            options=options,
        )
    return providers


def _parse_models(provider_id: str, raw: Any) -> dict[str, ConfiguredModel]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"provider {provider_id} models must be a non-empty object")
    models: dict[str, ConfiguredModel] = {}
    for model_id, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"model {provider_id}/{model_id} must be a JSON object")
        unknown = set(value) - {
            "description",
            "context_window_tokens",
            "max_output_tokens",
        }
        if unknown:
            raise ValueError(
                f"unknown fields for model {provider_id}/{model_id}: "
                f"{', '.join(sorted(unknown))}"
            )
        context_window_tokens = _positive_int(
            value.get("context_window_tokens"),
            f"model {provider_id}/{model_id} context_window_tokens",
        )
        max_output_tokens = _positive_int(
            value.get("max_output_tokens"),
            f"model {provider_id}/{model_id} max_output_tokens",
        )
        if max_output_tokens >= context_window_tokens:
            raise ValueError(
                f"model {provider_id}/{model_id} max_output_tokens must be smaller "
                "than context_window_tokens"
            )
        models[model_id] = ConfiguredModel(
            provider_id=provider_id,
            model_id=model_id,
            description=str(value.get("description") or model_id),
            context_window_tokens=context_window_tokens,
            max_output_tokens=max_output_tokens,
        )
    return models


def _load_credentials(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ValueError(f"credentials file must have mode 0600: {path}")
    raw = _read_json_object(path)
    credentials: dict[str, str] = {}
    for provider_id, value in raw.items():
        if not isinstance(value, dict) or value.get("type") != "api":
            raise ValueError(f"credential {provider_id} must have type 'api'")
        key = _required_string(value.get("key"), f"credential {provider_id} key")
        credentials[provider_id] = key
    return credentials


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"configuration file must contain a JSON object: {path}")
    return value


def _write_secure_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        path.chmod(0o600)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed
