from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterator

from .control import raise_if_cancelled
from .provider import (
    ModelCompleted,
    ModelError,
    ModelErrorKind,
    ModelErrorPhase,
    ModelEvent,
    ModelProfile,
    ModelRequest,
    ProviderAdapter,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_delay_seconds: float = 0.75
    max_delay_seconds: float = 8.0

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.initial_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays must not be negative")


@dataclass(frozen=True)
class RetryScheduled:
    error: ModelError
    failed_attempt: int
    next_attempt: int
    delay_seconds: float


InvocationEvent = ModelEvent | RetryScheduled


class ModelInvoker:
    """Harness-side policy for invoking one provider adapter."""

    def __init__(
        self,
        adapter: ProviderAdapter,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.adapter = adapter
        self.retry_policy = retry_policy or RetryPolicy()

    @property
    def profile(self) -> ModelProfile:
        return self.adapter.profile

    def stream(
        self,
        request: ModelRequest,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[InvocationEvent]:
        if request.tools and not self.profile.capabilities.tool_calling:
            raise ModelError(
                f"model {self.profile.model_id} does not support tool calling",
                kind=ModelErrorKind.INVALID_REQUEST,
                phase=ModelErrorPhase.REQUEST,
                provider_id=self.profile.provider_id,
                model_id=self.profile.model_id,
            )
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            raise_if_cancelled(cancel_event)
            emitted_delta = False
            completed = False
            try:
                for event in self.adapter.stream_once(request, cancel_event):
                    raise_if_cancelled(cancel_event)
                    if completed:
                        raise self._protocol_error(
                            "provider emitted an event after the completed event"
                        )
                    if isinstance(event, (TextDelta, ReasoningDelta, ToolCallDelta)):
                        emitted_delta = True
                    elif isinstance(event, ModelCompleted):
                        completed = True
                    else:
                        raise self._protocol_error(
                            f"provider emitted unsupported event {type(event).__name__}"
                        )
                    yield event
                if not completed:
                    raise self._protocol_error(
                        "provider stream ended without a completed event",
                        retryable=True,
                    )
                return
            except ModelError as exc:
                can_retry = (
                    exc.retryable
                    and not emitted_delta
                    and not completed
                    and attempt < self.retry_policy.max_attempts
                )
                if not can_retry:
                    raise
                delay = self._retry_delay(exc, attempt)
                yield RetryScheduled(
                    error=exc,
                    failed_attempt=attempt,
                    next_attempt=attempt + 1,
                    delay_seconds=delay,
                )
                self._wait(delay, cancel_event)

    def _retry_delay(self, error: ModelError, failed_attempt: int) -> float:
        if error.retry_after_seconds is not None:
            return min(error.retry_after_seconds, self.retry_policy.max_delay_seconds)
        delay = self.retry_policy.initial_delay_seconds * (2 ** (failed_attempt - 1))
        return min(delay, self.retry_policy.max_delay_seconds)

    def _wait(
        self,
        delay_seconds: float,
        cancel_event: threading.Event | None,
    ) -> None:
        if delay_seconds <= 0:
            return
        if cancel_event is not None:
            if cancel_event.wait(delay_seconds):
                raise_if_cancelled(cancel_event)
            return
        time.sleep(delay_seconds)

    def _protocol_error(self, message: str, *, retryable: bool = False) -> ModelError:
        return ModelError(
            message,
            kind=ModelErrorKind.PROTOCOL,
            phase=ModelErrorPhase.STREAM,
            retryable=retryable,
            provider_id=self.profile.provider_id,
            model_id=self.profile.model_id,
        )
