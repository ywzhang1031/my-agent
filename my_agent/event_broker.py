from __future__ import annotations

import queue
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable

from .protocol import AgentEventType


DEFAULT_SUBSCRIPTION_TTL_SECONDS = 120.0


@dataclass
class _Subscription:
    session_ids: frozenset[str] | None
    events: queue.Queue[AgentEventType] = field(default_factory=queue.Queue)
    last_access: float = field(default_factory=time.monotonic)


class EventBroker:
    """Fan out typed events to independent local client subscriptions."""

    def __init__(self, *, ttl_seconds: float = DEFAULT_SUBSCRIPTION_TTL_SECONDS) -> None:
        if ttl_seconds <= 0:
            raise ValueError("subscription ttl must be positive")
        self._ttl_seconds = ttl_seconds
        self._subscriptions: dict[str, _Subscription] = {}
        self._lock = threading.Lock()

    def subscribe(self, session_ids: Iterable[str] | None = None) -> str:
        selected = None if session_ids is None else frozenset(session_ids)
        if selected is not None and not selected:
            raise ValueError("event subscription must include at least one session")
        subscription_id = secrets.token_urlsafe(24)
        with self._lock:
            self._expire_unlocked()
            self._subscriptions[subscription_id] = _Subscription(selected)
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        with self._lock:
            return self._subscriptions.pop(subscription_id, None) is not None

    def publish(self, event: AgentEventType) -> None:
        with self._lock:
            self._expire_unlocked()
            subscribers = tuple(self._subscriptions.values())
        for subscription in subscribers:
            if (
                subscription.session_ids is None
                or event.session_id in subscription.session_ids
            ):
                subscription.events.put(event)

    def deliver(self, subscription_id: str, event: AgentEventType) -> None:
        subscription = self._get(subscription_id)
        if (
            subscription.session_ids is not None
            and event.session_id not in subscription.session_ids
        ):
            return
        subscription.events.put(event)

    def next_event(
        self,
        subscription_id: str,
        timeout: float | None = None,
    ) -> AgentEventType:
        subscription = self._get(subscription_id)
        event = subscription.events.get(timeout=timeout)
        with self._lock:
            current = self._subscriptions.get(subscription_id)
            if current is subscription:
                current.last_access = time.monotonic()
        return event

    def close(self) -> None:
        with self._lock:
            self._subscriptions.clear()

    @property
    def subscription_count(self) -> int:
        with self._lock:
            self._expire_unlocked()
            return len(self._subscriptions)

    def _get(self, subscription_id: str) -> _Subscription:
        with self._lock:
            self._expire_unlocked()
            subscription = self._subscriptions.get(subscription_id)
            if subscription is None:
                raise ValueError(f"event subscription not found: {subscription_id}")
            subscription.last_access = time.monotonic()
            return subscription

    def _expire_unlocked(self) -> None:
        cutoff = time.monotonic() - self._ttl_seconds
        expired = [
            subscription_id
            for subscription_id, subscription in self._subscriptions.items()
            if subscription.last_access < cutoff
        ]
        for subscription_id in expired:
            del self._subscriptions[subscription_id]
