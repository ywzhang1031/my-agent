from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol


class TurnInterrupted(RuntimeError):
    pass


def raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise TurnInterrupted("turn interrupted by user")


class ApprovalDecision(str, Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionRequest:
    action: str
    resource: str
    description: str
    details: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: secrets.token_hex(8))

    @property
    def scope_key(self) -> tuple[str, str]:
        return self.action, self.resource

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "action": self.action,
            "resource": self.resource,
            "description": self.description,
            "details": self.details,
        }


ApprovalNotifier = Callable[[], None]


class ApprovalBroker(Protocol):
    def authorize(
        self,
        request: PermissionRequest,
        *,
        cancel_event: threading.Event | None,
        on_waiting: ApprovalNotifier,
    ) -> ApprovalDecision:
        ...


class AutoApproveBroker:
    def authorize(
        self,
        request: PermissionRequest,
        *,
        cancel_event: threading.Event | None,
        on_waiting: ApprovalNotifier,
    ) -> ApprovalDecision:
        raise_if_cancelled(cancel_event)
        return ApprovalDecision.ALLOW_ONCE
