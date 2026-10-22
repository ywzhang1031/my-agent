from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .protocol import (
    AgentEventType,
    ItemCompletedEvent,
    ItemDeltaEvent,
    ItemFailedEvent,
    ItemStartedEvent,
    TurnAbortedEvent,
    TurnCompletedEvent,
    TurnContinuationRequiredEvent,
    TurnErrorEvent,
    TurnFailedEvent,
    TurnInterruptedEvent,
    TurnResumedEvent,
    TurnStartedEvent,
)


@dataclass
class ItemProjection:
    item_id: str
    turn_id: str
    item_type: str
    status: str
    step: int
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "turn_id": self.turn_id,
            "item_type": self.item_type,
            "status": self.status,
            "step": self.step,
            "content": self.content,
            "data": dict(self.data),
        }


@dataclass
class TurnProjection:
    turn_id: str
    task: str
    status: str = "in_progress"
    error: str | None = None
    items: dict[str, ItemProjection] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "task": self.task,
            "status": self.status,
            "error": self.error,
            "items": [item.to_dict() for item in self.items.values()],
        }


class SessionProjection:
    """Materialized UI view derived from the typed protocol event stream."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.turns: dict[str, TurnProjection] = {}
        self.latest_sequence = 0

    def apply(self, event: AgentEventType) -> None:
        if event.session_id != self.session_id:
            raise ValueError(
                f"projection session mismatch: {self.session_id} != {event.session_id}"
            )
        if event.sequence <= self.latest_sequence:
            raise ValueError(
                f"projection event sequence must increase: {event.sequence} <= "
                f"{self.latest_sequence}"
            )
        self.latest_sequence = event.sequence

        if isinstance(event, TurnStartedEvent):
            self.turns[event.turn_id] = TurnProjection(
                turn_id=event.turn_id,
                task=event.task,
            )
            return

        turn = self._turn(event.turn_id)
        if turn is None:
            return
        if isinstance(event, TurnResumedEvent):
            turn.status = "in_progress"
            turn.error = None
        elif isinstance(event, ItemStartedEvent):
            turn.items[event.item_id] = ItemProjection(
                item_id=event.item_id,
                turn_id=turn.turn_id,
                item_type=event.item_type,
                status="in_progress",
                step=event.step,
                data=dict(event.data),
            )
        elif isinstance(event, ItemDeltaEvent):
            item = self._item(turn, event.item_id, event.item_type, event.step)
            item.content += event.delta
            item.data.update(event.data)
        elif isinstance(event, ItemCompletedEvent):
            item = self._item(turn, event.item_id, event.item_type, event.step)
            item.status = "completed"
            item.content = event.content
            item.data.update(event.data)
        elif isinstance(event, ItemFailedEvent):
            item = self._item(turn, event.item_id, event.item_type, event.step)
            item.status = event.status
            item.data.update(event.data)
            item.data["error"] = event.error
        elif isinstance(event, TurnCompletedEvent):
            turn.status = "completed"
        elif isinstance(event, TurnContinuationRequiredEvent):
            turn.status = "waiting_user"
        elif isinstance(event, TurnErrorEvent):
            turn.status = "failed"
            turn.error = event.error
        elif isinstance(event, TurnFailedEvent):
            turn.status = "failed"
            turn.error = event.error
        elif isinstance(event, TurnInterruptedEvent):
            turn.status = "interrupted"
            turn.error = event.reason
        elif isinstance(event, TurnAbortedEvent):
            turn.status = "aborted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "latest_sequence": self.latest_sequence,
            "turns": [turn.to_dict() for turn in self.turns.values()],
        }

    def _turn(self, turn_id: str | None) -> TurnProjection | None:
        if turn_id is None:
            return None
        return self.turns.get(turn_id)

    @staticmethod
    def _item(
        turn: TurnProjection,
        item_id: str,
        item_type: str,
        step: int,
    ) -> ItemProjection:
        item = turn.items.get(item_id)
        if item is None:
            item = ItemProjection(
                item_id=item_id,
                turn_id=turn.turn_id,
                item_type=item_type,
                status="in_progress",
                step=step,
            )
            turn.items[item_id] = item
        return item
