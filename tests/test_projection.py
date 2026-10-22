from __future__ import annotations

import unittest

from my_agent.projection import SessionProjection
from my_agent.protocol import (
    AgentState,
    ItemCompletedEvent,
    ItemDeltaEvent,
    ItemStartedEvent,
    TurnCompletedEvent,
    TurnStartedEvent,
)


class SessionProjectionTests(unittest.TestCase):
    def test_projection_rebuilds_live_item_by_stable_identity(self) -> None:
        projection = SessionProjection("session-1")
        projection.apply(
            TurnStartedEvent(
                session_id="session-1",
                turn_id="session-1:1",
                sequence=1,
                state=AgentState.RUNNING_MODEL,
                task="Inspect app.py",
            )
        )
        projection.apply(
            ItemStartedEvent(
                session_id="session-1",
                turn_id="session-1:1",
                sequence=2,
                state=AgentState.RUNNING_MODEL,
                durable=False,
                item_id="assistant-1",
                item_type="assistant_message",
                step=1,
            )
        )
        projection.apply(
            ItemDeltaEvent(
                session_id="session-1",
                turn_id="session-1:1",
                sequence=3,
                state=AgentState.RUNNING_MODEL,
                durable=False,
                item_id="assistant-1",
                item_type="assistant_message",
                step=1,
                delta="Inspecting ",
            )
        )
        projection.apply(
            ItemDeltaEvent(
                session_id="session-1",
                turn_id="session-1:1",
                sequence=4,
                state=AgentState.RUNNING_MODEL,
                durable=False,
                item_id="assistant-1",
                item_type="assistant_message",
                step=1,
                delta="app.py",
            )
        )

        snapshot = projection.to_dict()

        self.assertEqual(snapshot["latest_sequence"], 4)
        self.assertEqual(snapshot["turns"][0]["status"], "in_progress")
        self.assertEqual(
            snapshot["turns"][0]["items"][0]["content"],
            "Inspecting app.py",
        )

    def test_terminal_events_replace_delta_content_and_complete_turn(self) -> None:
        projection = SessionProjection("session-1")
        projection.apply(
            TurnStartedEvent(
                session_id="session-1",
                turn_id="session-1:1",
                sequence=1,
                state=AgentState.RUNNING_MODEL,
                task="Inspect app.py",
            )
        )
        projection.apply(
            ItemCompletedEvent(
                session_id="session-1",
                turn_id="session-1:1",
                sequence=5,
                state=AgentState.RUNNING_MODEL,
                item_id="assistant-1",
                item_type="assistant_message",
                step=1,
                content="Inspection complete.",
            )
        )
        projection.apply(
            TurnCompletedEvent(
                session_id="session-1",
                turn_id="session-1:1",
                sequence=6,
                state=AgentState.COMPLETED,
                answer="Inspection complete.",
                steps=1,
                stopped_by="final_answer",
                mode_id="build",
            )
        )

        turn = projection.to_dict()["turns"][0]

        self.assertEqual(turn["status"], "completed")
        self.assertEqual(turn["items"][0]["status"], "completed")
        self.assertEqual(turn["items"][0]["content"], "Inspection complete.")


if __name__ == "__main__":
    unittest.main()
