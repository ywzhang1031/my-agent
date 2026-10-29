from __future__ import annotations

import hashlib
import json

from .database import Database
from .rpc import RequestEnvelope, ResponseEnvelope, response_from_dict, response_to_dict


MUTATING_METHODS = frozenset({
    "session.create", "session.rename", "turn.start", "turn.retry", "turn.recover",
    "turn.abort", "turn.interrupt", "plan.execute", "plan.revise", "goal.clear",
    "model.switch", "mode.switch", "permission.switch", "sandbox.switch",
    "approval.resolve", "provider.credential.set",
})


class RequestStore:
    def __init__(self, database: Database) -> None:
        self.database = database

    def begin(self, request: RequestEnvelope) -> ResponseEnvelope | None:
        # Persist only the digest, never raw params (which may contain an API key).
        canonical = json.dumps([request.method, request.params], sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT fingerprint, response FROM rpc_requests WHERE request_id=?", (request.request_id,)
            ).fetchone()
            if row is None:
                connection.execute("INSERT INTO rpc_requests VALUES (?, ?, NULL)",
                                   (request.request_id, fingerprint))
                return None
            if row[0] != fingerprint:
                return ResponseEnvelope.failure(request.request_id, code="idempotency_conflict",
                    message="request_id was already used with different parameters")
            if row[1] is not None:
                return response_from_dict(json.loads(row[1]))
            return ResponseEnvelope.failure(request.request_id, code="request_outcome_unknown",
                message="Request was already admitted but its outcome is unavailable. Inspect the session; do not resubmit with a new ID blindly.")

    def finish(self, response: ResponseEnvelope) -> None:
        with self.database.transaction() as connection:
            connection.execute("UPDATE rpc_requests SET response=? WHERE request_id=?",
                (json.dumps(response_to_dict(response), ensure_ascii=False), response.request_id))
