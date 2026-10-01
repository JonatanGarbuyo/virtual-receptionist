"""Runtime persistence for call summaries (in-memory stand-in)."""

from __future__ import annotations

from receptionist.boundaries import AuditEvent, CallSummary


class InMemoryCallRepository:
    def __init__(self) -> None:
        self._summaries: dict[str, CallSummary] = {}

    def save(self, summary: CallSummary) -> None:
        self._summaries[summary.call_id] = summary

    def get(self, call_id: str) -> CallSummary | None:
        return self._summaries.get(call_id)

    def list_all(self) -> list[CallSummary]:
        return list(self._summaries.values())


class InMemoryAuditLog:
    """Append-only audit stand-in. Records privileged actions, never transcripts."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self._events.append(event)

    def list_all(self) -> list[AuditEvent]:
        return list(self._events)
