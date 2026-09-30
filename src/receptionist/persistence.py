"""Runtime persistence for call summaries (in-memory stand-in)."""

from __future__ import annotations

from receptionist.boundaries import CallSummary


class InMemoryCallRepository:
    def __init__(self) -> None:
        self._summaries: dict[str, CallSummary] = {}

    def save(self, summary: CallSummary) -> None:
        self._summaries[summary.call_id] = summary

    def get(self, call_id: str) -> CallSummary | None:
        return self._summaries.get(call_id)

    def list_all(self) -> list[CallSummary]:
        return list(self._summaries.values())
