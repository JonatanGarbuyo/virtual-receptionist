"""Runtime persistence stand-ins (the runtime.db side of the model).

Configuration lives behind ConfigService (the config.db side); every
record here is operational data: call summaries, confirmed messages,
optional text transcripts, and the structured action audit. No audio,
no secrets, no conversational content beyond the explicitly authorized
message/transcript stores.
"""

from __future__ import annotations

from receptionist.boundaries import (
    AuditEvent,
    AuditLog,
    CallRepository,
    CallSummary,
    Clock,
    MessageDraft,
    MessageRecord,
    MessageRepository,
    TranscriptEntry,
    TranscriptStore,
)
from receptionist.policy import RetentionPolicy


class InMemoryCallRepository:
    def __init__(self) -> None:
        self._summaries: dict[str, CallSummary] = {}

    def save(self, summary: CallSummary) -> None:
        self._summaries[summary.call_id] = summary

    def get(self, call_id: str) -> CallSummary | None:
        return self._summaries.get(call_id)

    def list_all(self) -> list[CallSummary]:
        return list(self._summaries.values())

    def find_by_caller(self, caller_id: str) -> list[CallSummary]:
        return [s for s in self._summaries.values() if s.caller_id == caller_id]

    def find_in_range(self, start: float, end: float) -> list[CallSummary]:
        return [s for s in self._summaries.values() if start <= s.started_at <= end]

    def prune_before(self, cutoff: float) -> int:
        expired = [key for key, s in self._summaries.items() if s.started_at < cutoff]
        for key in expired:
            del self._summaries[key]
        return len(expired)


class InMemoryMessageRepository:
    """Confirmed messages only. The id is assigned here, on successful save."""

    def __init__(self, clock: Clock | None = None) -> None:
        self._messages: dict[str, MessageRecord] = {}
        self._clock = clock
        self._next_id = 1

    def save(self, draft: MessageDraft) -> MessageRecord:
        message_id = f"msg-{self._next_id}"
        self._next_id += 1
        record = MessageRecord(
            id=message_id,
            call_id=draft.call_id,
            caller_id=draft.caller_id,
            caller_name=draft.caller_name,
            text=draft.text,
            created_at=self._clock.now() if self._clock is not None else 0.0,
        )
        self._messages[message_id] = record
        return record

    def get(self, message_id: str) -> MessageRecord | None:
        return self._messages.get(message_id)

    def list_all(self) -> list[MessageRecord]:
        return list(self._messages.values())

    def prune_before(self, cutoff: float) -> int:
        expired = [key for key, m in self._messages.items() if m.created_at < cutoff]
        for key in expired:
            del self._messages[key]
        return len(expired)


class InMemoryTranscriptStore:
    """Text-only observational transcripts. Empty unless explicitly enabled."""

    def __init__(self) -> None:
        self._entries: list[TranscriptEntry] = []

    def append(self, entry: TranscriptEntry) -> None:
        self._entries.append(entry)

    def entries_for(self, call_id: str) -> list[TranscriptEntry]:
        return [e for e in self._entries if e.call_id == call_id]

    def prune_before(self, cutoff: float) -> int:
        kept = [e for e in self._entries if e.timestamp >= cutoff]
        removed = len(self._entries) - len(kept)
        self._entries = kept
        return removed


class InMemoryAuditLog:
    """Append-only audit stand-in. Records privileged actions, never transcripts."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self._events.append(event)

    def list_all(self) -> list[AuditEvent]:
        return list(self._events)

    def prune_before(self, cutoff: float) -> int:
        kept = [e for e in self._events if e.timestamp >= cutoff]
        removed = len(self._events) - len(kept)
        self._events = kept
        return removed


class RuntimeStorage:
    """The runtime.db side: one bundle for all operational repositories.

    Kept structurally separate from ConfigService (the config.db side) so
    caller data and retention pruning can never touch configuration.
    """

    def __init__(
        self,
        calls: CallRepository | None = None,
        messages: MessageRepository | None = None,
        transcripts: TranscriptStore | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self.calls = calls if calls is not None else InMemoryCallRepository()
        self.messages = messages if messages is not None else InMemoryMessageRepository()
        self.transcripts = transcripts if transcripts is not None else InMemoryTranscriptStore()
        self.audit = audit if audit is not None else InMemoryAuditLog()

    @classmethod
    def create(cls, clock: Clock | None = None) -> RuntimeStorage:
        storage = cls()
        storage.messages = InMemoryMessageRepository(clock=clock)
        return storage

    def prune_expired(self, retention: RetentionPolicy, now: float) -> dict[str, int]:
        """Delete records older than the policy. Idempotent: reruns delete 0."""
        return {
            "messages": self.messages.prune_before(now - retention.messages_days * 86400.0),
            "calls": self.calls.prune_before(now - retention.history_days * 86400.0),
            "transcripts": self.transcripts.prune_before(
                now - retention.transcripts_days * 86400.0
            ),
            "audit": self.audit.prune_before(now - retention.audit_days * 86400.0),
        }
