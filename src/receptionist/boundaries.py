"""Project-owned contracts for the ReceptionistCore / CallSession seam.

Production code depends only on these protocols, never on concrete
adapters. Tests provide deterministic fakes behind the same protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Clock(Protocol):
    """Controllable time source. Production uses wall clock; tests use a fake."""

    def now(self) -> float: ...


class CallIdGenerator(Protocol):
    """Issues call ids unique across process restarts.

    Production uses random UUIDs; tests use a deterministic counter.
    The core never invents ids itself and knows nothing about SQLite.
    """

    def next_id(self) -> str: ...


class TransferResult(Enum):
    """Normalized handoff outcome. Only these four values may cross the seam."""

    ACCEPTED_BY_PBX = "accepted_by_pbx"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    TRANSPORT_ERROR = "transport_error"


class TelephonyListener(Protocol):
    def on_answered(self, call_id: str) -> None: ...
    def on_caller_hangup(self, call_id: str) -> None: ...
    def on_hangup_completed(self, call_id: str) -> None: ...
    def on_transfer_result(self, call_id: str, result: TransferResult) -> None: ...


class TelephonyAdapter(Protocol):
    """Call-control boundary. No dial-by-URI capability exists on this contract."""

    def set_listener(self, listener: TelephonyListener) -> None: ...
    def answer(self, call_id: str) -> None: ...
    def reject(self, call_id: str) -> None: ...
    def hangup(self, call_id: str) -> None: ...
    def blind_transfer(self, call_id: str, pbx_target: str) -> None: ...


@dataclass(frozen=True)
class TransferRequest:
    """The only transfer shape the model/requester may produce.

    A symbolic destination id resolved by policy; never a number or URI.
    """

    destination_id: str


@dataclass(frozen=True)
class StartMessageCapture:
    """Typed request to open message capture. Carries no content."""


@dataclass(frozen=True)
class MessageTextFinal:
    """Final usable message text from the voice backend (STT final)."""

    text: str


@dataclass(frozen=True)
class MessageConfirmed:
    """The caller accepted the readback. Persist only after this."""


@dataclass(frozen=True)
class MessageRejected:
    """The caller rejected the draft. Discard it."""


class VoiceListener(Protocol):
    """Events flowing from the voice backend into one call session."""

    def on_transcript(self, text: str) -> None: ...
    def on_response(self, turn_id: int, text: str) -> None: ...
    def on_playback_finished(self, turn_id: int) -> None: ...
    def on_action_request(self, action: object) -> None: ...


class VoiceSession(Protocol):
    """One backend voice stream for one call."""

    def speak(self, text: str, turn_id: int) -> None: ...
    def close(self) -> None: ...


class VoiceBackend(Protocol):
    def open_session(self, call_id: str, listener: VoiceListener) -> VoiceSession: ...


class ConfigRepository(Protocol):
    """Key/value configuration source behind the service boundary."""

    def get(self, key: str) -> str | None: ...


class Policy(Protocol):
    """Admission policy: answer the inbound call or reject it."""

    def should_answer(self, caller_id: str) -> bool: ...


class CallOutcome(Enum):
    COMPLETED = "completed"
    CALLER_HANGUP = "caller_hangup"
    REJECTED = "rejected"
    TRANSFERRED = "transferred"


class StoreUnavailableError(Exception):
    """The runtime store cannot serve requests right now. Not retryable here."""


class TransientStoreError(Exception):
    """A possibly-transient store failure. Bounded retries may follow."""


@dataclass(frozen=True)
class MessageDraft:
    """Unpersisted caller message. The id is assigned by the store on save."""

    call_id: str
    caller_id: str
    caller_name: str | None
    text: str


@dataclass(frozen=True)
class MessageRecord:
    """Persisted confirmed message: text/structure only, never audio."""

    id: str
    call_id: str
    caller_id: str
    caller_name: str | None
    text: str
    created_at: float


class MessageRepository(Protocol):
    """Runtime persistence boundary for confirmed messages.

    Atomicity contract: save() either commits the record and returns it,
    or raises before committing anything. A TransientStoreError therefore
    implies no record was stored, which is what makes the session's bounded
    retry safe against duplicates. Stores must only raise
    TransientStoreError (retryable) or StoreUnavailableError (fallback).
    """

    def save(self, draft: MessageDraft) -> MessageRecord: ...
    def get(self, message_id: str) -> MessageRecord | None: ...
    def list_all(self) -> list[MessageRecord]: ...
    def prune_before(self, cutoff: float) -> int: ...


@dataclass(frozen=True)
class TranscriptEntry:
    """One observational transcript line: text only, never authority."""

    call_id: str
    timestamp: float
    speaker: str
    text: str


class TranscriptStore(Protocol):
    """Append-only text transcript boundary. Disabled by default."""

    def append(self, entry: TranscriptEntry) -> None: ...
    def entries_for(self, call_id: str) -> list[TranscriptEntry]: ...
    def prune_before(self, cutoff: float) -> int: ...


class AuditDecision(Enum):
    REQUESTED = "requested"
    ALLOWED = "allowed"
    DENIED = "denied"
    COMPLETED = "completed"


@dataclass(frozen=True)
class AuditEvent:
    """One structured privileged-action record. No transcript content."""

    timestamp: float
    call_id: str
    action: str
    destination: str
    decision: AuditDecision
    result: TransferResult | None = None
    detail: str = ""


class AuditLog(Protocol):
    """Append-only audit boundary for privileged actions."""

    def record(self, event: AuditEvent) -> None: ...
    def list_all(self) -> list[AuditEvent]: ...
    def prune_before(self, cutoff: float) -> int: ...


@dataclass(frozen=True)
class CallSummary:
    call_id: str
    caller_id: str
    started_at: float
    ended_at: float
    outcome: CallOutcome
    turn_count: int
    caller_name: str | None = None
    handoff_destination_id: str | None = None
    message_id: str | None = None
    failure_category: str | None = None


class CallRepository(Protocol):
    """Runtime persistence boundary for call summaries."""

    def save(self, summary: CallSummary) -> None: ...
    def get(self, call_id: str) -> CallSummary | None: ...
    def list_all(self) -> list[CallSummary]: ...
    def find_by_caller(self, caller_id: str) -> list[CallSummary]: ...
    def find_in_range(self, start: float, end: float) -> list[CallSummary]: ...
    def prune_before(self, cutoff: float) -> int: ...
