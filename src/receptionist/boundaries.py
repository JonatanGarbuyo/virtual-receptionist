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


class TelephonyListener(Protocol):
    def on_answered(self, call_id: str) -> None: ...
    def on_caller_hangup(self, call_id: str) -> None: ...
    def on_hangup_completed(self, call_id: str) -> None: ...


class TelephonyAdapter(Protocol):
    """Call-control boundary. No dial-by-URI capability exists on this contract."""

    def set_listener(self, listener: TelephonyListener) -> None: ...
    def answer(self, call_id: str) -> None: ...
    def reject(self, call_id: str) -> None: ...
    def hangup(self, call_id: str) -> None: ...


class VoiceListener(Protocol):
    """Events flowing from the voice backend into one call session."""

    def on_transcript(self, text: str) -> None: ...
    def on_response(self, turn_id: int, text: str) -> None: ...
    def on_playback_finished(self, turn_id: int) -> None: ...


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


@dataclass(frozen=True)
class CallSummary:
    call_id: str
    caller_id: str
    started_at: float
    ended_at: float
    outcome: CallOutcome
    turn_count: int


class CallRepository(Protocol):
    """Runtime persistence boundary for call summaries."""

    def save(self, summary: CallSummary) -> None: ...
    def get(self, call_id: str) -> CallSummary | None: ...
    def list_all(self) -> list[CallSummary]: ...
