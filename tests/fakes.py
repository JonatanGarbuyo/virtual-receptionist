"""Project-owned test doubles for the ReceptionistCore / CallSession seam.

All fakes are deterministic: synchronous listener callbacks, no threads,
no wall-clock sleeps, no network access. They stand in for the system
boundaries (telephony, voice backend, clock, policy) defined in
``receptionist.boundaries``.
"""

from __future__ import annotations

from receptionist.boundaries import (
    ProviderFailure,
    TelephonyListener,
    TransferResult,
    VoiceListener,
    VoiceSession,
)
from receptionist.persistence import InMemoryMessageRepository


class FakeClock:
    """Manually advanced clock. Tests control time via :meth:`advance`."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class FakeTelephony:
    """Synchronous telephony stand-in. Records call control, notifies listener.

    With ``auto_confirm=False``, answer/hangup are only recorded and the
    test completes them explicitly via :meth:`complete_answer` /
    :meth:`complete_hangup`, so intermediate states stay observable.
    """

    def __init__(self, auto_confirm: bool = True) -> None:
        self._listener: TelephonyListener | None = None
        self.auto_confirm = auto_confirm
        self.answered: list[str] = []
        self.rejected: list[str] = []
        self.hung_up: list[str] = []
        self.transfers: list[tuple[str, str]] = []

    def set_listener(self, listener: TelephonyListener) -> None:
        self._listener = listener

    def answer(self, call_id: str) -> None:
        self.answered.append(call_id)
        if self.auto_confirm:
            self.complete_answer(call_id)

    def complete_answer(self, call_id: str) -> None:
        assert self._listener is not None
        self._listener.on_answered(call_id)

    def reject(self, call_id: str) -> None:
        self.rejected.append(call_id)

    def hangup(self, call_id: str) -> None:
        self.hung_up.append(call_id)
        if self.auto_confirm:
            self.complete_hangup(call_id)

    def complete_hangup(self, call_id: str) -> None:
        assert self._listener is not None
        self._listener.on_hangup_completed(call_id)

    def simulate_caller_hangup(self, call_id: str) -> None:
        """Test driver: the caller hangs up from the far end."""
        assert self._listener is not None
        self._listener.on_caller_hangup(call_id)

    def blind_transfer(self, call_id: str, pbx_target: str) -> None:
        """Record a transfer. The test completes it via :meth:`complete_transfer`."""
        self.transfers.append((call_id, pbx_target))

    def complete_transfer(self, call_id: str, result: TransferResult) -> None:
        """Test driver: the PBX answers the transfer with a normalized result."""
        assert self._listener is not None
        self._listener.on_transfer_result(call_id, result)


class FakeVoiceSession(VoiceSession):
    """Per-call voice stand-in. Records spoken audio, delivers scripted events."""

    def __init__(self, call_id: str, listener: VoiceListener) -> None:
        self.call_id = call_id
        self._listener = listener
        self.spoken: list[tuple[str, int]] = []
        self.closed = False

    def speak(self, text: str, turn_id: int) -> None:
        self.spoken.append((text, turn_id))

    def close(self) -> None:
        self.closed = True

    # -- test drivers ----------------------------------------------------

    def deliver_caller_speech(self, text: str) -> None:
        """Test driver: caller audio arrives as recognized text."""
        self._listener.on_transcript(text)

    def deliver_response(self, text: str, turn_id: int) -> None:
        """Test driver: the backend completes inference for a turn."""
        self._listener.on_response(turn_id, text)

    def finish_playback(self, turn_id: int) -> None:
        """Test driver: playout of a spoken turn completes."""
        self._listener.on_playback_finished(turn_id)

    def deliver_action_request(self, action: object) -> None:
        """Test driver: the backend emits a typed model action request."""
        self._listener.on_action_request(action)

    def deliver_failure(self, turn_id: int, failure: ProviderFailure) -> None:
        """Test driver: the provider operation for a turn fails.

        Called on the voice-session object that owned the attempt, so a
        superseded session delivers a stale generation automatically.
        """
        self._listener.on_provider_failure(turn_id, failure)


class FakeVoiceBackend:
    """Opens one deterministic voice session per call."""

    def __init__(self) -> None:
        self.sessions: dict[str, FakeVoiceSession] = {}
        self.all_sessions: list[FakeVoiceSession] = []

    def open_session(self, call_id: str, listener: VoiceListener) -> FakeVoiceSession:
        session = FakeVoiceSession(call_id, listener)
        self.sessions[call_id] = session
        self.all_sessions.append(session)
        return session


class FakeCallIds:
    """Deterministic call id generator for tests: call-1, call-2, ..."""

    def __init__(self) -> None:
        self._next = 1

    def next_id(self) -> str:
        call_id = f"call-{self._next}"
        self._next += 1
        return call_id


class FakePolicy:
    """Admission policy stand-in. Records every decision."""

    def __init__(self, allow: bool = True) -> None:
        self._allow = allow
        self.decisions: list[str] = []

    def should_answer(self, caller_id: str) -> bool:
        self.decisions.append(caller_id)
        return self._allow


class FailingMessageRepository(InMemoryMessageRepository):
    """Message store with scriptable failures for persistence semantics tests."""

    def __init__(self, clock) -> None:
        super().__init__(clock=clock)
        self.fail_script: list[Exception] = []
        self.saves = 0
        self.on_save = None

    def save(self, draft):
        self.saves += 1
        if self.on_save is not None:
            self.on_save()
        if self.fail_script:
            raise self.fail_script.pop(0)
        return super().save(draft)


class FailingCallRepository:
    """Call history with a scriptable save failure. Reads stay healthy so
    only the write path degrades."""

    def __init__(self, delegate=None) -> None:
        from receptionist.persistence import InMemoryCallRepository

        self._delegate = delegate or InMemoryCallRepository()
        self.fail_save: Exception | None = None

    def save(self, summary) -> None:
        if self.fail_save is not None:
            raise self.fail_save
        self._delegate.save(summary)

    def get(self, call_id: str):
        return self._delegate.get(call_id)

    def list_all(self):
        return self._delegate.list_all()

    def find_by_caller(self, caller_id: str):
        return self._delegate.find_by_caller(caller_id)

    def find_in_range(self, start: float, end: float):
        return self._delegate.find_in_range(start, end)

    def prune_before(self, cutoff: float) -> int:
        return self._delegate.prune_before(cutoff)


class FailingAuditLog:
    """Audit log with a scriptable record failure."""

    def __init__(self, delegate=None) -> None:
        from receptionist.persistence import InMemoryAuditLog

        self._delegate = delegate or InMemoryAuditLog()
        self.fail_record: Exception | None = None

    def record(self, event) -> None:
        if self.fail_record is not None:
            raise self.fail_record
        self._delegate.record(event)

    def list_all(self):
        return self._delegate.list_all()

    def prune_before(self, cutoff: float) -> int:
        return self._delegate.prune_before(cutoff)


class FailingTranscriptStore:
    """Transcript sidecar with a scriptable append failure."""

    def __init__(self, delegate=None) -> None:
        from receptionist.persistence import InMemoryTranscriptStore

        self._delegate = delegate or InMemoryTranscriptStore()
        self.fail_append: Exception | None = None

    def append(self, entry) -> None:
        if self.fail_append is not None:
            raise self.fail_append
        self._delegate.append(entry)

    def entries_for(self, call_id: str):
        return self._delegate.entries_for(call_id)

    def prune_before(self, cutoff: float) -> int:
        return self._delegate.prune_before(cutoff)


class RaisingConfigRepository:
    """Config authority that is suddenly unavailable: every read raises."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def get(self, key: str):
        raise self._error
