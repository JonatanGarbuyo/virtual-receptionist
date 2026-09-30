"""Project-owned test doubles for the ReceptionistCore / CallSession seam.

All fakes are deterministic: synchronous listener callbacks, no threads,
no wall-clock sleeps, no network access. They stand in for the system
boundaries (telephony, voice backend, clock, policy) defined in
``receptionist.boundaries``.
"""

from __future__ import annotations

from receptionist.boundaries import (
    TelephonyListener,
    VoiceListener,
    VoiceSession,
)


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


class FakeVoiceBackend:
    """Opens one deterministic voice session per call."""

    def __init__(self) -> None:
        self.sessions: dict[str, FakeVoiceSession] = {}

    def open_session(self, call_id: str, listener: VoiceListener) -> FakeVoiceSession:
        session = FakeVoiceSession(call_id, listener)
        self.sessions[call_id] = session
        return session


class FakePolicy:
    """Admission policy stand-in. Records every decision."""

    def __init__(self, allow: bool = True) -> None:
        self._allow = allow
        self.decisions: list[str] = []

    def should_answer(self, caller_id: str) -> bool:
        self.decisions.append(caller_id)
        return self._allow
