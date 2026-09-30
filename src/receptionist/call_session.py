"""Per-call session: call lifecycle, active modes, and turn identity.

The session is driven by normalized telephony and voice-backend events.
Late events carrying an obsolete turn id are ignored so a cancelled turn
can never corrupt a newer one.
"""

from __future__ import annotations

from enum import Enum

from receptionist.boundaries import (
    CallOutcome,
    CallRepository,
    CallSummary,
    Clock,
    TelephonyAdapter,
    VoiceBackend,
    VoiceSession,
)


class CallState(Enum):
    INCOMING = "incoming"
    ANSWERING = "answering"
    ACTIVE = "active"
    TERMINATING = "terminating"
    ENDED = "ended"


class ActiveMode(Enum):
    GREETING = "greeting"
    LISTENING = "listening"
    INFERENCE = "inference"
    SPEAKING = "speaking"


class CallSession:
    """One inbound call. Observable via ``state``, ``mode``, ``history``."""

    def __init__(
        self,
        *,
        call_id: str,
        caller_id: str,
        telephony: TelephonyAdapter,
        voice: VoiceBackend,
        greeting: str,
        calls: CallRepository,
        clock: Clock,
    ) -> None:
        self.call_id = call_id
        self.caller_id = caller_id
        self._telephony = telephony
        self._voice_backend = voice
        # Opened lazily in request_answer, so refused calls never acquire it.
        # Non-None whenever the session leaves INCOMING via admission.
        self._voice_session: VoiceSession | None = None
        self._greeting = greeting
        self._calls = calls
        self._clock = clock
        self._state = CallState.INCOMING
        self._mode: ActiveMode | None = None
        self._history = [CallState.INCOMING]
        self._turn = 0
        self._turn_count = 0
        self._pending_outcome: CallOutcome | None = None
        self._started_at = clock.now()

    @property
    def state(self) -> CallState:
        return self._state

    @property
    def mode(self) -> ActiveMode | None:
        return self._mode

    @property
    def history(self) -> list[CallState]:
        return list(self._history)

    @property
    def current_turn(self) -> int:
        return self._turn

    def _transition(self, state: CallState) -> None:
        self._state = state
        self._history.append(state)

    # -- admission --------------------------------------------------------

    def request_answer(self) -> None:
        """Move INCOMING -> ANSWERING, acquire voice, ask telephony to answer."""
        if self._state is not CallState.INCOMING:
            return
        self._transition(CallState.ANSWERING)
        self._voice_session = self._voice_backend.open_session(self.call_id, self)
        self._telephony.answer(self.call_id)

    def reject(self) -> None:
        """Refuse the call. No media resources were acquired."""
        if self._state is not CallState.INCOMING:
            return
        self._telephony.reject(self.call_id)
        self._finish(CallOutcome.REJECTED)

    # -- telephony events (dispatched by the core) -------------------------

    def handle_answered(self) -> None:
        if self._state is not CallState.ANSWERING:
            return
        assert self._voice_session is not None  # opened in request_answer
        self._transition(CallState.ACTIVE)
        self._mode = ActiveMode.GREETING
        self._turn = 1
        self._voice_session.speak(self._greeting, self._turn)

    def handle_caller_hangup(self) -> None:
        if self._state in (CallState.TERMINATING, CallState.ENDED):
            return
        if self._state is CallState.INCOMING:
            self._telephony.reject(self.call_id)
            self._finish(CallOutcome.CALLER_HANGUP)
            return
        self._pending_outcome = CallOutcome.CALLER_HANGUP
        self._transition(CallState.TERMINATING)
        self._telephony.hangup(self.call_id)

    def handle_hangup_completed(self) -> None:
        if self._state is not CallState.TERMINATING:
            return
        self._finish(self._pending_outcome or CallOutcome.COMPLETED)

    # -- voice events ------------------------------------------------------

    def on_transcript(self, text: str) -> None:
        if self._state is not CallState.ACTIVE:
            return
        if self._mode not in (ActiveMode.LISTENING, ActiveMode.INFERENCE):
            # Barge-in is deferred: speech over the greeting or while the
            # assistant is speaking does not open a turn in this slice.
            return
        self._turn += 1
        self._turn_count += 1
        self._mode = ActiveMode.INFERENCE

    def on_response(self, turn_id: int, text: str) -> None:
        if self._state is not CallState.ACTIVE:
            return
        if self._mode is not ActiveMode.INFERENCE:
            return
        if turn_id != self._turn:
            return  # late event from an obsolete turn: ignore
        assert self._voice_session is not None  # opened in request_answer
        self._mode = ActiveMode.SPEAKING
        self._voice_session.speak(text, turn_id)

    def on_playback_finished(self, turn_id: int) -> None:
        if self._state is not CallState.ACTIVE:
            return
        if turn_id != self._turn:
            return  # late event from an obsolete turn: ignore
        if self._mode in (ActiveMode.GREETING, ActiveMode.SPEAKING):
            self._mode = ActiveMode.LISTENING

    # -- local actions -----------------------------------------------------

    def end_call(self) -> None:
        if self._state in (CallState.TERMINATING, CallState.ENDED):
            return
        if self._state is CallState.INCOMING:
            self.reject()
            return
        self._pending_outcome = CallOutcome.COMPLETED
        self._transition(CallState.TERMINATING)
        self._telephony.hangup(self.call_id)

    # -- internals ---------------------------------------------------------

    def _finish(self, outcome: CallOutcome) -> None:
        self._transition(CallState.ENDED)
        if self._voice_session is not None:
            self._voice_session.close()
        self._calls.save(
            CallSummary(
                call_id=self.call_id,
                caller_id=self.caller_id,
                started_at=self._started_at,
                ended_at=self._clock.now(),
                outcome=outcome,
                turn_count=self._turn_count,
            )
        )
