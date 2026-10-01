"""Per-call session: call lifecycle, active modes, and turn identity.

The session is driven by normalized telephony and voice-backend events.
Late events carrying an obsolete turn id are ignored so a cancelled turn
can never corrupt a newer one.
"""

from __future__ import annotations

from enum import Enum

from receptionist.boundaries import (
    AuditDecision,
    AuditEvent,
    AuditLog,
    CallOutcome,
    CallRepository,
    CallSummary,
    Clock,
    TelephonyAdapter,
    TransferRequest,
    TransferResult,
    VoiceBackend,
    VoiceSession,
)
from receptionist.policy import DestinationStatus, PolicyEngine


class CallState(Enum):
    INCOMING = "incoming"
    ANSWERING = "answering"
    ACTIVE = "active"
    TRANSFER_HANDOFF = "transfer_handoff"
    FALLBACK_HANDOFF = "fallback_handoff"
    TERMINATING = "terminating"
    HANDED_OFF = "handed_off"
    ENDED = "ended"


class ActiveMode(Enum):
    GREETING = "greeting"
    LISTENING = "listening"
    INFERENCE = "inference"
    SPEAKING = "speaking"


#: Brief deterministic apology spoken before terminating a call whose
#: handoff and fallback both failed. Kept constant, not configurable.
EXIT_APOLOGY = "Lo siento, no fue posible comunicarle. La llamada terminará."


from dataclasses import dataclass


@dataclass(frozen=True)
class PendingTransfer:
    destination_id: str


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
        policy_engine: PolicyEngine,
        audit: AuditLog,
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
        self._policy_engine = policy_engine
        self._audit = audit
        self._state = CallState.INCOMING
        self._mode: ActiveMode | None = None
        self._history = [CallState.INCOMING]
        self._turn = 0
        self._turn_count = 0
        self._attempts_used = 0
        self._pending_transfer: PendingTransfer | None = None
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

    def _over_call_limit(self) -> bool:
        return (
            self._clock.now() - self._started_at
            >= self._policy_engine.limits.max_call_seconds
        )

    def _terminate_if_over_limit(self) -> bool:
        """End the call when the hard deadline passed. True when it did."""
        if self._over_call_limit():
            self.end_call()
            return True
        return False

    def check_deadline(self) -> bool:
        """Enforce the hard call deadline outside conversational events.

        The runtime calls this periodically; under test the fake clock
        advances and this method terminates deterministically, with no
        wall-clock sleeps and no background timers in the core.
        """
        if self._state in (CallState.TERMINATING, CallState.ENDED):
            return False
        return self._terminate_if_over_limit()

    def on_transcript(self, text: str) -> None:
        if self._state is not CallState.ACTIVE:
            return
        if self._terminate_if_over_limit():
            return
        if self._mode not in (ActiveMode.LISTENING, ActiveMode.INFERENCE):
            # Barge-in is deferred: speech over the greeting or while the
            # assistant is speaking does not open a turn in this slice.
            return
        if self._turn_count >= self._policy_engine.limits.max_turns:
            self.end_call()
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
        if self._terminate_if_over_limit():
            return
        assert self._voice_session is not None  # opened in request_answer
        self._mode = ActiveMode.SPEAKING
        self._voice_session.speak(text, turn_id)

    def on_playback_finished(self, turn_id: int) -> None:
        if self._state is not CallState.ACTIVE:
            return
        if turn_id != self._turn:
            return  # late event from an obsolete turn: ignore
        if self._terminate_if_over_limit():
            return
        if self._mode in (ActiveMode.GREETING, ActiveMode.SPEAKING):
            self._mode = ActiveMode.LISTENING

    def on_action_request(self, action: object) -> None:
        if self._terminate_if_over_limit():
            return
        # Privileged actions arrive only as typed objects. Spoken text,
        # transcripts, and any other channel can never open this path.
        if not isinstance(action, TransferRequest):
            self._audit_action("", AuditDecision.DENIED, detail=f"malformed action: {type(action).__name__}")
            return
        self._audit_action(action.destination_id, AuditDecision.REQUESTED)
        if self._state is not CallState.ACTIVE or self._mode not in (
            ActiveMode.LISTENING,
            ActiveMode.INFERENCE,
        ):
            self._deny(action.destination_id, "out of state")
            return
        resolution = self._policy_engine.resolve(action.destination_id)
        if resolution.status is not DestinationStatus.OK:
            self._deny(action.destination_id, resolution.status.value)
            return
        if not self._policy_engine.transfer_allowed(self._attempts_used):
            self._deny(action.destination_id, "transfer attempts exceeded")
            return
        assert resolution.destination is not None
        self._attempts_used += 1
        self._pending_transfer = PendingTransfer(action.destination_id)
        self._audit_action(
            action.destination_id,
            AuditDecision.ALLOWED,
            detail=f"target {resolution.destination.target}",
        )
        self._transition(CallState.TRANSFER_HANDOFF)
        self._mode = None
        self._telephony.blind_transfer(self.call_id, resolution.destination.target)

    def handle_transfer_result(self, result: TransferResult) -> None:
        if self._state not in (CallState.TRANSFER_HANDOFF, CallState.FALLBACK_HANDOFF):
            return
        pending = self._pending_transfer
        destination_id = pending.destination_id if pending is not None else ""
        if not isinstance(result, TransferResult):
            # Fail closed: anomaly, not a normal result. The raw value is
            # never echoed into the audit record.
            self._audit_action(
                destination_id,
                AuditDecision.DENIED,
                detail=f"invalid transfer result: {type(result).__name__}",
            )
            return
        self._audit_action(destination_id, AuditDecision.COMPLETED, result=result)
        if result is TransferResult.ACCEPTED_BY_PBX:
            self._pending_transfer = None
            self._transition(CallState.HANDED_OFF)
            self._finish(CallOutcome.TRANSFERRED)
            return
        self._pending_transfer = None
        if self._state is CallState.TRANSFER_HANDOFF:
            fallback = self._policy_engine.fallback()
            if fallback is None:
                reason = self._policy_engine.resolve(
                    self._policy_engine.fallback_id
                ).status.value
                self._audit_action(
                    self._policy_engine.fallback_id,
                    AuditDecision.DENIED,
                    detail=f"fallback {reason}",
                )
            elif not self._policy_engine.transfer_allowed(self._attempts_used):
                self._audit_action(
                    fallback.id,
                    AuditDecision.DENIED,
                    detail="fallback skipped: transfer attempts exceeded",
                )
            else:
                self._attempts_used += 1
                self._pending_transfer = PendingTransfer(fallback.id)
                self._audit_action(fallback.id, AuditDecision.REQUESTED, detail="fallback")
                self._audit_action(
                    fallback.id, AuditDecision.ALLOWED, detail=f"target {fallback.target}"
                )
                self._transition(CallState.FALLBACK_HANDOFF)
                self._telephony.blind_transfer(self.call_id, fallback.target)
                return
        self._exit_after_failed_handoff()

    def _exit_after_failed_handoff(self) -> None:
        """Single deterministic exit: one brief apology, then terminate."""
        if self._voice_session is not None:
            self._voice_session.speak(EXIT_APOLOGY, self._turn)
        self._pending_outcome = CallOutcome.COMPLETED
        self._transition(CallState.TERMINATING)
        self._telephony.hangup(self.call_id)

    def _deny(self, destination_id: str, reason: str) -> None:
        self._audit_action(destination_id, AuditDecision.DENIED, detail=reason)

    def _audit_action(
        self,
        destination: str,
        decision: AuditDecision,
        result: TransferResult | None = None,
        detail: str = "",
    ) -> None:
        self._audit.record(
            AuditEvent(
                timestamp=self._clock.now(),
                call_id=self.call_id,
                action="transfer",
                destination=destination,
                decision=decision,
                result=result,
                detail=detail,
            )
        )

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
