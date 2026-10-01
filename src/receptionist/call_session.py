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
    MessageConfirmed,
    MessageDraft,
    MessageRejected,
    MessageTextFinal,
    StartMessageCapture,
    StoreUnavailableError,
    TelephonyAdapter,
    TransferRequest,
    TransferResult,
    TranscriptEntry,
    TransientStoreError,
    VoiceBackend,
    VoiceSession,
)
from receptionist.persistence import RuntimeStorage
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
    MESSAGE_CAPTURE = "message_capture"


#: Brief deterministic apology spoken before terminating a call whose
#: handoff and fallback both failed. Kept constant, not configurable.
EXIT_APOLOGY = "Lo siento, no fue posible comunicarle. La llamada terminará."

#: Success acknowledgement, spoken only after the message commit succeeds.
MESSAGE_SAVED_ACK = "Su mensaje ha sido guardado. Gracias."


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
        caller_name: str | None = None,
        telephony: TelephonyAdapter,
        voice: VoiceBackend,
        greeting: str,
        clock: Clock,
        policy_engine: PolicyEngine,
        runtime: RuntimeStorage,
        transcripts_enabled: bool = False,
    ) -> None:
        self.call_id = call_id
        self.caller_id = caller_id
        self._caller_name = caller_name
        self._telephony = telephony
        self._voice_backend = voice
        # Opened lazily in request_answer, so refused calls never acquire it.
        # Non-None whenever the session leaves INCOMING via admission.
        self._voice_session: VoiceSession | None = None
        self._greeting = greeting
        self._runtime = runtime
        self._clock = clock
        self._policy_engine = policy_engine
        self._transcripts_enabled = transcripts_enabled
        self._state = CallState.INCOMING
        self._mode: ActiveMode | None = None
        self._history = [CallState.INCOMING]
        self._turn = 0
        self._turn_count = 0
        self._attempts_used = 0
        self._pending_transfer: PendingTransfer | None = None
        self._pending_outcome: CallOutcome | None = None
        self._message_draft: MessageDraft | None = None
        self._message_id: str | None = None
        self._handoff_destination_id: str | None = None
        self._failure_category: str | None = None
        self._capture_started_at: float | None = None
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

    def _speak(self, text: str, turn_id: int) -> None:
        """Play assistant audio; record it as observational text when enabled."""
        assert self._voice_session is not None  # opened in request_answer
        self._voice_session.speak(text, turn_id)
        if self._transcripts_enabled:
            self._runtime.transcripts.append(
                TranscriptEntry(
                    call_id=self.call_id,
                    timestamp=self._clock.now(),
                    speaker="assistant",
                    text=text,
                )
            )

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
        self._transition(CallState.ACTIVE)
        self._mode = ActiveMode.GREETING
        self._turn = 1
        self._speak(self._greeting, self._turn)

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

    def check_timeouts(self) -> bool:
        """Enforce time-based bounds outside conversational events.

        Terminates past-deadline calls; exits expired message capture
        back to listening. Returns True when the call was terminated.
        """
        if self._state in (CallState.TERMINATING, CallState.ENDED):
            return False
        if self._terminate_if_over_limit():
            return True
        self._enforce_capture_expiry()
        return False

    def _enforce_capture_expiry(self) -> bool:
        """Leave expired message capture. True when it just expired."""
        if (
            self._mode is ActiveMode.MESSAGE_CAPTURE
            and self._capture_started_at is not None
            and self._clock.now() - self._capture_started_at
            >= self._policy_engine.limits.message_capture_seconds
        ):
            self._message_draft = None
            self._capture_started_at = None
            self._mode = ActiveMode.LISTENING
            return True
        return False

    def on_transcript(self, text: str) -> None:
        if self._state is not CallState.ACTIVE:
            return
        if self._transcripts_enabled:
            self._runtime.transcripts.append(
                TranscriptEntry(
                    call_id=self.call_id,
                    timestamp=self._clock.now(),
                    speaker="caller",
                    text=text,
                )
            )
        if self._terminate_if_over_limit():
            return
        if self._mode is ActiveMode.MESSAGE_CAPTURE:
            # Partial speech during capture is transient only: the usable
            # text arrives as a typed final event, never parsed from here.
            self._enforce_capture_expiry()
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
        self._mode = ActiveMode.SPEAKING
        self._speak(text, turn_id)

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
        if isinstance(action, TransferRequest):
            self._request_transfer(action.destination_id)
            return
        if isinstance(action, StartMessageCapture):
            self._start_capture()
            return
        if isinstance(action, MessageTextFinal):
            self._capture_text(action.text)
            return
        if isinstance(action, MessageConfirmed):
            self._confirm_capture()
            return
        if isinstance(action, MessageRejected):
            self._reject_capture()
            return
        # Privileged actions arrive only as typed objects. Spoken text,
        # transcripts, and any other channel can never open this path.
        self._audit_action("", AuditDecision.DENIED, detail=f"malformed action: {type(action).__name__}")

    def _request_transfer(self, destination_id: object) -> None:
        self._audit_action(
            destination_id if isinstance(destination_id, str) else "",
            AuditDecision.REQUESTED,
        )
        if self._state is not CallState.ACTIVE or self._mode not in (
            ActiveMode.LISTENING,
            ActiveMode.INFERENCE,
        ):
            self._deny(str(destination_id), "out of state")
            return
        resolution = self._policy_engine.resolve(destination_id)
        if resolution.status is not DestinationStatus.OK:
            self._deny(str(destination_id), resolution.status.value)
            return
        if not self._policy_engine.transfer_allowed(self._attempts_used):
            self._deny(str(destination_id), "transfer attempts exceeded")
            return
        assert resolution.destination is not None
        self._attempts_used += 1
        self._pending_transfer = PendingTransfer(str(destination_id))
        self._handoff_destination_id = str(destination_id)
        self._audit_action(
            str(destination_id),
            AuditDecision.ALLOWED,
            detail=f"target {resolution.destination.target}",
        )
        self._transition(CallState.TRANSFER_HANDOFF)
        self._mode = None
        self._telephony.blind_transfer(self.call_id, resolution.destination.target)

    # -- message capture ---------------------------------------------------

    def _start_capture(self) -> None:
        if self._state is not CallState.ACTIVE:
            return
        if self._mode is ActiveMode.MESSAGE_CAPTURE:
            return
        if self._mode not in (ActiveMode.LISTENING, ActiveMode.INFERENCE):
            return
        self._message_draft = None
        self._capture_started_at = self._clock.now()
        self._mode = ActiveMode.MESSAGE_CAPTURE

    def _capture_text(self, text: object) -> None:
        if self._state is not CallState.ACTIVE:
            return
        if self._mode is not ActiveMode.MESSAGE_CAPTURE:
            return
        if self._enforce_capture_expiry():
            return
        if not isinstance(text, str) or not text.strip():
            return
        self._message_draft = MessageDraft(
            call_id=self.call_id,
            caller_id=self.caller_id,
            caller_name=self._caller_name,
            text=text,
        )
        self._speak(
            f"Le confirmo su mensaje: {text}. ¿Es correcto?", self._turn
        )

    def _confirm_capture(self) -> None:
        if self._state is not CallState.ACTIVE:
            return
        if self._mode is not ActiveMode.MESSAGE_CAPTURE:
            return
        if self._enforce_capture_expiry():
            return
        if self._message_draft is None:
            return
        self._persist_confirmed_message()

    def _reject_capture(self) -> None:
        if self._state is not CallState.ACTIVE:
            return
        if self._mode is not ActiveMode.MESSAGE_CAPTURE:
            return
        if self._message_draft is None:
            return
        self._message_draft = None
        self._capture_started_at = None
        self._mode = ActiveMode.LISTENING

    def _persist_confirmed_message(self) -> None:
        assert self._message_draft is not None
        draft = self._message_draft
        max_retries = self._policy_engine.limits.message_persist_max_retries
        attempts = 0
        while True:
            try:
                record = self._runtime.messages.save(draft)
                break
            except TransientStoreError:
                attempts += 1
                if attempts > max_retries:
                    self._begin_message_fallback()
                    return
            except StoreUnavailableError:
                self._begin_message_fallback()
                return
        self._message_id = record.id
        self._message_draft = None
        self._capture_started_at = None
        self._mode = ActiveMode.LISTENING
        self._speak(MESSAGE_SAVED_ACK, self._turn)

    def _begin_message_fallback(self) -> None:
        self._message_draft = None
        self._capture_started_at = None
        self._failure_category = "message_persist_failed"
        if not self._attempt_fallback("message persist failed"):
            self._exit_after_failed_handoff()

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
            self._failure_category = "transfer_failed"
            if self._attempt_fallback(f"transfer {result.value}"):
                return
        self._exit_after_failed_handoff()

    def _attempt_fallback(self, origin: str) -> bool:
        """Start the configured fallback transfer. False when skipped."""
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
            return False
        if not self._policy_engine.transfer_allowed(self._attempts_used):
            self._audit_action(
                fallback.id,
                AuditDecision.DENIED,
                detail="fallback skipped: transfer attempts exceeded",
            )
            return False
        self._attempts_used += 1
        self._pending_transfer = PendingTransfer(fallback.id)
        self._handoff_destination_id = fallback.id
        self._audit_action(
            fallback.id, AuditDecision.REQUESTED, detail=f"fallback after {origin}"
        )
        self._audit_action(
            fallback.id, AuditDecision.ALLOWED, detail=f"target {fallback.target}"
        )
        self._transition(CallState.FALLBACK_HANDOFF)
        self._mode = None
        self._telephony.blind_transfer(self.call_id, fallback.target)
        return True

    def _exit_after_failed_handoff(self) -> None:
        """Single deterministic exit: one brief apology, then terminate."""
        if self._voice_session is not None:
            self._speak(EXIT_APOLOGY, self._turn)
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
        self._runtime.audit.record(
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
        self._runtime.calls.save(
            CallSummary(
                call_id=self.call_id,
                caller_id=self.caller_id,
                started_at=self._started_at,
                ended_at=self._clock.now(),
                outcome=outcome,
                turn_count=self._turn_count,
                caller_name=self._caller_name,
                handoff_destination_id=self._handoff_destination_id,
                message_id=self._message_id,
                failure_category=self._failure_category,
            )
        )
