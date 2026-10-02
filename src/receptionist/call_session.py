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
    ProviderFailure,
    ProviderFailureCategory,
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
from receptionist.alerting import (
    CODE_AUDIT_UNAVAILABLE,
    CODE_HISTORY_UNAVAILABLE,
    CODE_TRANSCRIPT_UNAVAILABLE,
    HealthComponent,
    HealthMonitor,
)
from receptionist.persistence import RuntimeStorage
from receptionist.policy import DestinationStatus, PolicyEngine
from receptionist.resilience import CircuitBreaker, ResilienceConfig, RetryPolicy


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

#: Short reprompt after one silence window. Kept constant, not configurable.
NO_INPUT_REPROMPT = "¿Sigue ahí? ¿En qué puedo ayudarle?"


class _VoiceAttemptListener:
    """Per-attempt voice listener. Every backend event is tagged with the
    generation captured when the attempt opened; events from any other
    generation are stale by construction and never reach the session, so
    a superseded attempt can neither speak again nor request actions."""

    def __init__(self, session: CallSession, epoch: int) -> None:
        self._session = session
        self._epoch = epoch

    def _live(self) -> bool:
        return self._session._voice_epoch == self._epoch

    def on_transcript(self, text: str) -> None:
        if self._live():
            self._session.on_transcript(text)

    def on_response(self, turn_id: int, text: str) -> None:
        if self._live():
            self._session.on_response(turn_id, text)

    def on_playback_finished(self, turn_id: int) -> None:
        if self._live():
            self._session.on_playback_finished(turn_id)

    def on_action_request(self, action: object) -> None:
        if self._live():
            self._session.on_action_request(action)

    def on_provider_failure(self, turn_id: int, failure: ProviderFailure) -> None:
        if self._live():
            self._session._on_provider_failure(turn_id, failure)


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
        resilience: ResilienceConfig | None = None,
        breaker: CircuitBreaker | None = None,
        monitor: HealthMonitor | None = None,
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
        self._resilience = resilience if resilience is not None else ResilienceConfig()
        self._retry_policy = RetryPolicy(self._resilience.provider_retries)
        self._breaker = breaker
        self._monitor = monitor
        self._voice_epoch = -1
        self._attempts_made = 0
        self._turn_started_at = clock.now()
        self._last_input_at: float | None = None
        self._no_input_warned = False
        self._provider_terminal_failure = False
        self._provider_evidence = False
        self._history_failed = False
        self._audit_failed = False
        self._transcript_failed = False
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

    @property
    def provider_terminal_failure(self) -> bool:
        """Whether a provider operation failed terminally on this call."""
        return self._provider_terminal_failure

    @property
    def provider_success_evidence(self) -> bool:
        """Whether the provider completed real work in the live generation
        (response text or a typed action while ACTIVE). Only this resets
        the breaker streak: mere call completion, hangup, or fallback-only
        admission is neutral and preserves failure evidence."""
        return self._provider_evidence

    def _note_provider_evidence(self) -> None:
        if self._state is CallState.ACTIVE:
            self._provider_evidence = True

    @property
    def observability_failed(self) -> bool:
        """Whether a history/audit/transcript write failed. Observability
        only: it never changes authorization or routing decisions."""
        return self._history_failed or self._audit_failed or self._transcript_failed

    def _transition(self, state: CallState) -> None:
        self._state = state
        self._history.append(state)

    def _report_unhealthy(self, component: HealthComponent, code: str, detail: str) -> None:
        if self._monitor is not None:
            self._monitor.report_unhealthy(component, code, detail)

    def _report_recovered(self, component: HealthComponent, code: str) -> None:
        if self._monitor is not None:
            self._monitor.report_recovered(component, code)

    def _speak(self, text: str, turn_id: int) -> None:
        """Play assistant audio; record it as observational text when enabled.

        A transcript sidecar failure never blocks audio: it is contained,
        flagged for degradation, and the call continues.
        """
        assert self._voice_session is not None  # opened in request_answer
        self._voice_session.speak(text, turn_id)
        if self._transcripts_enabled:
            # The transcript boundary declares no error type, so any sidecar
            # failure is contained here: audio already played, the call
            # continues, degradation is flagged, nothing is swallowed blindly.
            try:
                self._runtime.transcripts.append(
                    TranscriptEntry(
                        call_id=self.call_id,
                        timestamp=self._clock.now(),
                        speaker="assistant",
                        text=text,
                    )
                )
            except Exception:
                self._transcript_failed = True
                self._report_unhealthy(
                    HealthComponent.TRANSCRIPT,
                    CODE_TRANSCRIPT_UNAVAILABLE,
                    "transcript sidecar write failed",
                )
            else:
                self._report_recovered(
                    HealthComponent.TRANSCRIPT, CODE_TRANSCRIPT_UNAVAILABLE
                )

    def _open_voice_attempt(self) -> None:
        """Close the previous attempt (cancelling it) and open a fresh
        voice attempt with a new generation identity."""
        if self._voice_session is not None:
            self._voice_session.close()
        self._voice_epoch += 1
        self._voice_session = self._voice_backend.open_session(
            self.call_id, _VoiceAttemptListener(self, self._voice_epoch)
        )

    def _open_turn(self) -> None:
        """Start one logical turn. Attempts and retry budget belong to the
        turn; the deadline never resets on retry."""
        self._turn += 1
        self._turn_count += 1
        self._attempts_made = 0
        self._turn_started_at = self._clock.now()
        self._mode = ActiveMode.INFERENCE

    # -- admission --------------------------------------------------------

    def request_answer(self) -> None:
        """Move INCOMING -> ANSWERING, acquire voice, ask telephony to answer."""
        if self._state is not CallState.INCOMING:
            return
        self._transition(CallState.ANSWERING)
        self._open_voice_attempt()
        self._telephony.answer(self.call_id)

    def begin_fallback_only(self, reason: str) -> None:
        """Admission-time routing when the AI path is unavailable (open
        circuit or saturated capacity). No provider resources are opened;
        the call goes straight to the configured PBX fallback."""
        if self._state is not CallState.INCOMING:
            return
        self._failure_category = reason
        if not self._attempt_fallback(f"admission {reason}"):
            self._exit_after_failed_handoff()

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
        self._attempts_made = 0
        self._turn_started_at = self._clock.now()
        self._last_input_at = self._clock.now()
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
        back to listening; reprompts once after a silence window and
        falls back after the second. Returns True when the call was
        terminated.
        """
        if self._state in (CallState.TERMINATING, CallState.ENDED):
            return False
        if self._terminate_if_over_limit():
            return True
        self._enforce_capture_expiry()
        self._enforce_no_input()
        return False

    def _enforce_no_input(self) -> None:
        """One short reprompt after the first silence window, PBX fallback
        after the second. Only while listening: an outstanding inference,
        speech, or message capture has its own bounds."""
        if (
            self._state is not CallState.ACTIVE
            or self._mode is not ActiveMode.LISTENING
            or self._last_input_at is None
        ):
            return
        if (
            self._clock.now() - self._last_input_at
            < self._resilience.no_input_seconds
        ):
            return
        if not self._no_input_warned:
            self._no_input_warned = True
            self._last_input_at = self._clock.now()
            self._speak(NO_INPUT_REPROMPT, self._turn)
            return
        self._failure_category = "no_input"
        if not self._attempt_fallback("second no-input"):
            self._exit_after_failed_handoff()

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
            # Same containment as assistant-side transcripts (see _speak).
            try:
                self._runtime.transcripts.append(
                    TranscriptEntry(
                        call_id=self.call_id,
                        timestamp=self._clock.now(),
                        speaker="caller",
                        text=text,
                    )
                )
            except Exception:
                self._transcript_failed = True
                self._report_unhealthy(
                    HealthComponent.TRANSCRIPT,
                    CODE_TRANSCRIPT_UNAVAILABLE,
                    "transcript sidecar write failed",
                )
            else:
                self._report_recovered(
                    HealthComponent.TRANSCRIPT, CODE_TRANSCRIPT_UNAVAILABLE
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
        self._last_input_at = self._clock.now()
        self._no_input_warned = False
        self._open_turn()

    def on_response(self, turn_id: int, text: str) -> None:
        if self._state is not CallState.ACTIVE:
            return
        if self._mode is not ActiveMode.INFERENCE:
            return
        if turn_id != self._turn:
            return  # late event from an obsolete turn: ignore
        if self._terminate_if_over_limit():
            return
        self._note_provider_evidence()
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
        self._last_input_at = self._clock.now()
        self._no_input_warned = False
        if isinstance(action, TransferRequest):
            self._note_provider_evidence()
            self._request_transfer(action.destination_id)
            return
        if isinstance(action, StartMessageCapture):
            self._note_provider_evidence()
            self._start_capture()
            return
        if isinstance(action, MessageTextFinal):
            self._note_provider_evidence()
            self._capture_text(action.text)
            return
        if isinstance(action, MessageConfirmed):
            self._note_provider_evidence()
            self._confirm_capture()
            return
        if isinstance(action, MessageRejected):
            self._note_provider_evidence()
            self._reject_capture()
            return
        # Privileged actions arrive only as typed objects. Spoken text,
        # transcripts, and any other channel can never open this path.
        self._audit_action("", AuditDecision.DENIED, detail=f"malformed action: {type(action).__name__}")

    def _on_provider_failure(self, turn_id: int, failure: ProviderFailure) -> None:
        """Handle one provider failure for the current attempt, phase-aware.

        CANCELLED (app-owned) never retries and never feeds the breaker.
        INFERENCE failures retry bounded under the turn deadline, else go
        terminal. Failures in GREETING, SPEAKING, or LISTENING cannot be
        safely retried mid-phase (no playback resume, no backend
        switching), so a real one is terminal for the AI path toward PBX
        fallback. In MESSAGE_CAPTURE a failure means no usable final text
        can arrive: the draft is discarded with no save and no ACK, and
        the call routes to fallback. Stale generations never reach here.
        """
        if not isinstance(failure.category, ProviderFailureCategory):
            return  # not a normalized failure: never reason about it
        if self._state is not CallState.ACTIVE:
            return
        if turn_id != self._turn:
            return  # late event from an obsolete turn: ignore
        if failure.category is ProviderFailureCategory.CANCELLED:
            return
        if self._mode is ActiveMode.INFERENCE:
            self._handle_inference_failure(failure)
            return
        if self._mode is ActiveMode.MESSAGE_CAPTURE:
            self._message_draft = None
            self._capture_started_at = None
        if self._mode in (
            ActiveMode.GREETING,
            ActiveMode.SPEAKING,
            ActiveMode.LISTENING,
            ActiveMode.MESSAGE_CAPTURE,
        ):
            self._fail_provider_path("provider_failed")

    def _handle_inference_failure(self, failure: ProviderFailure) -> None:
        """Bounded retry of one outstanding inference, else terminal routing."""
        if (
            self._retry_policy.allows(failure.category, self._attempts_made)
            and not self._over_call_limit()
            and self._clock.now() - self._turn_started_at
            < self._resilience.turn_deadline_seconds
        ):
            self._attempts_made += 1
            self._open_voice_attempt()
            return
        self._fail_provider_path("provider_failed")

    def _fail_provider_path(self, failure_category: str) -> None:
        """Record one terminal conversational failure and leave the AI
        path exactly once, toward PBX fallback or a safe exit."""
        self._provider_terminal_failure = True
        if self._breaker is not None:
            self._breaker.record_failure()
        self._failure_category = failure_category
        if not self._attempt_fallback("provider failed"):
            self._exit_after_failed_handoff()

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
        """Single deterministic exit: one brief apology, then terminate.

        The apology is best-effort: when the voice path just failed
        terminally, speech may be unavailable, but termination never
        depends on the broken component. TERMINATING -> ENDED happens
        exactly once either way.
        """
        if self._voice_session is not None:
            try:
                self._speak(EXIT_APOLOGY, self._turn)
            except Exception:
                pass  # apology unavailable; terminate anyway, unconditionally
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
        """Record a privileged-action audit event. A store failure is
        contained and flagged: the audit trail is observability, never
        authority, so it cannot change the policy decision just made."""
        try:
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
        except (StoreUnavailableError, TransientStoreError):
            self._audit_failed = True
            self._report_unhealthy(
                HealthComponent.RUNTIME,
                CODE_AUDIT_UNAVAILABLE,
                "audit write failed",
            )
        else:
            self._report_recovered(HealthComponent.RUNTIME, CODE_AUDIT_UNAVAILABLE)

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
        try:
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
        except (StoreUnavailableError, TransientStoreError):
            # History is observability: a store write failure is flagged
            # for degradation, never allowed to break call teardown.
            self._history_failed = True
            self._report_unhealthy(
                HealthComponent.RUNTIME,
                CODE_HISTORY_UNAVAILABLE,
                "call history write failed",
            )
        else:
            self._report_recovered(
                HealthComponent.RUNTIME, CODE_HISTORY_UNAVAILABLE
            )
