"""Per-call session: call lifecycle, active modes, and turn identity.

The session is driven by normalized telephony and voice-backend events.
Late events carrying an obsolete turn id are ignored so a cancelled turn
can never corrupt a newer one.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum

from receptionist.boundaries import (
    AuditDecision,
    AuditEvent,
    AuditLog,
    AudioFrame,
    CallOutcome,
    CallRepository,
    CallSummary,
    CancelReason,
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

LOG = logging.getLogger("receptionist.session")

#: Success acknowledgement, spoken only after the message commit succeeds.
MESSAGE_SAVED_ACK = "Su mensaje ha sido guardado. Gracias."

#: Short reprompt after one silence window. Kept constant, not configurable.
NO_INPUT_REPROMPT = "¿Sigue ahí? ¿En qué puedo ayudarle?"

#: Bound on transient caller PCM retained per turn for a bounded retry
#: re-commit. Memory only, never persisted; oldest frames drop first.
_MAX_RETAINED_AUDIO_FRAMES = 600

#: Same retention bounded by bytes (~64 s at 16 kHz mono): frame count
#: alone cannot bound memory when adapters choose chunk sizes.
_MAX_RETAINED_AUDIO_BYTES = 2 * 1024 * 1024


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

    def on_transcript_sidecar(self, text: str) -> None:
        if self._live():
            self._session.on_transcript_sidecar(text)

    def on_response(self, turn_id: int, text: str) -> None:
        if self._live():
            self._session.on_response(turn_id, text)

    def on_audio(self, turn_id: int, frame: AudioFrame) -> None:
        if self._live():
            self._session.on_audio(turn_id, frame)

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
    """One inbound call. Observable via ``state``, ``mode``, ``history``.

    Threading: the telephony dispatcher drives one thread; voice-backend
    workers may deliver events on others, and media input may arrive on
    a third. Every voice/media entry point serializes on an internal
    lock, so a barge-in, hangup, or timeout can preempt an in-flight
    turn. Telephony dispatch itself stays single-threaded (core-owned).
    """

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
        barge_in_enabled: bool = True,
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
        self._barge_in_enabled = barge_in_enabled
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
        # Transient caller PCM for the audio path: frames pushed since
        # the last turn commit, plus the snapshot consumed by it. Memory
        # only, bounded, cleared on turn boundaries; never persisted.
        self._pending_turn_audio: list[AudioFrame] = []
        self._pending_audio_bytes = 0
        self._last_committed_audio: list[AudioFrame] = []
        # Whether the current turn was opened by an audio barge-in and
        # still awaits its app-owned end-of-turn commit.
        self._audio_turn_open = False
        # Whether playback generation finished while adapter playout was
        # still draining (barge-in stays armed until it drains).
        self._awaiting_drain = False
        # Whether caller PCM ingest already failed on this turn: the
        # commit then fails the provider path terminally instead of
        # retrying an empty buffer forever.
        self._ingest_failed = False
        self._lock = threading.RLock()

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
    def voice_session(self) -> VoiceSession | None:
        """The live backend session, if admission opened one. Read-only
        observability seam (tests, diagnostics); never drive the call
        through it."""
        return self._voice_session

    def drain_voice(self, timeout: float = 5.0) -> bool:
        """Bounded rendezvous with the backend worker (async backends).
        No-op for synchronous sessions. Test/ops seam."""
        session = self._voice_session
        wait = getattr(session, "wait_until_idle", None)
        if not callable(wait):
            return True
        try:
            return bool(wait(timeout))
        except Exception:
            return False

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

    def _report_unhealthy(self, component: HealthComponent, code: str) -> None:
        if self._monitor is not None:
            self._monitor.report_unhealthy(component, code)

    def _report_recovered(self, component: HealthComponent, code: str) -> None:
        if self._monitor is not None:
            self._monitor.report_recovered(component, code)

    def _record_transcript(self, speaker: str, text: str) -> None:
        # The transcript boundary declares no error type, so any sidecar
        # failure is contained here: audio already played, the call
        # continues, degradation is flagged, nothing is swallowed blindly.
        # A later success clears both the flag and the condition, so the
        # aggregate health cannot disagree with the monitor.
        try:
            self._runtime.transcripts.append(
                TranscriptEntry(
                    call_id=self.call_id,
                    timestamp=self._clock.now(),
                    speaker=speaker,
                    text=text,
                )
            )
        except Exception:
            self._transcript_failed = True
            self._report_unhealthy(
                HealthComponent.TRANSCRIPT,
                CODE_TRANSCRIPT_UNAVAILABLE,
            )
        else:
            self._transcript_failed = False
            self._report_recovered(
                HealthComponent.TRANSCRIPT, CODE_TRANSCRIPT_UNAVAILABLE
            )

    def _speak(self, text: str, turn_id: int) -> None:
        """Play assistant audio; record it as observational text when enabled.

        A transcript sidecar failure never blocks audio: it is contained,
        flagged for degradation, and the call continues.
        """
        assert self._voice_session is not None  # opened in request_answer
        self._voice_session.speak(text, turn_id)
        if self._transcripts_enabled:
            self._record_transcript("assistant", text)

    def _cancel_voice_output(self, reason: CancelReason) -> None:
        """Best-effort output cancellation on the live voice session.

        Cancelling must never break teardown or routing: legacy fakes
        without `cancel_output` simply skip it (`close`/generation
        invalidation still applies), and a raising backend is contained.
        """
        session = self._voice_session
        if session is not None:
            cancel = getattr(session, "cancel_output", None)
            if callable(cancel):
                try:
                    cancel(reason)
                except Exception:
                    pass
        # Telephony playout always flushes, even when there is no voice
        # session (or it lacks cancel_output): queued RTP media for a
        # cancelled turn must never keep playing.
        self._flush_telephony_playout()

    def _provides_playback(self) -> bool:
        """Whether the live voice session streams its own AssistantAudio.

        Playback-capable backends (the cascaded TTS path) deliver audio
        via `on_audio`; the session must not re-speak their sidecar
        text. Legacy text-only sessions keep the `_speak` path.
        """
        session = self._voice_session
        if session is None:
            return False
        return bool(getattr(session, "provides_playback", False))

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
        self._audio_turn_open = False

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
        self._cancel_voice_output(CancelReason.CALLER_HANGUP)
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

    def handle_media_failed(self, detail: str = "") -> None:
        """Application decision for a dead local media stack on a live
        dialog (adapter streak trip). The caller is still there and the
        SIP leg still signals, so this enters the normal provider
        failure path: PBX fallback handoff when configured, otherwise a
        deterministic exit. Only ACTIVE sessions decide; closing/closed
        sessions already own their outcome, and an in-flight failure is
        never double-handled (late/repeated signals are no-ops)."""
        del detail
        if self._state is not CallState.ACTIVE:
            return
        if self._provider_terminal_failure:
            return
        self._fail_provider_path("local_media_failure")

    # -- voice events ------------------------------------------------------

    def _over_call_limit(self) -> bool:
        return (
            self._clock.now() - self._started_at
            >= self._policy_engine.limits.max_call_seconds
        )

    def _terminate_if_over_limit(self) -> bool:
        """End the call when the hard deadline passed. True when it did."""
        if self._over_call_limit():
            self._cancel_voice_output(CancelReason.CALL_LIMIT)
            self.end_call()
            return True
        return False

    def check_timeouts(self) -> bool:
        """Enforce time-based bounds outside conversational events.

        Terminates past-deadline calls; exits expired message capture
        back to listening; reprompts once after a silence window and
        falls back after the second. Completes a deferred
        playback-finished transition whose playout drained without any
        inbound media (silence-suppressed peer): without this poll the
        session would stay SPEAKING forever and no-input handling would
        never start. Returns True when the call was terminated.
        """
        with self._lock:
            if self._state in (CallState.TERMINATING, CallState.ENDED):
                return False
            if self._terminate_if_over_limit():
                return True
            # Drain completion is independent of inbound RTP: the peer
            # may send nothing while listening, so the periodic tick is
            # the backstop that flips SPEAKING to LISTENING once TX
            # drained. Cancellation/handoff clear _awaiting_drain, and a
            # newer turn ignores the stale flag via turn identity, so a
            # late drain can never move a newer turn.
            self._maybe_finish_drain()
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
        with self._lock:
            self._on_transcript_locked(text)

    def _on_transcript_locked(self, text: str) -> None:
        if self._state is not CallState.ACTIVE:
            return
        if self._transcripts_enabled:
            self._record_transcript("caller", text)
        if self._terminate_if_over_limit():
            return
        if self._mode is ActiveMode.MESSAGE_CAPTURE:
            # Partial speech during capture is transient only: the usable
            # text arrives as a typed final event, never parsed from here.
            self._enforce_capture_expiry()
            return
        if self._mode in (ActiveMode.GREETING, ActiveMode.SPEAKING):
            # Barge-in (enabled by default): caller speech over assistant
            # audio cancels the current output, discards undelivered
            # audio, invalidates the generation, and reopens listening as
            # a fresh turn. Late events from the cancelled turn carry a
            # stale epoch/turn and are ignored downstream.
            if not self._barge_in_enabled:
                return
            self._barge_in(CancelReason.BARGE_IN, text_complete=True)
            return
        if self._mode not in (ActiveMode.LISTENING, ActiveMode.INFERENCE):
            return
        if self._turn_count >= self._policy_engine.limits.max_turns:
            self.end_call()
            return
        self._last_input_at = self._clock.now()
        self._no_input_warned = False
        # Text-driven turn: no PCM backs it, so any retained audio from
        # an earlier turn must not leak into a retry of this one.
        self._pending_turn_audio = []
        self._pending_audio_bytes = 0
        self._last_committed_audio = []
        self._open_turn()

    def _barge_in(self, reason: CancelReason, *, text_complete: bool) -> None:
        """Cancel assistant output and reopen the turn for new input.

        Text barge-in carries the complete user input (transcript), so
        the fresh turn opens straight into INFERENCE. Audio barge-in
        caught only speech onset: the fresh turn opens into LISTENING
        to accumulate caller audio until the app-owned end-of-turn.
        """
        self._cancel_voice_output(reason)
        self._open_voice_attempt()
        self._awaiting_drain = False
        self._pending_turn_audio = []
        self._pending_audio_bytes = 0
        self._last_committed_audio = []
        if self._turn_count >= self._policy_engine.limits.max_turns:
            self.end_call()
            return
        self._last_input_at = self._clock.now()
        self._no_input_warned = False
        if text_complete:
            self._audio_turn_open = False
            self._open_turn()
            return
        self._turn += 1
        self._turn_count += 1
        self._attempts_made = 0
        self._turn_started_at = self._clock.now()
        self._mode = ActiveMode.LISTENING
        self._audio_turn_open = True

    def on_transcript_sidecar(self, text: str) -> None:
        """Observational caller text (primary STT reuse). Recorded only
        when transcripts are enabled; never opens a turn, never feeds
        evidence, never authorizes anything."""
        with self._lock:
            if self._state is not CallState.ACTIVE:
                return
            if self._transcripts_enabled:
                self._record_transcript("caller", text)

    def push_caller_audio(self, frame: AudioFrame) -> None:
        """Caller-facing PCM entry point for the media adapter (#25).

        Frames carrying speech energy while the assistant is speaking
        barge in; otherwise they are buffered for the current turn.
        Silence frames never cancel output. Audio is transient: never
        persisted here.
        """
        with self._lock:
            self._push_caller_audio_locked(frame)

    def _push_caller_audio_locked(self, frame: AudioFrame) -> None:
        if self._state is not CallState.ACTIVE:
            return
        if not isinstance(frame, AudioFrame):
            return
        self._maybe_finish_drain()
        if self._mode is ActiveMode.MESSAGE_CAPTURE:
            self._enforce_capture_expiry()
            return
        from receptionist.audio import is_speech

        speech = is_speech(frame)
        if self._mode in (ActiveMode.GREETING, ActiveMode.SPEAKING):
            if speech and self._barge_in_enabled:
                self._barge_in(CancelReason.BARGE_IN, text_complete=False)
                if self._voice_session is not None and self._state is CallState.ACTIVE:
                    self._ingest_locked(frame)
            return
        if self._mode not in (ActiveMode.LISTENING, ActiveMode.INFERENCE):
            return
        if self._voice_session is not None:
            self._ingest_locked(frame)

    def _ingest_locked(self, frame: AudioFrame) -> None:
        """Buffer one caller frame; ingest breakage is observable (log
        + terminal commit), never a silent empty-turn retry loop."""
        assert self._voice_session is not None
        try:
            self._voice_session.push_audio(frame)
            self._retain_frame(frame)
            self._ingest_failed = False
        except Exception as error:
            self._ingest_failed = True
            LOG.warning(
                "caller audio ingest failed call_id=%s turn=%s error=%s",
                self.call_id,
                self._turn,
                type(error).__name__,
            )

    def _retain_frame(self, frame: AudioFrame) -> None:
        """Keep transient PCM for a possible bounded retry re-commit."""
        self._pending_turn_audio.append(frame)
        self._pending_audio_bytes += len(frame.pcm)
        while (len(self._pending_turn_audio) > _MAX_RETAINED_AUDIO_FRAMES
               or self._pending_audio_bytes > _MAX_RETAINED_AUDIO_BYTES):
            dropped = self._pending_turn_audio.pop(0)
            self._pending_audio_bytes -= len(dropped.pcm)

    def commit_caller_turn(self) -> None:
        """App-owned end-of-user-turn: run the backend pipeline for the
        buffered caller audio of a fresh turn. No-op outside listening.
        The pipeline runs asynchronously; completion arrives as voice
        events on the backend worker."""
        with self._lock:
            self._commit_caller_turn_locked()

    def _commit_caller_turn_locked(self) -> None:
        if self._state is not CallState.ACTIVE:
            return
        self._maybe_finish_drain()
        if self._mode is not ActiveMode.LISTENING:
            return
        if self._terminate_if_over_limit():
            return
        if self._ingest_failed and not self._pending_turn_audio:
            # Media ingest is broken and nothing was buffered: fail the
            # provider path terminally instead of retrying emptiness.
            self._ingest_failed = False
            self._fail_provider_path("caller_audio_unavailable")
            return
        if self._turn_count >= self._policy_engine.limits.max_turns:
            self.end_call()
            return
        self._last_input_at = self._clock.now()
        self._no_input_warned = False
        if self._audio_turn_open:
            # An audio barge-in already opened this turn into LISTENING;
            # the end-of-turn commits it without opening another one.
            self._audio_turn_open = False
            self._mode = ActiveMode.INFERENCE
            self._turn_started_at = self._clock.now()
        else:
            self._open_turn()
        self._last_committed_audio = self._pending_turn_audio
        self._pending_turn_audio = []
        self._pending_audio_bytes = 0
        assert self._voice_session is not None
        try:
            self._voice_session.commit_turn(self._turn)
        except Exception:
            self._fail_provider_path("provider_failed")

    def on_audio(self, turn_id: int, frame: AudioFrame) -> None:
        """AssistantAudio: TTS PCM for the live turn. Late frames from a
        cancelled turn are ignored; audio is never persisted."""
        with self._lock:
            self._on_audio_locked(turn_id, frame)

    def _on_audio_locked(self, turn_id: int, frame: AudioFrame) -> None:
        if self._state is not CallState.ACTIVE:
            return
        if turn_id != self._turn:
            return  # late audio from an obsolete turn: ignore
        if self._terminate_if_over_limit():
            return
        if self._mode is ActiveMode.INFERENCE:
            self._note_provider_evidence()
            self._mode = ActiveMode.SPEAKING
        elif self._mode is not ActiveMode.SPEAKING:
            return
        # Close the playout seam (#25): valid assistant audio for the live
        # turn flows to the telephony media path. Late/cancelled audio was
        # already discarded above, so everything reaching here plays. The
        # adapter owns resampling/codec conversion; the session never
        # re-synthesizes text and never opens a second playback path.
        self._playout_to_telephony(frame)

    def _playout_to_telephony(self, frame: AudioFrame) -> None:
        """Forward one validated assistant frame to telephony playout.

        Best-effort and contained: a telephony failure must never break
        the conversational state machine (the mode transition above
        already happened). Adapters without a media path (legacy doubles
        in older tests) simply skip via duck-typing.
        """
        send = getattr(self._telephony, "send_audio", None)
        if not callable(send):
            return
        try:
            send(self.call_id, frame)
        except Exception:
            LOG.warning(
                "assistant playout failed call_id=%s turn=%s",
                self.call_id,
                self._turn,
            )

    def _flush_telephony_playout(self) -> None:
        """Discard queued TTS/RTP playout for this call (barge-in,
        hangup, handoff, failure). The voice-backend cancellation stops
        synthesis; this stops audio already handed to telephony so a
        cancelled turn cannot keep playing for seconds."""
        flush = getattr(self._telephony, "flush_audio", None)
        if not callable(flush):
            return
        try:
            flush(self.call_id)
        except Exception:
            pass

    def on_response(self, turn_id: int, text: str) -> None:
        with self._lock:
            self._on_response_locked(turn_id, text)

    def _on_response_locked(self, turn_id: int, text: str) -> None:
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
        if self._provides_playback():
            # Playback-capable backend: AssistantAudio follows via
            # on_audio; this text is a transcript/debug sidecar only and
            # must not be re-spoken (no double synthesis).
            if self._transcripts_enabled:
                self._record_transcript("assistant", text)
            return
        self._speak(text, turn_id)

    def on_playback_finished(self, turn_id: int) -> None:
        with self._lock:
            if self._state is not CallState.ACTIVE:
                return
        if turn_id != self._turn:
            return  # late event from an obsolete turn: ignore
        if self._terminate_if_over_limit():
            return
        if self._mode in (ActiveMode.GREETING, ActiveMode.SPEAKING):
            # Generation end is not RTP drain: the adapter may still
            # hold seconds of queued playout. Stay speaking (barge-in
            # armed) until the queue actually drains; the flip happens
            # in _maybe_finish_drain on the next media/commit event or
            # periodic tick.
            if self._playout_drained():
                self._mode = ActiveMode.LISTENING
            else:
                self._awaiting_drain = True

    def _playout_drained(self) -> bool:
        """Whether no assistant playout is still queued in telephony."""
        pending = getattr(self._telephony, "playout_pending_bytes", None)
        if not callable(pending):
            return True
        try:
            return int(pending(self.call_id)) <= 0
        except Exception:
            return True

    def _maybe_finish_drain(self) -> None:
        """Complete a deferred playback-finished transition once RTP
        actually drained. Called on media/commit entry points and on the
        periodic timeout tick (caller must hold the session lock). Only
        acts while _awaiting_drain is set; cancellation, barge-in,
        handoff, and finish all clear the flag, and turn identity guards
        the playback-finished setter, so a late drain never moves a
        newer turn."""
        if not self._awaiting_drain:
            return
        if not self._playout_drained():
            return
        self._awaiting_drain = False
        if self._mode in (ActiveMode.GREETING, ActiveMode.SPEAKING):
            self._mode = ActiveMode.LISTENING

    def on_action_request(self, action: object) -> None:
        with self._lock:
            self._on_action_request_locked(action)

    def _on_action_request_locked(self, action: object) -> None:
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
        with self._lock:
            self._on_provider_failure_locked(turn_id, failure)

    def _on_provider_failure_locked(
        self, turn_id: int, failure: ProviderFailure
    ) -> None:
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
            self._recommit_audio_for_retry()
            return
        self._fail_provider_path("provider_failed")

    def _recommit_audio_for_retry(self) -> None:
        """Re-drive the failed audio turn on the fresh attempt.

        The retry budget belongs to CallSession (#22); without this the
        new attempt would idle with an empty buffer and the retry would
        be fiction. Text-driven turns retain no audio and skip silently.
        A re-commit that itself raises fails the provider path.
        """
        if not self._last_committed_audio:
            return
        session = self._voice_session
        if session is None:
            return
        push = getattr(session, "push_audio", None)
        commit = getattr(session, "commit_turn", None)
        if not callable(push) or not callable(commit):
            return
        try:
            for frame in self._last_committed_audio:
                push(frame)
            commit(self._turn)
        except Exception:
            self._fail_provider_path("provider_failed")

    def _fail_provider_path(self, failure_category: str) -> None:
        """Record one terminal conversational failure and leave the AI
        path exactly once, toward PBX fallback or a safe exit. In-flight
        provider output is cancelled first (best-effort, contained):
        no more inference/TTS burns on a path already declared dead.
        The fallback handoff cancels again with its own reason when it
        runs; both are idempotent by backend contract."""
        self._provider_terminal_failure = True
        if self._breaker is not None:
            self._breaker.record_failure()
        self._failure_category = failure_category
        self._cancel_voice_output(CancelReason.PROVIDER_FAILED)
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
        self._cancel_voice_output(CancelReason.TRANSFER_HANDOFF)
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
        self._cancel_voice_output(CancelReason.FALLBACK_HANDOFF)
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
        # Discard any stale queued assistant audio before the apology so
        # only the apology itself can play.
        self._flush_telephony_playout()
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
            )
        else:
            self._audit_failed = False
            self._report_recovered(HealthComponent.RUNTIME, CODE_AUDIT_UNAVAILABLE)

    # -- local actions -----------------------------------------------------

    def end_call(self) -> None:
        if self._state in (CallState.TERMINATING, CallState.ENDED):
            return
        if self._state is CallState.INCOMING:
            self.reject()
            return
        self._cancel_voice_output(CancelReason.SHUTDOWN)
        self._pending_outcome = CallOutcome.COMPLETED
        self._transition(CallState.TERMINATING)
        self._telephony.hangup(self.call_id)

    # -- internals ---------------------------------------------------------

    def _finish(self, outcome: CallOutcome) -> None:
        self._transition(CallState.ENDED)
        self._awaiting_drain = False
        self._flush_telephony_playout()
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
            )
        else:
            self._history_failed = False
            self._report_recovered(
                HealthComponent.RUNTIME, CODE_HISTORY_UNAVAILABLE
            )
