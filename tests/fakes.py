"""Project-owned test doubles for the ReceptionistCore / CallSession seam.

All fakes are deterministic: synchronous listener callbacks, no threads,
no wall-clock sleeps, no network access. They stand in for the system
boundaries (telephony, voice backend, clock, policy) defined in
``receptionist.boundaries``.
"""

from __future__ import annotations

from receptionist.boundaries import (
    AudioFrame,
    InboundCallHandler,
    ProviderFailure,
    TelephonyListener,
    TelephonyRegistrationState,
    TelephonyStatusListener,
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

    Media/doubles side (#25): records assistant playout per call
    (``playout``), supports bounded-queue semantics with ``flush_audio``,
    DTMF RX/TX drivers, hold/resume state, inbound admission through a
    project-owned handler, and registration-state hooks. Closed calls
    ignore late frames/callbacks, mirroring the production adapter.
    """

    #: Playout bound mirroring the production TX-queue policy shape:
    #: the fake keeps every frame (tests assert exact content) but
    #: tracks the byte bound so overflow behavior stays observable.
    PLAYOUT_MAX_BYTES = 32000

    def __init__(self, auto_confirm: bool = True) -> None:
        self._listener: TelephonyListener | None = None
        self._inbound: InboundCallHandler | None = None
        self._status: TelephonyStatusListener | None = None
        self.auto_confirm = auto_confirm
        self.answered: list[str] = []
        self.rejected: list[str] = []
        self.hung_up: list[str] = []
        self.transfers: list[tuple[str, str]] = []
        self.playout: dict[str, list[AudioFrame]] = {}
        self.playout_bytes: dict[str, int] = {}
        self.flushed: list[str] = []
        self.sent_dtmf: dict[str, list[str]] = {}
        self.held: set[str] = set()
        self.remote_held: set[str] = set()
        self.closed: set[str] = set()
        self.registration_state = TelephonyRegistrationState.STOPPING
        self.registration_transitions: list[TelephonyRegistrationState] = []
        self.refreshes = 0
        self.started = False
        self.shut_down = False
        self.shutdown_count = 0

    def set_listener(self, listener: TelephonyListener) -> None:
        self._listener = listener

    def set_inbound_handler(self, handler: InboundCallHandler | None) -> None:
        self._inbound = handler

    def set_status_listener(self, listener: TelephonyStatusListener | None) -> None:
        self._status = listener
        if listener is not None:
            listener.on_registration_state(self.registration_state)

    def start(self) -> None:
        self.started = True

    def refresh_registration(self) -> tuple[TelephonyRegistrationState, bool]:
        """Project-owned refresh seam: counts refreshes, keeps state,
        always fresh (the fake registrar answers immediately)."""
        self.refreshes += 1
        return self.registration_state, True

    def shutdown(self) -> None:
        self.shutdown_count += 1
        self.shut_down = True

    def report_registration(
        self, state: TelephonyRegistrationState, detail: str = ""
    ) -> None:
        """Test driver: the registrar answers / fails / is lost."""
        self.registration_state = state
        self.registration_transitions.append(state)
        if self._status is not None:
            self._status.on_registration_state(state, detail)

    def report_media(self, healthy: bool, call_id: str = "") -> None:
        """Test driver: the media path is lost / verified again."""
        if self._status is not None:
            report = getattr(self._status, "on_media_state", None)
            if callable(report):
                report(healthy, call_id)

    def answer(self, call_id: str) -> None:
        self.answered.append(call_id)
        if self.auto_confirm:
            self.complete_answer(call_id)

    def complete_answer(self, call_id: str) -> None:
        assert self._listener is not None
        self._listener.on_answered(call_id)

    def reject(self, call_id: str) -> None:
        self.rejected.append(call_id)
        self.closed.add(call_id)

    def hangup(self, call_id: str) -> None:
        self.hung_up.append(call_id)
        if self.auto_confirm:
            self.complete_hangup(call_id)

    def complete_hangup(self, call_id: str) -> None:
        assert self._listener is not None
        self.closed.add(call_id)
        self._listener.on_hangup_completed(call_id)

    def simulate_caller_hangup(self, call_id: str) -> None:
        """Test driver: the caller hangs up from the far end."""
        assert self._listener is not None
        if call_id in self.closed:
            return
        self.closed.add(call_id)
        self._listener.on_caller_hangup(call_id)

    def blind_transfer(self, call_id: str, pbx_target: str) -> None:
        """Record a transfer. The test completes it via :meth:`complete_transfer`."""
        self.transfers.append((call_id, pbx_target))

    def complete_transfer(self, call_id: str, result: TransferResult) -> None:
        """Test driver: the PBX answers the transfer with a normalized result."""
        assert self._listener is not None
        self._listener.on_transfer_result(call_id, result)

    # -- media ------------------------------------------------------

    def send_audio(self, call_id: str, frame: AudioFrame) -> None:
        """Record assistant playout. Late frames for closed calls drop,
        mirroring production exactly-once close semantics. After
        shutdown everything drops (the stack is gone)."""
        if self.shut_down or call_id in self.closed:
            return
        self.playout.setdefault(call_id, []).append(frame)
        self.playout_bytes[call_id] = self.playout_bytes.get(call_id, 0) + len(
            frame.pcm
        )

    def flush_audio(self, call_id: str) -> None:
        """Discard queued playout for one call (barge-in/cancel/hangup)."""
        self.flushed.append(call_id)
        self.playout.pop(call_id, None)
        self.playout_bytes.pop(call_id, None)

    def playout_pending_bytes(self, call_id: str) -> int:
        """Bytes still queued for playout (0 when none/unknown)."""
        return self.playout_bytes.get(call_id, 0)

    def drain_playout(self, call_id: str, max_bytes: int) -> int:
        """Test driver: playout consumes up to max_bytes (RTP drain).

        Returns bytes drained. Deterministic stand-in for the transmit
        clock: tests advance playout explicitly instead of sleeping.
        """
        pending = self.playout_bytes.get(call_id, 0)
        drained = min(pending, max(0, max_bytes))
        if drained:
            frames = self.playout.get(call_id, [])
            remaining = drained
            while frames and remaining > 0:
                head = len(frames[0].pcm)
                if head <= remaining:
                    remaining -= head
                    frames.pop(0)
                else:
                    break
            left = self.playout_bytes.get(call_id, 0) - drained
            if left <= 0:
                self.playout.pop(call_id, None)
                self.playout_bytes.pop(call_id, None)
            else:
                self.playout_bytes[call_id] = left
        return drained

    def simulate_caller_audio(self, call_id: str, frame: AudioFrame) -> None:
        """Test driver: decoded caller PCM arrives from the line."""
        assert self._listener is not None
        if self.shut_down or call_id in self.closed:
            return
        if call_id in self.held or call_id in self.remote_held:
            return  # on hold either way: no caller speech leaks through
        self._listener.on_caller_audio(call_id, frame)

    def send_dtmf(self, call_id: str, digits: str) -> None:
        self.sent_dtmf.setdefault(call_id, []).append(digits)

    def simulate_dtmf(self, call_id: str, digit: str) -> None:
        """Test driver: the far end sends one DTMF digit."""
        assert self._listener is not None
        if call_id in self.closed:
            return
        self._listener.on_dtmf(call_id, digit)

    def hold(self, call_id: str) -> None:
        self.held.add(call_id)

    def resume(self, call_id: str) -> None:
        """Resume from local hold, discarding playout queued while held
        (mirrors the adapter flushing stale media on resume)."""
        if call_id in self.held:
            self.held.discard(call_id)
            self.flush_audio(call_id)

    def simulate_remote_hold(self, call_id: str, held: bool) -> None:
        assert self._listener is not None
        if call_id in self.closed:
            return
        if held:
            self.remote_held.add(call_id)
        else:
            if call_id in self.remote_held:
                self.remote_held.discard(call_id)
                self.flush_audio(call_id)
        self._listener.on_remote_hold(call_id, held)

    def simulate_incoming(
        self, caller_id: str, caller_name: str | None = None
    ) -> str | None:
        """Test driver: a native INVITE arrives; admit through the
        project-owned inbound handler like the production adapter."""
        assert self._inbound is not None, "no inbound handler wired"
        return self._inbound.handle_incoming_call(caller_id, caller_name)


class FakeVoiceSession(VoiceSession):
    """Per-call voice stand-in. Records spoken audio, delivers scripted events."""

    provides_playback = False

    def __init__(self, call_id: str, listener: VoiceListener) -> None:
        self.call_id = call_id
        self._listener = listener
        self.spoken: list[tuple[str, int]] = []
        self.pushed_audio: list = []
        self.committed_turns: list[int] = []
        self.cancels: list = []
        self.audio_frames: list[tuple[int, object]] = []
        self.closed = False
        self.fail_speak: Exception | None = None

    def push_audio(self, frame) -> None:
        self.pushed_audio.append(frame)

    def commit_turn(self, turn_id: int) -> None:
        self.committed_turns.append(turn_id)

    def cancel_output(self, reason) -> None:
        self.cancels.append(reason)

    def speak(self, text: str, turn_id: int) -> None:
        if self.fail_speak is not None:
            raise self.fail_speak
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

    def deliver_audio(self, turn_id: int, frame) -> None:
        """Test driver: the backend streams one AssistantAudio frame."""
        self._listener.on_audio(turn_id, frame)

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


class FakeVoiceBackendWithReadiness(FakeVoiceBackend):
    """Voice backend with a controllable readiness hook for admission
    and recovery tests. Mirrors CascadedVoiceBackend.check_ready."""

    def __init__(self, ready: bool = True) -> None:
        super().__init__()
        self.backend_ready = ready

    def check_ready(self) -> tuple[bool, str]:
        if self.backend_ready:
            return True, "voice backend ready"
        return False, "voice backend not ready"


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
