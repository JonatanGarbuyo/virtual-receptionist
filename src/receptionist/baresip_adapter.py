"""baresip/libbaresip telephony adapter behind the project-owned contract (#25).

Production SIP/media engine for v0.1. libbaresip (through the
``baresip-python`` binding) owns SIP and RTP; this module only adapts:

- registration events -> :class:`TelephonyRegistrationState`,
- call lifecycle -> :class:`TelephonyListener` (normalized, exactly-once),
- decoded caller PCM -> project-owned :class:`AudioFrame` (16 kHz mono),
- assistant :class:`AudioFrame` -> line rate PCM for RTP,
- DTMF RX/TX, hold/resume, blind-transfer primitive, clean shutdown.

What never crosses this file's public surface: baresip ``Call`` objects,
native handles/pointers, ``struct auframe``, RTP packets, codec payloads,
SIP dialogs/headers, passwords. Core code cannot import ``baresip``: the
binding is imported lazily inside :meth:`start` so unit tests and
environments without the binding still import this module.

Threading model (this decides whether the binding is usable):

- one asyncio loop thread owns the baresip ``Runtime``/``UserAgent`` and
  runs every baresip coroutine (register, answer, transfer, ...). Public
  sync methods never block on it: they mutate maps/queues under a lock
  and submit work with ``run_coroutine_threadsafe`` (fire-and-forget).
- one media thread pumps TX queues (``CallAudio.write``, thread-safe per
  the binding docs) and drains RX (``CallAudio.read``) roughly every
  10 ms. It never runs inference: it resamples, builds frames, and hands
  them to the listener.
- SIP/stack callbacks land on the loop thread; media callbacks on the
  media thread; inbound admission runs on a serial worker thread (never
  the SIP thread). All listener invocations are serialized per call,
  are lightweight (record + return), and never call back into blocking
  adapter methods while holding native locks (public methods only take
  the adapter lock briefly and never await the loop). Lock order is
  always adapter-lock then at most one per-call callback lock, never
  the reverse.

Backpressure policy (explicit): each call owns a bounded TX queue of
``TX_QUEUE_MAX_BYTES`` (about 2 s at 8 kHz mono16). ``send_audio`` never
blocks; on overflow the oldest bytes drop (live audio stays live) and a
counter records it. ``flush_audio`` discards everything queued plus
anything the stack has not yet encoded (``flush_tx``) so a cancelled
turn stops within one frame (~20 ms).
"""

from __future__ import annotations

import asyncio
import collections
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from receptionist.audio import make_frame, resample_pcm16
from receptionist.boundaries import (
    VALID_DTMF_DIGITS,
    AudioFrame,
    InboundCallHandler,
    TelephonyAdapter,
    TelephonyListener,
    TelephonyRegistrationState,
    TelephonyStatusListener,
    TransferResult,
)
from receptionist.telephony_config import TelephonyConfig

LOG = logging.getLogger("receptionist.telephony")

#: Binding accepted for v0.1 after the #25 matrix review (see
#: docs/evidence/25-baresip-matrix/). Pinned at install time; the wheel
#: statically bundles libre/libbaresip so no system SIP stack is needed.
#: Enforced at startup (fail fast on drift) and declared in
#: tools/sip/requirements-telephony.txt.
BARESIP_PYTHON_PIN = "baresip-python==0.5.2a3"

#: Settle window for refresh_registration() (see its docstring): the
#: stack answers both the unregister leg and the re-register leg with
#: 200 OK, and ua.register() resolves on the first outcome. Waiting for
#: outcome quiescence keeps callers from observing the registrar in the
#: transient contact-deleted state. _SETTLE_QUIET_S of silence ends the
#: wait early; _SETTLE_CAP_S bounds it (an auto-refresh firing mid-settle
#: only extends the wait up to the cap, never fails the call).
_REFRESH_SETTLE_QUIET_S = 1.0
_REFRESH_SETTLE_CAP_S = 5.0
_REFRESH_SETTLE_POLL_S = 0.05


def _check_binding_version(module: Any) -> None:
    """Fail fast when the installed binding is not the validated one.

    The matrix evidence describes exactly one binding build; running
    production telephony against another is an unvalidated
    configuration, so refuse loudly instead of degrading silently.
    """
    expected = BARESIP_PYTHON_PIN.split("==", 1)[1]
    actual = str(getattr(module, "__version__", ""))
    if actual != expected:
        raise RuntimeError(
            f"unsupported baresip-python {actual!r}: "
            f"this adapter is validated against {expected!r} "
            f"({BARESIP_PYTHON_PIN})"
        )

#: App-side PCM rate: inbound line audio is normalized here for the STT
#: path; outbound assistant audio is converted to the negotiated line
#: rate by the adapter (VoiceBackend never resamples for telephony).
APP_PCM_RATE = 16000

#: Fallback line rate when the stack has not reported one yet (G.711 is
#: always 8 kHz; this only covers the pre-establishment window).
DEFAULT_LINE_RATE = 8000

#: Per-call TX queue bound (~2 s at 8 kHz mono16). TTS arriving faster
#: than RTP playout drops oldest-first; never blocks the producer, never
#: accumulates seconds of stale audio after barge-in.
TX_QUEUE_MAX_BYTES = 32000

#: RX delivery chunk: 20 ms of 8 kHz mono16, resampled to 20 ms of 16 kHz.
RX_READ_BYTES = 640

#: Media pump period in seconds.
MEDIA_PERIOD = 0.01

#: Per-call RX dispatch queue bound (frames of 20 ms): caller audio
#: read from the stack is enqueued here by the media pump and delivered
#: to the listener by a dedicated per-call dispatcher thread. The pump
#: itself never runs listener code, so a call blocked inside
#: ``on_caller_audio`` (barge-in cancelling STT/LLM output, which can
#: take seconds) cannot stall TX/RX for other calls. Full queues drop
#: oldest-first and count ``rx_dropped_frames`` (same policy as TX).
RX_DISPATCH_MAX_FRAMES = 50

#: RX dispatcher poll period: prompt delivery via the per-record event,
#: with a bounded poll fallback so a missed wakeup never stalls a call.
RX_DISPATCH_POLL_S = 0.05

#: Consecutive media-failure trip: this many back-to-back TX (or RX)
#: stack failures on one call converts dead media into one
#: project-owned media-unhealthy transition plus one application
#: media-failure signal (the application owns fallback/termination).
#: At the 10 ms pump period this is ~0.5 s of fully dead media;
#: isolated blips never trip it, and any successful media resets the
#: streak. Fires exactly once per streak (equality, not >=), so a
#: lingering failure cannot spam health transitions.
MEDIA_ERROR_STREAK_LIMIT = 50

#: Lifecycle-event queue bound (DTMF backlog only): DTMF snapshots
#: waiting on one call's dispatcher. Terminal and answered events
#: (established/closed/transfer/media_failed) always append, and so
#: does every hold/resume snapshot (full FIFO: holds are rare, tiny,
#: and causal order across DTMF must survive, so they are never
#: coalesced or evicted). DTMF order is preserved among kept digits,
#: dropping oldest only as a documented last resort under
#: pathological flood.
EV_DISPATCH_MAX_NONTERMINAL = 128

#: Lifecycle kinds that are never dropped or coalesced: the answered
#: signal and every terminal outcome (close, transfer result, media
#: failure). The enqueue path guarantees capacity for these even
#: under non-terminal backpressure.
_EV_NO_DROP_KINDS = frozenset(
    {"established", "closed", "transfer", "media_failed"}
)


@dataclass
class _CallRecord:
    """Adapter-owned per-call state. The native call object lives here
    and never leaves this module."""

    app_id: str
    native: Any = None
    tx: collections.deque = field(default_factory=collections.deque)
    tx_bytes: int = 0
    tx_dropped_bytes: int = 0
    rx_seq: int = 0
    # RX dispatch: frames the media pump has read but not yet delivered
    # to the listener. A dedicated per-call dispatcher thread (started at
    # establishment) drains this FIFO in order; the pump only enqueues,
    # so listener work -- including blocking barge-in cancellation --
    # never stalls the shared media pump. Bounded oldest-drop with a
    # drop counter; pending audio is meaningless after close and is
    # discarded with the record.
    rx_dispatch: collections.deque = field(default_factory=collections.deque)
    rx_dropped_frames: int = 0
    # Lifecycle events from the stack (established/closed/hold/dtmf/
    # transfer/media_failed) as small tuples, drained FIFO by the same
    # per-call dispatcher before RX frames. Signaling policy is split:
    # answered/terminal events are never dropped or coalesced (rare,
    # tiny, exactly-once close semantics), and hold/resume snapshots
    # are full FIFO too (causal order across DTMF must survive, so
    # holds are never merged); DTMF keeps order, dropping oldest only
    # as a last resort when its backlog exceeds
    # EV_DISPATCH_MAX_NONTERMINAL. The media gate itself (remote_hold
    # flag plus queue flushes) is applied synchronously in
    # _gate_remote_hold on the stack callback -- never deferred to the
    # dispatcher -- so no TX/RX crosses a HOLD boundary.
    ev_dispatch: collections.deque = field(default_factory=collections.deque)
    rx_dispatch_started: bool = False
    rx_thread: Any = None
    rx_ready: threading.Event = field(default_factory=threading.Event)
    tx_rate: int = DEFAULT_LINE_RATE
    rx_rate: int = DEFAULT_LINE_RATE
    local_hold: bool = False
    remote_hold: bool = False
    local_close: bool = False
    transfer_open: bool = False
    transfer_watchdog: Any = None
    # Decline-release legs skip the media pump entirely (no RX delivery,
    # no TX, no audio probing): the call is being released, media is
    # meaningless, and keeping audio idle also keeps the post-answer
    # media refresh from racing the release BYE (deterministic 200+BYE).
    decline_release: bool = False
    close_emitted: bool = False
    hangup_completed_emitted: bool = False
    remote_closed: bool = False
    established: bool = False
    listeners_attached: bool = False
    audio_restarts: int = 0
    tx_accepted_total: int = 0
    first_tx_at: float | None = None
    # Media failure counters: every TX write / RX read-or-resample
    # failure increments exactly one of these. They never raise, never
    # block, and feed the media-health hook; silence here would be a
    # live call with dead media and no signal.
    tx_errors: int = 0
    rx_errors: int = 0
    # Consecutive-failure streaks backing the bounded trip policy (see
    # MEDIA_ERROR_STREAK_LIMIT): any successful media resets the
    # streak for that direction.
    tx_error_streak: int = 0
    rx_error_streak: int = 0


class BaresipTelephonyAdapter(TelephonyAdapter):
    """Project-owned adapter over baresip/libbaresip (``baresip-python``).

    Construct with a :class:`TelephonyConfig`, wire with
    :func:`wire_baresip_core` (or set the listener/inbound/status hooks
    manually), then :meth:`start`. All call ids are application ids
    issued by ``CallIdGenerator`` via the core; the native <-> app
    mapping lives in ``_calls``/``_pending`` and is keyed by app id
    after admission (never by Caller-ID, never by SIP Call-ID).
    """

    def __init__(self, config: TelephonyConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        # Per-call callback locks: listener delivery for one call never
        # blocks another call's SIP/media callbacks. Re-entrant delivery
        # on the same thread stays safe (RLock: remote BYE ->
        # caller_hangup -> hangup -> hangup_completed). Lock order is
        # one-way only: a callback lock may be followed by _lock
        # (listener re-entrancy into adapter methods), but _lock is
        # never held across delivery -- snapshots happen first, then
        # the lock is released and _deliver() runs lock-free apart from
        # the per-call serializer. Entries are dropped with their call
        # record.
        self._cb_locks: dict[str, threading.RLock] = {}
        # RX dispatcher threads by call id (see _ensure_rx_dispatch):
        # joined boundedly at shutdown, self-exiting on record removal.
        self._rx_threads: dict[str, threading.Thread] = {}
        self._admit_queue: queue.Queue | None = None
        self._admit_thread: threading.Thread | None = None
        self._admit_inflight = 0
        self._listener: TelephonyListener | None = None
        self._inbound: InboundCallHandler | None = None
        self._status: TelephonyStatusListener | None = None
        self._calls: dict[str, _CallRecord] = {}
        self._pending: collections.deque = collections.deque()
        self._reg_state = TelephonyRegistrationState.STOPPING
        self._running = False
        self._started_once = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._media_thread: threading.Thread | None = None
        self._runtime: Any = None
        self._ua: Any = None
        self._declined_total = 0
        self._register_ok_count = 0
        # Bounded native-event timeline (diagnostics): (t, app_id, kind).
        self._event_log: collections.deque = collections.deque(maxlen=200)

    def event_timeline(self) -> list[tuple[float, str, str]]:
        """Copy of the recent native-event timeline (diagnostics only)."""
        with self._lock:
            return list(self._event_log)

    def _note_event(self, app_id: str, kind: str) -> None:
        with self._lock:
            self._event_log.append((time.time(), app_id, kind))

    # -- project-owned hooks -------------------------------------------

    def set_listener(self, listener: TelephonyListener) -> None:
        with self._lock:
            self._listener = listener

    def set_inbound_handler(self, handler: InboundCallHandler | None) -> None:
        with self._lock:
            self._inbound = handler

    def set_status_listener(self, listener: TelephonyStatusListener | None) -> None:
        with self._lock:
            self._status = listener
            state = self._reg_state
        if listener is not None:
            try:
                listener.on_registration_state(state)
            except Exception:
                pass

    @property
    def registration_state(self) -> TelephonyRegistrationState:
        with self._lock:
            return self._reg_state

    @property
    def config_summary(self) -> str:
        """Log-safe identity (no secrets)."""
        return self._config.redacted_summary

    # -- lifecycle ----------------------------------------------------

    def start(self, *, timeout: float = 15.0) -> TelephonyRegistrationState:
        """Bring up the stack and REGISTER. Blocks (bounded) until the
        registrar answers or ``timeout`` expires; returns the resulting
        registration state. Safe to call once; later calls are no-ops
        returning the current state."""
        with self._lock:
            if self._started_once:
                return self._reg_state
            self._started_once = True
            self._running = True
            self._set_reg_state_locked(TelephonyRegistrationState.STARTING)
        try:
            import baresip  # noqa: F401  -- lazy: unit tests never reach here
        except Exception as error:
            LOG.warning("telephony binding unavailable: %s", type(error).__name__)
            self._set_reg_state(TelephonyRegistrationState.REGISTRATION_FAILED)
            return self.registration_state
        ready = threading.Event()
        failure: list[str] = []

        def runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with self._lock:
                self._loop = loop
            try:
                loop.run_until_complete(self._async_start())
            except Exception as error:  # startup/register failure path
                failure.append(type(error).__name__)
                self._set_reg_state(TelephonyRegistrationState.REGISTRATION_FAILED)
            finally:
                ready.set()
                if self._is_running():
                    loop.run_forever()
                try:
                    loop.run_until_complete(self._async_stop_quietly())
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass

        thread = threading.Thread(
            target=runner, name="baresip-loop", daemon=True
        )
        with self._lock:
            self._loop_thread = thread
        thread.start()
        ready.wait(timeout)
        if not ready.is_set():
            self._set_reg_state(TelephonyRegistrationState.REGISTRATION_FAILED)
            return self.registration_state
        if failure or self.registration_state not in (
            TelephonyRegistrationState.REGISTERED,
        ):
            if self.registration_state is TelephonyRegistrationState.STARTING:
                self._set_reg_state(TelephonyRegistrationState.REGISTRATION_FAILED)
            return self.registration_state
        self._start_media_thread()
        self._start_admit_thread()
        return self.registration_state

    def refresh_registration(
        self, *, timeout: float = 15.0
    ) -> tuple[TelephonyRegistrationState, bool]:
        """Send one REGISTER refresh on the loop thread (bounded wait).

        This is the same stack path as automatic expiry refresh, exposed
        so operators and tests can verify re-registration deterministically
        without waiting out ``reg_interval``. Returns ``(state, fresh)``
        where ``fresh`` is True only if the registrar confirmed *this*
        refresh (a REGISTER_OK arrived while waiting) -- never a stale
        previous state. The outcome also surfaces through the status
        listener (REGISTERED, or REGISTRATION_FAILED/LOST); transport
        errors never propagate to the caller as vendor types.

        Stack note: the native ``ua_register()`` refreshes by first
        unregistering the old dialog (expires-0 REGISTER, itself answered
        with 200 OK and therefore a REGISTER_OK event) and then
        registering a new dialog (challenged, retried with auth, answered
        200 OK -- a second REGISTER_OK). ``ua.register()`` resolves on the
        *first* outcome, so this method additionally waits for outcome
        quiescence (no new REGISTER_OK for ``_SETTLE_QUIET_S``) bounded by
        ``_SETTLE_CAP_S`` before returning. Without the settle, a caller
        querying the registrar immediately after ``fresh=True`` can catch
        the contact deleted (unregister processed, re-register still in
        flight) -- a real, registrar-observable race, not a stale read.
        """
        loop = self._current_loop()
        with self._lock:
            ua = self._ua
            mark = self._register_ok_count
        if loop is None or ua is None or not self._is_running():
            return self.registration_state, False

        async def run() -> None:
            try:
                await ua.register()
            except Exception as error:
                name = type(error).__name__
                LOG.debug("registration refresh failed: %s", name)

        try:
            future = asyncio.run_coroutine_threadsafe(run(), loop)
            future.result(timeout=timeout)
        except Exception:
            pass
        # Settle past the re-register outcome(s): the first REGISTER_OK
        # may belong to the unregister leg. Quiescence (no new REGISTER_OK
        # for _SETTLE_QUIET_S, bounded by _SETTLE_CAP_S) is behaviour
        # agnostic -- it works whether the stack sends one outcome or
        # several -- and never fails the call, it only delays it.
        settle_end = time.monotonic() + _REFRESH_SETTLE_CAP_S
        last_change = time.monotonic()
        with self._lock:
            last_count = self._register_ok_count
        while time.monotonic() < settle_end:
            time.sleep(_REFRESH_SETTLE_POLL_S)
            with self._lock:
                count = self._register_ok_count
            if count != last_count:
                last_count = count
                last_change = time.monotonic()
            elif time.monotonic() - last_change >= _REFRESH_SETTLE_QUIET_S:
                break
        with self._lock:
            fresh = self._register_ok_count > mark
            state = self._reg_state
        return state, fresh

    def shutdown(self) -> None:
        """Idempotent shutdown: stop accepting, drop media queues, hang
        up live calls (best effort), unregister, free the stack, join
        worker threads bounded, silence further callbacks."""
        with self._lock:
            if not self._started_once:
                self._reg_state = TelephonyRegistrationState.STOPPED
                return
            already = not self._running
            self._running = False
            self._set_reg_state_locked(TelephonyRegistrationState.STOPPING)
            loop = self._loop
            loop_thread = self._loop_thread
            media_thread = self._media_thread
            admit_thread = self._admit_thread
            admit_queue = self._admit_queue
            rx_threads = list(self._rx_threads.values())
            for record in list(self._calls.values()):
                record.rx_ready.set()
            calls = list(self._calls.values())
            self._pending.clear()
        for record in calls:
            record.local_close = True
            self._close_native_best_effort(record)
        # Graceful (bounded): let the BYEs complete so closes are
        # observed instead of cut. Records stay mapped until their
        # close arrives (or the deadline); the map is cleared at the
        # end unconditionally. Callbacks keep flowing until loop stop
        # (listener still set); _is_running gates new work only.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self._lock:
                if not self._calls:
                    break
            time.sleep(0.1)
        if loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._async_shutdown_stack(), loop
                )
                future.result(timeout=10)
            except Exception:
                pass
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass
        if loop_thread is not None and loop_thread is not threading.current_thread():
            loop_thread.join(timeout=10)
        if media_thread is not None and media_thread is not threading.current_thread():
            media_thread.join(timeout=5)
        if admit_queue is not None:
            try:
                admit_queue.put(None)
            except Exception:
                pass
        if admit_thread is not None and admit_thread is not threading.current_thread():
            admit_thread.join(timeout=5)
        me = threading.current_thread()
        for rx_thread in rx_threads:
            if rx_thread is not me:
                rx_thread.join(timeout=5)
        with self._lock:
            self._loop = None
            self._loop_thread = None
            self._media_thread = None
            self._admit_thread = None
            self._admit_queue = None
            self._admit_inflight = 0
            self._rx_threads.clear()
            self._runtime = None
            self._ua = None
            self._calls.clear()
            self._pending.clear()
            self._cb_locks.clear()
            self._set_reg_state_locked(TelephonyRegistrationState.STOPPED)

    # -- call control (non-blocking, never raise to the core) ----------

    def answer(self, call_id: str) -> None:
        record = self._bind_pending_if_needed(call_id)
        with self._lock:
            record = record or self._calls.get(call_id)
        if record is None:
            return
        self._submit(record, lambda call: call.answer())

    #: Decline retry schedule (seconds): the native call handle
    #: registers asynchronously after CALL_INCOMING, so an immediate
    #: reject can lose the race even on a healthy dialog. Retries make
    #: 486 the deterministic outcome; only a genuinely gone leg (remote
    #: CANCEL winning every round) falls through to answer+release.
    _REJECT_RETRY_DELAYS = (0.1, 0.5, 2.0)

    def reject(self, call_id: str) -> None:
        """Decline an inbound call with a SIP rejection (486).

        The native ``reject()`` is attempted first, retried with
        backoff on ``StaleHandleError``; only a leg that never validates
        is released via public answer+BYE (:meth:`_answer_then_release`)
        so the slot never leaks. Either way the core outcome is
        identical (released, slot freed, exactly-once close, no AI
        resources opened). No patching, no unsafe handles, no vendor
        types cross the boundary either way.
        """
        record = self._bind_pending_if_needed(call_id)
        with self._lock:
            record = record or self._calls.get(call_id)
            if record is None:
                return
            if record.remote_closed:
                # Far end already gone (CANCEL before answer): the core
                # finishes on reject() alone; drop the closed record.
                self._calls.pop(call_id, None)
                self._drop_cb_lock(call_id)
                return
            record.local_close = True
            record.decline_release = True
        loop = self._current_loop()
        if loop is None or not self._is_running():
            return

        async def run() -> None:
            for attempt, delay in enumerate(
                (0.0,) + self._REJECT_RETRY_DELAYS
            ):
                if delay:
                    await asyncio.sleep(delay)
                try:
                    result = record.native.reject()
                    if asyncio.iscoroutine(result):
                        await result
                    return
                except Exception as error:
                    if type(error).__name__ != "StaleHandleError":
                        LOG.debug(
                            "telephony op failed: %s", type(error).__name__
                        )
                        return
                    LOG.debug("decline reject race, attempt %d", attempt)
            # Genuinely gone (or never validated): release via
            # answer+BYE so no slot leaks.
            self._answer_then_release(record)

        try:
            asyncio.run_coroutine_threadsafe(run(), loop)
        except Exception:
            pass

    def hangup(self, call_id: str) -> None:
        # Lock discipline: never hold _lock while delivering (listener
        # callbacks re-enter the adapter and take _lock; holding it
        # across delivery would ABBA-deadlock with the media thread).
        complete_now = False
        with self._lock:
            record = self._calls.get(call_id)
            if record is None:
                # Late hangup long after close: ignore (exactly-once).
                return
            record.local_close = True
            record.tx.clear()
            record.tx_bytes = 0
            if record.remote_closed:
                # Far end already closed: our BYE would be meaningless;
                # local cleanup is trivially complete.
                complete_now = True
        if complete_now:
            self._emit_hangup_completed(call_id)
            return
        self._submit(record, lambda call: call.hangup())

    def blind_transfer(self, call_id: str, pbx_target: str) -> None:
        """Emit the REFER primitive to an already-trusted PBX target.

        No parsing, no resolution, no caller/model-derived destinations:
        ``pbx_target`` arrives resolved by policy. Outcomes normalize to
        :class:`TransferResult` on the listener; synchronous failures
        (unknown call, unestablished leg) emit TRANSPORT_ERROR instead
        of raising, so the session can fall back deterministically."""
        if not isinstance(pbx_target, str) or not pbx_target.strip():
            self._emit_transfer_result(call_id, TransferResult.TRANSPORT_ERROR)
            return
        record = self._bind_pending_if_needed(call_id)
        with self._lock:
            record = record or self._calls.get(call_id)
            if record is None:
                LOG.info("blind_transfer: unknown call, ignoring")
                return
            if record.transfer_open:
                LOG.info("blind_transfer: transfer already open, ignoring")
                return
            record.transfer_open = True
            # Host-thread watchdog: if the binding never resolves the
            # transfer (stuck command, lost NOTIFY), report TIMEOUT so
            # the session can fall back instead of hanging forever. It
            # only fires while the transfer is still open; the normal
            # outcome cancels it. Loop-independent by construction.
            try:
                timer = threading.Timer(
                    70.0, self._transfer_watchdog_fired, args=(record.app_id,)
                )
                timer.daemon = True
                record.transfer_watchdog = timer
                timer.start()
            except Exception:
                pass
        self._submit_transfer(record, pbx_target.strip())

    def _transfer_watchdog_fired(self, app_id: str) -> None:
        with self._lock:
            record = self._calls.get(app_id)
            if record is None or not record.transfer_open:
                return
            record.transfer_open = False
            record.transfer_watchdog = None
        LOG.warning("transfer watchdog: no outcome, reporting TIMEOUT")
        self._emit_transfer_result(app_id, TransferResult.TIMEOUT)

    def hold(self, call_id: str) -> None:
        with self._lock:
            record = self._calls.get(call_id)
            if record is None or record.local_hold:
                return
            record.local_hold = True  # pause RX delivery immediately
        self._submit(record, lambda call: call.hold(), revert_hold=False)

    def resume(self, call_id: str) -> None:
        with self._lock:
            record = self._calls.get(call_id)
            if record is None or not record.local_hold:
                return
            record.local_hold = False
        # Fresh media after resume: discard anything queued or buffered
        # while held (stale TTS / pre-resume line audio must not blast
        # out post-hold).
        self._flush_record(record)
        self._submit(record, lambda call: call.resume(), revert_hold=True)

    def _flush_record(self, record: _CallRecord) -> None:
        """Discard queued TX and buffered RX for one call (hold/resume,
        barge-in paths). Best-effort, never raises."""
        with self._lock:
            record.tx.clear()
            record.tx_bytes = 0
            native = record.native
        if native is None:
            return
        try:
            native.audio.flush_tx()
        except Exception:
            pass
        for _ in range(64):
            try:
                chunk = native.audio.read(RX_READ_BYTES)
            except Exception:
                break
            if not chunk:
                break

    def send_dtmf(self, call_id: str, digits: str) -> None:
        if not isinstance(digits, str) or not digits:
            return
        clean = digits.upper()
        if any(ch not in VALID_DTMF_DIGITS for ch in clean):
            return
        with self._lock:
            record = self._calls.get(call_id)
        if record is None:
            return
        self._submit(record, lambda call, d=clean: call.send_dtmf(d))

    # -- media ----------------------------------------------------------

    def send_audio(self, call_id: str, frame: AudioFrame) -> None:
        """Queue assistant PCM for RTP playout. Non-blocking from any
        thread; invalid frames and unknown/closed calls drop silently."""
        if not isinstance(frame, AudioFrame):
            return
        if frame.sample_format != "pcm16" or frame.channels != 1:
            return
        with self._lock:
            record = self._calls.get(call_id)
            if record is None or record.close_emitted:
                return
            try:
                pcm = resample_pcm16(bytes(frame.pcm), frame.sample_rate, record.tx_rate)
            except Exception:
                return
            record.tx.append(pcm)
            record.tx_bytes += len(pcm)
            while record.tx_bytes > TX_QUEUE_MAX_BYTES and record.tx:
                dropped = record.tx.popleft()
                record.tx_bytes -= len(dropped)
                record.tx_dropped_bytes += len(dropped)

    def flush_audio(self, call_id: str) -> None:
        """Discard queued (and not-yet-encoded) playout for one call."""
        with self._lock:
            record = self._calls.get(call_id)
            native = record.native if record is not None else None
            if record is not None:
                record.tx.clear()
                record.tx_bytes = 0
        if native is None:
            return
        try:
            native.audio.flush_tx()
        except Exception:
            pass

    def playout_pending_bytes(self, call_id: str) -> int:
        """Bytes queued but not yet transmitted for one call (0 when
        unknown/closed). The session uses this to keep barge-in armed
        until RTP actually drains, instead of trusting generation end."""
        with self._lock:
            record = self._calls.get(call_id)
            if record is None:
                return 0
            return record.tx_bytes

    # -- native event ingress (loop thread) ------------------------------

    def _on_native_incoming(self, native_call: Any) -> None:
        """A native INVITE arrived (loop thread, returns immediately).

        The native call is stashed in the pending slot (reserving one
        admission unit toward ``max_calls``) and admission runs on the
        serial admission worker, never on the SIP thread: handler work
        (core admission, voice session setup) must not stall SIP
        dispatch. The re-entrant answer/reject inside the handler still
        binds the pending call (see _bind_pending_if_needed), which is
        thread-safe; FIFO worker order keeps bindings aligned.
        """
        if not self._is_running():
            self._decline_native(native_call)
            return
        with self._lock:
            queue_ = self._admit_queue
            if (
                queue_ is None
                or len(self._calls) + self._admit_inflight
                >= max(1, self._config.max_calls)
            ):
                over_limit = True
            else:
                over_limit = False
                self._pending.append(native_call)
                self._admit_inflight += 1
        if over_limit:
            self._decline_native(native_call)
            return
        try:
            queue_.put(native_call)  # type: ignore[union-attr]
        except Exception:
            with self._lock:
                try:
                    self._pending.remove(native_call)
                except ValueError:
                    pass
                self._admit_inflight = max(0, self._admit_inflight - 1)
            self._decline_native(native_call)

    def _admission_worker(self) -> None:
        """Serial admission worker (one daemon thread, FIFO).

        Runs the project-owned inbound handler off the SIP thread.
        Exactly one admission runs at a time, in arrival order, so each
        re-entrant answer binds the oldest pending native call (its
        own), never a sibling's.
        """
        while True:
            with self._lock:
                queue_ = self._admit_queue
            if queue_ is None:
                return
            try:
                native_call = queue_.get(timeout=0.5)
            except Exception:
                if not self._is_running():
                    return
                continue
            if native_call is None:  # shutdown sentinel
                return
            try:
                self._admit_one(native_call)
            except Exception:
                LOG.debug("admission failed")
            finally:
                with self._lock:
                    self._admit_inflight = max(0, self._admit_inflight - 1)

    def _admit_one(self, native_call: Any) -> None:
        """Admit one pending native call through the handler."""
        if not self._is_running():
            self._decline_native(native_call)
            return
        with self._lock:
            handler = self._inbound
        peer = str(getattr(native_call, "peer", "") or "")
        caller_id = peer  # untrusted far-end URI; never authentication
        app_id: str | None = None
        if handler is not None:
            try:
                app_id = handler.handle_incoming_call(caller_id, None)
            except Exception:
                LOG.warning("inbound admission failed; rejecting")
                app_id = None
        if not app_id:
            with self._lock:
                try:
                    self._pending.remove(native_call)
                except ValueError:
                    pass
            LOG.info("declining inbound without app admission (answer-then-release)")
            self._decline_native(native_call)
            return
        # The re-entrant answer()/reject()/blind_transfer() inside
        # handle_incoming_call normally bound already. If the handler
        # returned without any telephony demand (unusual), bind now so
        # later demands find the leg.
        self._bind_pending_if_needed(app_id, native=native_call)
        self._attach_call_listeners(app_id, native_call)

    def _on_native_closed(self, app_id: str, reason: str) -> None:
        """Exactly-once close fan-out.

        Local hangup -> ``hangup_completed`` and the record drops.
        Remote close -> ``caller_hangup``; the record stays marked
        ``remote_closed`` until the core's ``hangup()`` arrives, which
        completes locally (the BYE already happened). This gives the
        core the two-step sequence it expects in every interleaving,
        including CANCEL-before-answer. Transfer-accepted closes stay
        silent: the transfer result already ended the session.
        """
        with self._lock:
            record = self._calls.get(app_id)
            if record is None:
                return
            if record.close_emitted and (
                record.local_close or record.hangup_completed_emitted
            ):
                return
            record.tx.clear()
            record.tx_bytes = 0
            if record.transfer_open and reason == "Call transfered":
                # Transfer success observed as dialog teardown: report
                # ACCEPTED here (the transfer() return may arrive after
                # the record is gone, or never). Idempotent: the return
                # path skips emission when the record is already gone,
                # and the core ignores non-handoff results.
                record.transfer_open = False
                self._cancel_transfer_watchdog(record)
                self._calls.pop(app_id, None)
                closed_first = True
            else:
                closed_first = False
            local = record.local_close
            if local:
                record.close_emitted = True
                record.hangup_completed_emitted = True
                self._calls.pop(app_id, None)
            else:
                if record.close_emitted:
                    return
                record.close_emitted = True
                record.remote_closed = True
            media_lost = "rtp" in reason.lower()
            if local or closed_first:
                # Record drops with the close; its callback lock drops
                # only AFTER the terminal delivery below, so delivery
                # reuses the live serializer instead of resurrecting a
                # permanent entry for a removed call.
                self._calls.pop(app_id, None)
                drop_lock_after = True
            else:
                drop_lock_after = False
        if media_lost:
            # RTP-timeout teardown: the media path died (vanished peer,
            # expired NAT, partition). Report it to health alongside the
            # normal close routing below; recovery follows on the next
            # established call or re-registration.
            LOG.info("telephony media lost call_id=%s reason=%s", app_id, reason)
            self._report_media(False, app_id, "rtp timeout")
        if closed_first:
            self._deliver(
                app_id, "on_transfer_result", TransferResult.ACCEPTED_BY_PBX
            )
        elif local:
            self._deliver(app_id, "on_hangup_completed")
        else:
            self._deliver(app_id, "on_caller_hangup")
        if drop_lock_after:
            self._drop_cb_lock(app_id)

    def _emit_hangup_completed(self, app_id: str) -> None:
        """Complete a locally-requested hangup against an already
        remote-closed leg. Exactly once; drops the record."""
        with self._lock:
            record = self._calls.get(app_id)
            if record is None or record.hangup_completed_emitted:
                return
            record.hangup_completed_emitted = True
            self._calls.pop(app_id, None)
        self._deliver(app_id, "on_hangup_completed")
        # Drop after delivery: delivering first reuses the live
        # serializer instead of resurrecting a permanent entry.
        self._drop_cb_lock(app_id)

    def _on_native_dtmf(self, app_id: str, digit: str) -> None:
        with self._lock:
            known = app_id in self._calls
        if not known:
            return
        self._deliver(app_id, "on_dtmf", digit)

    def _gate_remote_hold(self, app_id: str, held: bool) -> bool:
        """Apply the remote-hold media gate synchronously (stack thread).

        Called directly from the native CALL_HOLD/CALL_RESUME callback
        before it returns, so no TX write or RX read/delivery can cross
        the HOLD boundary while the core notification still travels
        through the per-call dispatcher. Deliberately lightweight:
        in-memory flag plus queue mutation under the adapter lock only
        -- no listener, no core, no AI, no SQLite, no native I/O, never
        blocking. Returns True when the gate changed state.

        Entry (False -> True) also discards already-queued caller RX:
        a frame read before the HOLD must never reach
        ``on_caller_audio`` after it. Resume (True -> False) flushes
        stale TX/RX first, so only fresh media flows afterwards.
        Unknown ids (close already handled) are ignored.
        """
        with self._lock:
            record = self._calls.get(app_id)
            if record is None:
                return False
            if held and not record.remote_hold:
                record.remote_hold = True
                record.rx_dispatch.clear()
                return True
            if not held and record.remote_hold:
                record.tx.clear()
                record.tx_bytes = 0
                record.rx_dispatch.clear()
                record.remote_hold = False
                return True
            return False

    def _on_native_remote_hold(self, app_id: str, held: bool) -> None:
        with self._lock:
            record = self._calls.get(app_id)
            if record is None:
                return
            was_held = record.remote_hold
            record.remote_hold = held
        if was_held and not held:
            # Peer resumed: same freshness rule as local resume.
            self._flush_record(record)
        self._deliver(app_id, "on_remote_hold", held)

    def _on_native_established(self, app_id: str) -> None:
        with self._lock:
            record = self._calls.get(app_id)
            native = record.native if record is not None else None
            declining = record.decline_release if record is not None else False
        if record is None or native is None:
            return
        try:
            info = None if declining else native.audio.info()
        except Exception:
            info = None
        emit_answered = False
        with self._lock:
            if app_id in self._calls and not record.remote_closed:
                if info is not None:
                    record.tx_rate = int(info.tx_sample_rate or DEFAULT_LINE_RATE)
                    record.rx_rate = int(info.rx_sample_rate or DEFAULT_LINE_RATE)
                # Declined legs must never look answered, even on the
                # answer-then-release fallback path: the core already
                # finished (or never admitted) the call, and depending
                # on another module's synchronous deletion is fragile.
                if not record.established:
                    record.established = True
                    emit_answered = not record.decline_release
        if emit_answered:
            # Media path verified working again (or for the first time):
            # lift any media-lost degradation.
            self._report_media(True, app_id, "established")
            if not declining:
                # From here on the pump reads caller audio; deliver it on
                # the per-call dispatcher so listener work (barge-in
                # cancellation included) never stalls the shared pump.
                self._ensure_rx_dispatch(app_id)
            self._deliver(app_id, "on_answered")

    # -- internals -------------------------------------------------------

    def _bind_pending_if_needed(
        self, app_id: str, native: Any = None
    ) -> _CallRecord | None:
        """Bind the oldest pending native call to ``app_id``.

        Called from answer()/reject()/blind_transfer(), including
        re-entrantly from inside the inbound handler, or directly after
        it returns. Thread-safe; idempotent: an already-bound id
        returns its record."""
        with self._lock:
            existing = self._calls.get(app_id)
            if existing is not None:
                return existing
            if native is None:
                if not self._pending:
                    return None
                native = self._pending.popleft()
            else:
                try:
                    self._pending.remove(native)
                except ValueError:
                    pass
            record = _CallRecord(app_id=app_id, native=native)
            self._calls[app_id] = record
            return record

    def _submit(
        self, record: _CallRecord, op: Callable, *, revert_hold: bool | None = None
    ) -> None:
        loop = self._current_loop()
        if loop is None or not self._is_running():
            if revert_hold is not None:
                with self._lock:
                    if record.app_id in self._calls:
                        record.local_hold = revert_hold
            return

        async def run() -> None:
            try:
                result = op(record.native)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as error:
                name = type(error).__name__
                if revert_hold is not None:
                    with self._lock:
                        if record.app_id in self._calls:
                            record.local_hold = revert_hold
                LOG.debug("telephony op failed: %s", name)

        try:
            asyncio.run_coroutine_threadsafe(run(), loop)
        except Exception:
            pass

    def _submit_transfer(self, record: _CallRecord, target: str) -> None:
        loop = self._current_loop()
        if loop is None or not self._is_running():
            with self._lock:
                record.transfer_open = False
            self._emit_transfer_result(record.app_id, TransferResult.TRANSPORT_ERROR)
            return

        async def run() -> None:
            LOG.info("transfer: REFER submitted for %s", record.app_id)
            try:
                result = record.native.transfer(target)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as error:
                LOG.info(
                    "transfer: outcome %s for %s",
                    type(error).__name__,
                    record.app_id,
                )
                self._transfer_failed(record, error)
                return
            LOG.info("transfer: accepted for %s", record.app_id)
            with self._lock:
                already = record.app_id not in self._calls
                record.transfer_open = False
                self._cancel_transfer_watchdog(record)
            if not already:
                self._emit_transfer_result(record.app_id, TransferResult.ACCEPTED_BY_PBX)

        try:
            asyncio.run_coroutine_threadsafe(run(), loop)
        except Exception as error:
            LOG.info("transfer: submit failed (%s)", type(error).__name__)
            with self._lock:
                record.transfer_open = False
            self._emit_transfer_result(record.app_id, TransferResult.TRANSPORT_ERROR)

    def _cancel_transfer_watchdog(self, record: _CallRecord) -> None:
        timer = record.transfer_watchdog
        record.transfer_watchdog = None
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass

    def _transfer_failed(self, record: _CallRecord, error: Exception) -> None:
        with self._lock:
            record.transfer_open = False
            self._cancel_transfer_watchdog(record)
        status = getattr(error, "status", None)
        message = str(error).lower()
        if status in (408,) or "timeout" in message or "timed out" in message:
            result = TransferResult.TIMEOUT
        elif isinstance(status, int) and 400 <= status <= 699:
            result = TransferResult.REJECTED
        elif type(error).__name__ in ("TransferFailed",):
            result = TransferResult.REJECTED
        else:
            result = TransferResult.TRANSPORT_ERROR
        self._emit_transfer_result(record.app_id, result)

    def _emit_transfer_result(self, app_id: str, result: TransferResult) -> None:
        """Route a normalized transfer outcome to the call dispatcher.

        Always emitted, even if the call record is already gone:
        transfer results race dialog teardown (the close event and the
        outcome callback arrive in either order), and the core safely
        ignores unknown/ended ids. Exactly-once per transfer is
        enforced by the transfer_open guard and watchdog cancellation
        at the call sites, not here.

        Known records enqueue (loop/watchdog threads snapshot and
        return; slow core work runs on the per-call dispatcher, never
        on shared SIP dispatch). Gone records deliver immediately on
        the calling thread: the core drops unknown ids before any
        slow work, so nothing shared can stall.
        """
        with self._lock:
            known = app_id in self._calls
        if known:
            self._enqueue_native_event(app_id, ("transfer", result))
        else:
            self._deliver(app_id, "on_transfer_result", result)

    def _decline_native(self, native_call: Any) -> None:
        """SIP-level decline for a native leg with no application id
        (inbound handler returned None): ``reject()`` first (486), with
        the same answer-then-release fallback as :meth:`reject` on
        ``StaleHandleError``. The leg gets a synthetic ``declined-N``
        record so its close routes sanely (unknown to the core, which
        ignores it) and no slot leaks."""
        with self._lock:
            self._declined_total += 1
            app_id = f"declined-{self._declined_total}"
            record = _CallRecord(app_id=app_id, native=native_call)
            record.local_close = True
            record.decline_release = True
            self._calls[app_id] = record
        self._attach_call_listeners(app_id, native_call)
        loop = self._current_loop()
        if loop is None or not self._is_running():
            with self._lock:
                self._calls.pop(app_id, None)
                self._drop_cb_lock(app_id)
            return

        async def run() -> None:
            for attempt, delay in enumerate(
                (0.0,) + self._REJECT_RETRY_DELAYS
            ):
                if delay:
                    await asyncio.sleep(delay)
                try:
                    result = native_call.reject()
                    if asyncio.iscoroutine(result):
                        await result
                    return
                except Exception as error:
                    if type(error).__name__ != "StaleHandleError":
                        LOG.debug(
                            "telephony op failed: %s", type(error).__name__
                        )
                        return
                    LOG.debug("decline reject race, attempt %d", attempt)
            self._answer_then_release(record)

        try:
            asyncio.run_coroutine_threadsafe(run(), loop)
        except Exception:
            pass

    def _answer_then_release(self, record: _CallRecord) -> None:
        """Decline fallback (see :meth:`reject`): answer the leg, then
        BYE once established (bounded). Must run on the loop thread
        (called from loop-thread coroutines). Cleans the record if the
        close never arrives."""
        loop = self._current_loop()
        if loop is None:
            with self._lock:
                self._calls.pop(record.app_id, None)
                self._drop_cb_lock(record.app_id)
            return

        async def run() -> None:
            start = time.monotonic()
            try:
                result = record.native.answer()
                if asyncio.iscoroutine(result):
                    await result
                LOG.info(
                    "decline release: answered in %.1fs",
                    time.monotonic() - start,
                )
            except Exception as error:
                LOG.info(
                    "decline release: answer failed (%s)",
                    type(error).__name__,
                )
                with self._lock:
                    self._calls.pop(record.app_id, None)
                    self._drop_cb_lock(record.app_id)
                return
            try:
                await asyncio.wait_for(
                    self._wait_established(record), timeout=10.0
                )
            except Exception:
                pass
            try:
                result = record.native.hangup()
                if asyncio.iscoroutine(result):
                    await result
                LOG.info("decline release: BYE sent")
            except Exception as error:
                LOG.info(
                    "decline release: hangup failed (%s)",
                    type(error).__name__,
                )
            await asyncio.sleep(15.0)
            # Safety: never leak the slot if the close never arrived.
            with self._lock:
                self._calls.pop(record.app_id, None)
                self._drop_cb_lock(record.app_id)

        try:
            asyncio.run_coroutine_threadsafe(run(), loop)
        except Exception:
            pass

    async def _wait_established(self, record: _CallRecord) -> None:
        """Resolve once the record shows established (polled, loop-safe).

        Tight 10 ms poll: the decline release must BYE before the
        stack's post-answer media refresh (~90 ms) fires, so the wire
        order stays deterministic (200 -> ACK -> BYE, no refresh).
        """
        for _ in range(100):
            with self._lock:
                if record.established or record.remote_closed:
                    return
            await asyncio.sleep(0.01)

    def _close_native_best_effort(self, record: _CallRecord) -> None:
        try:
            record.tx.clear()
            record.tx_bytes = 0
        except Exception:
            pass
        loop = self._current_loop()
        native = record.native
        if loop is None or native is None:
            return

        async def run() -> None:
            try:
                result = native.hangup()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

        try:
            asyncio.run_coroutine_threadsafe(run(), loop).result(timeout=5)
        except Exception:
            pass

    def _attach_call_listeners(self, app_id: str, native_call: Any) -> None:
        with self._lock:
            record = self._calls.get(app_id)
            if record is None or record.listeners_attached:
                return
            record.listeners_attached = True
        try:
            from baresip.events import Event as _Event
        except Exception:
            return

        def on_event(event: Any) -> None:
            try:
                kind = event.event
            except Exception:
                return
            try:
                self._note_event(app_id, f"stack:{kind.name}")
            except Exception:
                pass
            # Snapshot and return: the per-call dispatcher owns all
            # downstream core work, never the shared SIP loop thread.
            # Remote hold additionally gates media synchronously here
            # (lightweight flag + queue mutation only): the pump must
            # stop TX/RX the moment the stack reports HOLD, not when
            # the dispatcher later delivers the notification.
            if kind is _Event.CALL_ESTABLISHED:
                self._enqueue_native_event(app_id, ("established",))
            elif kind is _Event.CALL_CLOSED:
                self._enqueue_native_event(
                    app_id, ("closed", str(getattr(event, "text", "") or ""))
                )
            elif kind is _Event.CALL_HOLD:
                self._gate_remote_hold(app_id, True)
                self._enqueue_native_event(app_id, ("hold", True))
            elif kind is _Event.CALL_RESUME:
                self._gate_remote_hold(app_id, False)
                self._enqueue_native_event(app_id, ("hold", False))
            elif kind is _Event.CALL_LOCAL_SDP:
                try:
                    self._note_event(app_id, f"local-sdp:{event.text}")
                except Exception:
                    pass

        def on_dtmf(digit_event: Any) -> None:
            digit = str(getattr(digit_event, "digit", "") or "").upper()
            if len(digit) == 1 and digit in VALID_DTMF_DIGITS:
                self._enqueue_native_event(app_id, ("dtmf", digit))

        try:
            native_call.on(on_event)
            native_call.on_dtmf(on_dtmf)
        except Exception:
            pass

    # -- loop-thread coroutines ------------------------------------------

    async def _async_start(self) -> None:
        import baresip as _baresip_module
        from baresip import Account, Config, Runtime, UserAgent
        from baresip.events import Event as _Event

        _check_binding_version(_baresip_module)
        cfg = self._config
        # sip_trace logs every SIP message under baresip.native.sip at
        # DEBUG (integration diagnostics only; credentials never appear
        # there -- auth material stays in headers the harness redacts).
        # sip_listen travels as the one typed escape hatch the binding
        # requires for loopback benches (no free-text channel exists).
        extra_lines = (
            f"sip_listen {cfg.sip_listen}\n" if cfg.sip_listen else ""
        )
        baresip_config = Config(
            net_interface=cfg.bind,
            audio_driver="aumem",
            max_concurrent_calls=cfg.max_calls,
            rtp_timeout=cfg.rtp_timeout,
            extra_config_text=extra_lines,
            sip_trace=cfg.sip_trace,
        )
        runtime = Runtime()
        await runtime.start(baresip_config)
        with self._lock:
            self._runtime = runtime
        account = Account(
            user=cfg.username,
            password=cfg.password,
            domain=cfg.domain,
            registrar=cfg.registrar,
            reg_interval=cfg.reg_interval,
            transport=cfg.transport,  # type: ignore[arg-type]
            audio_codecs=tuple(cfg.audio_codecs),
            dtmf_mode=cfg.dtmf_mode,  # type: ignore[arg-type]
            auth_user=cfg.auth_user,
        )
        ua = await UserAgent.create(runtime, account, transfer_policy="reject")
        with self._lock:
            self._ua = ua

        def on_runtime_event(event: Any) -> None:
            try:
                kind = event.event
            except Exception:
                return
            if kind is _Event.REGISTER_OK:
                with self._lock:
                    self._register_ok_count += 1
                self._set_reg_state(TelephonyRegistrationState.REGISTERED)
            elif kind is _Event.REGISTER_FAIL:
                with self._lock:
                    was_registered = (
                        self._reg_state is TelephonyRegistrationState.REGISTERED
                    )
                self._set_reg_state(
                    TelephonyRegistrationState.REGISTRATION_LOST
                    if was_registered
                    else TelephonyRegistrationState.REGISTRATION_FAILED
                )

        runtime.subscribe(on_runtime_event)
        ua.on_incoming(self._on_native_incoming)
        runtime.on_dead = lambda: self._set_reg_state(
            TelephonyRegistrationState.REGISTRATION_LOST
        )
        await ua.register()
        # register() returning means REGISTER_OK arrived (it awaits the
        # outcome); the event hook above already set REGISTERED, but set
        # it explicitly in case of ordering subtleties.
        self._set_reg_state(TelephonyRegistrationState.REGISTERED)
        LOG.info("telephony registered: %s", self.config_summary)

    async def _async_shutdown_stack(self) -> None:
        ua = self._ua
        runtime = self._runtime
        if ua is not None:
            try:
                await ua.unregister()
            except Exception:
                pass
        if runtime is not None:
            # Graceful first: refuse new calls and let live ones end
            # (bounded) before tearing the stack down. Closing with live
            # calls risks native teardown races.
            try:
                await asyncio.wait_for(runtime.drain(), timeout=8.0)
            except Exception:
                pass
            try:
                await runtime.close()
            except Exception:
                pass

    async def _async_stop_quietly(self) -> None:
        try:
            await self._async_shutdown_stack()
        except Exception:
            pass

    def _start_media_thread(self) -> None:
        with self._lock:
            if self._media_thread is not None or not self._running:
                return
            thread = threading.Thread(
                target=self._media_pump, name="baresip-media", daemon=True
            )
            self._media_thread = thread
        thread.start()

    def _start_admit_thread(self) -> None:
        with self._lock:
            if self._admit_thread is not None or not self._running:
                return
            self._admit_queue = queue.Queue()
            thread = threading.Thread(
                target=self._admission_worker, name="baresip-admit", daemon=True
            )
            self._admit_thread = thread
        thread.start()

    def _media_pump(self) -> None:
        """Single media thread: TX pump then RX drain per live call.

        RX drain only enqueues frames for the per-call dispatcher
        threads; no listener code runs here, so one call's blocking
        listener work can never stall another call's media.
        """
        while self._is_running():
            try:
                self._pump_once()
            except Exception:
                pass
            time.sleep(MEDIA_PERIOD)

    def _pump_once(self) -> None:
        with self._lock:
            records = list(self._calls.values())
        for record in records:
            if not self._is_running():
                return
            self._pump_tx(record)
            self._drain_rx(record)

    def _pump_tx(self, record: _CallRecord) -> None:
        native = record.native
        if native is None:
            return
        with self._lock:
            if record.decline_release:
                return
            if record.local_hold or record.remote_hold:
                # Held either way: nothing is transmitted. The queue
                # stays bounded (oldest drops) and resume flushes it.
                return
            if not record.tx or record.remote_closed:
                return
            chunk = record.tx.popleft()
            record.tx_bytes -= len(chunk)
        try:
            audio = native.audio
        except Exception:
            with self._lock:
                record.tx.appendleft(chunk)
                record.tx_bytes += len(chunk)
            self._note_media_error(record, "tx")
            return
        try:
            try:
                info = audio.info()
                record.tx_rate = int(info.tx_sample_rate or record.tx_rate)
            except Exception:
                pass
            accepted = audio.write(chunk)
            self._note_media_success(record, "tx")
            with self._lock:
                record.tx_accepted_total += max(0, accepted)
                if record.first_tx_at is None and accepted > 0:
                    record.first_tx_at = time.time()
            if accepted < len(chunk):
                remainder = chunk[accepted:]
                with self._lock:
                    record.tx.appendleft(remainder)
                    record.tx_bytes += len(remainder)
        except Exception as error:
            # Every write failure is counted and the chunk is re-queued
            # (bounded: the cap still applies, oldest drops first). The
            # old code dropped the chunk silently whenever the re-probe
            # succeeded -- a live call with vanishing audio and no
            # signal. AudioRestarted additionally counts renegotiations.
            # The consecutive streak feeds the bounded trip policy.
            with self._lock:
                record.tx_errors += 1
                if type(error).__name__ == "AudioRestarted":
                    record.audio_restarts += 1
            self._note_media_error(record, "tx")
            try:
                audio.info()
            except Exception:
                pass
            with self._lock:
                if record.app_id in self._calls:
                    record.tx.appendleft(chunk)
                    record.tx_bytes += len(chunk)
                    while record.tx_bytes > TX_QUEUE_MAX_BYTES and record.tx:
                        dropped = record.tx.popleft()
                        record.tx_bytes -= len(dropped)
                        record.tx_dropped_bytes += len(dropped)

    def _drain_rx(self, record: _CallRecord) -> None:
        native = record.native
        if native is None:
            return
        with self._lock:
            if record.decline_release:
                return
            if record.local_hold or record.remote_hold or record.close_emitted:
                # Held either way: caller speech is never delivered as
                # app input while held (spec: no garbage as caller
                # speech). Buffered line audio is flushed on resume.
                return
        try:
            audio = native.audio
        except Exception:
            with self._lock:
                record.rx_errors += 1
            self._note_media_error(record, "rx")
            return
        try:
            raw = audio.read(RX_READ_BYTES)
        except Exception:
            with self._lock:
                record.rx_errors += 1
            self._note_media_error(record, "rx")
            return
        if not raw:
            return
        try:
            info = audio.info()
            rx_rate = int(info.rx_sample_rate or DEFAULT_LINE_RATE)
            record.rx_rate = rx_rate
        except Exception:
            rx_rate = record.rx_rate
        try:
            pcm = resample_pcm16(bytes(raw), rx_rate, APP_PCM_RATE)
        except Exception:
            with self._lock:
                record.rx_errors += 1
            self._note_media_error(record, "rx")
            return
        with self._lock:
            if record.local_hold or record.remote_hold or record.close_emitted:
                return
            if record.app_id not in self._calls:
                return
            seq = record.rx_seq
            record.rx_seq += 1
        try:
            frame = make_frame(
                pcm,
                APP_PCM_RATE,
                call_id=record.app_id,
                sequence=seq,
                timestamp=time.time(),
            )
        except Exception:
            return
        # Late close racing this delivery is benign: the dispatcher
        # re-checks membership per frame, and unknown ids and
        # wrong-state frames are dropped by the core/session.
        with self._lock:
            if record.app_id not in self._calls:
                return
            if len(record.rx_dispatch) >= RX_DISPATCH_MAX_FRAMES:
                record.rx_dispatch.popleft()
                record.rx_dropped_frames += 1
            record.rx_dispatch.append(frame)
            record.rx_ready.set()
        self._note_media_success(record, "rx")

    def _ensure_rx_dispatch(self, app_id: str) -> None:
        """Start the per-call dispatcher thread (idempotent).

        Called on establishment and on the first enqueued stack event:
        from then on, lifecycle events and RX frames are delivered to
        the listener on this thread, FIFO per stream (lifecycle first).
        The thread is a daemon and exits on its own when the record
        leaves ``_calls`` or the adapter stops, so close paths never
        join it (in particular a listener re-entrantly hanging up from
        this thread cannot self-join).
        """
        with self._lock:
            record = self._calls.get(app_id)
            if record is None or record.rx_dispatch_started:
                return
            record.rx_dispatch_started = True
            thread = threading.Thread(
                target=self._rx_dispatch_loop,
                args=(app_id,),
                name=f"baresip-rx-{app_id}",
                daemon=True,
            )
            record.rx_thread = thread
            self._rx_threads[app_id] = thread
        thread.start()

    def _enqueue_native_event(self, app_id: str, event: tuple) -> None:
        """Snapshot one stack lifecycle event for per-call dispatch.

        The production seam below the vendor-typed stack callbacks and
        the media pump: the calling thread passes plain data (kind plus
        payload) and returns immediately, so slow core
        cancellation/fallback work can never stall shared SIP/media
        dispatch. Drops the event when the call is unknown (close
        already handled) or the adapter is stopping (no callbacks
        after STOPPED).

        Backpressure: answered/terminal kinds (established, closed,
        transfer, media_failed) always append, and so does every
        hold/resume snapshot -- holds are full FIFO, never coalesced
        or evicted, so causal order across DTMF always survives.
        DTMF keeps arrival order with its backlog bounded at
        EV_DISPATCH_MAX_NONTERMINAL, dropping oldest only as a
        documented last resort under pathological flood.
        """
        kind = event[0] if event else None
        with self._lock:
            if not self._running:
                return
            record = self._calls.get(app_id)
            if record is None:
                return
            if kind not in _EV_NO_DROP_KINDS and kind != "hold":
                # DTMF-only bound: drop the oldest pending DTMF while
                # over budget (holds and answered/terminal events are
                # never evicted by DTMF pressure).
                while (
                    sum(
                        1
                        for pending in record.ev_dispatch
                        if pending and pending[0] == "dtmf"
                    )
                    >= EV_DISPATCH_MAX_NONTERMINAL
                ):
                    for pending in record.ev_dispatch:
                        if pending and pending[0] == "dtmf":
                            record.ev_dispatch.remove(pending)
                            break
            record.ev_dispatch.append(event)
            record.rx_ready.set()
        self._ensure_rx_dispatch(app_id)

    def _drain_native_events(self, app_id: str) -> int:
        """Process every queued lifecycle event for one call, FIFO.

        Deterministic test seam (no threads, no sleeps): the
        production dispatcher threads run the same
        ``_dispatch_native_event`` unit of work. Returns the number of
        events processed.
        """
        processed = 0
        while True:
            with self._lock:
                record = self._calls.get(app_id)
                if record is None or not record.ev_dispatch:
                    return processed
                event = record.ev_dispatch.popleft()
            self._dispatch_native_event(app_id, event)
            processed += 1

    def _dispatch_native_event(self, app_id: str, event: tuple) -> None:
        """Process one queued lifecycle event on the call dispatcher."""
        kind = event[0] if event else None
        if kind == "established":
            self._on_native_established(app_id)
        elif kind == "closed":
            self._on_native_closed(app_id, str(event[1]) if len(event) > 1 else "")
        elif kind == "hold":
            self._on_native_remote_hold(app_id, bool(event[1]) if len(event) > 1 else False)
        elif kind == "dtmf":
            self._on_native_dtmf(app_id, str(event[1]) if len(event) > 1 else "")
        elif kind == "transfer":
            self._deliver(
                app_id,
                "on_transfer_result",
                event[1] if len(event) > 1 else TransferResult.TRANSPORT_ERROR,
            )
        elif kind == "media_failed":
            self._deliver(
                app_id, "on_media_failed", str(event[1]) if len(event) > 1 else ""
            )

    def _rx_dispatch_loop(self, app_id: str) -> None:
        """Dispatcher body: lifecycle events then RX frames, until close/stop."""
        try:
            while self._is_running():
                with self._lock:
                    if app_id not in self._calls:
                        return
                    record = self._calls[app_id]
                    ready = record.rx_ready
                ready.wait(timeout=RX_DISPATCH_POLL_S)
                ready.clear()
                while True:
                    with self._lock:
                        if app_id not in self._calls:
                            return
                        record = self._calls[app_id]
                        pending = list(record.ev_dispatch)
                        record.ev_dispatch.clear()
                    if not pending:
                        break
                    for native_event in pending:
                        self._dispatch_native_event(app_id, native_event)
                self._dispatch_once(app_id)
        finally:
            with self._lock:
                if self._rx_threads.get(app_id) is threading.current_thread():
                    self._rx_threads.pop(app_id, None)

    def _dispatch_once(self, app_id: str) -> None:
        """Deliver all currently enqueued RX frames for one call, FIFO.

        The shared unit of delivery work: dispatcher threads call this
        when signalled, and deterministic tests call it directly (no
        threads, no sleeps). Membership is re-checked per frame so a
        close racing the drain discards the tail instead of delivering
        caller audio into a dead call.
        """
        with self._lock:
            record = self._calls.get(app_id)
            if record is None:
                return
            frames = list(record.rx_dispatch)
            record.rx_dispatch.clear()
        for frame in frames:
            with self._lock:
                if app_id not in self._calls:
                    return
                record = self._calls[app_id]
                if (
                    record.local_hold
                    or record.remote_hold
                    or record.close_emitted
                ):
                    # Held or closing: caller PCM is never delivered
                    # (the synchronous hold gate plus the RX-drain gate
                    # already stop new frames; this covers any frame
                    # queued before the gate engaged).
                    return
            self._deliver(app_id, "on_caller_audio", frame)

    # -- small helpers ----------------------------------------------------

    def _is_running(self) -> bool:
        with self._lock:
            return self._running

    def _current_listener(self) -> TelephonyListener | None:
        with self._lock:
            return self._listener

    def _current_loop(self) -> asyncio.AbstractEventLoop | None:
        with self._lock:
            return self._loop

    def _cb_lock_for(self, call_id: str) -> threading.RLock:
        """The callback serializer for one call (created on demand)."""
        with self._lock:
            lock = self._cb_locks.get(call_id)
            if lock is None:
                lock = threading.RLock()
                self._cb_locks[call_id] = lock
            return lock

    def _drop_cb_lock(self, call_id: str) -> None:
        with self._lock:
            self._cb_locks.pop(call_id, None)

    def _deliver(self, call_id: str, method: str, *args: object) -> None:
        """Invoke one listener method for one call: serialized per call,
        lightweight, never raising into the stack. The single shared
        shape for every fan-out site (one place to get the locking
        right). Callbacks must return fast: heavy backend work
        (model inference, process restarts, storage, network) must be
        deferred by the listener, never run inline on SIP/media threads.

        Serializer entries live and die with their call record: delivery
        for a removed id (late transfer outcomes, terminal closes)
        still happens -- the core safely ignores unknown ids -- but
        never resurrects a permanent entry.
        """
        listener = self._current_listener()
        if listener is None:
            return
        with self._lock:
            known = call_id in self._calls
        if not known:
            if not self._is_running():
                return
            try:
                getattr(listener, method)(call_id, *args)
            except Exception:
                pass
            return
        with self._cb_lock_for(call_id):
            if not self._is_running():
                return
            try:
                getattr(listener, method)(call_id, *args)
            except Exception:
                pass

    def _set_reg_state(self, state: TelephonyRegistrationState) -> None:
        with self._lock:
            listener = self._status
            self._set_reg_state_locked(state)
        if listener is not None:
            try:
                listener.on_registration_state(state)
            except Exception:
                pass
            # A fresh registration epoch invalidates pre-restart media
            # evidence: recovery here keeps re-registration restorative.
            if state is TelephonyRegistrationState.REGISTERED:
                self._report_media(True, "", "re-registered")

    def _note_media_error(self, record: _CallRecord, direction: str) -> None:
        """Account one consecutive TX/RX stack failure for one call.

        The streak trips :data:`MEDIA_ERROR_STREAK_LIMIT` exactly once:
        one sanitized media-unhealthy transition plus one project-owned
        per-call media-failure signal, enqueued for the per-call
        dispatcher (never delivered inline: the pump thread snapshots
        and returns, so slow application cancellation/fallback can
        never stall shared media dispatch). The adapter never decides
        the business outcome: the application (core/session) chooses
        PBX fallback handoff or termination, and only its
        hangup/transfer closes the leg. Records already closing,
        remote-closed, or held never trip -- their media path is
        intentionally idle, and close routing owns their outcome. The
        RTP-timeout path stays independent: it observes vanished
        peers, this observes a broken local stack with a live dialog.
        """
        with self._lock:
            if record.app_id not in self._calls:
                return
            if record.close_emitted or record.remote_closed or record.decline_release:
                return
            if direction == "tx":
                record.tx_error_streak += 1
                streak = record.tx_error_streak
            else:
                record.rx_error_streak += 1
                streak = record.rx_error_streak
            trip = streak == MEDIA_ERROR_STREAK_LIMIT
        if trip:
            detail = f"persistent {direction} media failure"
            LOG.warning(
                "telephony media %s failure streak call_id=%s",
                direction,
                record.app_id,
            )
            self._report_media(False, record.app_id, detail)
            # Application decision, queued for the per-call dispatcher
            # like every other listener fan-out; never raises into the
            # pump, never runs core inline on shared dispatch.
            self._enqueue_native_event(record.app_id, ("media_failed", detail))

    def _note_media_success(self, record: _CallRecord, direction: str) -> None:
        """Reset one direction's failure streak: real media flowing
        proves the path alive, so an earlier blip streak is forgotten
        and a recovered path never trips."""
        with self._lock:
            if direction == "tx":
                record.tx_error_streak = 0
            else:
                record.rx_error_streak = 0

    def _report_media(self, healthy: bool, call_id: str, detail: str = "") -> None:
        with self._lock:
            listener = self._status
        if listener is None:
            return
        report = getattr(listener, "on_media_state", None)
        if not callable(report):
            return
        try:
            report(healthy, call_id, detail)
        except Exception:
            pass

    def _set_reg_state_locked(self, state: TelephonyRegistrationState) -> None:
        self._reg_state = state


class CoreInboundBridge(InboundCallHandler):
    """Composition seam: native INVITE -> core admission -> app call id.

    The adapter holds the native call in its pending slot while this
    runs; the core's re-entrant ``answer``/``reject`` binds it. Returns
    ``None`` (SIP-level decline) when admission raises.
    """

    def __init__(self, core: Any) -> None:
        self._core = core

    def handle_incoming_call(
        self, caller_id: str, caller_name: str | None
    ) -> str | None:
        try:
            session = self._core.incoming_call(caller_id, caller_name)
        except Exception:
            LOG.warning("inbound admission failed; declining at SIP level")
            return None
        return session.call_id


class CoreStatusBridge(TelephonyStatusListener):
    """Composition seam: adapter registration/media states -> core health."""

    def __init__(self, core: Any) -> None:
        self._core = core

    def on_registration_state(
        self, state: TelephonyRegistrationState, detail: str = ""
    ) -> None:
        try:
            self._core.report_telephony_state(state, detail)
        except Exception:
            pass

    def on_media_state(
        self, healthy: bool, call_id: str = "", detail: str = ""
    ) -> None:
        try:
            self._core.report_telephony_media(healthy, call_id, detail)
        except Exception:
            pass


def wire_baresip_core(core: Any, adapter: BaresipTelephonyAdapter) -> None:
    """Connect both directions without the core knowing baresip:

    adapter events -> core (as :class:`TelephonyListener`), inbound
    INVITE -> core admission (inbound bridge), registration states ->
    core health (status bridge).
    """
    adapter.set_listener(core)
    adapter.set_inbound_handler(CoreInboundBridge(core))
    adapter.set_status_listener(CoreStatusBridge(core))


def baresip_adapter_from_config(config_service: Any) -> BaresipTelephonyAdapter:
    """Composition root for the telephony path: read the declarative
    ``sip.*`` identity through ``ConfigService`` (canonical config.db)
    and build the adapter. This is the consumer that keeps
    ``ConfigService.telephony_config()`` live: unregistered SIP
    identities fail closed here (ValueError) instead of halfway through
    a call.
    """
    return BaresipTelephonyAdapter(config_service.telephony_config())
