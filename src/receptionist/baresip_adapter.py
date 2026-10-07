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
  media thread. All listener invocations are serialized through one
  callback lock, are lightweight (record + return), and never call back
  into blocking adapter methods while holding native locks (public
  methods only take the adapter lock briefly and never await the loop).

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
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from receptionist.audio import make_frame, resample_pcm16
from receptionist.boundaries import (
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
BARESIP_PYTHON_PIN = "baresip-python==0.5.2a3"

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

_VALID_DTMF = frozenset("0123456789ABCD*#")


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
    negotiated_codec: str = ""
    audio_restarts: int = 0
    tx_accepted_total: int = 0
    first_tx_at: float | None = None


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
        # Re-entrant: the core synchronously calls back into the adapter
        # (hangup/flush) from inside listener delivery on the same thread
        # (remote BYE -> caller_hangup -> hangup -> hangup_completed).
        # Cross-thread delivery stays serialized. Never take _lock while
        # holding _cb_lock: snapshots happen before delivery.
        self._cb_lock = threading.RLock()
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
        return self.registration_state

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
        with self._lock:
            self._loop = None
            self._loop_thread = None
            self._media_thread = None
            self._runtime = None
            self._ua = None
            self._calls.clear()
            self._pending.clear()
            self._set_reg_state_locked(TelephonyRegistrationState.STOPPED)

    # -- call control (non-blocking, never raise to the core) ----------

    def answer(self, call_id: str) -> None:
        record = self._bind_pending_if_needed(call_id)
        with self._lock:
            record = record or self._calls.get(call_id)
        if record is None:
            return
        self._submit(record, lambda call: call.answer())

    def reject(self, call_id: str) -> None:
        """Decline an inbound call.

        Upstream limitation (baresip-python 0.5.2a3, to be reported):
        ``Call.reject()``/``hangup()`` on a just-arrived inbound leg
        raise ``StaleHandleError`` -- the native handle does not
        validate for END commands before establishment (verified
        immediate and delayed) -- so a direct 486 is not obtainable
        deterministically through public APIs today. Decline is
        therefore always released via public primitives only: answer
        the leg and BYE it immediately (answer-then-release). Observable
        outcome is identical for the core (call released, slot freed,
        exactly-once close, no AI resources opened); the SIP trace
        shows 200+BYE instead of 486 until upstream fixes decline. No
        patching, no unsafe handles, no vendor types cross the boundary
        either way. Deterministic by construction: no first-try-486
        race that would make the wire outcome vary run to run.
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
                return
            record.local_close = True
            record.decline_release = True
        self._answer_then_release(record)

    def hangup(self, call_id: str) -> None:
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
        self._submit(record, lambda call: call.resume(), revert_hold=True)

    def send_dtmf(self, call_id: str, digits: str) -> None:
        if not isinstance(digits, str) or not digits:
            return
        clean = digits.upper()
        if any(ch not in _VALID_DTMF for ch in clean):
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

    # -- native event ingress (loop thread) ------------------------------

    def _on_native_incoming(self, native_call: Any) -> None:
        """A native INVITE arrived. Hold it in the pending slot, extract
        untrusted caller metadata, and synchronously admit through the
        project-owned handler. The handler's re-entrant answer/reject
        binds the pending call (see _bind_pending_if_needed)."""
        if not self._is_running():
            self._decline_native(native_call)
            return
        with self._lock:
            if len(self._calls) >= max(1, self._config.max_calls):
                over_limit = True
            else:
                over_limit = False
                self._pending.append(native_call)
            handler = self._inbound
        if over_limit:
            self._decline_native(native_call)
            return
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
        listener = self._current_listener()
        if listener is None:
            return
        with self._cb_lock:
            if not self._is_running():
                return
            try:
                if closed_first:
                    listener.on_transfer_result(
                        app_id, TransferResult.ACCEPTED_BY_PBX
                    )
                elif local:
                    listener.on_hangup_completed(app_id)
                else:
                    listener.on_caller_hangup(app_id)
            except Exception:
                pass

    def _emit_hangup_completed(self, app_id: str) -> None:
        """Complete a locally-requested hangup against an already
        remote-closed leg. Exactly once; drops the record."""
        with self._lock:
            record = self._calls.get(app_id)
            if record is None or record.hangup_completed_emitted:
                return
            record.hangup_completed_emitted = True
            self._calls.pop(app_id, None)
        listener = self._current_listener()
        if listener is None:
            return
        with self._cb_lock:
            if not self._is_running():
                return
            try:
                listener.on_hangup_completed(app_id)
            except Exception:
                pass

    def _on_native_dtmf(self, app_id: str, digit: str) -> None:
        with self._lock:
            known = app_id in self._calls
        if not known:
            return
        listener = self._current_listener()
        if listener is None:
            return
        with self._cb_lock:
            if not self._is_running():
                return
            try:
                listener.on_dtmf(app_id, digit)
            except Exception:
                pass

    def _on_native_remote_hold(self, app_id: str, held: bool) -> None:
        with self._lock:
            record = self._calls.get(app_id)
            if record is None:
                return
            record.remote_hold = held
        listener = self._current_listener()
        if listener is None:
            return
        with self._cb_lock:
            if not self._is_running():
                return
            try:
                listener.on_remote_hold(app_id, held)
            except Exception:
                pass

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
                if not record.established:
                    record.established = True
                    emit_answered = True
        if not emit_answered:
            return
        listener = self._current_listener()
        if listener is None:
            return
        with self._cb_lock:
            if not self._is_running():
                return
            try:
                listener.on_answered(app_id)
            except Exception:
                pass

    # -- internals -------------------------------------------------------

    def _bind_pending_if_needed(
        self, app_id: str, native: Any = None
    ) -> _CallRecord | None:
        """Bind the oldest pending native call to ``app_id``.

        Called re-entrantly from answer()/reject()/blind_transfer() while
        the inbound handler is still on the stack, or directly after it
        returns. Idempotent: an already-bound id returns its record."""
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
        """Deliver a normalized transfer outcome. Always emitted, even if
        the call record is already gone: transfer results race dialog
        teardown (the close event and the outcome callback arrive in
        either order), and the core safely ignores unknown/ended ids.
        Exactly-once per transfer is enforced by the transfer_open guard
        and watchdog cancellation at the call sites, not here."""
        listener = self._current_listener()
        if listener is None:
            return
        with self._cb_lock:
            if not self._is_running():
                return
            try:
                listener.on_transfer_result(app_id, result)
            except Exception:
                pass

    def _decline_native(self, native_call: Any) -> None:
        """SIP-level decline for a native leg with no application id
        (inbound handler returned None). Same upstream limitation as
        :meth:`reject`: try ``reject()``, fall back to answer-then-BYEs
        on ``StaleHandleError``. The leg gets a synthetic
        ``declined-N`` record so its close routes sanely (unknown to the
        core, which ignores it) and no slot leaks."""
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
            return

        async def run() -> None:
            # Decline is always answer-then-release (see reject()): a
            # direct reject() is not deterministic on this binding.
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
            if kind is _Event.CALL_ESTABLISHED:
                self._on_native_established(app_id)
            elif kind is _Event.CALL_CLOSED:
                self._on_native_closed(app_id, str(getattr(event, "text", "") or ""))
            elif kind is _Event.CALL_HOLD:
                self._on_native_remote_hold(app_id, True)
            elif kind is _Event.CALL_RESUME:
                self._on_native_remote_hold(app_id, False)
            elif kind is _Event.CALL_LOCAL_SDP:
                try:
                    self._note_event(app_id, f"local-sdp:{event.text}")
                except Exception:
                    pass

        def on_dtmf(digit_event: Any) -> None:
            digit = str(getattr(digit_event, "digit", "") or "").upper()
            if len(digit) == 1 and digit in _VALID_DTMF:
                self._on_native_dtmf(app_id, digit)

        try:
            native_call.on(on_event)
            native_call.on_dtmf(on_dtmf)
        except Exception:
            pass

    # -- loop-thread coroutines ------------------------------------------

    async def _async_start(self) -> None:
        from baresip import Account, Config, Runtime, UserAgent
        from baresip.events import Event as _Event

        cfg = self._config
        # sip_trace logs every SIP message under baresip.native.sip at
        # DEBUG (integration diagnostics only; credentials never appear
        # there -- auth material stays in headers the harness redacts).
        baresip_config = Config(
            net_interface=cfg.bind,
            audio_driver="aumem",
            max_concurrent_calls=cfg.max_calls,
            rtp_timeout=cfg.rtp_timeout,
            extra_config_text=cfg.extra_config_text,
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

    def _media_pump(self) -> None:
        """Single media thread: TX pump then RX drain per live call."""
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
            if not record.tx or record.local_hold or record.remote_closed:
                return
            chunk = record.tx.popleft()
            record.tx_bytes -= len(chunk)
        try:
            audio = native.audio
        except Exception:
            with self._lock:
                record.tx.appendleft(chunk)
                record.tx_bytes += len(chunk)
            return
        try:
            try:
                info = audio.info()
                record.tx_rate = int(info.tx_sample_rate or record.tx_rate)
            except Exception:
                pass
            accepted = audio.write(chunk)
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
            # AudioRestarted/AudioNotActive: re-queue for the next epoch
            # when media is back; drop when the call is gone.
            if type(error).__name__ == "AudioRestarted":
                with self._lock:
                    record.audio_restarts += 1
            try:
                audio.info()
            except Exception:
                with self._lock:
                    if record.app_id in self._calls:
                        record.tx.appendleft(chunk)
                        record.tx_bytes += len(chunk)
                        while (
                            record.tx_bytes > TX_QUEUE_MAX_BYTES and record.tx
                        ):
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
            if record.local_hold or record.close_emitted:
                return
        try:
            audio = native.audio
        except Exception:
            return
        try:
            raw = audio.read(RX_READ_BYTES)
        except Exception:
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
            return
        with self._lock:
            if record.local_hold or record.close_emitted:
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
        listener = self._current_listener()
        if listener is None:
            return
        # Late close racing this delivery is benign: unknown ids and
        # wrong-state frames are dropped by the core/session.
        with self._cb_lock:
            if not self._is_running():
                return
            try:
                listener.on_caller_audio(record.app_id, frame)
            except Exception:
                pass

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

    def _set_reg_state(self, state: TelephonyRegistrationState) -> None:
        with self._lock:
            listener = self._status
            self._set_reg_state_locked(state)
        if listener is not None:
            try:
                listener.on_registration_state(state)
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
    """Composition seam: adapter registration states -> core health."""

    def __init__(self, core: Any) -> None:
        self._core = core

    def on_registration_state(
        self, state: TelephonyRegistrationState, detail: str = ""
    ) -> None:
        try:
            self._core.report_telephony_state(state, detail)
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
