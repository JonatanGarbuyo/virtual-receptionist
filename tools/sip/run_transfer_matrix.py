"""Focused #26 transfer matrix: symbolic transfer -> real PBX handoff.

Reuses the #25 harness (pinned Asterisk image, raw-baresip counterparty,
provenance gate) without duplicating the media/registration matrix. Five
scenarios, each driven THROUGH CallSession with a typed TransferRequest
carrying only a symbolic destination id:

  M1  primary accepted            -> HANDED_OFF, fallback never called
  M2  primary rejected + fallback -> HANDED_OFF (fallback exactly once)
  M3  primary timeout + fallback  -> HANDED_OFF (fallback exactly once)
  M4  primary transport-error     -> HANDED_OFF (fallback exactly once)
  M5  primary + fallback rejected -> apology + TERMINATING -> ENDED

Outcome mechanisms (exactly as evidenced, no hidden fabrication):

  accepted:        real REFER to registered auto-answering 702 (PBX bridges)
  rejected:        real REFER to unassigned sip:999@... (PBX 404/NOTIFY fail)
  timeout:         real REFER to 299 (rings 60 s, never answers); the real
                   adapter watchdog (short declarative sip.transfer_timeout,
                   never a sleep) reports TIMEOUT
  transport-error: fault injected at the owned native seam (record.native
                   replaced by a stub whose transfer() raises OSError on a
                   real established leg); the REAL adapter normalizes it to
                   TRANSPORT_ERROR and the REAL fallback REFER completes.
                   CallSession never fabricates the enum.

Usage: python3 tools/sip/run_transfer_matrix.py [--keep] [--allow-dirty]
         [--transfer-timeout 8] [--codecs pcmu]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tools", "sip"))

import run_matrix as H

REPO = H.REPO
AST_NAME = "vr-asterisk-26"
AST_IMAGE = H.AST_IMAGE
EVIDENCE = os.path.join(REPO, "docs", "evidence", "26-transfer-matrix", "evidence.json")

SIP_USER = H.SIP_USER
SIP_PASS = H.SIP_PASS
CALLER_USER = H.CALLER_USER
CALLER_PASS = H.CALLER_PASS
DOMAIN = H.DOMAIN
AST_SIP = H.AST_SIP
ADAPTER_SIP = H.ADAPTER_SIP

LOG = logging.getLogger("transfer-matrix")


class TransferListener(H.RecordingListener):
    """Append-only outcome recording: every normalized outcome per call
    with a timestamp. Never overwrites, so a fast fallback outcome can
    never hide the primary's (the #26 rerun lesson)."""

    def __init__(self) -> None:
        super().__init__()
        self.outcome_lists: dict[str, list] = {}

    def on_transfer_result(self, call_id: str, result) -> None:
        super().on_transfer_result(call_id, result)
        with self._lock:
            self.outcome_lists.setdefault(call_id, []).append(
                (time.monotonic(), result)
            )

    def outcomes(self, call_id: str) -> list:
        with self._lock:
            return [result for _, result in self.outcome_lists.get(call_id, [])]

    def outcome_events(self, call_id: str) -> list:
        """(monotonic timestamp, result) pairs, append-only."""
        with self._lock:
            return list(self.outcome_lists.get(call_id, []))


def wait_nth_outcome(listener, call_id, n: int, timeout: float):
    """Block until n outcomes are recorded for one call (fail-safe wait;
    correctness comes from the recorded content, never the delay)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(listener.outcomes(call_id)) >= n:
            return listener.outcomes(call_id)[n - 1]
        time.sleep(0.05)
    current = listener.outcomes(call_id)
    return current[n - 1] if len(current) >= n else None


def scenario_record() -> dict:
    return {
        "destination_ids": [],
        "primary_outcome": None,
        "fallback_outcomes": [],
        "fallback_attempts": 0,
        "final_state": None,
        "history": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--codecs", default="pcmu")
    parser.add_argument("--transfer-timeout", type=float, default=8.0)
    parser.add_argument("--only", default="",
                        help="run a single scenario (M1..M5), debug only")
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="run despite uncommitted code/harness changes (development "
        "only: evidence is then NOT attributable to tested_code_sha)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    checks = H.Check()
    evidence: dict = {
        "checks": checks.items,
        "scenarios": {},
        "sip_timelines": {},
        "keep": args.keep,
    }

    from receptionist.baresip_adapter import (
        BaresipTelephonyAdapter,
        wire_baresip_core,
    )
    from receptionist.boundaries import TelephonyRegistrationState, TransferResult
    from receptionist.call_session import CallState
    from receptionist.config import ConfigService, InMemoryConfigRepository
    from receptionist.core import ReceptionistCore
    from receptionist.health import HealthStatus
    from receptionist.persistence import (
        InMemoryAuditLog,
        InMemoryCallRepository,
        InMemoryMessageRepository,
        InMemoryTranscriptStore,
        RuntimeStorage,
    )
    from receptionist.policy import Destination as _Destination
    from receptionist.policy import Limits, PolicyEngine, RetentionPolicy
    from receptionist.resilience import ResilienceConfig
    from receptionist.telephony_config import TelephonyConfig
    from fakes import FakePolicy, FakeVoiceBackend

    codec = args.codecs
    run_id = f"m26-{int(time.time())}-{os.getpid()}"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        head_sha = H.sh("git", "rev-parse", "HEAD", timeout=15).stdout.strip()
    except Exception:
        head_sha = "unknown"
    evidence["meta"] = {
        "os": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "codec_run": codec,
        "transfer_timeout_seconds": args.transfer_timeout,
        "run_id": run_id,
        "started_at": started_at,
        "head_sha": head_sha,
    }
    _code_sha, _dirty_inputs = H.code_provenance()
    evidence["meta"]["tested_code_sha"] = _code_sha
    evidence["meta"]["worktree_dirty"] = _dirty_inputs
    evidence["meta"]["allow_dirty"] = bool(args.allow_dirty)
    _clean = not _dirty_inputs
    checks.record("worktree-clean", "matrix runs from a clean input tree",
                  _clean, "clean" if _clean else f"dirty_inputs={_dirty_inputs[:10]}")
    if _dirty_inputs and not args.allow_dirty:
        checks.record("worktree-abort", "no run on dirty input tree", False,
                      "commit code first or re-run with --allow-dirty")
        return _finish(evidence, checks, 1)
    try:
        import baresip

        evidence["meta"]["baresip_python"] = baresip.__version__
    except Exception:
        evidence["meta"]["baresip_python"] = "missing"

    import socket as _socket

    busy = []
    for _port in (5060, 5070, 5071, 5076):
        _sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        try:
            _sock.bind(("127.0.0.1", _port))
        except OSError:
            busy.append(_port)
        finally:
            _sock.close()
    if busy:
        checks.record("preflight-ports", "loopback SIP ports free", False,
                      f"busy={busy}; kill stray probes first")
        return _finish(evidence, checks, 1)

    r = H.sh("docker", "build", "-q", "-t", AST_IMAGE,
             os.path.join(REPO, "tools", "sip", "asterisk"), timeout=600)
    checks.record("img-asterisk", "asterisk image builds", r.returncode == 0,
                  (r.stderr or "")[-300:])
    ver = H.docker("run", "--rm", AST_IMAGE, "asterisk", "-V", timeout=60)
    evidence["meta"]["asterisk"] = (ver.stdout or ver.stderr or "").strip()
    # Fail fast when the pinned image does not build. worktree-clean is
    # excluded: with --allow-dirty it fails by design (unattributable
    # evidence) while the run itself must still proceed.
    if any(item["id"].startswith("img-") for item in checks.failed) and not args.keep:
        return _finish(evidence, checks, 1)

    H.docker("rm", "-f", AST_NAME, timeout=30)
    r = H.docker("run", "-d", "--name", AST_NAME, "--network", "host",
                 AST_IMAGE, timeout=60)
    checks.record("pbx-start", "asterisk container starts", r.returncode == 0,
                  (r.stderr or "")[-200:])

    # Point the shared helpers at this run's container.
    H.AST_NAME = AST_NAME
    ready = H.wait_asterisk_ready()
    checks.record("pbx-ready", "asterisk PJSIP ready with 700", ready)
    if not ready:
        return _finish(evidence, checks, 1)

    adapter = BaresipTelephonyAdapter(
        TelephonyConfig(
            username=SIP_USER,
            domain=DOMAIN,
            password=SIP_PASS,
            registrar=DOMAIN,
            bind="127.0.0.1",
            transport="udp",
            reg_interval=600,
            audio_codecs=(codec,),
            dtmf_mode="rtpevent",
            max_calls=2,
            rtp_timeout=30,
            transfer_timeout_seconds=args.transfer_timeout,
            sip_trace=True,
            sip_listen="127.0.0.1:5070",
        )
    )

    sip_lines: list[str] = []
    capturing = [True]

    class _SipTrace(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if not capturing[0]:
                return
            try:
                text = record.getMessage()
            except Exception:
                return
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or "Authorization" in stripped:
                    continue
                import re as _re

                if _re.match(
                    r"^(INVITE|ACK|BYE|CANCEL|REFER|NOTIFY|INFO|OPTIONS|"
                    r"REGISTER|UPDATE|PRACK|MESSAGE|SUBSCRIBE|SIP/2\.0)",
                    stripped,
                ):
                    sip_lines.append(stripped[:160])

    logging.getLogger("baresip.native.sip").addHandler(_SipTrace())
    logging.getLogger("baresip.native.sip").setLevel(logging.DEBUG)

    listener = TransferListener()
    # One process-wide call-id sequence: two sequential cores must never
    # reissue the same app id while the shared listener still records it.
    call_ids = H.CounterIds()

    def make_core(destinations: dict, fallback_id: str):
        voice = FakeVoiceBackend()
        core = ReceptionistCore(
            telephony=adapter,
            voice=voice,
            config_service=ConfigService(
                InMemoryConfigRepository({"greeting": "Hola", "language": "es"})
            ),
            policy=FakePolicy(),
            clock=H.WallClock(),
            resilience=ResilienceConfig(max_ai_sessions=2),
            policy_engine=PolicyEngine(
                destinations=destinations,
                fallback_id=fallback_id,
                limits=Limits(),
            ),
            runtime=RuntimeStorage(
                calls=InMemoryCallRepository(),
                messages=InMemoryMessageRepository(clock=H.WallClock()),
                transcripts=InMemoryTranscriptStore(),
                audit=InMemoryAuditLog(),
            ),
            retention=RetentionPolicy(),
            call_ids=call_ids,
        )
        adapter.set_listener(listener)
        wire_baresip_core(core, adapter)
        core_listener = adapter._listener  # noqa: SLF001

        class FanOut:
            def on_answered(self, call_id: str) -> None:
                listener.on_answered(call_id)
                core_listener.on_answered(call_id)

            def on_caller_hangup(self, call_id: str) -> None:
                listener.on_caller_hangup(call_id)
                core_listener.on_caller_hangup(call_id)

            def on_hangup_completed(self, call_id: str) -> None:
                listener.on_hangup_completed(call_id)
                core_listener.on_hangup_completed(call_id)

            def on_transfer_result(
                self, call_id: str, result, attempt_id=None
            ) -> None:
                listener.on_transfer_result(call_id, result)
                core_listener.on_transfer_result(call_id, result, attempt_id)

            def on_caller_audio(self, call_id: str, frame) -> None:
                listener.on_caller_audio(call_id, frame)
                core_listener.on_caller_audio(call_id, frame)

            def on_dtmf(self, call_id: str, digit: str) -> None:
                listener.on_dtmf(call_id, digit)
                core_listener.on_dtmf(call_id, digit)

            def on_remote_hold(self, call_id: str, held: bool) -> None:
                listener.on_remote_hold(call_id, held)
                core_listener.on_remote_hold(call_id, held)

            def on_media_failed(self, call_id: str, detail: str = "") -> None:
                listener.on_media_failed(call_id, detail)
                core_listener.on_media_failed(call_id, detail)

        adapter.set_listener(FanOut())
        return core, voice

    # Synthetic destinations only (never production data). The adapter
    # only ever sees the resolved PBX-local targets below.
    dest_ok = _Destination(id="primaria-ok", target=f"sip:702@{AST_SIP}",
                           kind="extension", label="Primaria OK")
    # Timeout primary: the when-sent target is irrelevant because the
    # attempt is held open by a stuck transfer command (see M3): the
    # synthetic 299 target documents intent only.
    dest_ring = _Destination(id="primaria-lenta", target=f"sip:299@{AST_SIP}",
                             kind="extension", label="Primaria lenta")
    dest_bad_domain = _Destination(id="primaria-rota",
                                   target="sip:702@nonexistent.invalid",
                                   kind="extension", label="Primaria rota")
    dest_bogus = _Destination(id="primaria-inexistente",
                              target=f"sip:999@{AST_SIP}",
                              kind="extension", label="Primaria inexistente")
    dest_fb_ok = _Destination(id="guardia", target=f"sip:702@{AST_SIP}",
                              kind="extension", label="Guardia")
    dest_fb_bad = _Destination(id="guardia-caida", target=f"sip:998@{AST_SIP}",
                               kind="extension", label="Guardia caída")

    coreA, voiceA = make_core(
        {d.id: d for d in (dest_ok, dest_ring, dest_bad_domain, dest_bogus,
                           dest_fb_ok, dest_fb_bad)},
        fallback_id="guardia",
    )
    coreA.start()
    state = adapter.start(timeout=20)
    checks.record("reg-initial", "REGISTER 200 OK against Asterisk",
                  state is TelephonyRegistrationState.REGISTERED, state.value)
    checks.record("ready-telephony", "service READY once registered",
                  coreA.health.status is HealthStatus.READY,
                  coreA.health.status.value)
    if state is not TelephonyRegistrationState.REGISTERED:
        return _finish(evidence, checks, 1)

    caller = H.CallerProc([codec])
    started = caller.start()
    checks.record("caller-reg", "counterparty REGISTERs as 701", started)
    if not started:
        return _finish(evidence, checks, 1)

    caller702 = H.CallerProc([codec], listen="127.0.0.1:5076",
                             user="702", password="vr-test-702")
    ok702 = caller702.start()
    armed702 = False
    if ok702:
        caller702.cmd({"op": "arm_autoanswer"})
        armed702 = caller702.expect(
            lambda e: e.get("event") == "autoanswer_armed", 10) is not None
    checks.record("xfer-target-ready",
                  "transfer target 702 registered + answering",
                  ok702 and armed702, f"reg={ok702} armed={armed702}")
    if not (ok702 and armed702):
        return _finish(evidence, checks, 1)



    from receptionist.boundaries import TransferRequest as _TR

    def run_leg(tag: str, core, voice):
        """Dial a fresh leg through Asterisk, drive it to LISTENING."""
        app = H.dial_identified(caller, lambda: caller.dial(tag), core)
        if app is None:
            H.diagnose_dial(evidence, f"{tag}-dial", adapter, core, caller)
            return None, None
        if not listener.wait_answered(app, 15):
            return None, None
        session = core.get_session(app)
        live = voice.sessions.get(app)
        if session is None or live is None:
            return None, None
        live.finish_playback(session.current_turn)
        if session.mode is None or session.mode.value != "listening":
            # Greeting drain fallback: one tick is enough on a live leg.
            time.sleep(0.5)
        return app, session

    only = {s.strip() for s in (args.only or "").split(",") if s.strip()}
    debug = os.getenv("VR_TRANSFER_DEBUG", "") == "1"

    def want(name: str) -> bool:
        return not only or name in only

    if debug:
        logging.getLogger("receptionist.telephony").setLevel(logging.DEBUG)
        logging.getLogger("receptionist.session").setLevel(logging.DEBUG)
        _orig_closed = adapter._on_native_closed  # noqa: SLF001

        def _logged_closed(app_id: str, reason: str) -> None:
            try:
                with adapter._lock:  # noqa: SLF001
                    record = adapter._calls.get(app_id)  # noqa: SLF001
                    info = (
                        f"open={record.transfer_open} "
                        f"attempt={record.transfer_attempt_id} "
                        f"local={record.local_close} "
                        f"stashed={record.pending_close_reason}"
                        if record is not None
                        else "no-record"
                    )
            except Exception:
                info = "lock-error"
            LOG.info("DBG closed app=%s reason=%r %s", app_id, reason, info)
            return _orig_closed(app_id, reason)

        adapter._on_native_closed = _logged_closed  # noqa: SLF001
        _orig_emit = adapter._emit_transfer_result  # noqa: SLF001

        def _logged_emit(app_id: str, result, attempt_id=None) -> None:
            LOG.info("DBG emit app=%s result=%s attempt=%s",
                     app_id, result, attempt_id)
            return _orig_emit(app_id, result, attempt_id)

        adapter._emit_transfer_result = _logged_emit  # noqa: SLF001
        _orig_bt = adapter.blind_transfer  # noqa: SLF001

        def _logged_bt(call_id: str, target: str, attempt_id=None) -> None:
            try:
                with adapter._lock:  # noqa: SLF001
                    record = adapter._calls.get(call_id)  # noqa: SLF001
                    info = (
                        f"open={record.transfer_open} "
                        f"attempt={record.transfer_attempt_id} "
                        f"native={type(record.native).__name__}"
                        if record is not None
                        else "no-record"
                    )
            except Exception:
                info = "lock-error"
            LOG.info("DBG blind_transfer app=%s target=%s attempt=%s %s",
                     call_id, target, attempt_id, info)
            return _orig_bt(call_id, target, attempt_id)

        adapter.blind_transfer = _logged_bt  # noqa: SLF001
        _orig_failed = adapter._transfer_failed  # noqa: SLF001

        def _logged_failed(record, error, attempt_id=None) -> None:
            LOG.info("DBG transfer_failed app=%s error=%s status=%s attempt=%s",
                     record.app_id, type(error).__name__,
                     getattr(error, "status", None), attempt_id)
            return _orig_failed(record, error, attempt_id)

        adapter._transfer_failed = _logged_failed  # noqa: SLF001

    def history_counts(session) -> dict:
        return {
            "transfer_handoff": session.history.count(CallState.TRANSFER_HANDOFF),
            "fallback_handoff": session.history.count(CallState.FALLBACK_HANDOFF),
            "handed_off": session.history.count(CallState.HANDED_OFF),
            "ended": session.history.count(CallState.ENDED),
        }

    def cleanup_tag(tag: str) -> None:
        try:
            caller.cmd({"op": "hangup", "tag": tag})
        except Exception:
            pass
        # Free 702-side capacity: answered transfer legs hold INCOMING-*
        # calls on the target UA; release them best-effort between runs.
        for index in range(6):
            try:
                caller702.cmd({"op": "hangup", "tag": f"INCOMING-{index}"})
            except Exception:
                pass

    def audit_trail(core) -> list:
        try:
            events = core._runtime.audit.list_all()  # noqa: SLF001
        except Exception:
            return []
        return [
            {
                "decision": e.decision.value,
                "destination": e.destination,
                "result": str(e.result) if e.result is not None else None,
                "detail": e.detail,
            }
            for e in events
        ]

    def finalize(rec: dict, session, core, audit_start: int = 0) -> dict:
        counts = history_counts(session)
        rec["fallback_attempts"] = counts["fallback_handoff"]
        rec["final_state"] = session.state.value
        rec["history"] = [s.value for s in session.history]
        rec["history_counts"] = counts
        try:
            rec["transfer_attempts_used"] = session._attempts_used  # noqa: SLF001
        except Exception:
            pass
        # Per-scenario slice: the core is shared across M1..M4, so only
        # entries appended after this scenario started belong to it.
        rec["audit"] = audit_trail(core)[audit_start:]
        rec["outcomes"] = [str(r) for r in listener.outcomes(session.call_id)]
        return rec

    # -- M1: primary accepted ------------------------------------------
    rec = scenario_record()
    app, session = ((None, None) if not want("M1")
                      else run_leg("M1", coreA, voiceA))
    if not want("M1"):
        checks.record("M1-accepted", "primary accepted -> HANDED_OFF", True,
                      "skipped by --only")
        app, session = None, None
    if app is None or session is None:
        if want("M1"):
            checks.record("M1-accepted", "primary accepted -> HANDED_OFF", False,
                          "dial/answer failed")
    else:
        rec["destination_ids"] = ["primaria-ok"]
        started = time.monotonic()
        audit_start = len(audit_trail(coreA))
        voiceA.sessions[app].deliver_action_request(_TR("primaria-ok"))
        result = wait_nth_outcome(listener, app, 1, 40)
        target_up = caller702.expect(
            lambda e: e.get("event") == "established"
            and str(e.get("tag", "")).startswith("INCOMING"), 30) is not None
        gone = H.wait_for(lambda: coreA.get_session(app) is None, 20)
        rec["duration_seconds"] = round(time.monotonic() - started, 1)
        counts = history_counts(session)
        rec["primary_outcome"] = str(result)
        finalize(rec, session, coreA, audit_start)
        checks.record(
            "M1-accepted", "primary accepted -> HANDED_OFF, fallback never called",
            result is TransferResult.ACCEPTED_BY_PBX and target_up and gone
            and counts["fallback_handoff"] == 0
            and counts["handed_off"] == 1 and counts["ended"] == 1,
            f"result={result} target_up={target_up} gone={gone} counts={counts}")
    evidence["scenarios"]["M1-primary-accepted"] = rec
    cleanup_tag("M1")

    # -- M2: primary rejected + fallback accepted ----------------------
    rec = scenario_record()
    app, session = ((None, None) if not want("M2")
                      else run_leg("M2", coreA, voiceA))
    if not want("M2"):
        checks.record("M2-rejected-fallback", "rejected -> fallback accepted", True, "skipped by --only")
        app, session = None, None
    if app is None or session is None:
        if want("M2"):
            checks.record("M2-rejected-fallback", "rejected -> fallback accepted", False,
                      "dial/answer failed")
    else:
        rec["destination_ids"] = ["primaria-inexistente", "guardia"]
        started = time.monotonic()
        audit_start = len(audit_trail(coreA))
        voiceA.sessions[app].deliver_action_request(_TR("primaria-inexistente"))
        first = wait_nth_outcome(listener, app, 1, 40)
        second = wait_nth_outcome(listener, app, 2, 40)
        target_up = caller702.expect(
            lambda e: e.get("event") == "established"
            and str(e.get("tag", "")).startswith("INCOMING"), 30) is not None
        gone = H.wait_for(lambda: coreA.get_session(app) is None, 20)
        rec["duration_seconds"] = round(time.monotonic() - started, 1)
        counts = history_counts(session)
        rec["primary_outcome"] = str(first)
        rec["fallback_outcomes"] = [str(second)]
        finalize(rec, session, coreA, audit_start)
        checks.record(
            "M2-rejected-fallback", "rejected -> fallback once -> HANDED_OFF",
            first is TransferResult.REJECTED
            and second is TransferResult.ACCEPTED_BY_PBX
            and counts["fallback_handoff"] == 1
            and counts["handed_off"] == 1 and counts["ended"] == 1
            and target_up and gone,
            f"first={first} second={second} target_up={target_up} "
            f"gone={gone} counts={counts}")
    evidence["scenarios"]["M2-rejected-fallback-accepted"] = rec
    cleanup_tag("M2")

    # -- M3: primary timeout + fallback accepted -----------------------
    rec = scenario_record()
    app, session = ((None, None) if not want("M3")
                      else run_leg("M3", coreA, voiceA))
    if not want("M3"):
        checks.record("M3-timeout-fallback", "timeout -> fallback accepted", True, "skipped by --only")
        app, session = None, None
    if app is None or session is None:
        if want("M3"):
            checks.record("M3-timeout-fallback", "timeout -> fallback accepted", False,
                      "dial/answer failed")
    else:
        rec["destination_ids"] = ["primaria-lenta", "guardia"]
        started = time.monotonic()
        action_at = time.monotonic()
        audit_start = len(audit_trail(coreA))
        # Timeout mechanism (documented fault at the owned native seam):
        # a stuck transfer command never resolves, so the attempt stays
        # without result on a REAL established dialog until the REAL
        # declarative watchdog fires. Real PBX stalls proved unusable:
        # Asterisk fast-fails every tried shape (404-style NOTIFY
        # failure, or a binding-level TransferFailed that also wedges
        # the dialog for the fallback). The fallback REFER below is a
        # real REFER over real SIP on the pristine dialog.
        import asyncio as _asyncio

        with adapter._lock:  # noqa: SLF001 (harness fault injection)
            record = adapter._calls.get(app)  # noqa: SLF001
            real_native = record.native if record is not None else None

        class _HangOnce:
            def __init__(self) -> None:
                self.armed = True

            def transfer(self, target: str):
                if self.armed:
                    self.armed = False

                    async def _hang():
                        await _asyncio.sleep(30)
                        raise OSError("stuck transfer finally gave up")

                    return _hang()
                return real_native.transfer(target)

        with adapter._lock:  # noqa: SLF001
            if record is not None:
                record.native = _HangOnce()
        rec["timeout_mechanism"] = (
            "single-shot stuck transfer command on a real leg; "
            "real declarative watchdog; real fallback REFER"
        )
        voiceA.sessions[app].deliver_action_request(_TR("primaria-lenta"))
        first = wait_nth_outcome(listener, app, 1, args.transfer_timeout + 30)
        # Watchdog proof: the first outcome must arrive at ~the
        # configured transfer timeout (a fast PBX failure would arrive
        # in <1 s and disprove the timeout shape).
        timed = [(round(t - action_at, 2), str(r))
                 for t, r in listener.outcome_events(app)]
        rec["outcome_timeline_seconds"] = timed
        second = wait_nth_outcome(listener, app, 2, 40)
        target_up = caller702.expect(
            lambda e: e.get("event") == "established"
            and str(e.get("tag", "")).startswith("INCOMING"), 30) is not None
        gone = H.wait_for(lambda: coreA.get_session(app) is None, 20)
        rec["duration_seconds"] = round(time.monotonic() - started, 1)
        counts = history_counts(session)
        rec["primary_outcome"] = str(first)
        rec["fallback_outcomes"] = [str(second)]
        finalize(rec, session, coreA, audit_start)
        watchdog_ok = (
            timed and timed[0][1] == str(TransferResult.TIMEOUT)
            and timed[0][0] >= args.transfer_timeout - 1.0
        )
        checks.record(
            "M3-timeout-fallback", "timeout -> fallback once -> HANDED_OFF",
            watchdog_ok
            and first is TransferResult.TIMEOUT
            and second is TransferResult.ACCEPTED_BY_PBX
            and counts["fallback_handoff"] == 1
            and counts["handed_off"] == 1 and counts["ended"] == 1
            and target_up and gone,
            f"timeline={timed} first={first} second={second} "
            f"target_up={target_up} gone={gone} counts={counts}")
    evidence["scenarios"]["M3-timeout-fallback-accepted"] = rec
    cleanup_tag("M3")

    # -- M4: primary transport-error + fallback accepted ---------------
    # Transport fault injected at the owned native seam on a real
    # established leg: a single-shot stub fails exactly the primary
    # REFER with OSError, then delegates every later call to the real
    # leg, so there is no restore race. The REAL adapter normalizes to
    # TRANSPORT_ERROR and the REAL fallback REFER completes. The session
    # never fabricates the enum.
    rec = scenario_record()
    app, session = ((None, None) if not want("M4")
                      else run_leg("M4", coreA, voiceA))
    if not want("M4"):
        checks.record("M4-transport-fallback", "transport-error -> fallback", True, "skipped by --only")
        app, session = None, None
    if app is None or session is None:
        if want("M4"):
            checks.record("M4-transport-fallback", "transport-error -> fallback", False,
                      "dial/answer failed")
    else:
        rec["destination_ids"] = ["primaria-rota", "guardia"]
        started = time.monotonic()
        audit_start = len(audit_trail(coreA))
        with adapter._lock:  # noqa: SLF001 (harness fault injection)
            record = adapter._calls.get(app)  # noqa: SLF001
            real_native = record.native if record is not None else None

        class _FailOnce:
            def __init__(self) -> None:
                self.armed = True

            def transfer(self, target: str):
                if self.armed:
                    self.armed = False
                    raise OSError("injected leg transport failure")
                return real_native.transfer(target)

        with adapter._lock:  # noqa: SLF001
            if record is not None:
                record.native = _FailOnce()
        voiceA.sessions[app].deliver_action_request(_TR("primaria-rota"))
        first = wait_nth_outcome(listener, app, 1, 30)
        second = wait_nth_outcome(listener, app, 2, 40)
        target_up = caller702.expect(
            lambda e: e.get("event") == "established"
            and str(e.get("tag", "")).startswith("INCOMING"), 30) is not None
        gone = H.wait_for(lambda: coreA.get_session(app) is None, 20)
        rec["duration_seconds"] = round(time.monotonic() - started, 1)
        counts = history_counts(session)
        rec["primary_outcome"] = str(first)
        rec["fallback_outcomes"] = [str(second)]
        rec["transport_mechanism"] = (
            "single-shot native-seam OSError on a real leg, "
            "delegating afterwards"
        )
        finalize(rec, session, coreA, audit_start)
        checks.record(
            "M4-transport-fallback",
            "transport-error -> fallback once -> HANDED_OFF",
            first is TransferResult.TRANSPORT_ERROR
            and second is TransferResult.ACCEPTED_BY_PBX
            and counts["fallback_handoff"] == 1
            and counts["handed_off"] == 1 and counts["ended"] == 1
            and target_up and gone,
            f"first={first} second={second} target_up={target_up} "
            f"gone={gone} counts={counts}")
    evidence["scenarios"]["M4-transport-error-fallback-accepted"] = rec
    cleanup_tag("M4")

    # -- M5: primary + fallback rejected -> deterministic exit ---------
    if want("M5"):
        coreB, voiceB = make_core(
            {d.id: d for d in (dest_bogus, dest_fb_bad, dest_fb_ok)},
            fallback_id="guardia-caida",
        )
        coreB.start()
    rec = scenario_record()
    app, session = ((None, None) if not want("M5")
                      else run_leg("M5", coreB, voiceB))
    if not want("M5"):
        checks.record("M5-double-failure", "double failure -> apology + ENDED",
                      True, "skipped by --only")
        app, session = None, None
    if app is None or session is None:
        if want("M5"):
            checks.record("M5-double-failure", "double failure -> apology + ENDED",
                          False, "dial/answer failed")
    else:
        rec["destination_ids"] = ["primaria-inexistente", "guardia-caida"]
        started = time.monotonic()
        audit_start = len(audit_trail(coreB))
        voiceB.sessions[app].deliver_action_request(_TR("primaria-inexistente"))
        first = wait_nth_outcome(listener, app, 1, 40)
        second = wait_nth_outcome(listener, app, 2, 40)
        # Deterministic exit: apology spoken, our BYE observed, ENDED.
        bye_seen = caller.expect(
            lambda e: e.get("event") == "closed" and e.get("tag") == "M5",
            20) is not None
        gone = H.wait_for(lambda: coreB.get_session(app) is None, 20)
        rec["duration_seconds"] = round(time.monotonic() - started, 1)
        counts = history_counts(session)
        rec["primary_outcome"] = str(first)
        rec["fallback_outcomes"] = [str(second)]
        finalize(rec, session, coreB, audit_start)
        checks.record(
            "M5-double-failure",
            "primary+fallback rejected -> apology/terminate -> ENDED",
            first is TransferResult.REJECTED
            and second is TransferResult.REJECTED
            and counts["fallback_handoff"] == 1
            and counts["handed_off"] == 0 and counts["ended"] == 1
            and bye_seen and gone,
            f"first={first} second={second} bye_seen={bye_seen} "
            f"gone={gone} counts={counts}")
    evidence["scenarios"]["M5-double-failure-exit"] = rec

    capturing[0] = False
    evidence["sip_timelines"]["adapter_trace"] = H.collapse_timeline(sip_lines)
    # Full directional trace (capped): collapsed timelines lose the
    # REFER/BYE/NOTIFY interleaving that decides transfer races.
    evidence["sip_timelines"]["adapter_trace_full"] = sip_lines[-300:]
    H.snapshot_adapter_events(evidence, adapter)
    try:
        caller.stop()
    except Exception:
        pass
    try:
        caller702.stop()
    except Exception:
        pass
    try:
        adapter.shutdown()
    except Exception:
        pass
    return _finish(evidence, checks, 0 if not checks.failed else 1)


def _finish(evidence: dict, checks: H.Check, code: int) -> int:
    if not evidence.get("keep"):
        try:
            H.docker("rm", "-f", AST_NAME, timeout=30)
        except Exception:
            pass
    evidence["checks"] = checks.items
    evidence["meta"]["finished_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    evidence["summary"] = {
        "total": len(checks.items),
        "failed": len(checks.failed),
    }
    _redact(evidence)
    os.makedirs(os.path.dirname(EVIDENCE), exist_ok=True)
    with open(EVIDENCE, "w") as handle:
        json.dump(evidence, handle, indent=2, default=str)
    print(f"evidence: {EVIDENCE} failed={len(checks.failed)}")
    for item in checks.failed:
        print(f"FAILED {item['id']}: {item['detail']}")
    return code


def _redact(evidence: dict) -> None:
    """Privacy gate: fail the run if secrets or production-like data leak.

    Synthetic passwords share the ``vr-test-`` prefix and must never
    appear anywhere in the evidence; neither may Authorization headers.
    All SIP targets in this matrix are synthetic fixtures (loopback,
    ``nonexistent.invalid``), never customer data.
    """

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and key.lower() in (
                    "password", "secret", "authorization", "auth_token",
                    "bot_token",
                ):
                    raise AssertionError(f"secret leaked under key {key!r}")
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)
        elif isinstance(node, str):
            lowered = node.lower()
            if "authorization" in lowered:
                raise AssertionError(
                    f"auth header leaked in evidence: {node[:60]!r}")
            if "vr-test-" in node:
                raise AssertionError(
                    f"credential leaked in evidence: {node[:60]!r}")

    _walk(evidence)


if __name__ == "__main__":
    raise SystemExit(main())
