"""Real #25 matrix: SIPp + disposable Asterisk + baresip adapter.

Runs the full telephony matrix against real software and writes
``docs/evidence/25-baresip-matrix/evidence.json``. Gated: needs docker
and the pinned images (built here). Nothing here is a mock: REGISTER,
INVITE/answer/reject/BYE, re-INVITE hold, DTMF, bidirectional PCM under
forced PCMU then PCMA, 2-call isolation, REFER, PBX restart, shutdown.

Roles: SIPp = deterministic protocol scripts (direct to the adapter's
loopback listener); Asterisk = real registrar/PBX semantics; a raw
baresip counterparty (tools/sip/caller.py, own process) = the far end
for media/DTMF/hold/transfer assertions.

Usage: python3 tools/sip/run_matrix.py [--keep] [--pcma-only]
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import platform
import queue
import re
import struct
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))

AST_NAME = "vr-asterisk-25"
AST_IMAGE = "vr-asterisk:25"
SIPP_IMAGE = "vr-sipp:25"
EVIDENCE = os.path.join(REPO, "docs", "evidence", "25-baresip-matrix", "evidence.json")

SIP_USER = "700"
SIP_PASS = "vr-test-700"
CALLER_USER = "701"
CALLER_PASS = "vr-test-701"
DOMAIN = "127.0.0.1"
AST_SIP = "127.0.0.1:5060"
ADAPTER_SIP = "127.0.0.1:5070"

LOG = logging.getLogger("matrix")


def sh(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def docker(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return sh("docker", *args, timeout=timeout)


class Check:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def record(self, cid: str, name: str, ok: bool, detail: str = "") -> None:
        self.items.append(
            {"id": cid, "name": name, "result": "pass" if ok else "fail", "detail": detail}
        )
        LOG.info("%s %s: %s", "PASS" if ok else "FAIL", cid, detail or name)

    @property
    def failed(self) -> list[dict]:
        return [i for i in self.items if i["result"] != "pass"]


class WallClock:
    def now(self) -> float:
        return time.time()


class CounterIds:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 1

    def next_id(self) -> str:
        with self._lock:
            cid = f"call-{self._next}"
            self._next += 1
            return cid


class RecordingListener:
    """Thread-safe TelephonyListener recording everything with events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.answered: dict[str, threading.Event] = {}
        self.caller_hangup: dict[str, int] = collections.Counter()
        self.hangup_completed: dict[str, int] = collections.Counter()
        self.transfers: dict[str, object] = {}
        self.transfer_events: dict[str, threading.Event] = {}
        self.audio: dict[str, list] = collections.defaultdict(list)
        self.audio_bytes: dict[str, int] = collections.Counter()
        self.dtmf: list[tuple[str, str]] = []
        self.dtmf_events: dict[tuple[str, str], threading.Event] = {}
        self.holds: list[tuple[str, bool]] = []
        self.hold_events: dict[tuple[str, bool], threading.Event] = {}
        self.media_failed: list[tuple[str, str]] = []
        self.threads: dict[str, set[str]] = collections.defaultdict(set)

    def _note(self, call_id: str) -> None:
        self.threads[call_id].add(threading.current_thread().name)

    def _event(self, mapping: dict, key) -> threading.Event:
        with self._lock:
            return mapping.setdefault(key, threading.Event())

    def on_answered(self, call_id: str) -> None:
        with self._lock:
            self._note(call_id)
            self.answered.setdefault(call_id, threading.Event()).set()

    def on_caller_hangup(self, call_id: str) -> None:
        with self._lock:
            self._note(call_id)
            self.caller_hangup[call_id] += 1

    def on_hangup_completed(self, call_id: str) -> None:
        with self._lock:
            self._note(call_id)
            self.hangup_completed[call_id] += 1

    def on_transfer_result(self, call_id: str, result) -> None:
        with self._lock:
            self.transfers[call_id] = result
            self.transfer_events.setdefault(call_id, threading.Event()).set()

    def on_caller_audio(self, call_id: str, frame) -> None:
        with self._lock:
            self._note(call_id)
            self.audio[call_id].append(frame)
            self.audio_bytes[call_id] += len(frame.pcm)

    def on_dtmf(self, call_id: str, digit: str) -> None:
        with self._lock:
            self.dtmf.append((call_id, digit))
            self.dtmf_events.setdefault((call_id, digit), threading.Event()).set()

    def on_remote_hold(self, call_id: str, held: bool) -> None:
        with self._lock:
            self.holds.append((call_id, held))
            self.hold_events.setdefault((call_id, held), threading.Event()).set()

    def on_media_failed(self, call_id: str, detail: str = "") -> None:
        with self._lock:
            self._note(call_id)
            self.media_failed.append((call_id, detail))

    # -- wait helpers (fail-safe timeouts; correctness via content) ----
    def wait_answered(self, call_id: str, timeout: float = 15) -> bool:
        # Race-free: the event may already be set before we wait.
        with self._lock:
            evt = self.answered.get(call_id)
            if evt is not None and evt.is_set():
                return True
        return self._event(self.answered, call_id).wait(timeout)

    def wait_transfer(self, call_id: str, timeout: float = 30):
        # Race-free: a fast outcome may precede the wait.
        with self._lock:
            if call_id in self.transfers:
                return self.transfers.get(call_id)
        if not self._event(self.transfer_events, call_id).wait(timeout):
            return None
        with self._lock:
            return self.transfers.get(call_id)

    def wait_dtmf(self, call_id: str, digit: str, timeout: float = 10) -> bool:
        with self._lock:
            if (call_id, digit) in self.dtmf:
                return True
        return self._event(self.dtmf_events, (call_id, digit)).wait(timeout)

    def wait_hold(self, call_id: str, held: bool, timeout: float = 10) -> bool:
        with self._lock:
            if (call_id, held) in self.holds:
                return True
        return self._event(self.hold_events, (call_id, held)).wait(timeout)


class CallerProc:
    """JSON-line driver for tools/sip/caller.py."""

    def __init__(
        self,
        codecs: list[str],
        listen: str = "127.0.0.1:5071",
        user: str = CALLER_USER,
        password: str = CALLER_PASS,
    ) -> None:
        self._listen = listen
        self._user = user
        self._password = password
        # Counterparty stderr is captured to a file (never DEVNULL):
        # dial/establish failures need the far-end traceback to be
        # diagnosable, and the tail lands in the evidence on failure.
        self._stderr_path = (
            f"/tmp/vr-caller-{user}-{listen.replace('.', '_').replace(':', '_')}.log"
        )
        self._stderr_file = open(self._stderr_path, "w")
        self._proc = subprocess.Popen(
            [sys.executable, os.path.join(REPO, "tools", "sip", "caller.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
        )
        self._queue: queue.Queue = queue.Queue()
        self._errors: queue.Queue = queue.Queue()
        self._stash: collections.deque = collections.deque()
        self._codecs = codecs
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            try:
                self._queue.put(json.loads(line))
            except Exception:
                pass

    def cmd(self, obj: dict) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def expect(self, pred, timeout: float = 20) -> dict | None:
        """Wait for the first event matching pred; non-matching events
        stay buffered for later expects (never dropped)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            while self._stash:
                evt = self._stash.popleft()
                if pred(evt):
                    return evt
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                evt = self._queue.get(timeout=min(0.2, remaining))
            except queue.Empty:
                continue
            if pred(evt):
                return evt
            self._stash.append(evt)
        return None

    def start(self, register: bool = True, listen: str | None = None) -> bool:
        self.cmd(
            {
                "op": "start",
                "user": self._user,
                "password": self._password,
                "domain": DOMAIN,
                "codecs": self._codecs,
                "listen": listen or self._listen,
                "register": register,
            }
        )
        want = "registered" if register else "started"
        return self.expect(lambda e: e.get("event") == want, 25) is not None

    def dial(self, tag: str) -> bool:
        return self.dial_direct(tag, f"sip:{SIP_USER}@{AST_SIP}")

    def dial_direct(self, tag: str, uri: str) -> bool:
        self.cmd({"op": "dial", "uri": uri, "tag": tag})
        return (
            self.expect(
                lambda e: e.get("event") == "established" and e.get("tag") == tag, 25
            )
            is not None
        )

    def stop(self) -> None:
        try:
            self.cmd({"op": "stop"})
        except Exception:
            pass
        try:
            self._proc.wait(timeout=20)
        except Exception:
            self._proc.kill()
        try:
            self._stderr_file.close()
        except Exception:
            pass

    def stderr_tail(self, lines: int = 25) -> str:
        """Last lines of counterparty stderr for failure evidence."""
        try:
            with open(self._stderr_path, errors="replace") as handle:
                content = handle.readlines()
        except OSError:
            return ""
        return "".join(content[-lines:])[-2000:]

    @property
    def returncode(self):
        return self._proc.poll()


def asterisk_cli(*args: str) -> str:
    proc = docker("exec", AST_NAME, "asterisk", "-rx", " ".join(args), timeout=30)
    return proc.stdout or ""


def wait_asterisk_ready(timeout: float = 60) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        proc = docker("exec", AST_NAME, "asterisk", "-rx", "core show version", timeout=15)
        if "Asterisk" in (proc.stdout or ""):
            eps = asterisk_cli("pjsip", "show", "endpoints")
            if "700" in eps:
                return True
        time.sleep(1)
    return False


def run_sipp(
    scenario: str,
    target: str,
    extra: list[str] | None = None,
    timeout: int = 10,
) -> tuple[int, str]:
    scen_dir = os.path.join(REPO, "tools", "sip", "sipp", "scenarios")
    # -s sets the user part of the Request-URI (sip:700@...): baresip
    # answers 404 to a user-less R-URI since no UA matches. Both streams
    # merged: SIPp reports failures on stderr. Full message log goes to a
    # host-mounted file: the reliable timeline source (no logger fragility).
    msg_file = f"/tmp/sipp-{os.path.splitext(scenario)[0]}.log"
    try:
        os.remove(msg_file)
    except OSError:
        pass
    # NOTE on Call-IDs: scenarios keep sipp's native [call_id]. An
    # earlier attempt to force unique-per-run ids via -inf + [field0]
    # broke sipp deterministically (its response-to-call correlation
    # keys on the internally generated id; overriding the header makes
    # every response "unmappable" and the run wedges with 0/0). Shared
    # ids across runs are safe: fresh processes/stacks each time, and
    # From-tags differ per scenario.
    # Unique loopback SIP port per scenario: a stuck sipp holder must
    # never poison later runs, and parallel leftovers cannot collide.
    local_ports = {
        "uac_answer_bye.xml": "5061",
        "uac_expect_reject.xml": "5062",
        "uac_hold_resume.xml": "5063",
        "uac_rtp_timeout.xml": "5064",
    }
    cmd = [
        "docker", "run", "--rm", "--network", "host",
        "-v", f"{scen_dir}:/scenarios:ro",
        "-v", "/tmp:/tmp",
        SIPP_IMAGE, "-sf", f"/scenarios/{scenario}", "-m", "1",
        "-timeout", str(timeout), "-timeout_error", "-s", SIP_USER,
        "-p", local_ports.get(scenario, "5061"),
        "-trace_msg", "-message_file", f"/tmp/{os.path.basename(msg_file)}",
    ] + (extra or []) + [target]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    finally:
        # A wedged sipp must never linger: reap by ancestor image so the
        # next phase starts from a clean slate (ports + dialogs).
        try:
            stale = sh("docker", "ps", "-q", "--filter", f"ancestor={SIPP_IMAGE}",
                       timeout=30).stdout.split()
            for cid in stale:
                docker("rm", "-f", cid, timeout=30)
        except Exception:
            pass
    combined = ((proc.stdout or "") + "\n" + (proc.stderr or ""))
    timeline = timeline_of_file(msg_file)
    if timeline:
        combined += "\nSIPT\n" + "\n".join(timeline[-30:])
    return proc.returncode, combined[-4000:]


def timeline_of_file(path: str) -> list[str]:
    """Method/status timeline from a SIPp message log, secrets redacted."""
    timeline: list[str] = []
    try:
        with open(path, errors="replace") as handle:
            content = handle.read()
    except OSError:
        return timeline
    current_dir = ""
    for line in content.splitlines():
        if line.startswith("UDP message sent"):
            current_dir = ">"
        elif line.startswith("UDP message received"):
            current_dir = "<"
        stripped = line.strip()
        if not stripped or "Authorization" in stripped:
            continue
        match = re.match(
            r"^(INVITE|ACK|BYE|CANCEL|REFER|NOTIFY|INFO|OPTIONS|"
            r"REGISTER|UPDATE|PRACK|MESSAGE|SUBSCRIBE|SIP/2\.0 \d{3})",
            stripped,
        )
        if match:
            timeline.append(f"{current_dir} {stripped[:150]}")
    return timeline


def pcm_rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    count = len(pcm) // 2
    values = struct.unpack(f"<{count}h", pcm)
    import math

    return math.sqrt(sum(v * v for v in values) / count) / 32768.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--codecs", default="pcmu")
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="run despite uncommitted code/harness changes (development "
        "only: evidence is then NOT attributable to tested_code_sha)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    checks = Check()
    evidence: dict = {
        "checks": checks.items,
        "sip_timelines": {},
        "threads": {},
        "keep": args.keep,
    }

    from receptionist.audio import make_frame, tone_pcm
    from receptionist.baresip_adapter import (
        BaresipTelephonyAdapter,
        wire_baresip_core,
    )
    from receptionist.boundaries import TelephonyRegistrationState, TransferResult
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
    from receptionist.policy import Limits, PolicyEngine, RetentionPolicy
    from receptionist.telephony_config import TelephonyConfig
    from fakes import FakePolicy, FakeVoiceBackend

    codec = args.codecs
    run_id = f"m25-{int(time.time())}-{os.getpid()}"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        head_sha = sh("git", "rev-parse", "HEAD", timeout=15).stdout.strip()
    except Exception:
        head_sha = "unknown"
    evidence["meta"] = {
        "os": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "codec_run": codec,
        "run_id": run_id,
        "started_at": started_at,
        "head_sha": head_sha,
    }
    # -- code provenance gate (M4) --------------------------------------
    # Auditable two-step flow: commit code/scenario/harness first, then
    # run matrices from the clean tree so tested_code_sha names exactly
    # what ran. head_sha above and tested_code_sha must agree on a clean
    # tree; dirty input trees fail fast (override with --allow-dirty
    # for iteration; that evidence is explicitly unattributable).
    # Evidence outputs (docs/evidence/) are run products, not inputs,
    # and never invalidate a run.
    _code_sha, _dirty_inputs = code_provenance()
    evidence["meta"]["tested_code_sha"] = _code_sha
    evidence["meta"]["worktree_dirty"] = _dirty_inputs
    evidence["meta"]["allow_dirty"] = bool(args.allow_dirty)
    _clean = not _dirty_inputs
    checks.record("worktree-clean", "matrix runs from a clean input tree",
                   _clean,
                   "clean" if _clean else
                   f"dirty_inputs={_dirty_inputs[:10]}")
    if _dirty_inputs and not args.allow_dirty:
        checks.record("worktree-abort", "no run on dirty input tree",
                       False,
                       f"commit code first or re-run with --allow-dirty "
                       f"(evidence then not attributable to {_code_sha})")
        return finish(evidence, checks, 1)
    try:
        import baresip

        evidence["meta"]["baresip_python"] = baresip.__version__
    except Exception:
        evidence["meta"]["baresip_python"] = "missing"

    # Preflight: every loopback SIP/RTP port must be free. A stray
    # probe holding one produces cascading nonsense (bind fails deep
    # inside the stack); fail fast with the culprit instead.
    import socket as _socket

    busy = []
    for _port in (5060, 5070, 5071, 5072, 5074, 5075, 5076):
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
        return finish(evidence, checks, 1)

    # -- build pinned images -----------------------------------------
    r = sh("docker", "build", "-q", "-t", AST_IMAGE,
           os.path.join(REPO, "tools", "sip", "asterisk"), timeout=600)
    checks.record("img-asterisk", "asterisk image builds", r.returncode == 0,
                   (r.stderr or "")[-300:])
    r = sh("docker", "build", "-q", "-t", SIPP_IMAGE,
           os.path.join(REPO, "tools", "sip", "sipp"), timeout=600)
    checks.record("img-sipp", "sipp image builds", r.returncode == 0,
                   (r.stderr or "")[-300:])
    ver = docker("run", "--rm", AST_IMAGE, "asterisk", "-V", timeout=60)
    evidence["meta"]["asterisk"] = (ver.stdout or ver.stderr or "").strip()
    ver = docker("run", "--rm", "--entrypoint", "sipp", SIPP_IMAGE, "-v", timeout=60)
    evidence["meta"]["sipp"] = ((ver.stdout or "") + (ver.stderr or "")).strip().splitlines()[0:1]
    # Fail fast when the pinned images do not build. worktree-clean is
    # excluded: with --allow-dirty it fails by design (unattributable
    # evidence) while the run itself must still proceed.
    if any(item["id"].startswith("img-") for item in checks.failed) and not args.keep:
        return finish(evidence, checks, 1)

    # -- start disposable PBX -----------------------------------------
    docker("rm", "-f", AST_NAME, timeout=30)
    r = docker("run", "-d", "--name", AST_NAME, "--network", "host", AST_IMAGE, timeout=60)
    checks.record("pbx-start", "asterisk container starts", r.returncode == 0,
                   (r.stderr or "")[-200:])
    ready = wait_asterisk_ready()
    checks.record("pbx-ready", "asterisk PJSIP ready with 700", ready)
    if not ready:
        return finish(evidence, checks, 1)

    # -- adapter + core -------------------------------------------------
    from receptionist.resilience import ResilienceConfig

    telephony = FakePolicy()
    voice = FakeVoiceBackend()
    clock = WallClock()
    call_ids = CounterIds()
    # Test-only widening to 2 AI sessions so two simultaneous legs stay
    # conversational (production default is 1 with the second call on PBX
    # fallback). Telephony 2-call isolation is what is under test.
    resilience = ResilienceConfig(max_ai_sessions=2)
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
            rtp_timeout=5,
            sip_trace=True,
            sip_listen="127.0.0.1:5070",
        )
    )
    from receptionist.policy import Destination as _Destination

    core = ReceptionistCore(
        telephony=adapter,
        voice=voice,
        config_service=ConfigService(
            InMemoryConfigRepository({"greeting": "Hola", "language": "es"})
        ),
        policy=telephony,
        clock=clock,
        resilience=resilience,
        # One trusted PBX-local transfer target for the end-to-end
        # transfer proof (symbolic id resolved by policy, never by the
        # caller/model; the adapter only ever sees the resolved target).
        policy_engine=PolicyEngine(
            destinations={
                "xfer702": _Destination(
                    id="xfer702",
                    target=f"sip:702@{AST_SIP}",
                    kind="extension",
                )
            },
            fallback_id="none",
            limits=Limits(),
        ),
        runtime=RuntimeStorage(
            calls=InMemoryCallRepository(),
            messages=InMemoryMessageRepository(clock=clock),
            transcripts=InMemoryTranscriptStore(),
            audit=InMemoryAuditLog(),
        ),
        retention=RetentionPolicy(),
        call_ids=call_ids,
    )
    op_error_count = [0]

    class _OpErrorWatch(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "telephony op failed" in record.getMessage():
                op_error_count[0] += 1

    logging.getLogger("receptionist.telephony").addHandler(_OpErrorWatch())

    # Real SIP timeline: capture the adapter's message trace (DEBUG) and
    # keep methods/statuses with credentials redacted.
    sip_lines: list[str] = []
    capturing = [False]

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
                if re.match(
                    r"^(INVITE|ACK|BYE|CANCEL|REFER|NOTIFY|INFO|OPTIONS|"
                    r"REGISTER|UPDATE|PRACK|MESSAGE|SUBSCRIBE|SIP/2\.0)",
                    stripped,
                ):
                    sip_lines.append(stripped[:160])

    logging.getLogger("baresip.native.sip").addHandler(_SipTrace())
    logging.getLogger("baresip.native.sip").setLevel(logging.DEBUG)

    def count_op_errors() -> int:
        return op_error_count[0]

    listener = RecordingListener()
    adapter.set_listener(listener)
    wire_baresip_core(core, adapter)
    # wire replaces the listener with core; keep recording via chained listener
    chain = listener
    core_listener = adapter._listener  # noqa: SLF001 (harness white-box)

    class FanOut:
        def __init__(self, first, second) -> None:
            self._a = first
            self._b = second

        def on_answered(self, call_id: str) -> None:
            self._a.on_answered(call_id)
            self._b.on_answered(call_id)

        def on_caller_hangup(self, call_id: str) -> None:
            self._a.on_caller_hangup(call_id)
            self._b.on_caller_hangup(call_id)

        def on_hangup_completed(self, call_id: str) -> None:
            self._a.on_hangup_completed(call_id)
            self._b.on_hangup_completed(call_id)

        def on_transfer_result(self, call_id: str, result) -> None:
            self._a.on_transfer_result(call_id, result)
            self._b.on_transfer_result(call_id, result)

        def on_caller_audio(self, call_id: str, frame) -> None:
            self._a.on_caller_audio(call_id, frame)
            self._b.on_caller_audio(call_id, frame)

        def on_dtmf(self, call_id: str, digit: str) -> None:
            self._a.on_dtmf(call_id, digit)
            self._b.on_dtmf(call_id, digit)

        def on_remote_hold(self, call_id: str, held: bool) -> None:
            self._a.on_remote_hold(call_id, held)
            self._b.on_remote_hold(call_id, held)

        def on_media_failed(self, call_id: str, detail: str = "") -> None:
            self._a.on_media_failed(call_id, detail)
            self._b.on_media_failed(call_id, detail)

    adapter.set_listener(FanOut(chain, core_listener))
    core.start()
    state = adapter.start(timeout=20)
    checks.record(
        "reg-initial", "REGISTER 200 OK against Asterisk",
        state is TelephonyRegistrationState.REGISTERED, state.value,
    )
    checks.record(
        "ready-telephony", "service READY once registered",
        core.health.status is HealthStatus.READY, core.health.status.value,
    )
    evidence["reg_aor_initial"] = asterisk_cli("pjsip", "show", "aor", "700")[-1200:]
    if state is not TelephonyRegistrationState.REGISTERED:
        # Fail fast: nothing downstream can work without registration
        # (e.g. a stale probe still holding the loopback SIP port).
        return finish(evidence, checks, 1)

    # -- negative registration (own processes) ---------------------------
    r = sh(sys.executable, os.path.join(REPO, "tools", "sip", "reg_probe.py"),
           SIP_USER, "wrong-password", DOMAIN, DOMAIN, timeout=60)
    bad = "STATE=registration_failed" in (r.stdout or "")
    checks.record("reg-badcreds", "invalid credentials fail cleanly", bad,
                   (r.stdout or "")[-200:])
    r = sh(sys.executable, os.path.join(REPO, "tools", "sip", "reg_probe.py"),
           SIP_USER, SIP_PASS, DOMAIN, "127.0.0.1:5099", timeout=60)
    down = "STATE=registration_failed" in (r.stdout or "")
    checks.record("reg-pbxdown", "unreachable PBX fails cleanly, no password leak",
                   down and "vr-test" not in (r.stdout or "") + (r.stderr or ""),
                   (r.stdout or "")[-200:])

    # -- SIPp s1: answer + remote BYE --------------------------------------
    capturing[0] = True
    rc, out = run_sipp("uac_answer_bye.xml", ADAPTER_SIP)
    s1_ok = rc == 0
    checks.record("sipp-answer-bye", "INVITE answered, remote BYE completed", s1_ok,
                   out[-500:])
    evidence["sip_timelines"]["s1"] = timeline_of(out)
    time.sleep(1)
    hangups = sum(listener.caller_hangup.values())
    checks.record("evt-remote-bye", "caller_hangup exactly once per SIPp call",
                   s1_ok and hangups == 1, f"caller_hangup total={hangups}")
    sessions_alive = len(core._sessions)  # noqa: SLF001
    checks.record("evt-cleanup", "session evicted after remote BYE",
                   sessions_alive == 0, f"sessions={sessions_alive}")

    # -- SIPp s2: decline ---------------------------------------------------
    from receptionist.boundaries import InboundCallHandler

    class DeclineAll(InboundCallHandler):
        def handle_incoming_call(self, caller_id: str, caller_name):
            return None

    adapter.set_inbound_handler(DeclineAll())
    rc, out = run_sipp("uac_expect_reject.xml", ADAPTER_SIP)
    checks.record("sipp-reject", "declined INVITE gets 486, no media",
                   rc == 0, out[-500:])
    evidence["sip_timelines"]["s2"] = timeline_of(out)
    from receptionist.baresip_adapter import CoreInboundBridge

    adapter.set_inbound_handler(CoreInboundBridge(core))

    # -- SIPp s3: remote hold/resume -----------------------------------------
    rc, out = run_sipp("uac_hold_resume.xml", ADAPTER_SIP)
    holds = [h for h in listener.holds if h[0].startswith("call-")]
    got_hold = any(h[1] for h in holds[-4:]) and any(not h[1] for h in holds[-4:])
    checks.record("sipp-hold", "re-INVITE hold/resume answered + observed",
                   rc == 0 and got_hold, f"rc={rc} holds={holds[-4:]}")
    evidence["sip_timelines"]["s3"] = timeline_of(out)

    # -- SIPp s4: RTP-timeout fault -----------------------------------------
    # Scripted fault/media scenario per #25: an established dialog with
    # zero RTP from the peer must be closed by the adapter's RTP
    # timeout. Observable outcome: the adapter emits caller_hangup for
    # the faulted call, the telephony media condition degrades, and
    # the session is evicted. The scenario itself fails (rc != 0) when
    # no BYE arrives, so silence can never count as coverage.
    from receptionist.alerting import (
        CODE_TELEPHONY_MEDIA_LOST as _MEDIA_LOST_S4,
    )
    from receptionist.alerting import HealthComponent as _HC_S4

    answered_before_s4 = set(listener.answered)
    rc, out = run_sipp("uac_rtp_timeout.xml", ADAPTER_SIP, timeout=25)
    evidence["sip_timelines"]["s4"] = timeline_of(out)
    fault_ids = sorted(set(listener.answered) - answered_before_s4)
    app_f = fault_ids[-1] if fault_ids else None
    hung_f = app_f is not None and listener.caller_hangup.get(app_f, 0) >= 1
    media_bad_s4 = any(
        c.component is _HC_S4.TELEPHONY and c.code == _MEDIA_LOST_S4
        for c in core.monitor.active_conditions()
    )
    evicted_f = app_f is not None and core.get_session(app_f) is None
    checks.record("sipp-rtp-timeout",
                   "no-RTP dialog closed by RTP timeout + media health",
                   rc == 0 and hung_f and media_bad_s4 and evicted_f,
                   f"rc={rc} app={app_f} hung={hung_f} "
                   f"media_lost={media_bad_s4} evicted={evicted_f}")

    # NOTE (no s5 CANCEL scenario): CANCEL-before-answer against a fast
    # responder is inherently racy for strict SIPp (the 200/CANCEL order
    # varies run to run and sipp scenarios must match exactly). The
    # pre-answer caller-hangup contract is proven deterministically at
    # the session seam instead
    # (test_call_session.PreAnswerCallerHangupTest: INCOMING ->
    # reject, CALLER_HANGUP outcome, no answer, no voice resources),
    # and remote hangup after answer is covered by s1/BYE, PBX-outage,
    # and shutdown-peer checks below.
    capturing[0] = False
    # Prefer the adapter-side trace (complete, redacted) over SIPp stdout.
    traced = collapse_timeline(sip_lines)
    if traced:
        evidence["sip_timelines"]["adapter_trace"] = traced
    time.sleep(1)

    # -- counterparty: media/DTMF/hold/transfer through Asterisk ---------------
    caller = CallerProc([codec])
    started = caller.start()
    checks.record("caller-reg", "counterparty REGISTERs as 701", started)
    if not started:
        return finish(evidence, checks, 1)
    app_a = dial_identified(caller, lambda: caller.dial("A"), core)
    ok = app_a is not None
    if not ok:
        diagnose_dial(evidence, "A", adapter, core, caller)
    checks.record("invite-routed", "INVITE via Asterisk answered + established", ok)
    both = ok and app_a is not None and listener.wait_answered(app_a, 5)
    checks.record("evt-answered", "adapter on_answered with app call id", bool(both),
                   f"app={app_a}")

    # PCM RX: caller tone -> adapter frames -> core voice layer
    app_b: str | None = None
    if both:
        caller.cmd({"op": "tone", "tag": "A", "seconds": 3.0, "freq": 440.0})
        got = wait_for(lambda: listener.audio_bytes.get(app_a, 0) >= 32000, 20)
        # Sequencing guard: the caller tone must FULLY stop (plus a
        # silence margin) before assistant TX starts, otherwise its tail
        # barge-ins the fresh turn and the harness measures stale audio
        # instead of the TX path. Product barge-in works as designed;
        # the harness must not overlap the two directions.
        caller.expect(
            lambda e: e.get("event") == "tone_done" and e.get("tag") == "A", 20)
        time.sleep(1.5)
        frames = listener.audio.get(app_a, [])
        joined = b"".join(f.pcm for f in frames[-100:])
        rms = pcm_rms(joined) if joined else 0.0
        vframe = voice.sessions.get(app_a)
        pushed = sum(len(f.pcm) for f in (vframe.pushed_audio if vframe else []))
        checks.record("pcm-rx", f"caller PCM reaches app seam ({codec})",
                       got and rms > 0.1 and pushed > 0,
                       f"bytes={listener.audio_bytes.get(app_a, 0)} rms={rms:.3f} voice_pushed={pushed}")
        sr_ok = all(f.sample_rate == 16000 and f.channels == 1 for f in frames[-10:])
        checks.record("pcm-rx-format", "inbound normalized to 16kHz mono pcm16",
                       bool(frames) and sr_ok, f"frames={len(frames)}")
        # Negotiated codec, registrar-side: the PBX reports the actual
        # on-the-wire codec per leg (offers alone prove nothing).
        expected_wire = {"pcmu": "ulaw", "pcma": "alaw"}.get(codec, codec)
        wire_codecs = parse_channel_codecs(asterisk_cli("pjsip", "show", "channelstats"))
        leg_codecs = sorted({c for ch, c in wire_codecs if ch.startswith("700-")})
        checks.record("codec-negotiated", f"wire codec is {expected_wire} ({codec})",
                       leg_codecs == [expected_wire],
                       f"legs={wire_codecs}")

        # PCM TX: assistant frames -> RTP -> caller reads tone
        session = core.get_session(app_a)
        live = voice.sessions.get(app_a)
        if session is not None and live is not None:
            from receptionist.call_session import ActiveMode as _Mode

            live.finish_playback(session.current_turn)
            live.deliver_caller_speech("hola")
            turn = session.current_turn
            chunk = tone_pcm(duration_seconds=0.02, sample_rate=16000)
            live.deliver_audio(turn, make_frame(chunk, 16000, sequence=0))
            if session.mode is not _Mode.SPEAKING:
                # Re-drive once with a fresh turn; diagnose if it persists.
                live = voice.sessions.get(app_a)
                if live is not None:
                    live.deliver_caller_speech("hola de nuevo")
                    turn = session.current_turn
                    live.deliver_audio(turn, make_frame(chunk, 16000, sequence=0))
            tx_primed = session.mode is _Mode.SPEAKING
            checks.record("pcm-tx-primed", "assistant turn reaches SPEAKING + playout",
                           tx_primed,
                           f"mode={session.mode} epoch={session._voice_epoch} "  # noqa: SLF001
                           f"live_current={live is session.voice_session}")
            reader_done: dict = {}
            start_seq = 1 if tx_primed else 0
            def read_caller() -> None:
                caller.cmd({"op": "read", "tag": "A", "seconds": 4.0, "drain": 1.0})
                evt = caller.expect(
                    lambda e: e.get("event") == "read_done" and e.get("tag") == "A", 25)
                if evt:
                    reader_done.update(evt)

            rt = threading.Thread(target=read_caller, daemon=True)
            rt.start()
            # Pre-roll: start streaming slightly before the reader's drain
            # ends so the measured window is live audio, not idle backlog.
            time.sleep(1.0)
            for seq in range(start_seq, start_seq + 200):
                live.deliver_audio(turn, make_frame(chunk, 16000, sequence=seq))
                time.sleep(0.02)
            rt.join(timeout=30)
            rms_tx = float(reader_done.get("rms_tail", reader_done.get("rms", 0.0)))
            with adapter._lock:  # noqa: SLF001
                dbg_rec = adapter._calls.get(app_a)  # noqa: SLF001
                tx_debug = {
                    "tx_bytes": dbg_rec.tx_bytes if dbg_rec else None,
                    "tx_dropped": dbg_rec.tx_dropped_bytes if dbg_rec else None,
                    "tx_rate": dbg_rec.tx_rate if dbg_rec else None,
                    "established": dbg_rec.established if dbg_rec else None,
                }
            try:
                native_stats = dbg_rec.native.audio.stats()  # noqa: SLF001
                tx_debug["native"] = {
                    "tx_silence": native_stats.tx_silence_frames,
                    "tx_starved": native_stats.tx_starved_frames,
                    "tx_buffered": native_stats.tx_buffered,
                    "tx_rejected": native_stats.tx_rejected_bytes,
                    "tx_flushed": native_stats.tx_flushed_bytes,
                }
            except Exception as error:
                tx_debug["native_error"] = type(error).__name__
            tx_debug["accepted_total"] = dbg_rec.tx_accepted_total if dbg_rec else None
            tx_debug["restarts"] = dbg_rec.audio_restarts if dbg_rec else None
            tx_debug["first_tx_at"] = dbg_rec.first_tx_at if dbg_rec else None
            evidence["tx_debug"] = tx_debug
            evidence["adapter_events"] = [
                {"t": round(t, 2), "call": c, "kind": k}
                for (t, c, k) in adapter.event_timeline()
            ]
            checks.record("pcm-tx", "assistant PCM reaches the caller over RTP",
                           rms_tx > 0.08 and int(reader_done.get("bytes", 0)) > 8000,
                           f"rms_tail={rms_tx} rms={reader_done.get('rms')} "
                           f"bytes={reader_done.get('bytes')} "
                           f"rx_cap={reader_done.get('rx_capacity')} "
                           f"adapter={tx_debug}")

        # DTMF both ways (RFC4733 through Asterisk)
        caller.cmd({"op": "dtmf", "tag": "A", "digits": "5"})
        rx = listener.wait_dtmf(app_a, "5", 12)
        checks.record("dtmf-rx", "caller DTMF reaches app with call id", rx,
                       f"dtmf={listener.dtmf[-3:]}")
        adapter.send_dtmf(app_a, "8")
        tx_evt = caller.expect(
            lambda e: e.get("event") == "dtmf" and e.get("tag") == "A"
            and e.get("digit") == "8", 12)
        checks.record("dtmf-tx", "adapter DTMF reaches the caller", tx_evt is not None,
                       str(tx_evt))

        # Local hold through the B2BUA leg: the re-INVITE must succeed
        # (no error), TX must pause while held and resume after. Asterisk
        # terminates hold locally (MOH) and does not forward re-INVITEs
        # end to end -- verified empirically -- so peer-observable hold
        # is proven on the direct leg (s3 + xfer-leg below) instead.
        op_errors_before = count_op_errors()
        adapter.hold(app_a)
        time.sleep(1)
        with adapter._lock:  # noqa: SLF001
            held_flag = adapter._calls.get(app_a).local_hold  # noqa: SLF001
        adapter.send_audio(app_a, make_frame(
            tone_pcm(duration_seconds=0.5, sample_rate=16000), 16000))
        rx_before = listener.audio_bytes.get(app_a, 0)
        time.sleep(1)
        with adapter._lock:  # noqa: SLF001
            rec_a = adapter._calls.get(app_a)  # noqa: SLF001
            queued_while_held = rec_a.tx_bytes if rec_a else -1
        rx_frozen = listener.audio_bytes.get(app_a, 0) == rx_before
        adapter.resume(app_a)
        time.sleep(1)
        with adapter._lock:  # noqa: SLF001
            rec_a2 = adapter._calls.get(app_a)  # noqa: SLF001
            resumed_flag = rec_a2.local_hold if rec_a2 else None
            drained = (rec_a2.tx_bytes if rec_a2 else 0) < queued_while_held
        checks.record("hold-local", "local hold pauses TX, resume drains, no errors",
                       held_flag is True and resumed_flag is False
                       and queued_while_held > 0 and drained
                       and count_op_errors() == op_errors_before,
                       f"held={held_flag} queued={queued_while_held} "
                       f"drained={drained} resumed={resumed_flag}")
        checks.record("hold-local-media", "no caller audio crosses while held",
                       rx_frozen, f"rx_frozen={rx_frozen}")
        # Caller re-INVITE through Asterisk succeeds at the SIP layer
        # (answered by the B2BUA leg); the session stays ACTIVE throughout.
        caller.cmd({"op": "hold", "tag": "A"})
        h_done = caller.expect(
            lambda e: e.get("event") == "hold_done" and e.get("tag") == "A", 15)
        caller.cmd({"op": "resume", "tag": "A"})
        r_done = caller.expect(
            lambda e: e.get("event") == "resume_done" and e.get("tag") == "A", 15)
        sess_a = core.get_session(app_a)
        checks.record("hold-remote-leg", "re-INVITE hold/resume through PBX succeeds",
                       h_done is not None and r_done is not None
                       and sess_a is not None and sess_a.state.value == "active",
                       f"hold_done={bool(h_done)} resume_done={bool(r_done)}")

    # Local hangup (AC3): the app ends a live leg; the caller must
    # observe our BYE and the session must complete and evict. Uses
    # a dedicated leg so the A/B isolation test below is untouched.
    app_e = dial_identified(caller, lambda: caller.dial("E"), core)
    hung_local = False
    if app_e is None:
        diagnose_dial(evidence, "E-local-hangup", adapter, core, caller)
    else:
        if listener.wait_answered(app_e, 15):
            session_e = core.get_session(app_e)
            if session_e is not None:
                session_e.end_call()
                hung_local = caller.expect(
                    lambda e: e.get("event") == "closed" and e.get("tag") == "E",
                    15,
                ) is not None
                hung_local = hung_local and wait_for(
                    lambda: core.get_session(app_e) is None, 15
                )
    checks.record("local-hangup", "app hangup BYEs the caller, session evicted",
                   hung_local, f"app={app_e}")

    # second call: isolation
    app_b = dial_identified(caller, lambda: caller.dial("B"), core)
    if app_b is None:
        diagnose_dial(evidence, "B-twocalls", adapter, core, caller)
    ok_b = app_b is not None
    checks.record("two-calls", "two simultaneous call objects, distinct app ids",
                   ok_b and app_b is not None and app_b != app_a,
                   f"a={app_a} b={app_b}")
    if ok_b and app_b:
        caller.cmd({"op": "tone", "tag": "B", "seconds": 2.0, "freq": 880.0})
        got_b = wait_for(lambda: listener.audio_bytes.get(app_b, 0) >= 16000, 15)
        checks.record("two-calls-audio", "per-call audio routing stays separate",
                       got_b, f"b_bytes={listener.audio_bytes.get(app_b, 0)}")
        caller.cmd({"op": "hangup", "tag": "A"})
        gone_a = wait_for(lambda: listener.caller_hangup.get(app_a, 0) >= 1, 15)
        time.sleep(1)
        still_b = core.get_session(app_b) is not None
        checks.record("two-calls-hangup", "hangup A keeps B alive",
                       gone_a and still_b, f"a_hangups={listener.caller_hangup.get(app_a, 0)} b_alive={still_b}")

    # -- direct leg: end-to-end hold forwarding + REFER primitive ------------
    # TRUE peer dialog (no B2BUA): an unregistered counterparty process
    # dials the adapter listener directly, so re-INVITEs and REFER travel
    # end to end over real SIP/RTP. A registered caller would proxy-route
    # through Asterisk even when dialling the listener address.
    callerB = CallerProc([codec])
    direct_ok = False
    app_x = None
    if callerB.start(register=False, listen="127.0.0.1:5072"):
        app_x = dial_identified(
            callerB,
            lambda: callerB.dial_direct("X", f"sip:{SIP_USER}@{ADAPTER_SIP}"),
            core,
        )
        if app_x is None:
            diagnose_dial(evidence, "X-direct", adapter, core, callerB)
        direct_ok = app_x is not None and listener.wait_answered(app_x, 5)
    checks.record("direct-leg", "direct INVITE answered with app call id",
                   direct_ok, f"app={app_x}")
    if direct_ok and app_x:
        # Stream caller audio across the hold window so the freeze
        # assertion is meaningful (an idle line proves nothing).
        callerB.cmd({"op": "tone", "tag": "X", "seconds": 8.0, "freq": 550.0})
        flowing = wait_for(
            lambda: listener.audio_bytes.get(app_x, 0) >= 16000, 15
        )
        callerB.cmd({"op": "hold", "tag": "X"})
        x_hold = listener.wait_hold(app_x, True, 15)
        bytes_at_hold = listener.audio_bytes.get(app_x, 0)
        time.sleep(1.5)
        frozen = listener.audio_bytes.get(app_x, 0) == bytes_at_hold
        # Pace re-INVITEs: a resume racing the hold transaction's tail
        # can be answered 491 and lost (real peers pace the same way).
        callerB.cmd({"op": "resume", "tag": "X"})
        x_resume = listener.wait_hold(app_x, False, 15)
        checks.record("hold-remote-direct", "remote re-INVITE observed end to end",
                       x_hold and x_resume, f"hold={x_hold} resume={x_resume}")
        checks.record("hold-remote-media", "no caller audio crosses while held",
                       bool(flowing) and x_hold and frozen,
                       f"flowing={flowing} bytes_at_hold={bytes_at_hold} "
                       f"bytes_later={listener.audio_bytes.get(app_x, 0)}")
        try:
            callerB.cmd({"op": "hangup", "tag": "X"})
        except Exception:
            pass
        wait_for(lambda: listener.caller_hangup.get(app_x, 0) >= 1
                 or listener.hangup_completed.get(app_x, 0) >= 1, 15)
    try:
        callerB.stop()
    except Exception:
        pass

    # -- real PBX transfer completes (702 answers): ACCEPTED_BY_PBX --------
    # A second counterparty registers as 702 with auto-answer armed; the
    # adapter REFERs live leg B to it; Asterisk bridges caller->702 and
    # reports success. End-to-end proof over real SIP/RTP with a real PBX.
    caller702 = CallerProc([codec], listen="127.0.0.1:5076",
                            user="702", password="vr-test-702")
    ok702 = caller702.start()
    armed702 = False
    if ok702:
        caller702.cmd({"op": "arm_autoanswer"})
        armed702 = caller702.expect(
            lambda e: e.get("event") == "autoanswer_armed", 10) is not None
    checks.record("xfer-target-ready", "transfer target 702 registered + answering",
                   ok702 and armed702, f"reg={ok702} armed={armed702}")
    if app_b and core.get_session(app_b) is not None and armed702:
        # Drive the transfer THROUGH the session (typed action request)
        # so the core reaches TRANSFER_HANDOFF and ends the session on
        # acceptance, exactly like the production #26 path will.
        from receptionist.boundaries import TransferRequest as _TR

        voice.sessions[app_b].deliver_action_request(_TR(destination_id="xfer702"))
        result = listener.wait_transfer(app_b, 75)
        target_up = caller702.expect(
            lambda e: e.get("event") == "established"
            and str(e.get("tag", "")).startswith("INCOMING"), 60) is not None
        b_gone = wait_for(lambda: core.get_session(app_b) is None, 20)
        checks.record("transfer", "PBX transfer completes -> ACCEPTED_BY_PBX",
                       result is TransferResult.ACCEPTED_BY_PBX
                       and target_up and b_gone,
                       f"result={result} target_up={target_up} b_gone={b_gone}")
    try:
        caller702.stop()
    except Exception:
        pass

    # -- real-PBX transfer failure normalizes to REJECTED -------------------
    # Fresh leg T via Asterisk: REFER it to 201 and let the real PBX
    # report the outcome (proven: 202 + NOTIFY sipfrag failure).
    app_t = dial_identified(caller, lambda: caller.dial("T"), core)
    if app_t is None:
        diagnose_dial(evidence, "T-xfer-reject", adapter, core, caller)
    if app_t is not None:
        listener.wait_answered(app_t, 10)
    if app_t and core.get_session(app_t) is not None:
        adapter.blind_transfer(app_t, f"sip:201@{AST_SIP}")
        result_t = listener.wait_transfer(app_t, 75)
        checks.record("transfer-rejected", "real PBX failure normalizes to REJECTED",
                       result_t is TransferResult.REJECTED, f"result={result_t}")
        wait_for(lambda: core.get_session(app_t) is None, 20)
    # hang up any leftover asterisk legs before the restart phase
    leftovers = []
    if both and app_a and core.get_session(app_a) is not None:
        leftovers.append("A")
    if app_b and core.get_session(app_b) is not None:
        leftovers.append("B")
    if app_t and core.get_session(app_t) is not None:
        leftovers.append("T")
    for tag in leftovers:
        try:
            caller.cmd({"op": "hangup", "tag": tag})
        except Exception:
            pass
    wait_for(lambda: len(core._sessions) == 0, 20)  # noqa: SLF001

    # -- explicit re-REGISTER (public primitive, registrar-verified) -------
    # refresh_registration() is the product seam (same stack path as
    # expiry refresh). Registrar-side proof: the 700 contact must still
    # be present with a bumped expiration afterwards. The AOR view
    # carries per-contact expiration for flap diagnosis.
    contacts_before = asterisk_cli("pjsip", "show", "contacts")
    aor_before = asterisk_cli("pjsip", "show", "aor", "700")
    state, fresh = adapter.refresh_registration(timeout=20)
    # The stack refreshes as unregister-old-dialog + register-new-dialog;
    # the registrar transiently shows no 700 contact between the two legs
    # (the adapter settle-wait covers the common case). Poll briefly so a
    # residual ms-scale race cannot flake the check, while a persistently
    # missing contact still fails loudly.
    contacts_after = ""
    contact_seen = False
    poll_end = time.monotonic() + 10.0
    while time.monotonic() < poll_end:
        contacts_after = asterisk_cli("pjsip", "show", "contacts")
        contact_seen = "700/" in contacts_after
        if contact_seen:
            break
        time.sleep(0.5)
    aor_after = asterisk_cli("pjsip", "show", "aor", "700")
    re_reg = (
        fresh
        and state is TelephonyRegistrationState.REGISTERED
        and contact_seen
    )
    evidence["reg_contacts_before"] = contacts_before[-800:]
    evidence["reg_contacts_after"] = contacts_after[-800:]
    evidence["reg_aor_before"] = aor_before[-1200:]
    evidence["reg_aor_after"] = aor_after[-1200:]
    checks.record("reg-reregister", "re-REGISTER 200 OK, registrar contact present",
                   re_reg,
                   f"state={state.value} fresh={fresh} contact_seen={contact_seen}")

    # -- PBX outage: call closes, registration lost, recovery --------------------
    # NOTE: every counterparty needs its own loopback SIP port (baresip
    # binds its listener at start); sharing one fails with EADDRINUSE.
    caller2 = CallerProc([codec], listen="127.0.0.1:5074")
    live_call = caller2.start()
    app_c = dial_identified(caller2, lambda: caller2.dial("C"), core) if live_call else None
    if live_call and app_c is None:
        diagnose_dial(evidence, "C-restart", adapter, core, caller2)
    # Deterministic outage test: the call must be ESTABLISHED on both
    # sides (plus media settle) before stopping the PBX; otherwise we
    # would measure setup-race teardown instead of outage survival.
    # dial() itself only returns True after the caller observes
    # establishment; the adapter side is confirmed below.
    c_up = app_c is not None and listener.wait_answered(app_c, 20)
    if c_up:
        time.sleep(1.5)
    live_call = live_call and app_c is not None and c_up
    checks.record("pbx-call-up", "outage leg established before PBX stop",
                   bool(live_call), f"app={app_c}")
    # Hard kill (SIGKILL, not graceful stop): no BYE goes out, RTP just
    # stops, so the adapter-side RTP timeout is what must close the call.
    # A graceful `docker stop` would let Asterisk BYE cleanly, which only
    # proves remote-hangup handling (already covered by s1).
    docker("kill", AST_NAME, timeout=60)
    closed = wait_for(
        lambda: app_c is not None and listener.caller_hangup.get(app_c, 0) >= 1, 30)
    checks.record("pbx-restart-call", "PBX outage closes the live call (RTP timeout)",
                   bool(live_call) and closed, f"app={app_c}")
    # Media health (story 23): the RTP-timeout teardown must surface as
    # a telephony media condition, distinct from registration loss.
    from receptionist.alerting import (
        CODE_TELEPHONY_MEDIA_LOST as _MEDIA_LOST,
    )
    from receptionist.alerting import HealthComponent as _HC

    def _media_active() -> bool:
        try:
            return any(
                c.component is _HC.TELEPHONY and c.code == _MEDIA_LOST
                for c in core.monitor.active_conditions()
            )
        except Exception:
            return False

    media_seen = wait_for(_media_active, 15)
    checks.record("media-health", "RTP-timeout teardown degrades telephony health",
                   bool(live_call) and closed and media_seen,
                   f"media_lost_active={media_seen}")
    adapter.refresh_registration(timeout=20)
    lost = wait_for(
        lambda: adapter.registration_state
        in (TelephonyRegistrationState.REGISTRATION_LOST,
            TelephonyRegistrationState.REGISTRATION_FAILED), 25)
    checks.record("reg-lost", "registration loss surfaces + health leaves READY",
                   lost and core.health.status is not HealthStatus.READY,
                   f"state={adapter.registration_state.value} health={core.health.status.value}")
    docker("start", AST_NAME, timeout=60)
    wait_asterisk_ready(60)
    recovered_state, recovered_fresh = adapter.refresh_registration(timeout=25)
    recovered = (
        recovered_fresh
        and recovered_state is TelephonyRegistrationState.REGISTERED
    )
    ok_health = wait_for(lambda: core.health.status is HealthStatus.READY, 15)
    checks.record("reg-recovery", "PBX back -> re-registered + READY", recovered and ok_health,
                   f"recovered={recovered} fresh={recovered_fresh} health={core.health.status.value}")
    try:
        caller2.stop()
    except Exception:
        pass

    # -- shutdown under load ------------------------------------------------------
    caller3 = CallerProc([codec], listen="127.0.0.1:5075")
    live_d = caller3.start()
    app_d = dial_identified(caller3, lambda: caller3.dial("D"), core) if live_d else None
    if live_d and app_d is None:
        diagnose_dial(evidence, "D-shutdown", adapter, core, caller3)
    live_d = live_d and app_d is not None and listener.wait_answered(app_d, 20)
    if live_d and app_d:
        time.sleep(1.0)
        caller3.cmd({"op": "tone", "tag": "D", "seconds": 8.0, "freq": 440.0})
    adapter.shutdown()
    adapter.shutdown()
    import threading as _th

    leftovers = [t.name for t in _th.enumerate() if t.name.startswith("baresip")]
    time.sleep(1)
    # Caller-side observation (robust): our shutdown BYE must close the
    # far end's leg. Adapter-side close events are best-effort during
    # teardown (loop stops); the peer's BYE receipt is the proof.
    peer_closed = caller3.expect(
        lambda e: e.get("event") == "closed" and e.get("tag") == "D", 10
    ) is not None
    hangup_seen = (
        peer_closed
        or listener.caller_hangup.get(app_d or "", 0) >= 1
        or listener.hangup_completed.get(app_d or "", 0) >= 1
    )
    checks.record("shutdown", "shutdown ends media call, joins threads, idempotent",
                   bool(live_d) and not leftovers and hangup_seen,
                   f"leftovers={leftovers} call_closed={hangup_seen} app={app_d}")
    try:
        caller.stop()
    except Exception:
        pass
    try:
        caller3.stop()
    except Exception:
        pass
    alive = []
    for proc in (caller, callerB, caller2, caller3):
        try:
            proc._proc.wait(timeout=20)  # noqa: SLF001
        except Exception:
            pass
        if proc._proc.poll() is None:  # noqa: SLF001
            alive.append(proc._proc.pid)  # noqa: SLF001
            try:
                proc._proc.kill()  # noqa: SLF001
            except Exception:
                pass
    checks.record("no-zombies", "counterparty processes exit cleanly", not alive,
                   f"alive_pids={alive}")

    with listener._lock:  # noqa: SLF001
        evidence["threads"] = {k: sorted(v) for k, v in listener.threads.items()}
    snapshot_adapter_events(evidence, adapter)
    if not args.keep:
        docker("rm", "-f", AST_NAME, timeout=30)
    return finish(evidence, checks, 0 if not checks.failed else 1)


def latest_app_call(listener: RecordingListener, core, exclude: set | None = None):
    """Newest live core session, preferring ones the adapter announced."""
    exclude = exclude or set()
    with listener._lock:  # noqa: SLF001
        announced = sorted(c for c in listener.answered if c not in exclude)
    for cid in reversed(announced):
        if core.get_session(cid) is not None:
            return cid
    live = sorted(cid for cid in core._sessions if cid not in exclude)  # noqa: SLF001
    return live[-1] if live else None


def dial_identified(caller, dial_fn, core, timeout: float = 25) -> str | None:
    """Dial and return exactly the new session id (snapshot diff).

    latest_app_call-style guessing misattributes legs when older
    sessions are still live; diffing the session set is exact.
    """
    before = set(core._sessions)  # noqa: SLF001
    if not dial_fn():
        return None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        after = set(core._sessions)  # noqa: SLF001
        new = sorted(after - before)
        if new:
            return new[-1]
        time.sleep(0.3)
    return None


def wait_for(pred, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.2)
    return bool(pred())


def parse_channel_codecs(channelstats: str) -> list[tuple[str, str]]:
    """Parse `pjsip show channelstats` into (channel, codec) pairs."""
    pairs: list[tuple[str, str]] = []
    for line in (channelstats or "").splitlines():
        parts = line.split()
        # Shape: <bridge> <channel like 700-0000000b> <uptime> <codec> ...
        if (
            len(parts) >= 4
            and re.fullmatch(r"[A-Za-z0-9]+-[0-9a-fA-F]+", parts[1] or "")
            and parts[3].lower() in ("ulaw", "alaw", "g722", "opus")
        ):
            pairs.append((parts[1], parts[3].lower()))
    return pairs


def diagnose_dial(evidence: dict, name: str, adapter, core, caller) -> None:
    """Failure diagnostics for a dial that produced no session: adapter
    load, registrar contacts, PBX channels, and counterparty stderr.
    A bare `app=None` with no cause is not an acceptable artifact."""
    detail: dict = {}
    try:
        with adapter._lock:  # noqa: SLF001
            detail["adapter_calls"] = sorted(adapter._calls)  # noqa: SLF001
            detail["adapter_pending"] = len(adapter._pending)  # noqa: SLF001
            detail["reg_state"] = adapter.registration_state.value
    except Exception as error:
        detail["adapter_error"] = type(error).__name__
    try:
        detail["core_sessions"] = sorted(core._sessions)  # noqa: SLF001
    except Exception as error:
        detail["core_error"] = type(error).__name__
    try:
        detail["pjsip_channels"] = asterisk_cli("pjsip", "show", "channels")[-800:]
        detail["contacts"] = asterisk_cli("pjsip", "show", "contacts")[-800:]
    except Exception as error:
        detail["asterisk_error"] = type(error).__name__
    try:
        detail["caller_stderr"] = caller.stderr_tail()
    except Exception:
        pass
    evidence.setdefault("dial_diagnostics", {})[name] = detail


def collapse_timeline(lines: list[str]) -> list[str]:
    seen: list[str] = []
    for entry in lines:
        if not seen or seen[-1] != entry:
            seen.append(entry)
    return seen[-80:]


def timeline_of(sipp_output: str) -> list[str]:
    timeline: list[str] = []
    for line in (sipp_output or "").splitlines():
        match = re.search(r"(INVITE|ACK|BYE|CANCEL|REFER|NOTIFY|INFO|OPTIONS|REGISTER|SIP/2\.0 \d{3})", line)
        if match and "Authorization" not in line:
            timeline.append(line.strip()[:160])
    seen: list[str] = []
    for entry in timeline:
        if not seen or seen[-1] != entry:
            seen.append(entry)
    return seen[-40:]


# Evidence outputs are run products, not inputs: they never invalidate
# a run. Everything else must be committed (see --allow-dirty).
_PROVENANCE_IGNORED_PREFIXES = ("docs/evidence/",)


def code_provenance() -> tuple[str, list[str]]:
    """(HEAD sha, dirty input paths) for auditable matrix evidence.

    `tested_code_sha` names the exact code tree under test. A run made
    after editing code/scenario/harness but before committing would be
    stamped with the parent SHA while testing something else entirely;
    the dirty list exposes exactly that situation.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=30, cwd=REPO,
        )
        sha = (proc.stdout or "").strip() or "unknown"
    except Exception:
        sha = "unknown"
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            timeout=30, cwd=REPO,
        )
        dirty = []
        for line in (proc.stdout or "").splitlines():
            path = line[3:].strip().split(" -> ")[-1]
            if path and not path.startswith(_PROVENANCE_IGNORED_PREFIXES):
                dirty.append(line.strip())
    except Exception:
        dirty = []
    return sha, dirty


def snapshot_adapter_events(evidence: dict, adapter) -> None:
    try:
        evidence["adapter_events"] = [
            {"t": round(t, 2), "call": c, "kind": k}
            for (t, c, k) in adapter.event_timeline()
        ]
        with adapter._lock:  # noqa: SLF001
            evidence["adapter_calls_left"] = sorted(adapter._calls.keys())  # noqa: SLF001
    except Exception:
        pass


def finish(evidence: dict, checks: Check, code: int) -> int:
    if not evidence.get("keep"):
        try:
            docker("rm", "-f", AST_NAME, timeout=30)
        except Exception:
            pass
    evidence["checks"] = checks.items
    evidence["meta"]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    evidence["summary"] = {
        "total": len(checks.items),
        "failed": len(checks.failed),
    }
    os.makedirs(os.path.dirname(EVIDENCE), exist_ok=True)
    with open(EVIDENCE, "w") as handle:
        json.dump(evidence, handle, indent=2, default=str)
    print(f"evidence: {EVIDENCE} failed={len(checks.failed)}")
    for item in checks.failed:
        print(f"FAILED {item['id']}: {item['detail']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
