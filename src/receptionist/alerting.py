"""Health-transition alerting (#23): actionable, spam-free operations signals.

Layering, from the core outward:

1. Subsystems change state (breaker, config authority, storage, ...).
2. The core reports typed conditions to HealthMonitor (never the reverse).
3. The monitor edge-triggers HealthTransitions per (component, code),
   records every one locally, and fans out to AlertSinks.
4. Sinks are output-only: they render sanitized transitions through
   injected transports and can never act on the call path.

Alert delivery never decides health: a failing sink produces a local
delivery diagnostic, never a second external alert, so failures cannot
recurse. No wall clock (timestamps come from the injected Clock), no
threads, no queues, no network except inside the stdlib transports.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from receptionist.boundaries import Clock


SERVICE_NAME = "virtual-receptionist"
TRANSITION_SCHEMA_VERSION = 1

# Stable reason codes. Identity is (component, code); human detail text
# never dedups, so rewordings cannot spam.
CODE_CIRCUIT_OPEN = "provider.circuit_open"
CODE_CONFIG_UNAVAILABLE = "config.unavailable"
CODE_CONFIG_INCOMPLETE = "config.missing_required"
CODE_HISTORY_UNAVAILABLE = "runtime.history_unavailable"
CODE_AUDIT_UNAVAILABLE = "runtime.audit_unavailable"
CODE_TRANSCRIPT_UNAVAILABLE = "transcript.unavailable"
CODE_CAPACITY_SATURATED = "capacity.saturated"
CODE_VOICE_BACKEND_NOT_READY = "voice_backend.not_ready"
CODE_TELEPHONY_NOT_REGISTERED = "telephony.not_registered"
CODE_TELEPHONY_LOST = "telephony.registration_lost"
CODE_TELEPHONY_MEDIA_LOST = "telephony.media_lost"

#: Project-owned safe phrases, one per code. Transition detail comes
#: only from this registry, never from caller text, exception strings,
#: paths, URLs, or credentials: unsanitized detail is unrepresentable.
DETAIL_BY_CODE = {
    CODE_CIRCUIT_OPEN: "failure threshold reached",
    CODE_CONFIG_UNAVAILABLE: "configuration authority unreadable",
    CODE_CONFIG_INCOMPLETE: "required configuration missing",
    CODE_HISTORY_UNAVAILABLE: "call history write failed",
    CODE_AUDIT_UNAVAILABLE: "audit write failed",
    CODE_TRANSCRIPT_UNAVAILABLE: "transcript sidecar write failed",
    CODE_CAPACITY_SATURATED: "ai capacity saturated",
    CODE_VOICE_BACKEND_NOT_READY: "voice backend not ready",
    CODE_TELEPHONY_NOT_REGISTERED: "telephony endpoint not registered",
    CODE_TELEPHONY_LOST: "telephony registration lost",
    CODE_TELEPHONY_MEDIA_LOST: "telephony media path lost",
}


class HealthComponent(Enum):
    """What is affected, as a stable identity (never parsed from text)."""

    CONFIGURATION = "configuration"
    PROVIDER = "provider"
    RUNTIME = "runtime"
    TRANSCRIPT = "transcript"
    CAPACITY = "capacity"
    TELEPHONY = "telephony"


class TransitionKind(Enum):
    UNHEALTHY = "unhealthy"
    RECOVERED = "recovered"


@dataclass(frozen=True)
class HealthTransition:
    """One edge-triggered health change. Sanitized by construction: the
    stable (component, code) identity plus a project-owned registry
    phrase. Never carries Caller-ID, transcripts, message text,
    knowledge, prompts, model output, audio, credentials, tokens,
    headers, secrets, or raw exception text — free text cannot enter."""

    timestamp: float
    component: HealthComponent
    code: str
    kind: TransitionKind
    detail: str = ""
    service: str = SERVICE_NAME
    schema_version: int = TRANSITION_SCHEMA_VERSION


@dataclass(frozen=True)
class ActiveCondition:
    """One currently-unhealthy (component, code) with first-seen time."""

    component: HealthComponent
    code: str
    detail: str
    first_seen: float


@dataclass(frozen=True)
class DeliveryDiagnostic:
    """Local-only record of a failed sink delivery. Sinks are identified
    by position (\"sink-0\", ...) so a hostile sink name can never inject
    secrets into diagnostics. The error is the exception type name only:
    never exception strings (they may embed secrets), never payloads.
    Never fanned out, so delivery failures cannot recurse into more
    external alerts."""

    timestamp: float
    sink_name: str
    error_kind: str


class AlertSink(Protocol):
    """Output-only health consumer. Receives sanitized transitions; must
    not touch sessions, policy, config, transcripts, or health. May
    raise on delivery failure: the monitor isolates every sink."""

    @property
    def name(self) -> str: ...

    def send(self, transition: HealthTransition) -> None: ...


#: Delivery queue policy: the monitor FIFO is intentionally unbounded.
#: Edge-triggering means it never holds one entry per failed call, so no
#: emitted transition is ever dropped and the external stream always
#: preserves the full unhealthy/recovery sequence per condition.

_LOGGER = logging.getLogger("virtual-receptionist.health")


def _log_transition(transition: HealthTransition) -> None:
    _LOGGER.info(
        "health_transition %s",
        json.dumps(
            {
                "service": transition.service,
                "schema_version": transition.schema_version,
                "timestamp": transition.timestamp,
                "component": transition.component.value,
                "code": transition.code,
                "status": transition.kind.value,
                "detail": transition.detail,
            }
        ),
    )


def _log_diagnostic(diagnostic: DeliveryDiagnostic) -> None:
    _LOGGER.warning(
        "alert_delivery_failed %s",
        json.dumps(
            {
                "service": SERVICE_NAME,
                "timestamp": diagnostic.timestamp,
                "sink": diagnostic.sink_name,
                "error": diagnostic.error_kind,
            }
        ),
    )


@dataclass
class HealthMonitor:
    """Edge-triggered transition detection plus decoupled fan-out.

    The single place that turns subsystem reports into transitions:
    first report of a (component, code) emits UNHEALTHY, repeats are
    silent, recovery emits RECOVERED once (externally only when
    notify_recovery is set; locally always). Every emitted transition
    is recorded locally and emitted as a structured log line before any
    sink runs, so the local record exists even when all sinks fail.

    Thread-safe: one small lock guards condition state, history appends,
    and worker lifecycle, and is never held during sink.send(). External
    delivery never blocks reporters: emitted transitions wait on one
    unbounded FIFO queue drained by a single worker, so the external
    order always matches history() order while slow endpoints cannot
    stall call admission. The queue is unbounded by design: the monitor
    is edge-triggered, so it never holds one entry per failed call, and
    no emitted transition is ever dropped. Use drain() in tests and
    close() at shutdown for explicit lifecycle. Sink exceptions (and
    hostile sink objects) are contained per sink and diagnosed locally
    without recursion.
    """

    clock: Clock
    sinks: list[AlertSink] = field(default_factory=list)
    notify_recovery: bool = True

    def __post_init__(self) -> None:
        self.sinks = [sink for sink in self.sinks if sink is not None]
        self._lock = threading.Lock()
        self._active: dict[tuple[HealthComponent, str], ActiveCondition] = {}
        self._history: list[HealthTransition] = []
        self._diagnostics: list[DeliveryDiagnostic] = []
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._started_workers = 0
        self._closed = False

    @property
    def started_workers(self) -> int:
        """How many delivery workers were ever started. Lifecycle
        observability: concurrent reporters must never start more than
        one effective worker."""
        with self._lock:
            return self._started_workers

    def active_conditions(self) -> tuple[ActiveCondition, ...]:
        """Currently-unhealthy conditions, with stable identities."""
        with self._lock:
            return tuple(self._active.values())

    def history(self) -> tuple[HealthTransition, ...]:
        """Every locally recorded transition, oldest first."""
        with self._lock:
            return tuple(self._history)

    def delivery_diagnostics(self) -> tuple[DeliveryDiagnostic, ...]:
        """Local-only delivery failure records. Never fanned out."""
        with self._lock:
            return tuple(self._diagnostics)

    def drain(self, timeout: float = 5.0) -> None:
        """Wait (bounded) until every accepted transition is delivered.
        Deterministic tests call this before asserting what sinks
        received. Delivery order always matches history() order."""
        with self._lock:
            worker = self._worker
        if worker is None:
            return
        done = threading.Event()
        deadline = time.monotonic() + timeout
        self._queue.put(("flush", done))
        remaining = deadline - time.monotonic()
        if remaining > 0:
            done.wait(timeout=remaining)

    def close(self) -> None:
        """Shutdown: no further external deliveries; drain what is queued,
        bounded. Single worker at most; joining it drains everything
        ahead of the stop marker."""
        with self._lock:
            self._closed = True
            worker, self._worker = self._worker, None
        if worker is not None:
            self._queue.put(("stop", None))
            worker.join(timeout=5.0)

    def _now(self) -> float:
        try:
            return self.clock.now()
        except Exception:
            return 0.0

    def report_unhealthy(
        self, component: HealthComponent, code: str
    ) -> HealthTransition | None:
        """Report a subsystem problem. Returns the emitted transition, or
        None when this (component, code) is already active (no re-alert).
        Detail comes from the project-owned registry, never arguments.
        Atomic: concurrent reporters of one condition emit exactly once."""
        with self._lock:
            key = (component, code)
            if key in self._active:
                return None
            now = self._now()
            self._active[key] = ActiveCondition(
                component=component,
                code=code,
                detail=DETAIL_BY_CODE.get(code, ""),
                first_seen=now,
            )
            transition = HealthTransition(
                timestamp=now,
                component=component,
                code=code,
                kind=TransitionKind.UNHEALTHY,
                detail=DETAIL_BY_CODE.get(code, ""),
            )
            self._history.append(transition)
            _log_transition(transition)
            self._enqueue_locked(transition)
            return transition

    def report_recovered(
        self, component: HealthComponent, code: str
    ) -> HealthTransition | None:
        """Clear one active condition. Returns the recovery transition, or
        None when nothing was active (no recovery is fabricated). Atomic:
        concurrent recoveries of one condition emit at most once."""
        with self._lock:
            key = (component, code)
            if key not in self._active:
                return None
            del self._active[key]
            transition = HealthTransition(
                timestamp=self._now(),
                component=component,
                code=code,
                kind=TransitionKind.RECOVERED,
            )
            self._history.append(transition)
            _log_transition(transition)
            if self.notify_recovery:
                self._enqueue_locked(transition)
            return transition

    def _enqueue_locked(self, transition: HealthTransition) -> None:
        """Queue one accepted transition. Runs under the state lock so
        external order always matches history order. Never blocks: the
        queue is unbounded and no emitted transition is ever dropped."""
        if self._closed or not self.sinks:
            return
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._drain_queue, daemon=True, name="health-alerts"
            )
            self._started_workers += 1
            self._worker.start()
        self._queue.put_nowait(("transition", transition))

    def _drain_queue(self) -> None:
        while True:
            kind, payload = self._queue.get()
            try:
                if kind == "stop":
                    return
                if kind == "flush":
                    payload.set()
                elif kind == "transition":
                    self._deliver(payload)
            finally:
                self._queue.task_done()

    def _deliver(self, transition: HealthTransition) -> None:
        for index, sink in enumerate(self.sinks):
            try:
                sink.send(transition)
            except Exception as error:
                # The whole per-sink body is guarded: send, hostile
                # objects, and anything else ends here as one safe record.
                diagnostic = DeliveryDiagnostic(
                    timestamp=self._now(),
                    sink_name=f"sink-{index}",
                    error_kind=type(error).__name__,
                )
                self._diagnostics.append(diagnostic)
                _log_diagnostic(diagnostic)


class AlertError(Exception):
    """Invalid alerting configuration. Raised at assembly, never during
    call handling or delivery (delivery failures are diagnosed locally)."""


class TransportError(Exception):
    """Normalized transport failure. Carries a status summary only, never
    URLs, headers, payloads, or raw library error text."""


class SmtpTransport(Protocol):
    """Boundary for sending email. Tests inject a fake; production uses
    SmtplibTransport. Never receives secrets outside its arguments."""

    def send_email(
        self,
        sender: str,
        recipients: tuple[str, ...],
        subject: str,
        body: str,
        timeout_seconds: float,
    ) -> None: ...


class HttpTransport(Protocol):
    """Boundary for outbound HTTPS posts (webhook, Telegram). Tests
    inject a fake; production uses UrllibTransport."""

    def post(
        self, url: str, payload: dict, headers: dict, timeout_seconds: float
    ) -> None: ...


@dataclass(frozen=True)
class EmailSettings:
    """Independent email channel configuration. Credentials live here for
    transport use only; they are never rendered into any alert body, and
    repr/str redact them so config dumps cannot leak."""

    enabled: bool = False
    host: str = ""
    port: int = 587
    use_tls: bool = True
    username: str = ""
    password: str = field(default="", repr=False)
    sender: str = ""
    recipients: tuple[str, ...] = ()
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class WebhookSettings:
    """Independent generic-webhook configuration. The URL comes only from
    trusted configuration; events and callers can never steer it. The
    token is transport-only and redacted from repr/str."""

    enabled: bool = False
    url: str = ""
    auth_scheme: str = "Bearer"
    auth_token: str = field(default="", repr=False)
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class TelegramSettings:
    """Independent Telegram Bot API configuration. The token addresses
    the transport only, is never rendered anywhere, and is redacted from
    repr/str."""

    enabled: bool = False
    bot_token: str = field(default="", repr=False)
    chat_id: str = ""
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class AlertSettings:
    """All alerting configuration plus recovery policy. Recovery
    transitions are always recorded locally; they reach external sinks
    only when notify_recovery is set. Repr reports channels and flags
    only, never nested secrets."""

    email: EmailSettings = field(default_factory=EmailSettings)
    webhook: WebhookSettings = field(default_factory=WebhookSettings)
    telegram: TelegramSettings = field(default_factory=TelegramSettings)
    notify_recovery: bool = True

    def __repr__(self) -> str:
        channels = (
            f"email(enabled={self.email.enabled})",
            f"webhook(enabled={self.webhook.enabled})",
            f"telegram(enabled={self.telegram.enabled})",
        )
        return (
            f"AlertSettings({', '.join(channels)}, "
            f"notify_recovery={self.notify_recovery})"
        )


def _render_subject(transition: HealthTransition) -> str:
    return (
        f"[HEALTH] {transition.kind.value} "
        f"{transition.component.value}: {transition.code}"
    )


def _render_body(transition: HealthTransition) -> str:
    lines = [
        f"service: {transition.service}",
        f"component: {transition.component.value}",
        f"code: {transition.code}",
        f"status: {transition.kind.value}",
        f"timestamp: {transition.timestamp}",
    ]
    if transition.detail:
        lines.append(f"detail: {transition.detail}")
    return "\n".join(lines)


def _payload(transition: HealthTransition) -> dict:
    """Stable webhook schema: sanitized transition fields only."""
    return {
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "service": transition.service,
        "timestamp": transition.timestamp,
        "component": transition.component.value,
        "code": transition.code,
        "status": transition.kind.value,
        "detail": transition.detail,
    }


class EmailSink:
    """Renders sanitized transitions to email through an SMTP transport."""

    def __init__(self, transport: SmtpTransport, settings: EmailSettings) -> None:
        self._transport = transport
        self._settings = settings

    @property
    def name(self) -> str:
        return "email"

    def send(self, transition: HealthTransition) -> None:
        self._transport.send_email(
            self._settings.sender,
            self._settings.recipients,
            _render_subject(transition),
            _render_body(transition),
            self._settings.timeout_seconds,
        )


class WebhookSink:
    """Posts the stable sanitized schema to the configured trusted URL."""

    def __init__(self, transport: HttpTransport, settings: WebhookSettings) -> None:
        self._transport = transport
        self._settings = settings

    @property
    def name(self) -> str:
        return "webhook"

    def send(self, transition: HealthTransition) -> None:
        headers: dict = {}
        if self._settings.auth_token:
            headers["Authorization"] = f"{self._settings.auth_scheme} {self._settings.auth_token}"
        self._transport.post(
            self._settings.url,
            _payload(transition),
            headers,
            self._settings.timeout_seconds,
        )


class TelegramSink:
    """Sends a short operational message through the Bot API transport."""

    def __init__(self, transport: HttpTransport, settings: TelegramSettings) -> None:
        self._transport = transport
        self._settings = settings

    @property
    def name(self) -> str:
        return "telegram"

    def send(self, transition: HealthTransition) -> None:
        text = (
            f"[{transition.kind.value}] {transition.component.value}: "
            f"{transition.code}"
        )
        if transition.detail:
            text += f" ({transition.detail})"
        self._transport.post(
            f"https://api.telegram.org/bot{self._settings.bot_token}/sendMessage",
            {"chat_id": self._settings.chat_id, "text": text[:400]},
            {},
            self._settings.timeout_seconds,
        )


class SmtplibTransport:
    """Real SMTP delivery over the standard library. Bounded by timeout;
    every failure raises for the monitor to diagnose locally (by type
    name only, so credentials in error text can never leak)."""

    def __init__(self, host: str, port: int, use_tls: bool, username: str, password: str) -> None:
        self._host = host
        self._port = port
        self._use_tls = use_tls
        self._username = username
        self._password = password

    def send_email(
        self,
        sender: str,
        recipients: tuple[str, ...],
        subject: str,
        body: str,
        timeout_seconds: float,
    ) -> None:
        import smtplib

        message = (
            f"From: {sender}\nTo: {', '.join(recipients)}\n"
            f"Subject: {subject}\n\n{body}"
        )
        with smtplib.SMTP(host=self._host, port=self._port, timeout=timeout_seconds) as client:
            if self._use_tls:
                client.starttls()
            if self._username:
                client.login(self._username, self._password)
            client.sendmail(sender, list(recipients), message)


class _NoFollowRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect before headers or payload move. Following a
    301/302/303 would resend the Authorization header to a host the
    operator never configured; a redirecting endpoint surfaces as a
    delivery failure instead."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise TransportError(f"redirect refused (http {code})")


class UrllibTransport:
    """Real outbound HTTPS posts over the standard library. Bounded by
    timeout; failures raise for local diagnosis by type name only.

    Redirects are never followed: the configured trusted URL is the only
    destination, so a redirect response can never carry the Authorization
    header (or the alert body) to an untrusted host. A redirecting
    endpoint surfaces as a delivery error instead.
    """

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPHandler,
            urllib.request.HTTPSHandler,
            _NoFollowRedirectHandler(),
        )

    def post(
        self, url: str, payload: dict, headers: dict, timeout_seconds: float
    ) -> None:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            self._opener.open(request, timeout=timeout_seconds)
        except urllib.error.HTTPError as error:
            raise TransportError(f"http status {error.code}") from error


class AlertConfigRepository(Protocol):
    """Configuration-side persistence boundary for alert settings.
    Lives in config.db, behind ConfigService, like every config domain."""

    def load(self) -> AlertSettings: ...

def _require_timeout(timeout_seconds: float, channel: str) -> None:
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise AlertError(f"{channel} channel needs a positive timeout")


def build_alert_sinks(
    settings: AlertSettings,
    smtp_transport: SmtpTransport | None = None,
    http_transport: HttpTransport | None = None,
) -> list[AlertSink]:
    """Deterministic factory: only enabled, complete channels become
    sinks. Enabled-but-incomplete (or out-of-range) channels fail closed
    with AlertError instead of silently alerting nowhere. Transports
    default to the stdlib implementations; tests inject fakes (no real
    network)."""
    sinks: list[AlertSink] = []
    email = settings.email
    if email.enabled:
        if not email.host.strip() or not email.sender.strip() or not [
            r for r in email.recipients if r.strip()
        ]:
            raise AlertError("email channel enabled without host/sender/recipients")
        if not 1 <= email.port <= 65535:
            raise AlertError("email channel needs a port in 1..65535")
        _require_timeout(email.timeout_seconds, "email")
        sinks.append(
            EmailSink(
                transport=smtp_transport
                or SmtplibTransport(
                    email.host, email.port, email.use_tls, email.username, email.password
                ),
                settings=email,
            )
        )
    webhook = settings.webhook
    if webhook.enabled:
        if not webhook.url.strip() or not (
            webhook.url.startswith("https://") or webhook.url.startswith("http://")
        ):
            raise AlertError("webhook channel enabled without an http(s) url")
        _require_timeout(webhook.timeout_seconds, "webhook")
        sinks.append(
            WebhookSink(
                transport=http_transport or UrllibTransport(), settings=webhook
            )
        )
    telegram = settings.telegram
    if telegram.enabled:
        if not telegram.bot_token or not telegram.chat_id.strip():
            raise AlertError("telegram channel enabled without bot token/chat id")
        _require_timeout(telegram.timeout_seconds, "telegram")
        sinks.append(
            TelegramSink(
                transport=http_transport or UrllibTransport(), settings=telegram
            )
        )
    return sinks


def build_monitor(
    settings: AlertSettings,
    clock: Clock,
    smtp_transport: SmtpTransport | None = None,
    http_transport: HttpTransport | None = None,
) -> HealthMonitor:
    """Productive composition seam: settings (canonically from
    config.db via ConfigService.alert_settings()) become live sinks on
    one monitor, honoring the persisted notify_recovery flag. This is the
    chain production assembly calls; no packaging, UI, or threads beyond
    the monitor's own bounded delivery."""
    return HealthMonitor(
        clock=clock,
        sinks=build_alert_sinks(
            settings, smtp_transport=smtp_transport, http_transport=http_transport
        ),
        notify_recovery=settings.notify_recovery,
    )
