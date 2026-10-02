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


class HealthComponent(Enum):
    """What is affected, as a stable identity (never parsed from text)."""

    CONFIGURATION = "configuration"
    PROVIDER = "provider"
    RUNTIME = "runtime"
    TRANSCRIPT = "transcript"
    CAPACITY = "capacity"


class TransitionKind(Enum):
    UNHEALTHY = "unhealthy"
    RECOVERED = "recovered"


@dataclass(frozen=True)
class HealthTransition:
    """One edge-triggered health change. Sanitized at creation: short
    stable codes plus a safe detail string. Never carries Caller-ID,
    transcripts, message text, knowledge, prompts, model output, audio,
    credentials, tokens, headers, secrets, or raw exception text."""

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
    """Local-only record of a failed sink delivery. Carries the sink name
    and the exception type name only: never exception strings (they may
    embed secrets), never payloads. Never fanned out, so delivery
    failures cannot recurse into more external alerts."""

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


@dataclass
class HealthMonitor:
    """Edge-triggered transition detection plus isolated fan-out.

    The single place that turns subsystem reports into transitions:
    first report of a (component, code) emits UNHEALTHY, repeats are
    silent, recovery emits RECOVERED once (externally only when
    notify_recovery is set; locally always). Every emitted transition
    is recorded locally before any sink runs, so the structured record
    exists even when all sinks fail. Sink exceptions are contained per
    sink and diagnosed locally without recursion.
    """

    clock: Clock
    sinks: list[AlertSink] = field(default_factory=list)
    notify_recovery: bool = True

    def __post_init__(self) -> None:
        self._active: dict[tuple[HealthComponent, str], ActiveCondition] = {}
        self._history: list[HealthTransition] = []
        self._diagnostics: list[DeliveryDiagnostic] = []

    def active_conditions(self) -> tuple[ActiveCondition, ...]:
        """Currently-unhealthy conditions, with stable identities."""
        return tuple(self._active.values())

    def history(self) -> tuple[HealthTransition, ...]:
        """Every locally recorded transition, oldest first."""
        return tuple(self._history)

    def delivery_diagnostics(self) -> tuple[DeliveryDiagnostic, ...]:
        """Local-only delivery failure records. Never fanned out."""
        return tuple(self._diagnostics)

    def report_unhealthy(
        self, component: HealthComponent, code: str, detail: str = ""
    ) -> HealthTransition | None:
        """Report a subsystem problem. Returns the emitted transition, or
        None when this (component, code) is already active (no re-alert,
        however the detail text differs)."""
        key = (component, code)
        if key in self._active:
            return None
        condition = ActiveCondition(
            component=component,
            code=code,
            detail=detail,
            first_seen=self.clock.now(),
        )
        self._active[key] = condition
        return self._emit(
            HealthTransition(
                timestamp=condition.first_seen,
                component=component,
                code=code,
                kind=TransitionKind.UNHEALTHY,
                detail=detail,
            )
        )

    def report_recovered(
        self, component: HealthComponent, code: str
    ) -> HealthTransition | None:
        """Clear one active condition. Returns the recovery transition, or
        None when nothing was active (no recovery is fabricated)."""
        key = (component, code)
        if key not in self._active:
            return None
        del self._active[key]
        transition = HealthTransition(
            timestamp=self.clock.now(),
            component=component,
            code=code,
            kind=TransitionKind.RECOVERED,
        )
        self._history.append(transition)
        if self.notify_recovery:
            self._fan_out(transition)
        return transition

    def _emit(self, transition: HealthTransition) -> HealthTransition:
        self._history.append(transition)
        self._fan_out(transition)
        return transition

    def _fan_out(self, transition: HealthTransition) -> None:
        for sink in self.sinks:
            try:
                sink.send(transition)
            except Exception as error:
                self._diagnostics.append(
                    DeliveryDiagnostic(
                        timestamp=self.clock.now(),
                        sink_name=sink.name,
                        error_kind=type(error).__name__,
                    )
                )


class AlertError(Exception):
    """Invalid alerting configuration. Raised at assembly, never during
    call handling or delivery (delivery failures are diagnosed locally)."""


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
    transport use only; they are never rendered into any alert body."""

    enabled: bool = False
    host: str = ""
    port: int = 587
    use_tls: bool = True
    username: str = ""
    password: str = ""
    sender: str = ""
    recipients: tuple[str, ...] = ()
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class WebhookSettings:
    """Independent generic-webhook configuration. The URL comes only from
    trusted configuration; events and callers can never steer it."""

    enabled: bool = False
    url: str = ""
    auth_scheme: str = "Bearer"
    auth_token: str = ""
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class TelegramSettings:
    """Independent Telegram Bot API configuration. The token addresses
    the transport only and is never rendered anywhere."""

    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class AlertSettings:
    """All alerting configuration plus recovery policy. Recovery
    transitions are always recorded locally; they reach external sinks
    only when notify_recovery is set."""

    email: EmailSettings = field(default_factory=EmailSettings)
    webhook: WebhookSettings = field(default_factory=WebhookSettings)
    telegram: TelegramSettings = field(default_factory=TelegramSettings)
    notify_recovery: bool = True


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


class UrllibTransport:
    """Real outbound HTTPS posts over the standard library. Bounded by
    timeout; failures raise for local diagnosis by type name only."""

    def post(
        self, url: str, payload: dict, headers: dict, timeout_seconds: float
    ) -> None:
        import json
        import urllib.request

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds):
            pass


class AlertConfigRepository(Protocol):
    """Configuration-side persistence boundary for alert settings.
    Lives in config.db, behind ConfigService, like every config domain."""

    def load(self) -> AlertSettings: ...


def build_alert_sinks(    settings: AlertSettings,
    smtp_transport: SmtpTransport | None = None,
    http_transport: HttpTransport | None = None,
) -> list[AlertSink]:
    """Deterministic factory: only enabled, complete channels become
    sinks. Enabled-but-incomplete channels fail closed with AlertError
    instead of silently alerting nowhere. Transports default to the
    stdlib implementations; tests inject fakes (no real network)."""
    sinks: list[AlertSink] = []
    email = settings.email
    if email.enabled:
        if not email.host or not email.sender or not email.recipients:
            raise AlertError("email channel enabled without host/sender/recipients")
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
        if not webhook.url:
            raise AlertError("webhook channel enabled without url")
        sinks.append(
            WebhookSink(
                transport=http_transport or UrllibTransport(), settings=webhook
            )
        )
    telegram = settings.telegram
    if telegram.enabled:
        if not telegram.bot_token or not telegram.chat_id:
            raise AlertError("telegram channel enabled without bot token/chat id")
        sinks.append(
            TelegramSink(
                transport=http_transport or UrllibTransport(), settings=telegram
            )
        )
    return sinks
