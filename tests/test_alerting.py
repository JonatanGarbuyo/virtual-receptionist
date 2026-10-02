"""Health-transition alerting (#23, spec #17).

Edge-triggered, deduped health transitions with sanitized payloads,
independent fan-out to output-only sinks, and a structured local record
that always exists. FakeClock everywhere; no network, no wall clock.
"""

import unittest

from receptionist.alerting import (
    EmailSettings,
    EmailSink,
    HealthComponent,
    HealthMonitor,
    HealthTransition,
    SmtplibTransport,
    TelegramSettings,
    TelegramSink,
    TransitionKind,
    UrllibTransport,
    WebhookSettings,
    WebhookSink,
    build_alert_sinks,
)

from fakes import FakeClock


class RecordingSink:
    """Output-only test sink. Records sanitized transitions, never acts."""

    def __init__(self, name: str = "recorder") -> None:
        self.name = name
        self.received: list[HealthTransition] = []

    def send(self, transition: HealthTransition) -> None:
        self.received.append(transition)


class RaisingSink:
    """Sink whose delivery always fails, without leaking anything."""

    def __init__(self, name: str = "raising") -> None:
        self.name = name
        self.attempts = 0

    def send(self, transition: HealthTransition) -> None:
        self.attempts += 1
        raise RuntimeError("transport down")


def make_monitor(*sinks, **kwargs) -> HealthMonitor:
    return HealthMonitor(clock=FakeClock(), sinks=list(sinks), **kwargs)


class EdgeTriggeringTest(unittest.TestCase):
    def test_healthy_to_unhealthy_emits_once(self) -> None:
        sink = RecordingSink()
        monitor = make_monitor(sink)
        first = monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open", "threshold reached"
        )
        self.assertIsNotNone(first)
        self.assertEqual(first.kind, TransitionKind.UNHEALTHY)
        self.assertEqual(len(sink.received), 1)

    def test_repeated_same_condition_never_reemits(self) -> None:
        sink = RecordingSink()
        monitor = make_monitor(sink)
        monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open", "threshold reached"
        )
        for detail in ("still open", "open with other words", "OPEN!!!", ""):
            repeated = monitor.report_unhealthy(
                HealthComponent.PROVIDER, "provider.circuit_open", detail
            )
            self.assertIsNone(repeated)
        # Detail text is never the identity: still exactly one alert.
        self.assertEqual(len(sink.received), 1)
        self.assertEqual(len(monitor.history()), 1)

    def test_recovery_emits_once_then_silence(self) -> None:
        sink = RecordingSink()
        monitor = make_monitor(sink)
        monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open", "open"
        )
        recovered = monitor.report_recovered(
            HealthComponent.PROVIDER, "provider.circuit_open"
        )
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.kind, TransitionKind.RECOVERED)
        self.assertEqual(len(sink.received), 2)
        self.assertIsNone(
            monitor.report_recovered(HealthComponent.PROVIDER, "provider.circuit_open")
        )
        self.assertEqual(len(sink.received), 2)

    def test_recovery_without_prior_unhealthy_emits_nothing(self) -> None:
        sink = RecordingSink()
        monitor = make_monitor(sink)
        self.assertIsNone(
            monitor.report_recovered(HealthComponent.PROVIDER, "provider.circuit_open")
        )
        self.assertEqual(sink.received, [])
        self.assertEqual(monitor.history(), ())

    def test_recovery_disabled_stays_local_only(self) -> None:
        sink = RecordingSink()
        monitor = make_monitor(sink, notify_recovery=False)
        monitor.report_unhealthy(
            HealthComponent.RUNTIME, "runtime.history_unavailable", "disk full"
        )
        self.assertEqual(len(sink.received), 1)
        recovered = monitor.report_recovered(
            HealthComponent.RUNTIME, "runtime.history_unavailable"
        )
        self.assertIsNotNone(recovered)
        # Local record exists; the external sink is not notified.
        self.assertEqual(len(sink.received), 1)
        self.assertEqual(len(monitor.history()), 2)

    def test_conditions_are_tracked_per_component_and_code(self) -> None:
        monitor = make_monitor()
        monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open", "open"
        )
        monitor.report_unhealthy(
            HealthComponent.RUNTIME, "runtime.history_unavailable", "down"
        )
        active = {(c.component, c.code) for c in monitor.active_conditions()}
        self.assertEqual(
            active,
            {
                (HealthComponent.PROVIDER, "provider.circuit_open"),
                (HealthComponent.RUNTIME, "runtime.history_unavailable"),
            },
        )
        monitor.report_recovered(HealthComponent.PROVIDER, "provider.circuit_open")
        active = {(c.component, c.code) for c in monitor.active_conditions()}
        self.assertEqual(
            active, {(HealthComponent.RUNTIME, "runtime.history_unavailable")}
        )

    def test_transition_carries_stable_identity_and_timestamp(self) -> None:
        clock = FakeClock(start=1234.5)
        monitor = HealthMonitor(clock=clock, sinks=[])
        transition = monitor.report_unhealthy(
            HealthComponent.CONFIGURATION, "config.unavailable", "db locked"
        )
        assert transition is not None
        self.assertEqual(transition.timestamp, 1234.5)
        self.assertEqual(transition.component, HealthComponent.CONFIGURATION)
        self.assertEqual(transition.code, "config.unavailable")
        self.assertEqual(transition.detail, "db locked")
        self.assertTrue(transition.service)
        self.assertGreaterEqual(transition.schema_version, 1)


class FanOutIsolationTest(unittest.TestCase):
    def test_one_failing_sink_does_not_stop_the_others(self) -> None:
        first = RecordingSink("a")
        failing = RaisingSink("b")
        third = RecordingSink("c")
        monitor = make_monitor(first, failing, third)
        monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open", "open"
        )
        self.assertEqual(len(first.received), 1)
        self.assertEqual(failing.attempts, 1)
        self.assertEqual(len(third.received), 1)

    def test_delivery_failure_is_diagnosed_locally_without_recursion(self) -> None:
        first = RecordingSink("a")
        failing = RaisingSink("b")
        monitor = make_monitor(first, failing)
        monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open", "open"
        )
        # The failure is recorded locally exactly once...
        diagnostics = monitor.delivery_diagnostics()
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].sink_name, "b")
        # ...and never fanned out as a second external alert.
        self.assertEqual(len(first.received), 1)
        # No recursion: a failing sink never triggers further delivery.
        self.assertEqual(failing.attempts, 1)

    def test_diagnostic_never_carries_exception_strings(self) -> None:
        class LeakySink:
            name = "leaky"

            def send(self, transition: HealthTransition) -> None:
                raise RuntimeError("auth failed for SUPER_SECRET_123")

        monitor = make_monitor(RecordingSink(), LeakySink())
        monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open", "open"
        )
        diagnostics = monitor.delivery_diagnostics()
        self.assertEqual(len(diagnostics), 1)
        self.assertNotIn("SUPER_SECRET_123", repr(diagnostics[0]))
        self.assertNotIn("SUPER_SECRET_123", diagnostics[0].error_kind)

    def test_local_record_exists_even_when_every_sink_fails(self) -> None:
        monitor = make_monitor(RaisingSink("a"), RaisingSink("b"))
        transition = monitor.report_unhealthy(
            HealthComponent.CONFIGURATION, "config.unavailable", "down"
        )
        self.assertIsNotNone(transition)
        self.assertEqual(len(monitor.history()), 1)
        self.assertEqual(len(monitor.delivery_diagnostics()), 2)

    def test_sink_order_is_stable(self) -> None:
        order: list[str] = []

        class OrderedSink:
            def __init__(self, name: str) -> None:
                self.name = name

            def send(self, transition: HealthTransition) -> None:
                order.append(self.name)

        monitor = make_monitor(OrderedSink("one"), OrderedSink("two"), OrderedSink("three"))
        monitor.report_unhealthy(
            HealthComponent.CAPACITY, "capacity.saturated", "full"
        )
        self.assertEqual(order, ["one", "two", "three"])


class FakeSmtpTransport:
    """Captures email without touching the network."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send_email(
        self,
        sender: str,
        recipients: tuple[str, ...],
        subject: str,
        body: str,
        timeout_seconds: float,
    ) -> None:
        self.sent.append(
            {
                "sender": sender,
                "recipients": recipients,
                "subject": subject,
                "body": body,
                "timeout_seconds": timeout_seconds,
            }
        )


class FakeHttpTransport:
    """Captures HTTP posts without touching the network."""

    def __init__(self) -> None:
        self.posts: list[dict] = []

    def post(
        self,
        url: str,
        payload: dict,
        headers: dict,
        timeout_seconds: float,
    ) -> None:
        self.posts.append(
            {"url": url, "payload": payload, "headers": headers,
             "timeout_seconds": timeout_seconds}
        )


def unhealthy_transition() -> HealthTransition:
    return HealthTransition(
        timestamp=100.0,
        component=HealthComponent.PROVIDER,
        code="provider.circuit_open",
        kind=TransitionKind.UNHEALTHY,
        detail="threshold reached",
    )


class EmailSinkTest(unittest.TestCase):
    def test_sends_to_configured_recipients_with_sanitized_body(self) -> None:
        transport = FakeSmtpTransport()
        sink = EmailSink(
            transport=transport,
            settings=EmailSettings(
                enabled=True,
                host="smtp.example.com",
                port=587,
                use_tls=True,
                username="bot",
                password="SUPER_SECRET_123",
                sender="recepcionista@example.com",
                recipients=("admin@example.com", "guardia@example.com"),
                timeout_seconds=7.0,
            ),
        )
        self.assertEqual(sink.name, "email")
        sink.send(unhealthy_transition())
        (sent,) = transport.sent
        self.assertEqual(sent["sender"], "recepcionista@example.com")
        self.assertEqual(sent["recipients"], ("admin@example.com", "guardia@example.com"))
        self.assertEqual(sent["timeout_seconds"], 7.0)
        self.assertIn("provider.circuit_open", sent["subject"])
        self.assertIn("provider.circuit_open", sent["body"])
        self.assertNotIn("SUPER_SECRET_123", sent["subject"])
        self.assertNotIn("SUPER_SECRET_123", sent["body"])

    def test_disabled_channel_builds_no_sink(self) -> None:
        from receptionist.alerting import AlertSettings

        sinks = build_alert_sinks(
            AlertSettings(
                email=EmailSettings(enabled=False, host="smtp.example.com"),
            ),
            smtp_transport=FakeSmtpTransport(),
            http_transport=FakeHttpTransport(),
        )
        self.assertEqual(sinks, [])

    def test_enabled_but_incomplete_channel_is_rejected(self) -> None:
        from receptionist.alerting import AlertError, AlertSettings

        with self.assertRaises(AlertError):
            build_alert_sinks(
                AlertSettings(email=EmailSettings(enabled=True, host="")),
                smtp_transport=FakeSmtpTransport(),
                http_transport=FakeHttpTransport(),
            )


class WebhookSinkTest(unittest.TestCase):
    def test_posts_stable_schema_to_trusted_url_only(self) -> None:
        transport = FakeHttpTransport()
        sink = WebhookSink(
            transport=transport,
            settings=WebhookSettings(
                enabled=True,
                url="https://ops.example.com/hooks/recepcionista",
                auth_token="SUPER_SECRET_123",
                timeout_seconds=5.0,
            ),
        )
        self.assertEqual(sink.name, "webhook")
        sink.send(unhealthy_transition())
        (post,) = transport.posts
        # The URL comes only from trusted configuration, never the event.
        self.assertEqual(post["url"], "https://ops.example.com/hooks/recepcionista")
        self.assertEqual(post["timeout_seconds"], 5.0)
        payload = post["payload"]
        self.assertEqual(
            set(payload),
            {"schema_version", "service", "timestamp", "component", "code",
             "status", "detail"},
        )
        self.assertEqual(payload["component"], "provider")
        self.assertEqual(payload["code"], "provider.circuit_open")
        self.assertEqual(payload["status"], "unhealthy")
        self.assertNotIn("SUPER_SECRET_123", repr(payload))
        # The secret travels only as a transport header, never in payload.
        self.assertIn("SUPER_SECRET_123", post["headers"].get("Authorization", ""))
        self.assertNotIn("SUPER_SECRET_123", repr(post["url"]))

    def test_caller_text_cannot_reach_payload_or_url(self) -> None:
        transport = FakeHttpTransport()
        sink = WebhookSink(
            transport=transport,
            settings=WebhookSettings(
                enabled=True, url="https://ops.example.com/hooks/x"
            ),
        )
        sink.send(
            HealthTransition(
                timestamp=1.0,
                component=HealthComponent.RUNTIME,
                code="runtime.audit_unavailable",
                kind=TransitionKind.UNHEALTHY,
                detail="write failed",
            )
        )
        (post,) = transport.posts
        self.assertEqual(post["url"], "https://ops.example.com/hooks/x")
        self.assertNotIn("+34910000001", repr(post["payload"]))


class TelegramSinkTest(unittest.TestCase):
    def test_sends_short_operational_message(self) -> None:
        transport = FakeHttpTransport()
        sink = TelegramSink(
            transport=transport,
            settings=TelegramSettings(
                enabled=True,
                bot_token="SUPER_SECRET_123",
                chat_id="-12345",
                timeout_seconds=5.0,
            ),
        )
        self.assertEqual(sink.name, "telegram")
        sink.send(unhealthy_transition())
        (post,) = transport.posts
        # Token is transport addressing only, never rendered content.
        self.assertIn("SUPER_SECRET_123", post["url"])
        self.assertNotIn("SUPER_SECRET_123", repr(post["payload"]))
        self.assertEqual(post["payload"]["chat_id"], "-12345")
        text = post["payload"]["text"]
        self.assertIn("provider", text)
        self.assertIn("provider.circuit_open", text)
        self.assertNotIn("+34910000001", text)
        self.assertLessEqual(len(text), 400)


class AlertConfigTest(unittest.TestCase):
    def test_settings_roundtrip_in_config_db(self) -> None:
        import os
        import sqlite3
        import tempfile

        from receptionist.sqlite_storage import SQLiteAlertRepository

        with tempfile.TemporaryDirectory() as tmp:
            repo = SQLiteAlertRepository(
                sqlite3.connect(os.path.join(tmp, "config.db"))
            )
            repo.save_email(
                EmailSettings(
                    enabled=True,
                    host="smtp.example.com",
                    port=587,
                    use_tls=True,
                    username="bot",
                    password="SUPER_SECRET_123",
                    sender="r@example.com",
                    recipients=("a@example.com", "b@example.com"),
                    timeout_seconds=7.0,
                )
            )
            repo.save_webhook(
                WebhookSettings(
                    enabled=True,
                    url="https://ops.example.com/hooks/x",
                    auth_token="SUPER_SECRET_123",
                )
            )
            repo.save_telegram(
                TelegramSettings(
                    enabled=True, bot_token="SUPER_SECRET_123", chat_id="-1"
                )
            )
            repo.save_notify_recovery(False)
            loaded = repo.load()
            self.assertTrue(loaded.email.enabled)
            self.assertEqual(loaded.email.host, "smtp.example.com")
            self.assertEqual(
                loaded.email.recipients, ("a@example.com", "b@example.com")
            )
            self.assertEqual(loaded.email.password, "SUPER_SECRET_123")
            self.assertTrue(loaded.webhook.enabled)
            self.assertEqual(loaded.webhook.url, "https://ops.example.com/hooks/x")
            self.assertTrue(loaded.telegram.enabled)
            self.assertEqual(loaded.telegram.chat_id, "-1")
            self.assertFalse(loaded.notify_recovery)

    def test_config_service_defaults_and_passthrough(self) -> None:
        import os
        import sqlite3
        import tempfile

        from receptionist.config import ConfigService, InMemoryConfigRepository
        from receptionist.sqlite_storage import SQLiteAlertRepository

        bare = ConfigService(InMemoryConfigRepository({"greeting": "hola"}))
        defaults = bare.alert_settings()
        self.assertFalse(defaults.email.enabled)
        self.assertFalse(defaults.webhook.enabled)
        self.assertFalse(defaults.telegram.enabled)
        self.assertTrue(defaults.notify_recovery)

        with tempfile.TemporaryDirectory() as tmp:
            repo = SQLiteAlertRepository(
                sqlite3.connect(os.path.join(tmp, "config.db"))
            )
            repo.save_webhook(
                WebhookSettings(enabled=True, url="https://ops.example.com/h")
            )
            wired = ConfigService(
                InMemoryConfigRepository({"greeting": "hola"}), alerts=repo
            )
            self.assertTrue(wired.alert_settings().webhook.enabled)

    def test_assembler_builds_only_complete_enabled_channels(self) -> None:
        from receptionist.alerting import AlertError, AlertSettings

        smtp = FakeSmtpTransport()
        http = FakeHttpTransport()
        settings = AlertSettings(
            email=EmailSettings(
                enabled=True,
                host="smtp.example.com",
                sender="r@example.com",
                recipients=("a@example.com",),
            ),
            webhook=WebhookSettings(enabled=False, url=""),
            telegram=TelegramSettings(
                enabled=True, bot_token="tok", chat_id="1"
            ),
        )
        sinks = build_alert_sinks(settings, smtp_transport=smtp, http_transport=http)
        self.assertEqual([s.name for s in sinks], ["email", "telegram"])
        with self.assertRaises(AlertError):
            build_alert_sinks(
                AlertSettings(
                    telegram=TelegramSettings(enabled=True, bot_token="", chat_id="")
                ),
                smtp_transport=smtp,
                http_transport=http,
            )


class CoreAlertIntegrationTest(unittest.TestCase):
    """The core reports subsystem transitions; edge-triggering, fan-out,
    and sanitization hold end to end. Deterministic via FakeClock."""

    def make_core(self, sinks=None, resilience=None, **kwargs):
        from receptionist.config import ConfigService, InMemoryConfigRepository
        from receptionist.core import ReceptionistCore
        from receptionist.health import HealthStatus as _H  # noqa: F401
        from receptionist.persistence import (
            InMemoryAuditLog,
            InMemoryCallRepository,
            InMemoryMessageRepository,
            InMemoryTranscriptStore,
            RuntimeStorage,
        )
        from receptionist.policy import Destination, Limits, PolicyEngine, RetentionPolicy

        from fakes import FakeCallIds, FakeClock, FakePolicy, FakeTelephony, FakeVoiceBackend

        telephony = FakeTelephony()
        voice = FakeVoiceBackend()
        clock = FakeClock()
        engine = PolicyEngine(
            destinations={
                "ventas": Destination(
                    id="ventas", target="SIP/201", kind="extension", enabled=True
                ),
                "recepcion": Destination(
                    id="recepcion", target="SIP/100", kind="extension", enabled=True
                ),
            },
            fallback_id="recepcion",
            limits=Limits(),
        )
        monitor = HealthMonitor(clock=clock, sinks=list(sinks or []))
        core = ReceptionistCore(
            telephony=telephony,
            voice=voice,
            config_service=ConfigService(
                InMemoryConfigRepository(
                    {
                        "greeting": "Bienvenido",
                        "language": "es",
                        "transcripts_enabled": "true",
                    }
                )
            ),
            policy=FakePolicy(),
            clock=clock,
            policy_engine=engine,
            runtime=RuntimeStorage(
                calls=kwargs.get("calls") or InMemoryCallRepository(),
                messages=InMemoryMessageRepository(clock=clock),
                transcripts=kwargs.get("transcripts") or InMemoryTranscriptStore(),
                audit=kwargs.get("audit") or InMemoryAuditLog(),
            ),
            retention=RetentionPolicy(),
            call_ids=FakeCallIds(),
            resilience=resilience,
            monitor=monitor,
        )
        core.start()
        return core, telephony, voice, clock, monitor

    def open_breaker(self, core, telephony, voice):
        from receptionist.boundaries import ProviderFailure, ProviderFailureCategory
        from receptionist.boundaries import TransferResult

        for caller in ("+34910000001", "+34910000002"):
            session = core.incoming_call(caller)
            current = voice.sessions[session.call_id]
            current.finish_playback(session.current_turn)
            current.deliver_caller_speech("hola")
            turn = session.current_turn
            current.deliver_failure(
                turn, ProviderFailure(category=ProviderFailureCategory.TIMEOUT)
            )
            telephony.complete_transfer(session.call_id, TransferResult.ACCEPTED_BY_PBX)
        assert core.breaker.is_open

    def open_breaker_once(self, core, telephony, voice):
        """Single terminal failure; for breakers with threshold 1."""
        from receptionist.boundaries import ProviderFailure, ProviderFailureCategory
        from receptionist.boundaries import TransferResult

        session = core.incoming_call("+34910000001")
        current = voice.sessions[session.call_id]
        current.finish_playback(session.current_turn)
        current.deliver_caller_speech("hola")
        turn = session.current_turn
        current.deliver_failure(
            turn, ProviderFailure(category=ProviderFailureCategory.TIMEOUT)
        )
        telephony.complete_transfer(session.call_id, TransferResult.ACCEPTED_BY_PBX)
        assert core.breaker.is_open

    def test_breaker_alerts_once_then_recovery(self) -> None:
        from receptionist.boundaries import ProviderFailure, ProviderFailureCategory
        from receptionist.boundaries import TransferResult
        from receptionist.resilience import ResilienceConfig

        sink = RecordingSink()
        core, telephony, voice, clock, monitor = self.make_core(
            sinks=[sink],
            resilience=ResilienceConfig(provider_retries=0, breaker_threshold=2),
        )
        # Failures below threshold: no circuit transition.
        first = core.incoming_call("+34910000001")
        current = voice.sessions[first.call_id]
        current.finish_playback(first.current_turn)
        current.deliver_caller_speech("hola")
        turn = first.current_turn
        current.deliver_failure(
            turn, ProviderFailure(category=ProviderFailureCategory.TIMEOUT)
        )
        telephony.complete_transfer(first.call_id, TransferResult.ACCEPTED_BY_PBX)
        self.assertEqual(sink.received, [])
        # Threshold opens: exactly one provider transition...
        second = core.incoming_call("+34910000002")
        current = voice.sessions[second.call_id]
        current.finish_playback(second.current_turn)
        current.deliver_caller_speech("hola")
        turn = second.current_turn
        current.deliver_failure(
            turn, ProviderFailure(category=ProviderFailureCategory.TIMEOUT)
        )
        telephony.complete_transfer(second.call_id, TransferResult.ACCEPTED_BY_PBX)
        core.tick()
        provider_alerts = [
            t for t in sink.received if t.code == "provider.circuit_open"
        ]
        self.assertEqual(len(provider_alerts), 1)
        # ...and ten bypassed calls add nothing more.
        for i in range(10):
            bypassed = core.incoming_call(f"+349100001{i}")
            telephony.complete_transfer(
                bypassed.call_id, TransferResult.ACCEPTED_BY_PBX
            )
        self.assertEqual(
            len([t for t in sink.received if t.code == "provider.circuit_open"]), 1
        )
        # Successful probe: exactly one recovery.
        clock.advance(60.0)
        self.assertTrue(core.report_provider_probe(True))
        recoveries = [
            t
            for t in sink.received
            if t.code == "provider.circuit_open"
            and t.kind == TransitionKind.RECOVERED
        ]
        self.assertEqual(len(recoveries), 1)

    def test_multi_cause_no_false_global_recovery(self) -> None:
        from receptionist.boundaries import StoreUnavailableError
        from receptionist.boundaries import TransferResult
        from receptionist.resilience import ResilienceConfig

        from fakes import FailingCallRepository

        sink = RecordingSink()
        failing_calls = FailingCallRepository()
        failing_calls.fail_save = StoreUnavailableError("history disk full")
        core, telephony, voice, clock, monitor = self.make_core(
            sinks=[sink],
            resilience=ResilienceConfig(
                provider_retries=0, breaker_threshold=1,
                breaker_probe_cooldown_seconds=60.0,
            ),
            calls=failing_calls,
        )
        self.open_breaker_once(core, telephony, voice)
        core.tick()
        # Runtime outage on top of the open provider.
        failing = core.incoming_call("+34910000009")
        telephony.complete_transfer(failing.call_id, TransferResult.ACCEPTED_BY_PBX)
        codes = {(t.component, t.code, t.kind) for t in sink.received}
        self.assertIn(
            (HealthComponent.PROVIDER, "provider.circuit_open", TransitionKind.UNHEALTHY),
            codes,
        )
        self.assertIn(
            (
                HealthComponent.RUNTIME,
                "runtime.history_unavailable",
                TransitionKind.UNHEALTHY,
            ),
            codes,
        )
        # Provider recovers: its recovery emits, runtime stays active,
        # and no global READY recovery is fabricated.
        clock.advance(60.0)
        self.assertTrue(core.report_provider_probe(True))
        kinds = {
            (t.code, t.kind)
            for t in sink.received
            if t.code == "provider.circuit_open"
        }
        self.assertIn(("provider.circuit_open", TransitionKind.RECOVERED), kinds)
        active = {(c.component, c.code) for c in monitor.active_conditions()}
        self.assertIn(
            (HealthComponent.RUNTIME, "runtime.history_unavailable"), active
        )
        self.assertNotIn(
            (HealthComponent.RUNTIME, "runtime.history_unavailable", TransitionKind.RECOVERED),
            {(t.component, t.code, t.kind) for t in sink.received},
        )

    def test_config_loss_alerts_once_and_recovers_once(self) -> None:
        from receptionist.config import ConfigService
        from receptionist.health import HealthStatus

        from fakes import RaisingConfigRepository

        sink = RecordingSink()
        core, telephony, voice, clock, monitor = self.make_core(sinks=[sink])
        live = core.incoming_call("+34910000001")
        core._config = ConfigService(RaisingConfigRepository(RuntimeError("down")))
        refused = core.incoming_call("+34910000002")
        self.assertIn(refused.call_id, telephony.rejected)
        config_alerts = [
            t for t in sink.received if t.code == "config.unavailable"
        ]
        self.assertEqual(len(config_alerts), 1)
        # Repeated refusals while still down: no more alerts.
        for _ in range(5):
            core.incoming_call("+34910000003")
        self.assertEqual(
            len([t for t in sink.received if t.code == "config.unavailable"]), 1
        )
        # Restore: exactly one recovery.
        from receptionist.config import InMemoryConfigRepository

        core._config = ConfigService(
            InMemoryConfigRepository({"greeting": "Bienvenido", "language": "es"})
        )
        core.incoming_call("+34910000004")
        recoveries = [
            t
            for t in sink.received
            if t.code == "config.unavailable" and t.kind == TransitionKind.RECOVERED
        ]
        self.assertEqual(len(recoveries), 1)
        self.assertEqual(core.health.status, HealthStatus.READY)
        telephony.simulate_caller_hangup(live.call_id)

    def test_runtime_outage_alerts_once_and_recovers_on_success(self) -> None:
        from receptionist.boundaries import StoreUnavailableError
        from receptionist.boundaries import TransferResult

        from fakes import FailingCallRepository

        sink = RecordingSink()
        failing_calls = FailingCallRepository()
        failing_calls.fail_save = StoreUnavailableError("history disk full")
        core, telephony, voice, clock, monitor = self.make_core(
            sinks=[sink], calls=failing_calls
        )
        for caller in ("+34910000001", "+34910000002"):
            session = core.incoming_call(caller)
            telephony.simulate_caller_hangup(session.call_id)
        history_alerts = [
            t for t in sink.received if t.code == "runtime.history_unavailable"
        ]
        self.assertEqual(len(history_alerts), 1)
        # Storage fixed: the next successful write recovers (evidence).
        failing_calls.fail_save = None
        healed = core.incoming_call("+34910000003")
        telephony.simulate_caller_hangup(healed.call_id)
        recoveries = [
            t
            for t in sink.received
            if t.code == "runtime.history_unavailable"
            and t.kind == TransitionKind.RECOVERED
        ]
        self.assertEqual(len(recoveries), 1)

    def test_transcript_failure_alerts_without_blocking(self) -> None:
        from receptionist.boundaries import TransferRequest

        from fakes import FailingTranscriptStore

        sink = RecordingSink()
        failing_transcripts = FailingTranscriptStore()
        failing_transcripts.fail_append = RuntimeError("sidecar down")
        core, telephony, voice, clock, monitor = self.make_core(
            sinks=[sink], transcripts=failing_transcripts
        )
        session = core.incoming_call("+34910000001")
        current = voice.sessions[session.call_id]
        self.assertTrue(current.spoken)
        transcript_alerts = [
            t for t in sink.received if t.code == "transcript.unavailable"
        ]
        self.assertEqual(len(transcript_alerts), 1)
        # Sidecar fixed: the next successful append recovers.
        failing_transcripts.fail_append = None
        current.finish_playback(session.current_turn)
        current.deliver_caller_speech("hola")
        recoveries = [
            t
            for t in sink.received
            if t.code == "transcript.unavailable"
            and t.kind == TransitionKind.RECOVERED
        ]
        self.assertEqual(len(recoveries), 1)
        # Non-blocking: transfer still lands, breaker untouched.
        current.deliver_action_request(TransferRequest(destination_id="ventas"))
        self.assertEqual(telephony.transfers, [(session.call_id, "SIP/201")])
        self.assertFalse(core.breaker.is_open)

    def test_broken_sinks_change_no_call_behavior(self) -> None:
        from receptionist.boundaries import (
            MessageConfirmed,
            MessageTextFinal,
            StartMessageCapture,
            TransferRequest,
            TransferResult,
        )
        from receptionist.call_session import MESSAGE_SAVED_ACK

        core, telephony, voice, clock, monitor = self.make_core(
            sinks=[RaisingSink("dead-a"), RaisingSink("dead-b")]
        )
        session = core.incoming_call("+34910000001")
        current = voice.sessions[session.call_id]
        current.finish_playback(session.current_turn)
        current.deliver_action_request(TransferRequest(destination_id="ventas"))
        self.assertEqual(telephony.transfers, [(session.call_id, "SIP/201")])
        telephony.complete_transfer(session.call_id, TransferResult.ACCEPTED_BY_PBX)
        second = core.incoming_call("+34910000002")
        voice2 = voice.sessions[second.call_id]
        voice2.finish_playback(second.current_turn)
        voice2.deliver_action_request(StartMessageCapture())
        voice2.deliver_action_request(MessageTextFinal(text="Llámeme mañana."))
        voice2.deliver_action_request(MessageConfirmed())
        spoken = [text for text, _ in voice2.spoken]
        self.assertIn(MESSAGE_SAVED_ACK, spoken)
        self.assertFalse(core.breaker.is_open)
        # Force one alertable condition: delivery fails are diagnosed
        # locally, never raised, never recursed.
        from receptionist.boundaries import StoreUnavailableError

        from fakes import FailingCallRepository

        failing = FailingCallRepository()
        failing.fail_save = StoreUnavailableError("history disk full")
        core._runtime.calls = failing
        doomed = core.incoming_call("+34910000003")
        telephony.simulate_caller_hangup(doomed.call_id)
        self.assertTrue(monitor.delivery_diagnostics())
        from receptionist.call_session import CallState

        self.assertEqual(doomed.state, CallState.ENDED)

    def test_startup_transitions(self) -> None:
        from receptionist.config import ConfigService, InMemoryConfigRepository
        from receptionist.core import ReceptionistCore
        from receptionist.health import HealthStatus
        from receptionist.persistence import RuntimeStorage
        from receptionist.policy import Destination, Limits, PolicyEngine, RetentionPolicy

        from fakes import (
            FakeCallIds,
            FakeClock,
            FakePolicy,
            FakeTelephony,
            FakeVoiceBackend,
            RaisingConfigRepository,
        )

        def build_core(config_service):
            clock = FakeClock()
            monitor = HealthMonitor(clock=clock, sinks=[RecordingSink()])
            core = ReceptionistCore(
                telephony=FakeTelephony(),
                voice=FakeVoiceBackend(),
                config_service=config_service,
                policy=FakePolicy(),
                clock=clock,
                policy_engine=PolicyEngine(
                    destinations={}, fallback_id="recepcion", limits=Limits()
                ),
                runtime=RuntimeStorage.create(clock=clock),
                retention=RetentionPolicy(),
                call_ids=FakeCallIds(),
                monitor=monitor,
            )
            return core, monitor

        # Clean STARTING -> READY: no recovery alert is fabricated.
        core, monitor = build_core(
            ConfigService(InMemoryConfigRepository({"greeting": "h", "language": "es"}))
        )
        core.start()
        self.assertEqual(core.health.status, HealthStatus.READY)
        self.assertEqual(monitor.history(), ())
        # STARTING -> NOT_READY: exactly one unhealthy transition...
        broken, broken_monitor = build_core(
            ConfigService(RaisingConfigRepository(RuntimeError("db locked")))
        )
        broken.start()
        self.assertEqual(broken.health.status, HealthStatus.NOT_READY)
        self.assertEqual(len(broken_monitor.history()), 1)
        self.assertEqual(
            broken_monitor.history()[0].kind, TransitionKind.UNHEALTHY
        )
        # ...and repeated startup checks of the same condition stay silent.
        broken.start()
        broken.start()
        self.assertEqual(len(broken_monitor.history()), 1)

    def test_caller_data_never_reaches_transitions(self) -> None:
        from receptionist.boundaries import ProviderFailure, ProviderFailureCategory

        sink = RecordingSink()
        core, telephony, voice, clock, monitor = self.make_core(sinks=[sink])
        session = core.incoming_call("+34910000001", caller_name="García")
        current = voice.sessions[session.call_id]
        current.finish_playback(session.current_turn)
        current.deliver_caller_speech("hola")
        current.deliver_failure(
            session.current_turn,
            ProviderFailure(category=ProviderFailureCategory.TIMEOUT),
        )
        telephony.simulate_caller_hangup(session.call_id)
        blob = repr(monitor.history())
        self.assertNotIn("+34910000001", blob)
        self.assertNotIn("García", blob)
        self.assertNotIn("hola", blob)

    def test_capacity_saturation_alerts_and_recovers(self) -> None:
        from receptionist.boundaries import TransferResult
        from receptionist.call_session import CallState

        sink = RecordingSink()
        core, telephony, voice, clock, monitor = self.make_core(sinks=[sink])
        first = core.incoming_call("+34910000001")
        saturated = core.incoming_call("+34910000002")
        self.assertEqual(saturated.state, CallState.FALLBACK_HANDOFF)
        capacity_alerts = [
            t for t in sink.received if t.code == "capacity.saturated"
        ]
        self.assertEqual(len(capacity_alerts), 1)
        telephony.complete_transfer(saturated.call_id, TransferResult.ACCEPTED_BY_PBX)
        telephony.simulate_caller_hangup(first.call_id)
        third = core.incoming_call("+34910000003")
        self.assertIn(third.call_id, voice.sessions)
        recoveries = [
            t
            for t in sink.received
            if t.code == "capacity.saturated" and t.kind == TransitionKind.RECOVERED
        ]
        self.assertEqual(len(recoveries), 1)

    def test_transition_timestamps_follow_fake_clock(self) -> None:
        sink = RecordingSink()
        core, telephony, voice, clock, monitor = self.make_core(sinks=[sink])
        clock.advance(42.0)
        transition = monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open", "open"
        )
        assert transition is not None
        self.assertEqual(transition.timestamp, 42.0)
        self.assertEqual(sink.received[0].timestamp, 42.0)


if __name__ == "__main__":
    unittest.main()
