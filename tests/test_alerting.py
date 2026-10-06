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
            HealthComponent.PROVIDER, "provider.circuit_open"
        )
        self.assertIsNotNone(first)
        self.assertEqual(first.kind, TransitionKind.UNHEALTHY)
        # Detail is project-owned registry text, never caller input.
        self.assertEqual(first.detail, "failure threshold reached")
        monitor.drain()
        self.assertEqual(len(sink.received), 1)

    def test_repeated_same_condition_never_reemits(self) -> None:
        sink = RecordingSink()
        monitor = make_monitor(sink)
        monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open"
        )
        for _ in range(20):
            repeated = monitor.report_unhealthy(
                HealthComponent.PROVIDER, "provider.circuit_open"
            )
            self.assertIsNone(repeated)
        monitor.drain()
        self.assertEqual(len(sink.received), 1)
        self.assertEqual(len(monitor.history()), 1)

    def test_free_text_cannot_enter_a_transition(self) -> None:
        sink = RecordingSink()
        monitor = make_monitor(sink)
        with self.assertRaises(TypeError):
            monitor.report_unhealthy(
                HealthComponent.PROVIDER,
                "provider.circuit_open",
                "password=hunter2 caller=+54911SECRET",  # type: ignore[call-arg]
            )
        self.assertEqual(monitor.history(), ())
        monitor.drain()
        self.assertEqual(sink.received, [])

    def test_recovery_emits_once_then_silence(self) -> None:
        sink = RecordingSink()
        monitor = make_monitor(sink)
        monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open"
        )
        recovered = monitor.report_recovered(
            HealthComponent.PROVIDER, "provider.circuit_open"
        )
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.kind, TransitionKind.RECOVERED)
        monitor.drain()
        self.assertEqual(len(sink.received), 2)
        self.assertIsNone(
            monitor.report_recovered(HealthComponent.PROVIDER, "provider.circuit_open")
        )
        monitor.drain()
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
            HealthComponent.RUNTIME, "runtime.history_unavailable"
        )
        monitor.drain()
        self.assertEqual(len(sink.received), 1)
        recovered = monitor.report_recovered(
            HealthComponent.RUNTIME, "runtime.history_unavailable"
        )
        self.assertIsNotNone(recovered)
        # Local record exists; the external sink is not notified.
        monitor.drain()
        self.assertEqual(len(sink.received), 1)
        self.assertEqual(len(monitor.history()), 2)

    def test_conditions_are_tracked_per_component_and_code(self) -> None:
        monitor = make_monitor()
        monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open"
        )
        monitor.report_unhealthy(
            HealthComponent.RUNTIME, "runtime.history_unavailable"
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
        from receptionist.alerting import SERVICE_NAME, TRANSITION_SCHEMA_VERSION

        clock = FakeClock(start=1234.5)
        monitor = HealthMonitor(clock=clock, sinks=[])
        transition = monitor.report_unhealthy(
            HealthComponent.CONFIGURATION, "config.unavailable"
        )
        assert transition is not None
        self.assertEqual(transition.timestamp, 1234.5)
        self.assertEqual(transition.component, HealthComponent.CONFIGURATION)
        self.assertEqual(transition.code, "config.unavailable")
        self.assertEqual(transition.detail, "configuration authority unreadable")
        self.assertEqual(transition.service, SERVICE_NAME)
        self.assertEqual(transition.schema_version, TRANSITION_SCHEMA_VERSION)

    def test_transitions_are_emitted_as_structured_logs(self) -> None:
        import json

        monitor = make_monitor()
        with self.assertLogs("virtual-receptionist.health", level="INFO") as logs:
            monitor.report_unhealthy(
                HealthComponent.CAPACITY, "capacity.saturated"
            )
        (line,) = [r for r in logs.output if "health_transition" in r]
        payload = json.loads(line.split("health_transition ", 1)[1])
        self.assertEqual(payload["component"], "capacity")
        self.assertEqual(payload["code"], "capacity.saturated")
        self.assertEqual(payload["status"], "unhealthy")


class FanOutIsolationTest(unittest.TestCase):
    def test_one_failing_sink_does_not_stop_the_others(self) -> None:
        first = RecordingSink("a")
        failing = RaisingSink("b")
        third = RecordingSink("c")
        monitor = make_monitor(first, failing, third)
        monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open"
        )
        monitor.drain()
        self.assertEqual(len(first.received), 1)
        self.assertEqual(failing.attempts, 1)
        self.assertEqual(len(third.received), 1)

    def test_delivery_failure_is_diagnosed_locally_without_recursion(self) -> None:
        first = RecordingSink("a")
        failing = RaisingSink("b")
        monitor = make_monitor(first, failing)
        monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open"
        )
        monitor.drain()
        # The failure is recorded locally exactly once, by position...
        diagnostics = monitor.delivery_diagnostics()
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].sink_name, "sink-1")
        # ...and never fanned out as a second external alert.
        self.assertEqual(len(first.received), 1)
        # No recursion: a failing sink never triggers further delivery.
        self.assertEqual(failing.attempts, 1)

    def test_diagnostic_never_carries_exception_strings(self) -> None:
        class LeakySink:
            def send(self, transition: HealthTransition) -> None:
                raise RuntimeError("auth failed for SUPER_SECRET_123")

        monitor = make_monitor(RecordingSink(), LeakySink())
        monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open"
        )
        monitor.drain()
        diagnostics = monitor.delivery_diagnostics()
        self.assertEqual(len(diagnostics), 1)
        self.assertNotIn("SUPER_SECRET_123", repr(diagnostics[0]))

    def test_hostile_sink_name_never_reaches_diagnostics(self) -> None:
        class HostileName:
            @property
            def name(self):  # noqa: D102 (intentionally hostile)
                return "hook SMTP_SECRET_ABC"

            def send(self, transition: HealthTransition) -> None:
                raise RuntimeError("down")

        monitor = make_monitor(RecordingSink(), HostileName())
        monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open"
        )
        monitor.drain()
        diagnostics = monitor.delivery_diagnostics()
        self.assertEqual(len(diagnostics), 1)
        blob = repr(diagnostics) + repr(monitor.history())
        self.assertNotIn("SMTP_SECRET_ABC", blob)

    def test_broken_clock_none_sink_and_missing_send_are_contained(self) -> None:
        class BrokenClock:
            def now(self) -> float:
                raise RuntimeError("clock broken")

        class NoSend:
            name = "nosend"

        monitor = HealthMonitor(clock=BrokenClock(), sinks=[RecordingSink(), None, NoSend()])  # type: ignore[list-item]
        transition = monitor.report_unhealthy(
            HealthComponent.CAPACITY, "capacity.saturated"
        )
        assert transition is not None
        self.assertEqual(transition.timestamp, 0.0)
        monitor.drain()
        # None filtered, missing send diagnosed, recorder still served.
        self.assertEqual(len(monitor.delivery_diagnostics()), 1)

    def test_local_record_exists_even_when_every_sink_fails(self) -> None:
        monitor = make_monitor(RaisingSink("a"), RaisingSink("b"))
        transition = monitor.report_unhealthy(
            HealthComponent.CONFIGURATION, "config.unavailable"
        )
        self.assertIsNotNone(transition)
        self.assertEqual(len(monitor.history()), 1)
        monitor.drain()
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
            HealthComponent.CAPACITY, "capacity.saturated"
        )
        monitor.drain()
        self.assertEqual(order, ["one", "two", "three"])

    def test_slow_sinks_never_block_the_reporter(self) -> None:
        import time

        class SlowSink:
            name = "slow"

            def send(self, transition: HealthTransition) -> None:
                time.sleep(0.3)

        monitor = make_monitor(SlowSink(), SlowSink())
        started = time.monotonic()
        monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open"
        )
        elapsed = time.monotonic() - started
        # Two 0.3 s deliveries would stall a synchronous reporter ≥ 0.6 s.
        self.assertLess(elapsed, 0.3)
        monitor.drain()

    def test_external_order_matches_history_under_adversarial_schedule(self) -> None:
        import threading

        received: list[TransitionKind] = []
        release = threading.Event()

        class BlockingSink:
            name = "blocking"

            def send(self, transition: HealthTransition) -> None:
                if transition.kind == TransitionKind.UNHEALTHY:
                    release.wait(timeout=5.0)
                received.append(transition.kind)

        monitor = make_monitor(BlockingSink())
        monitor.report_unhealthy(HealthComponent.PROVIDER, "provider.circuit_open")
        # The worker is stuck inside the first delivery; recovery queues behind.
        monitor.report_recovered(HealthComponent.PROVIDER, "provider.circuit_open")
        release.set()
        monitor.drain()
        kinds = [t.kind for t in monitor.history()]
        self.assertEqual(kinds, [TransitionKind.UNHEALTHY, TransitionKind.RECOVERED])
        self.assertEqual(received, [TransitionKind.UNHEALTHY, TransitionKind.RECOVERED])

    def test_backlog_preserves_full_sequence_without_drops(self) -> None:
        import threading

        release = threading.Event()
        received: list[str] = []

        class BlockingSink:
            name = "blocking"

            def send(self, transition: HealthTransition) -> None:
                release.wait(timeout=10.0)
                received.append(f"{transition.code}:{transition.kind.value}")

        monitor = make_monitor(BlockingSink())
        monitor.report_unhealthy(HealthComponent.PROVIDER, "provider.circuit_open")
        # Flood while the worker is stuck: every report still returns fast
        # and nothing emitted is ever dropped.
        import time

        started = time.monotonic()
        for _ in range(40):
            monitor.report_recovered(HealthComponent.PROVIDER, "provider.circuit_open")
            monitor.report_unhealthy(HealthComponent.PROVIDER, "provider.circuit_open")
        self.assertLess(time.monotonic() - started, 5.0)
        release.set()
        monitor.drain()
        # External stream equals local history exactly: no gaps, no reorder.
        self.assertEqual(
            received,
            [f"{t.code}:{t.kind.value}" for t in monitor.history()],
        )
        self.assertEqual(
            [d for d in monitor.delivery_diagnostics()], []
        )

    def test_concurrent_reports_of_one_condition_emit_once(self) -> None:
        import threading

        sink = RecordingSink()
        monitor = make_monitor(sink)
        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def report() -> None:
            try:
                barrier.wait(timeout=5.0)
                monitor.report_unhealthy(
                    HealthComponent.PROVIDER, "provider.circuit_open"
                )
            except Exception as error:  # noqa: BLE001 (asserted empty below)
                errors.append(error)

        threads = [threading.Thread(target=report) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
        self.assertEqual(errors, [])
        unhealthy = [
            t
            for t in monitor.history()
            if t.kind == TransitionKind.UNHEALTHY
        ]
        self.assertEqual(len(unhealthy), 1)
        monitor.drain()
        self.assertEqual(len(sink.received), 1)
        self.assertLessEqual(monitor.started_workers, 1)

    def test_concurrent_recoveries_emit_at_most_once(self) -> None:
        import threading

        sink = RecordingSink()
        monitor = make_monitor(sink)
        monitor.report_unhealthy(HealthComponent.PROVIDER, "provider.circuit_open")
        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def recover() -> None:
            try:
                barrier.wait(timeout=5.0)
                monitor.report_recovered(
                    HealthComponent.PROVIDER, "provider.circuit_open"
                )
            except Exception as error:  # noqa: BLE001 (asserted empty below)
                errors.append(error)

        threads = [threading.Thread(target=recover) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
        self.assertEqual(errors, [])
        recoveries = [
            t for t in monitor.history() if t.kind == TransitionKind.RECOVERED
        ]
        self.assertEqual(len(recoveries), 1)
        monitor.drain()
        self.assertEqual(len(sink.received), 2)


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
                detail="audit write failed",
            )
        )
        (post,) = transport.posts
        self.assertEqual(post["url"], "https://ops.example.com/hooks/x")
        self.assertNotIn("+34910000001", repr(post["payload"]))

    def test_redirect_never_carries_authorization_off_host(self) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from receptionist.alerting import TransportError, UrllibTransport

        seen: list[dict] = []

        class Target(BaseHTTPRequestHandler):
            def _record(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                seen.append(
                    {
                        "method": self.command,
                        "headers": dict(self.headers),
                        "path": self.path,
                    }
                )
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                self._record()

            def do_GET(self) -> None:
                # urllib converts a followed 302 POST into GET: a target
                # that only speaks POST would hide the leak. Record both.
                self._record()

            def log_message(self, *args: object) -> None:
                pass

        class Redirector(BaseHTTPRequestHandler):
            target = ""

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                self.send_response(302)
                self.send_header("Location", self.target)
                self.end_headers()

            def log_message(self, *args: object) -> None:
                pass

        target_server = HTTPServer(("127.0.0.1", 0), Target)
        redirect_server = HTTPServer(("127.0.0.1", 0), Redirector)
        Redirector.target = (
            f"http://127.0.0.1:{target_server.server_port}/stolen"
        )
        for server in (target_server, redirect_server):
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
        try:
            transport = UrllibTransport()
            sink = WebhookSink(
                transport=transport,
                settings=WebhookSettings(
                    enabled=True,
                    url=f"http://127.0.0.1:{redirect_server.server_port}/hook",
                    auth_token="SUPER_SECRET_123",
                ),
            )
            with self.assertRaises(TransportError) as raised:
                sink.send(unhealthy_transition())
            # The redirect target receives zero requests, by any method.
            self.assertEqual(seen, [])
            # The caller sees a normalized failure: no URL, no token.
            self.assertNotIn("SUPER_SECRET_123", str(raised.exception))
            self.assertNotIn("127.0.0.1", str(raised.exception))
        finally:
            target_server.shutdown()
            redirect_server.shutdown()


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


class SettingsReprTest(unittest.TestCase):
    def test_secrets_redacted_from_repr_and_str(self) -> None:
        from receptionist.alerting import AlertSettings

        settings = AlertSettings(
            email=EmailSettings(password="SMTP_SECRET_ABC"),
            webhook=WebhookSettings(
                url="https://h/x", auth_token="WEBHOOK_SECRET_DEF"
            ),
            telegram=TelegramSettings(bot_token="TELEGRAM_SECRET_GHI", chat_id="1"),
        )
        for text in (repr(settings), str(settings)):
            self.assertNotIn("SMTP_SECRET_ABC", text)
            self.assertNotIn("WEBHOOK_SECRET_DEF", text)
            self.assertNotIn("TELEGRAM_SECRET_GHI", text)
        for piece in (
            repr(settings.email),
            str(settings.email),
            repr(settings.webhook),
            str(settings.webhook),
            repr(settings.telegram),
            str(settings.telegram),
        ):
            self.assertNotIn("SECRET", piece)
        # Channels and flags stay visible for operations.
        self.assertIn("email(enabled=False)", repr(settings))
        self.assertIn("notify_recovery=True", repr(settings))

    def test_channel_validation_ranges(self) -> None:
        from receptionist.alerting import AlertError, AlertSettings

        smtp = FakeSmtpTransport()
        http = FakeHttpTransport()
        bad = [
            AlertSettings(
                email=EmailSettings(
                    enabled=True, host="h", sender="s", recipients=("r",), port=0
                )
            ),
            AlertSettings(
                email=EmailSettings(
                    enabled=True, host="h", sender="s", recipients=("r",), port=70000
                )
            ),
            AlertSettings(
                email=EmailSettings(
                    enabled=True, host="h", sender="s", recipients=("  ",),
                    timeout_seconds=0,
                )
            ),
            AlertSettings(webhook=WebhookSettings(enabled=True, url="ftp://x/y")),
            AlertSettings(
                telegram=TelegramSettings(enabled=True, bot_token="t", chat_id="  ")
            ),
        ]
        for settings in bad:
            with self.assertRaises(AlertError):
                build_alert_sinks(settings, smtp_transport=smtp, http_transport=http)

    def test_build_monitor_wires_settings_end_to_end(self) -> None:
        from receptionist.alerting import AlertSettings, build_monitor

        http = FakeHttpTransport()
        monitor = build_monitor(
            AlertSettings(
                webhook=WebhookSettings(enabled=True, url="https://ops.example.com/h"),
                notify_recovery=False,
            ),
            FakeClock(),
            http_transport=http,
        )
        self.assertFalse(monitor.notify_recovery)
        monitor.report_unhealthy(HealthComponent.PROVIDER, "provider.circuit_open")
        monitor.drain()
        self.assertEqual(len(http.posts), 1)
        monitor.report_recovered(HealthComponent.PROVIDER, "provider.circuit_open")
        monitor.drain()
        self.assertEqual(len(http.posts), 1)


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

    def make_core(self, sinks=None, resilience=None, config_repo=None, **kwargs):
        from receptionist.config import ConfigService, InMemoryConfigRepository
        from receptionist.core import ReceptionistCore
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
                config_repo
                or InMemoryConfigRepository(
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

    def open_breaker(self, core, telephony, voice, calls=2):
        from receptionist.boundaries import ProviderFailure, ProviderFailureCategory
        from receptionist.boundaries import TransferResult

        for n in range(calls):
            session = core.incoming_call(f"+3491000000{n}")
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
        monitor.drain()
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
        monitor.drain()
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
        monitor.drain()
        self.assertEqual(
            len([t for t in sink.received if t.code == "provider.circuit_open"]), 1
        )
        # Successful probe: exactly one recovery.
        clock.advance(60.0)
        self.assertTrue(core.report_provider_probe(True))
        monitor.drain()
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
        self.open_breaker(core, telephony, voice, calls=1)
        core.tick()
        # Runtime outage on top of the open provider.
        failing = core.incoming_call("+34910000009")
        telephony.complete_transfer(failing.call_id, TransferResult.ACCEPTED_BY_PBX)
        monitor.drain()
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
        monitor.drain()
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

        sink = RecordingSink()

        class MutableConfigRepository:
            def __init__(self) -> None:
                self.broken = False
                self.values = {"greeting": "Bienvenido", "language": "es"}

            def get(self, key: str):
                if self.broken:
                    raise RuntimeError("config.db lost")
                return self.values.get(key)

        repository = MutableConfigRepository()
        core, telephony, voice, clock, monitor = self.make_core(
            sinks=[sink], config_repo=repository
        )
        live = core.incoming_call("+34910000001")
        repository.broken = True
        refused = core.incoming_call("+34910000002")
        self.assertIn(refused.call_id, telephony.rejected)
        monitor.drain()
        config_alerts = [
            t for t in sink.received if t.code == "config.unavailable"
        ]
        self.assertEqual(len(config_alerts), 1)
        # Repeated refusals while still down: no more alerts.
        for _ in range(5):
            core.incoming_call("+34910000003")
        monitor.drain()
        self.assertEqual(
            len([t for t in sink.received if t.code == "config.unavailable"]), 1
        )
        # Restore: exactly one recovery.
        repository.broken = False
        core.incoming_call("+34910000004")
        monitor.drain()
        recoveries = [
            t
            for t in sink.received
            if t.code == "config.unavailable" and t.kind == TransitionKind.RECOVERED
        ]
        self.assertEqual(len(recoveries), 1)
        self.assertEqual(core.health.status, HealthStatus.READY)
        telephony.simulate_caller_hangup(live.call_id)

    def test_config_cross_transitions(self) -> None:
        """unavailable -> available+incomplete -> complete, and back."""
        from receptionist.health import HealthStatus

        sink = RecordingSink()

        class MutableConfigRepository:
            def __init__(self) -> None:
                self.broken = False
                self.values = {"greeting": "Bienvenido", "language": "es"}

            def get(self, key: str):
                if self.broken:
                    raise RuntimeError("config.db lost")
                return self.values.get(key)

        repository = MutableConfigRepository()
        core, telephony, voice, clock, monitor = self.make_core(
            sinks=[sink], config_repo=repository
        )
        # DB back but a required key gone: available yet incomplete.
        repository.broken = True
        core.incoming_call("+34910000001")
        repository.broken = False
        repository.values.pop("language")
        refused = core.incoming_call("+34910000002")
        monitor.drain()
        self.assertIn(refused.call_id, telephony.rejected)
        self.assertEqual(core.health.status, HealthStatus.NOT_READY)
        self.assertNotIn(refused.call_id, voice.sessions)
        codes = {(t.code, t.kind) for t in sink.received}
        self.assertIn(("config.unavailable", TransitionKind.RECOVERED), codes)
        self.assertIn(("config.missing_required", TransitionKind.UNHEALTHY), codes)
        # Operator fixes config at runtime: recovery, no hung alert.
        repository.values["language"] = "es"
        revived = core.incoming_call("+34910000003")
        monitor.drain()
        self.assertIn(revived.call_id, voice.sessions)
        codes = {(t.code, t.kind) for t in sink.received}
        self.assertIn(("config.missing_required", TransitionKind.RECOVERED), codes)

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
        monitor.drain()
        history_alerts = [
            t for t in sink.received if t.code == "runtime.history_unavailable"
        ]
        self.assertEqual(len(history_alerts), 1)
        # Storage fixed: the next successful write recovers (evidence).
        failing_calls.fail_save = None
        healed = core.incoming_call("+34910000003")
        telephony.simulate_caller_hangup(healed.call_id)
        monitor.drain()
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
        monitor.drain()
        transcript_alerts = [
            t for t in sink.received if t.code == "transcript.unavailable"
        ]
        self.assertEqual(len(transcript_alerts), 1)
        # Sidecar fixed: the next successful append recovers.
        failing_transcripts.fail_append = None
        current.finish_playback(session.current_turn)
        current.deliver_caller_speech("hola")
        monitor.drain()
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
        failing_core, failing_telephony, _, _, failing_monitor = self.make_core(
            sinks=[RaisingSink("dead-a"), RaisingSink("dead-b")], calls=failing
        )
        doomed = failing_core.incoming_call("+34910000003")
        failing_telephony.simulate_caller_hangup(doomed.call_id)
        failing_monitor.drain()
        self.assertTrue(failing_monitor.delivery_diagnostics())
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
        monitor.drain()
        capacity_alerts = [
            t for t in sink.received if t.code == "capacity.saturated"
        ]
        self.assertEqual(len(capacity_alerts), 1)
        # Recovery fires when the slot frees, not on later traffic.
        telephony.simulate_caller_hangup(first.call_id)
        monitor.drain()
        recoveries = [
            t
            for t in sink.received
            if t.code == "capacity.saturated" and t.kind == TransitionKind.RECOVERED
        ]
        self.assertEqual(len(recoveries), 1)
        telephony.complete_transfer(saturated.call_id, TransferResult.ACCEPTED_BY_PBX)
        third = core.incoming_call("+34910000003")
        self.assertIn(third.call_id, voice.sessions)
        monitor.drain()
        self.assertEqual(
            len([t for t in sink.received if t.code == "capacity.saturated"]), 2
        )

    def test_aggregate_agrees_with_monitor_provider_then_runtime(self) -> None:
        """Order A→B→recover A: no RECOVERED alert with DEGRADED aggregate,
        no READY aggregate with an active condition."""
        from receptionist.boundaries import StoreUnavailableError
        from receptionist.boundaries import TransferResult
        from receptionist.health import HealthStatus
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
        self.open_breaker(core, telephony, voice, calls=1)
        failing = core.incoming_call("+34910000009")
        telephony.complete_transfer(failing.call_id, TransferResult.ACCEPTED_BY_PBX)
        self.assertEqual(core.health.status, HealthStatus.DEGRADED)
        clock.advance(60.0)
        self.assertTrue(core.report_provider_probe(True))
        monitor.drain()
        # Provider recovery emitted, but the aggregate still degrades on
        # the active runtime condition: never a false global recovery.
        self.assertTrue(monitor.active_conditions())
        self.assertEqual(core.health.status, HealthStatus.DEGRADED)
        self.assertNotIn(
            (HealthComponent.RUNTIME, "runtime.history_unavailable", TransitionKind.RECOVERED),
            {(t.component, t.code, t.kind) for t in sink.received},
        )
        # Runtime heals too: empty conditions agree with READY.
        failing_calls.fail_save = None
        healed = core.incoming_call("+34910000010")
        telephony.simulate_caller_hangup(healed.call_id)
        monitor.drain()
        self.assertEqual(monitor.active_conditions(), ())
        self.assertEqual(core.health.status, HealthStatus.READY)

    def test_aggregate_agrees_with_monitor_runtime_then_provider(self) -> None:
        """Order B→A→recover B: runtime recovery first, provider still open."""
        from receptionist.boundaries import StoreUnavailableError
        from receptionist.boundaries import TransferResult
        from receptionist.health import HealthStatus
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
        doomed = core.incoming_call("+34910000001")
        telephony.simulate_caller_hangup(doomed.call_id)
        self.open_breaker(core, telephony, voice, calls=1)
        core.tick()
        self.assertEqual(core.health.status, HealthStatus.DEGRADED)
        # Runtime heals while the provider stays open: still DEGRADED.
        failing_calls.fail_save = None
        healed = core.incoming_call("+34910000002")
        telephony.simulate_caller_hangup(healed.call_id)
        monitor.drain()
        active = {(c.component, c.code) for c in monitor.active_conditions()}
        self.assertNotIn(
            (HealthComponent.RUNTIME, "runtime.history_unavailable"), active
        )
        self.assertIn((HealthComponent.PROVIDER, "provider.circuit_open"), active)
        self.assertEqual(core.health.status, HealthStatus.DEGRADED)
        # Provider heals: empty agrees with READY.
        clock.advance(60.0)
        self.assertTrue(core.report_provider_probe(True))
        monitor.drain()
        self.assertEqual(monitor.active_conditions(), ())
        self.assertEqual(core.health.status, HealthStatus.READY)

    def test_hostile_sinks_cannot_break_admission(self) -> None:
        from receptionist.call_session import CallState

        class BadName:
            @property
            def name(self):
                raise RuntimeError("name exploded")

            def send(self, transition: HealthTransition) -> None:
                raise RuntimeError("delivery down")

        core, telephony, voice, clock, monitor = self.make_core(
            sinks=[BadName(), None]
        )
        first = core.incoming_call("+34910000001")
        self.assertIn(first.call_id, voice.sessions)
        saturated = core.incoming_call("+34910000002")
        # State transitioned and no exception escaped, despite hostile sinks.
        self.assertEqual(saturated.state, CallState.FALLBACK_HANDOFF)
        monitor.drain()
        self.assertEqual(len(monitor.delivery_diagnostics()), 1)

    def test_productive_wiring_from_config_db_to_sink(self) -> None:
        import os
        import sqlite3
        import tempfile

        from receptionist.alerting import build_monitor
        from receptionist.config import ConfigService, InMemoryConfigRepository
        from receptionist.sqlite_storage import SQLiteAlertRepository

        with tempfile.TemporaryDirectory() as tmp:
            repo = SQLiteAlertRepository(
                sqlite3.connect(os.path.join(tmp, "config.db"))
            )
            repo.save_webhook(
                WebhookSettings(enabled=True, url="https://ops.example.com/prod")
            )
            service = ConfigService(
                InMemoryConfigRepository({"greeting": "h", "language": "es"}),
                alerts=repo,
            )
            http = FakeHttpTransport()
            monitor = build_monitor(
                service.alert_settings(), FakeClock(), http_transport=http
            )
            monitor.report_unhealthy(
                HealthComponent.PROVIDER, "provider.circuit_open"
            )
            monitor.drain()
            (post,) = http.posts
            self.assertEqual(post["url"], "https://ops.example.com/prod")
            self.assertEqual(post["payload"]["code"], "provider.circuit_open")
            monitor.close()

    def test_core_builds_monitor_from_config_db_without_injected_monitor(self) -> None:
        """Productive route: no prebuilt HealthMonitor is passed. Settings
        persist in config.db; a real transition reaches the fake
        transports; persisted notify_recovery=False keeps recovery local."""
        import os
        import sqlite3
        import tempfile

        from receptionist.boundaries import (
            ProviderFailure,
            ProviderFailureCategory,
            TransferResult,
        )
        from receptionist.config import ConfigService, InMemoryConfigRepository
        from receptionist.core import ReceptionistCore
        from receptionist.health import HealthStatus
        from receptionist.persistence import RuntimeStorage
        from receptionist.policy import Destination, Limits, PolicyEngine, RetentionPolicy
        from receptionist.resilience import ResilienceConfig

        from fakes import (
            FakeCallIds,
            FakeClock,
            FakePolicy,
            FakeTelephony,
            FakeVoiceBackend,
        )
        from receptionist.sqlite_storage import SQLiteAlertRepository

        with tempfile.TemporaryDirectory() as tmp:
            repo = SQLiteAlertRepository(
                sqlite3.connect(os.path.join(tmp, "config.db"))
            )
            repo.save_webhook(
                WebhookSettings(enabled=True, url="https://ops.example.com/prod")
            )
            repo.save_email(
                EmailSettings(
                    enabled=True,
                    host="smtp.example.com",
                    sender="r@example.com",
                    recipients=("ops@example.com",),
                )
            )
            repo.save_telegram(
                TelegramSettings(enabled=True, bot_token="tok", chat_id="1")
            )
            repo.save_notify_recovery(False)
            service = ConfigService(
                InMemoryConfigRepository({"greeting": "h", "language": "es"}),
                alerts=repo,
            )
            clock = FakeClock()
            telephony = FakeTelephony()
            voice = FakeVoiceBackend()
            smtp = FakeSmtpTransport()
            http = FakeHttpTransport()
            core = ReceptionistCore(
                telephony=telephony,
                voice=voice,
                config_service=service,
                policy=FakePolicy(),
                clock=clock,
                policy_engine=PolicyEngine(
                    destinations={
                        "recepcion": Destination(
                            id="recepcion", target="SIP/100",
                            kind="extension", enabled=True,
                        )
                    },
                    fallback_id="recepcion",
                    limits=Limits(),
                ),
                runtime=RuntimeStorage.create(clock=clock),
                retention=RetentionPolicy(),
                call_ids=FakeCallIds(),
                resilience=ResilienceConfig(
                    provider_retries=0, breaker_threshold=1,
                    breaker_probe_cooldown_seconds=60.0,
                ),
                smtp_transport=smtp,
                http_transport=http,
            )
            core.start()
            self.assertFalse(core.monitor.notify_recovery)
            self.assertEqual(
                sorted(s.name for s in core.monitor.sinks),
                ["email", "telegram", "webhook"],
            )
            # A real provider outage reaches every configured transport.
            session = core.incoming_call("+34910000001")
            current = voice.sessions[session.call_id]
            current.finish_playback(session.current_turn)
            current.deliver_caller_speech("hola")
            current.deliver_failure(
                session.current_turn,
                ProviderFailure(category=ProviderFailureCategory.TIMEOUT),
            )
            telephony.complete_transfer(session.call_id, TransferResult.ACCEPTED_BY_PBX)
            core.tick()
            core.monitor.drain()
            self.assertEqual(len(smtp.sent), 1)
            urls = [p["url"] for p in http.posts]
            self.assertIn("https://ops.example.com/prod", urls)
            self.assertTrue(any("api.telegram.org" in url for url in urls))
            self.assertEqual(len(http.posts), 2)
            # Recovery stays local-only per the persisted setting.
            clock.advance(60.0)
            self.assertTrue(core.report_provider_probe(True))
            core.monitor.drain()
            self.assertEqual(len(smtp.sent), 1)
            self.assertEqual(len(http.posts), 2)
            recoveries = [
                t
                for t in core.monitor.history()
                if t.code == "provider.circuit_open"
                and t.kind == TransitionKind.RECOVERED
            ]
            self.assertEqual(len(recoveries), 1)
            self.assertEqual(core.health.status, HealthStatus.READY)
            core.close()

    def test_transition_timestamps_follow_fake_clock(self) -> None:
        sink = RecordingSink()
        core, telephony, voice, clock, monitor = self.make_core(sinks=[sink])
        clock.advance(42.0)
        transition = monitor.report_unhealthy(
            HealthComponent.PROVIDER, "provider.circuit_open"
        )
        assert transition is not None
        self.assertEqual(transition.timestamp, 42.0)
        monitor.drain()
        self.assertEqual(sink.received[0].timestamp, 42.0)


if __name__ == "__main__":
    unittest.main()
