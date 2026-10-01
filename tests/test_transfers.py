"""Behavior tests for symbolic transfers and PBX fallback policy (#19).

Same seam as #18 (ReceptionistCore / CallSession), same deterministic
fakes. The model/requester may only ask for typed actions carrying a
symbolic destination id; policy resolves it to a trusted PBX-local
target. No test in this file touches the network, real models, or the
wall clock.
"""

import unittest

from receptionist.boundaries import (
    AuditDecision,
    CallOutcome,
    TransferRequest,
    TransferResult,
)
from receptionist.call_session import ActiveMode, CallState
from receptionist.config import ConfigService, InMemoryConfigRepository
from receptionist.core import ReceptionistCore
from receptionist.health import HealthStatus
from receptionist.persistence import InMemoryAuditLog, InMemoryCallRepository, RuntimeStorage
from receptionist.policy import Destination, Limits, PolicyEngine, RetentionPolicy

from fakes import FakeClock, FakePolicy, FakeTelephony, FakeVoiceBackend


GREETING = "Bienvenido, ¿en qué puedo ayudarle?"


def default_destinations() -> dict[str, Destination]:
    return {
        "ventas": Destination(id="ventas", target="SIP/201", kind="extension", enabled=True),
        "soporte": Destination(id="soporte", target="Queue(soporte)", kind="queue", enabled=True),
        "movil-guardia": Destination(
            id="movil-guardia", target="Local/guardia@pbx", kind="virtual", enabled=True
        ),
        "cobros": Destination(id="cobros", target="SIP/202", kind="extension", enabled=False),
        "recepcion": Destination(
            id="recepcion", target="SIP/100", kind="extension", enabled=True
        ),
    }


def make_transfer_core(
    destinations: dict[str, Destination] | None = None,
    fallback_id: str = "recepcion",
    limits: Limits | None = None,
    config_values: dict | None = None,
    auto_confirm: bool = True,
) -> tuple:
    telephony = FakeTelephony(auto_confirm=auto_confirm)
    voice = FakeVoiceBackend()
    clock = FakeClock()
    calls = InMemoryCallRepository()
    audit = InMemoryAuditLog()
    engine = PolicyEngine(
        destinations=dict(destinations) if destinations is not None else default_destinations(),
        fallback_id=fallback_id,
        limits=limits or Limits(),
    )
    values = {"greeting": GREETING, "language": "es"} if config_values is None else config_values
    core = ReceptionistCore(
        telephony=telephony,
        voice=voice,
        config_service=ConfigService(InMemoryConfigRepository(dict(values))),
        policy=FakePolicy(),
        clock=clock,
        policy_engine=engine,
        runtime=RuntimeStorage(calls=calls, audit=audit),
        retention=RetentionPolicy(),
    )
    return core, telephony, voice, clock, calls, audit, engine


def start_call(core, voice, caller_id: str = "+34910000001"):
    """Drive a call to ACTIVE/LISTENING and return (session, backend)."""
    session = core.incoming_call(caller_id)
    backend = voice.sessions[session.call_id]
    backend.finish_playback(1)
    return session, backend


def audit_trail(audit) -> list[tuple]:
    return [
        (e.decision, e.destination, e.result, e.detail) for e in audit.list_all()
    ]


class ValidTransferTest(unittest.TestCase):
    def test_valid_symbolic_transfer_is_attempted_with_resolved_target(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)

        backend.deliver_action_request(TransferRequest("ventas"))

        self.assertEqual(session.state, CallState.TRANSFER_HANDOFF)
        # The PBX target comes from configuration, never from the model.
        self.assertEqual(telephony.transfers, [("call-1", "SIP/201")])
        trail = audit_trail(audit)
        self.assertEqual(trail[0][:2], (AuditDecision.REQUESTED, "ventas"))
        self.assertEqual(trail[1][:2], (AuditDecision.ALLOWED, "ventas"))
        self.assertIn("SIP/201", trail[1][3])

    def test_accepted_handoff_ends_receptionist_responsibility(self) -> None:
        core, telephony, voice, _, calls, audit, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("soporte"))

        telephony.complete_transfer("call-1", TransferResult.ACCEPTED_BY_PBX)

        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(
            session.history,
            [
                CallState.INCOMING,
                CallState.ANSWERING,
                CallState.ACTIVE,
                CallState.TRANSFER_HANDOFF,
                CallState.HANDED_OFF,
                CallState.ENDED,
            ],
        )
        # The PBX owns the call now: no local hangup, voice closed.
        self.assertEqual(telephony.hung_up, [])
        self.assertTrue(backend.closed)
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.TRANSFERRED)
        completed = [e for e in audit.list_all() if e.decision is AuditDecision.COMPLETED]
        self.assertEqual(len(completed), 1)
        self.assertIs(completed[0].result, TransferResult.ACCEPTED_BY_PBX)

    def test_configured_kinds_transfer_verbatim(self) -> None:
        for destination_id, expected_target in [
            ("ventas", "SIP/201"),
            ("soporte", "Queue(soporte)"),
            ("movil-guardia", "Local/guardia@pbx"),
        ]:
            with self.subTest(destination=destination_id):
                core, telephony, voice, _, _, _, _ = make_transfer_core()
                core.start()
                _, backend = start_call(core, voice)

                backend.deliver_action_request(TransferRequest(destination_id))

                self.assertEqual(telephony.transfers, [("call-1", expected_target)])


class FailClosedTest(unittest.TestCase):
    def test_unknown_destination_is_denied(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)

        backend.deliver_action_request(TransferRequest("logistica"))

        self.assertEqual(session.state, CallState.ACTIVE)
        self.assertEqual(session.mode, ActiveMode.LISTENING)
        self.assertEqual(telephony.transfers, [])
        trail = audit_trail(audit)
        self.assertEqual(
            [(d, dest) for d, dest, _, _ in trail],
            [(AuditDecision.REQUESTED, "logistica"), (AuditDecision.DENIED, "logistica")],
        )
        self.assertIn("unknown", trail[1][3])

    def test_disabled_destination_is_denied(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)

        backend.deliver_action_request(TransferRequest("cobros"))

        self.assertEqual(session.state, CallState.ACTIVE)
        self.assertEqual(telephony.transfers, [])
        trail = audit_trail(audit)
        self.assertEqual(trail[1][0], AuditDecision.DENIED)
        self.assertIn("disabled", trail[1][3])

    def test_malformed_ids_are_denied(self) -> None:
        for raw_id in [
            "sip:evil@example.com",
            "+34666123456",
            "tel:123",
            "",
            "   ",
            "../etc/passwd",
            "ventas norte",
            "Ventas",
        ]:
            with self.subTest(raw_id=raw_id):
                core, telephony, voice, _, _, audit, _ = make_transfer_core()
                core.start()
                session, backend = start_call(core, voice)

                backend.deliver_action_request(TransferRequest(raw_id))

                self.assertEqual(session.state, CallState.ACTIVE)
                self.assertEqual(telephony.transfers, [])
                trail = audit_trail(audit)
                self.assertEqual(trail[-1][0], AuditDecision.DENIED)
                self.assertIn(trail[-1][3], ("malformed", "unknown"))

    def test_non_string_destination_id_is_denied(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)

        backend.deliver_action_request(TransferRequest(12345))  # type: ignore[arg-type]

        self.assertEqual(session.state, CallState.ACTIVE)
        self.assertEqual(telephony.transfers, [])
        self.assertEqual(audit.list_all()[-1].decision, AuditDecision.DENIED)

    def test_unknown_action_type_is_denied(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)

        backend.deliver_action_request("transfer ventas")

        self.assertEqual(session.state, CallState.ACTIVE)
        self.assertEqual(session.mode, ActiveMode.LISTENING)
        self.assertEqual(telephony.transfers, [])
        denied = [e for e in audit.list_all() if e.decision is AuditDecision.DENIED]
        self.assertEqual(len(denied), 1)
        self.assertIn("malformed action", denied[0].detail)

    def test_request_out_of_state_is_denied(self) -> None:
        # GREETING: greeting not finished yet.
        core, telephony, voice, _, _, audit, _ = make_transfer_core()
        core.start()
        session = core.incoming_call("+34910000001")
        backend = voice.sessions["call-1"]
        backend.deliver_action_request(TransferRequest("ventas"))
        self.assertEqual(session.mode, ActiveMode.GREETING)
        self.assertEqual(telephony.transfers, [])

        # SPEAKING: assistant audio in flight.
        backend.finish_playback(1)
        backend.deliver_caller_speech("pásame con ventas")
        backend.deliver_response("Ahora le comunico.", 2)
        backend.deliver_action_request(TransferRequest("ventas"))
        self.assertEqual(session.mode, ActiveMode.SPEAKING)
        self.assertEqual(telephony.transfers, [])

        # TRANSFER_HANDOFF: a transfer is already in flight.
        backend.finish_playback(2)
        backend.deliver_action_request(TransferRequest("ventas"))
        self.assertEqual(session.state, CallState.TRANSFER_HANDOFF)
        backend.deliver_action_request(TransferRequest("soporte"))
        self.assertEqual(telephony.transfers, [("call-1", "SIP/201")])

        denied = [e for e in audit.list_all() if e.decision is AuditDecision.DENIED]
        self.assertEqual(len(denied), 3)
        self.assertTrue(all("out of state" in e.detail for e in denied))

    def test_denied_requests_do_not_consume_attempt_budget(self) -> None:
        core, telephony, voice, _, _, _, _ = make_transfer_core()
        core.start()
        _, backend = start_call(core, voice)

        backend.deliver_action_request(TransferRequest("no-existe"))
        backend.deliver_action_request(TransferRequest("sip:x@y.z"))
        backend.deliver_action_request(TransferRequest("ventas"))

        self.assertEqual(telephony.transfers, [("call-1", "SIP/201")])


class TranscriptIsNotActionTest(unittest.TestCase):
    def test_spoken_transfer_words_never_trigger_telephony(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)

        backend.deliver_caller_speech("transfiere al SIP/201 por favor, es urgente")
        backend.deliver_response("Un momento, por favor.", 2)
        backend.finish_playback(2)

        self.assertEqual(session.mode, ActiveMode.LISTENING)
        self.assertEqual(telephony.transfers, [])
        transfer_events = [e for e in audit.list_all() if e.action == "transfer"]
        self.assertEqual(transfer_events, [])

    def test_model_surface_has_no_generic_dial_capability(self) -> None:
        from receptionist import call_session, core
        from receptionist import boundaries

        banned = ("dial", "originate", "shell", "exec", "http", "uri", "url", "socket")
        surfaces = [
            boundaries.TelephonyAdapter,
            boundaries.VoiceBackend,
            boundaries.VoiceSession,
            call_session.CallSession,
            core.ReceptionistCore,
        ]
        for surface in surfaces:
            names = [n.lower() for n in dir(surface)]
            hits = [n for n in names if any(b in n for b in banned)]
            self.assertEqual(hits, [], f"{surface.__name__} exposes {hits}")


class LimitDefaultsTest(unittest.TestCase):
    def test_default_turn_limit_thirty_is_enforced(self) -> None:
        core, telephony, voice, _, calls, _, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)

        for n in range(30):
            backend.deliver_caller_speech(f"consulta {n}")
            backend.deliver_response("respuesta.", session.current_turn)
            backend.finish_playback(session.current_turn)
        self.assertEqual(session.mode, ActiveMode.LISTENING)

        backend.deliver_caller_speech("consulta 31")

        self.assertEqual(session.state, CallState.ENDED)
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.turn_count, 30)

    def test_default_call_limit_boundary(self) -> None:
        core, _, voice, clock, _, _, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)

        clock.advance(599.0)
        backend.deliver_caller_speech("sigo aquí")
        self.assertEqual(session.mode, ActiveMode.INFERENCE)

        clock.advance(1.0)
        backend.deliver_caller_speech("sigo aquí todavía")
        self.assertEqual(session.state, CallState.ENDED)


class TurnLimitTest(unittest.TestCase):
    def test_third_utterance_ends_call_when_max_turns_is_two(self) -> None:
        core, telephony, voice, _, calls, _, _ = make_transfer_core(
            limits=Limits(max_turns=2)
        )
        core.start()
        session, backend = start_call(core, voice)

        for text in ("primera consulta", "segunda consulta"):
            backend.deliver_caller_speech(text)
            backend.deliver_response("respuesta.", session.current_turn)
            backend.finish_playback(session.current_turn)
        self.assertEqual(session.mode, ActiveMode.LISTENING)

        backend.deliver_caller_speech("tercera consulta")

        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(
            session.history[-2:], [CallState.TERMINATING, CallState.ENDED]
        )
        self.assertEqual(telephony.hung_up, ["call-1"])
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.COMPLETED)
        self.assertEqual(summary.turn_count, 2)


class CallDurationLimitTest(unittest.TestCase):
    def test_call_over_limit_terminates_on_next_event(self) -> None:
        core, telephony, voice, clock, calls, _, _ = make_transfer_core(
            limits=Limits(max_call_seconds=60.0)
        )
        core.start()
        session, backend = start_call(core, voice)

        clock.advance(59.0)
        backend.deliver_caller_speech("sigo aquí")
        self.assertEqual(session.mode, ActiveMode.INFERENCE)

        clock.advance(2.0)
        backend.deliver_caller_speech("sigo aquí todavía")

        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(telephony.hung_up, ["call-1"])
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.ended_at, 61.0)

    def test_exact_limit_boundary_terminates(self) -> None:
        core, _, voice, clock, _, _, _ = make_transfer_core(
            limits=Limits(max_call_seconds=60.0)
        )
        core.start()
        session, backend = start_call(core, voice)

        clock.advance(60.0)
        backend.deliver_caller_speech("hola")

        self.assertEqual(session.state, CallState.ENDED)


class TransferBudgetTest(unittest.TestCase):
    def test_zero_budget_denies_every_request(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_transfer_core(
            limits=Limits(max_transfer_attempts=0)
        )
        core.start()
        session, backend = start_call(core, voice)

        backend.deliver_action_request(TransferRequest("ventas"))

        self.assertEqual(session.state, CallState.ACTIVE)
        self.assertEqual(session.mode, ActiveMode.LISTENING)
        self.assertEqual(telephony.transfers, [])
        trail = audit_trail(audit)
        self.assertEqual(trail[1][0], AuditDecision.DENIED)
        self.assertIn("attempts exceeded", trail[1][3])


EXIT_APOLOGY = "Lo siento, no fue posible comunicarle. La llamada terminará."


class FallbackTest(unittest.TestCase):
    def test_failed_transfer_routes_to_configured_fallback(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))

        telephony.complete_transfer("call-1", TransferResult.TIMEOUT)

        self.assertEqual(session.state, CallState.FALLBACK_HANDOFF)
        self.assertEqual(
            telephony.transfers, [("call-1", "SIP/201"), ("call-1", "SIP/100")]
        )
        trail = audit_trail(audit)
        self.assertEqual(
            [(d, dest) for d, dest, _, _ in trail],
            [
                (AuditDecision.REQUESTED, "ventas"),
                (AuditDecision.ALLOWED, "ventas"),
                (AuditDecision.COMPLETED, "ventas"),
                (AuditDecision.REQUESTED, "recepcion"),
                (AuditDecision.ALLOWED, "recepcion"),
            ],
        )
        self.assertIs(trail[2][2], TransferResult.TIMEOUT)
        self.assertIn("fallback", trail[3][3])

    def test_all_non_accepted_results_route_to_fallback(self) -> None:
        for result in (
            TransferResult.REJECTED,
            TransferResult.TIMEOUT,
            TransferResult.TRANSPORT_ERROR,
        ):
            with self.subTest(result=result):
                core, telephony, voice, _, _, _, _ = make_transfer_core()
                core.start()
                _, backend = start_call(core, voice)
                backend.deliver_action_request(TransferRequest("ventas"))

                telephony.complete_transfer("call-1", result)

                self.assertEqual(
                    telephony.transfers,
                    [("call-1", "SIP/201"), ("call-1", "SIP/100")],
                )

    def test_accepted_fallback_hands_off(self) -> None:
        core, telephony, voice, _, calls, audit, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        telephony.complete_transfer("call-1", TransferResult.TIMEOUT)

        telephony.complete_transfer("call-1", TransferResult.ACCEPTED_BY_PBX)

        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(
            session.history,
            [
                CallState.INCOMING,
                CallState.ANSWERING,
                CallState.ACTIVE,
                CallState.TRANSFER_HANDOFF,
                CallState.FALLBACK_HANDOFF,
                CallState.HANDED_OFF,
                CallState.ENDED,
            ],
        )
        self.assertEqual(telephony.hung_up, [])
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.TRANSFERRED)
        completed = [e for e in audit.list_all() if e.decision is AuditDecision.COMPLETED]
        self.assertEqual(
            [e.result for e in completed],
            [TransferResult.TIMEOUT, TransferResult.ACCEPTED_BY_PBX],
        )

    def test_failed_fallback_takes_single_exit_path(self) -> None:
        core, telephony, voice, _, calls, audit, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        telephony.complete_transfer("call-1", TransferResult.REJECTED)

        telephony.complete_transfer("call-1", TransferResult.TRANSPORT_ERROR)

        self.assertEqual(backend.spoken[-1], (EXIT_APOLOGY, session.current_turn))
        self.assertEqual(telephony.hung_up, ["call-1"])
        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(
            session.history[-3:],
            [CallState.FALLBACK_HANDOFF, CallState.TERMINATING, CallState.ENDED],
        )
        # No third recovery chain: exactly two telephony transfer ops.
        self.assertEqual(len(telephony.transfers), 2)
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.COMPLETED)

    def test_budget_one_skips_fallback_straight_to_exit(self) -> None:
        core, telephony, voice, _, _, _, _ = make_transfer_core(
            limits=Limits(max_transfer_attempts=1)
        )
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))

        telephony.complete_transfer("call-1", TransferResult.TIMEOUT)

        self.assertEqual(len(telephony.transfers), 1)
        self.assertEqual(backend.spoken[-1][0], EXIT_APOLOGY)
        self.assertEqual(session.state, CallState.ENDED)

    def test_unresolvable_fallback_goes_straight_to_exit(self) -> None:
        core, telephony, voice, _, _, _, _ = make_transfer_core(
            fallback_id="inexistente"
        )
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))

        telephony.complete_transfer("call-1", TransferResult.TIMEOUT)

        self.assertEqual(len(telephony.transfers), 1)
        self.assertEqual(backend.spoken[-1][0], EXIT_APOLOGY)
        self.assertEqual(session.state, CallState.ENDED)


class TransferRaceTest(unittest.TestCase):
    def test_stale_result_after_hangup_is_ignored(self) -> None:
        core, telephony, voice, _, calls, audit, _ = make_transfer_core(
            auto_confirm=False
        )
        core.start()
        session, backend = start_call(core, voice)
        telephony.complete_answer("call-1")
        backend.finish_playback(1)
        backend.deliver_action_request(TransferRequest("ventas"))

        telephony.simulate_caller_hangup("call-1")
        self.assertEqual(session.state, CallState.TERMINATING)
        telephony.complete_transfer("call-1", TransferResult.ACCEPTED_BY_PBX)

        self.assertEqual(session.state, CallState.TERMINATING)
        self.assertEqual(
            [e for e in audit.list_all() if e.decision is AuditDecision.COMPLETED],
            [],
        )
        telephony.complete_hangup("call-1")
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.CALLER_HANGUP)

    def test_repeated_result_is_ignored(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        telephony.complete_transfer("call-1", TransferResult.ACCEPTED_BY_PBX)

        telephony.complete_transfer("call-1", TransferResult.ACCEPTED_BY_PBX)

        self.assertEqual(session.state, CallState.ENDED)
        completed = [e for e in audit.list_all() if e.decision is AuditDecision.COMPLETED]
        self.assertEqual(len(completed), 1)

    def test_result_for_unknown_call_is_ignored(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)

        telephony.complete_transfer("call-99", TransferResult.ACCEPTED_BY_PBX)

        self.assertEqual(session.state, CallState.ACTIVE)
        self.assertEqual(
            [e for e in audit.list_all() if e.decision is AuditDecision.COMPLETED],
            [],
        )

    def test_direct_result_without_pending_transfer_is_ignored(self) -> None:
        core, _, voice, _, _, audit, _ = make_transfer_core()
        core.start()
        session, _ = start_call(core, voice)

        session.handle_transfer_result(TransferResult.ACCEPTED_BY_PBX)

        self.assertEqual(session.state, CallState.ACTIVE)
        self.assertEqual(session.mode, ActiveMode.LISTENING)
        self.assertEqual(
            [e for e in audit.list_all() if e.decision is AuditDecision.COMPLETED],
            [],
        )


class AuditCompletenessTest(unittest.TestCase):
    def test_full_chain_is_audited_without_transcript_content(self) -> None:
        core, telephony, voice, clock, _, audit, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)
        clock.advance(5.0)
        backend.deliver_caller_speech("soy el señor García, pásame con ventas por favor")
        backend.deliver_action_request(TransferRequest("ventas"))
        clock.advance(4.0)
        telephony.complete_transfer("call-1", TransferResult.ACCEPTED_BY_PBX)

        trail = audit_trail(audit)
        self.assertEqual(
            [(d, dest) for d, dest, _, _ in trail],
            [
                (AuditDecision.REQUESTED, "ventas"),
                (AuditDecision.ALLOWED, "ventas"),
                (AuditDecision.COMPLETED, "ventas"),
            ],
        )
        self.assertEqual([e.timestamp for e in audit.list_all()], [5.0, 5.0, 9.0])
        leaked = [
            e
            for e in audit.list_all()
            if "García" in e.destination or "García" in e.detail
        ]
        self.assertEqual(leaked, [])

    def test_denied_paths_leave_no_allowed_or_completed(self) -> None:
        core, _, voice, _, _, audit, _ = make_transfer_core()
        core.start()
        _, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("nadie"))

        decisions = [e.decision for e in audit.list_all()]
        self.assertEqual(
            decisions, [AuditDecision.REQUESTED, AuditDecision.DENIED]
        )


class DeadlineTickTest(unittest.TestCase):
    def test_tick_terminates_silent_call_past_deadline(self) -> None:
        core, telephony, voice, clock, calls, _, _ = make_transfer_core()
        core.start()
        session, _ = start_call(core, voice)

        clock.advance(601.0)
        core.tick()

        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(
            session.history[-2:], [CallState.TERMINATING, CallState.ENDED]
        )
        self.assertEqual(telephony.hung_up, ["call-1"])
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.COMPLETED)
        self.assertEqual(summary.ended_at, 601.0)

    def test_tick_before_deadline_is_noop(self) -> None:
        core, telephony, voice, clock, _, _, _ = make_transfer_core()
        core.start()
        session, _ = start_call(core, voice)

        clock.advance(599.0)
        core.tick()

        self.assertEqual(session.state, CallState.ACTIVE)
        self.assertEqual(session.mode, ActiveMode.LISTENING)
        self.assertEqual(telephony.hung_up, [])

    def test_tick_terminates_handoff_waiting_past_deadline(self) -> None:
        core, telephony, voice, clock, calls, audit, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))

        clock.advance(601.0)
        core.tick()
        telephony.complete_transfer("call-1", TransferResult.ACCEPTED_BY_PBX)

        self.assertEqual(session.state, CallState.ENDED)
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.COMPLETED)
        self.assertEqual(
            [e for e in audit.list_all() if e.decision is AuditDecision.COMPLETED],
            [],
        )

    def test_action_request_past_deadline_terminates_without_transfer(self) -> None:
        core, telephony, voice, clock, _, _, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)

        clock.advance(601.0)
        backend.deliver_action_request(TransferRequest("ventas"))

        self.assertEqual(telephony.transfers, [])
        self.assertEqual(session.state, CallState.ENDED)


class TransferResultValidationTest(unittest.TestCase):
    def test_non_enum_result_is_ignored_and_audited(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_transfer_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))

        telephony.complete_transfer("call-1", "ACCEPTED_BY_PBX")  # type: ignore[arg-type]

        self.assertEqual(session.state, CallState.TRANSFER_HANDOFF)
        self.assertEqual(len(telephony.transfers), 1)
        denied = [e for e in audit.list_all() if e.decision is AuditDecision.DENIED]
        self.assertEqual(len(denied), 1)
        self.assertEqual(denied[0].destination, "ventas")
        self.assertIn("invalid transfer result", denied[0].detail)
        self.assertNotIn("ACCEPTED_BY_PBX", denied[0].detail)
        self.assertEqual(
            [e for e in audit.list_all() if e.decision is AuditDecision.COMPLETED],
            [],
        )

        telephony.complete_transfer("call-1", TransferResult.ACCEPTED_BY_PBX)
        self.assertEqual(session.state, CallState.ENDED)


class FallbackSkipAuditTest(unittest.TestCase):
    def test_skipped_fallback_by_budget_is_audited(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_transfer_core(
            limits=Limits(max_transfer_attempts=1)
        )
        core.start()
        _, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        telephony.complete_transfer("call-1", TransferResult.TIMEOUT)

        denied = [e for e in audit.list_all() if e.decision is AuditDecision.DENIED]
        self.assertEqual(len(denied), 1)
        self.assertEqual(denied[0].destination, "recepcion")
        self.assertIn("attempts exceeded", denied[0].detail)

    def test_unresolvable_fallback_is_audited(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_transfer_core(
            fallback_id="inexistente"
        )
        core.start()
        _, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        telephony.complete_transfer("call-1", TransferResult.TIMEOUT)

        denied = [e for e in audit.list_all() if e.decision is AuditDecision.DENIED]
        self.assertEqual(len(denied), 1)
        self.assertEqual(denied[0].destination, "inexistente")
        self.assertIn("fallback unknown", denied[0].detail)


if __name__ == "__main__":
    unittest.main()
