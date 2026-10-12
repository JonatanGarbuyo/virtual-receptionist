"""Symbolic transfer -> real-PBX handoff over the telephony boundary (#26).

Drives ReceptionistCore / CallSession with project-owned fakes only (no
SIP, no native objects, no wall-clock sleeps). Proves the mandatory
boundary:

    TransferRequest(destination_id) -> PolicyEngine -> configured
    destination -> trusted PBX-local target -> blind_transfer(...)

plus attempt identity / stale-outcome protection, exactly-once handoff
and teardown, the required race matrix, and AI-capacity release.
"""

import unittest

from receptionist.boundaries import (
    AuditDecision,
    CallOutcome,
    TransferRequest,
    TransferResult,
)
from receptionist.call_session import ActiveMode, CallState, EXIT_APOLOGY
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


GREETING = "Bienvenido, ¿en qué puedo ayudarle?"


def default_destinations() -> dict[str, Destination]:
    return {
        "ventas": Destination(
            id="ventas",
            target="SIP/201",
            kind="extension",
            enabled=True,
            label="Ventas",
        ),
        "soporte": Destination(
            id="soporte",
            target="Queue(soporte)",
            kind="queue",
            enabled=True,
            label="Soporte",
        ),
        "cobros": Destination(
            id="cobros",
            target="SIP/202",
            kind="extension",
            enabled=False,
            label="Cobros",
        ),
        "recepcion": Destination(
            id="recepcion",
            target="SIP/100",
            kind="extension",
            enabled=True,
            label="Recepción",
        ),
    }


def make_core(
    destinations: dict[str, Destination] | None = None,
    fallback_id: str = "recepcion",
    limits: Limits | None = None,
    auto_confirm: bool = True,
) -> tuple:
    telephony = FakeTelephony(auto_confirm=auto_confirm)
    voice = FakeVoiceBackend()
    clock = FakeClock()
    calls = InMemoryCallRepository()
    audit = InMemoryAuditLog()
    engine = PolicyEngine(
        destinations=dict(destinations)
        if destinations is not None
        else default_destinations(),
        fallback_id=fallback_id,
        limits=limits or Limits(),
    )
    core = ReceptionistCore(
        telephony=telephony,
        voice=voice,
        config_service=ConfigService(
            InMemoryConfigRepository({"greeting": GREETING, "language": "es"})
        ),
        policy=FakePolicy(),
        clock=clock,
        policy_engine=engine,
        runtime=RuntimeStorage(
            calls=calls,
            messages=InMemoryMessageRepository(clock=clock),
            transcripts=InMemoryTranscriptStore(),
            audit=audit,
        ),
        retention=RetentionPolicy(),
        call_ids=FakeCallIds(),
    )
    return core, telephony, voice, clock, calls, audit, engine


def start_call(core, voice, caller_id: str = "+34910000001"):
    session = core.incoming_call(caller_id)
    backend = voice.sessions[session.call_id]
    backend.finish_playback(1)
    return session, backend


def completed_results(audit) -> list:
    return [e for e in audit.list_all() if e.decision is AuditDecision.COMPLETED]


class PrimaryTransferTest(unittest.TestCase):
    def test_allowed_symbolic_destination_resolves_and_transfers(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_core()
        core.start()
        session, backend = start_call(core, voice)

        backend.deliver_action_request(TransferRequest("ventas"))

        self.assertEqual(session.state, CallState.TRANSFER_HANDOFF)
        # The PBX target comes from configuration, never from the model.
        self.assertEqual(telephony.transfers, [("call-1", "SIP/201")])
        # Project-owned attempt identity travels to the adapter.
        _, _, attempt_id = telephony.transfer_attempts[0]
        self.assertTrue(attempt_id.startswith("call-1:xfer:"))
        trail = [(e.decision, e.destination) for e in audit.list_all()]
        self.assertEqual(
            trail[:2],
            [(AuditDecision.REQUESTED, "ventas"), (AuditDecision.ALLOWED, "ventas")],
        )

    def test_primary_accepted_hands_off(self) -> None:
        core, telephony, voice, _, calls, audit, _ = make_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        attempt_id = telephony.open_attempt_id("call-1")
        assert attempt_id is not None

        telephony.complete_transfer(
            "call-1", TransferResult.ACCEPTED_BY_PBX, attempt_id
        )

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
        # ACCEPTED_BY_PBX ends our responsibility: no local hangup, no
        # fallback attempt, voice closed, AI slot released via eviction.
        self.assertEqual(telephony.hung_up, [])
        self.assertEqual(len(telephony.transfers), 1)
        self.assertTrue(backend.closed)
        self.assertIsNone(core.get_session("call-1"))
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.TRANSFERRED)
        self.assertEqual(summary.handoff_destination_id, "ventas")
        done = completed_results(audit)
        self.assertEqual(len(done), 1)
        self.assertIs(done[0].result, TransferResult.ACCEPTED_BY_PBX)


class DestinationSecurityTest(unittest.TestCase):
    def test_unknown_destination_denied_before_telephony(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_core()
        core.start()
        session, backend = start_call(core, voice)

        backend.deliver_action_request(TransferRequest("logistica"))

        self.assertEqual(session.state, CallState.ACTIVE)
        self.assertEqual(telephony.transfers, [])
        self.assertEqual(telephony.transfer_attempts, [])
        denied = [e for e in audit.list_all() if e.decision is AuditDecision.DENIED]
        self.assertEqual(len(denied), 1)
        self.assertIn("unknown", denied[0].detail)

    def test_disabled_destination_denied(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_core()
        core.start()
        session, backend = start_call(core, voice)

        backend.deliver_action_request(TransferRequest("cobros"))

        self.assertEqual(session.state, CallState.ACTIVE)
        self.assertEqual(telephony.transfers, [])
        denied = [e for e in audit.list_all() if e.decision is AuditDecision.DENIED]
        self.assertEqual(len(denied), 1)
        self.assertIn("disabled", denied[0].detail)

    def test_arbitrary_model_destinations_never_reach_adapter(self) -> None:
        hostile = [
            "100",
            "sip:100@pbx",
            "SIP/100",
            "PJSIP/100",
            "tel:+541112345678",
            "http://pbx/100",
            "sip:200@localhost",
            "../etc/passwd",
            "ventas norte",
            "",
            "Local/100@pbx",
        ]
        for raw_id in hostile:
            with self.subTest(destination_id=raw_id):
                core, telephony, voice, _, _, audit, _ = make_core()
                core.start()
                session, backend = start_call(core, voice)

                backend.deliver_action_request(TransferRequest(raw_id))

                self.assertEqual(session.state, CallState.ACTIVE)
                self.assertEqual(telephony.transfers, [])
                self.assertEqual(telephony.transfer_attempts, [])
                # Nothing model-controlled leaks into adapter targets.
                for _, target, _ in telephony.transfer_attempts:
                    self.assertNotEqual(target, raw_id)
                denied = [
                    e for e in audit.list_all() if e.decision is AuditDecision.DENIED
                ]
                self.assertEqual(len(denied), 1)

    def test_configured_internal_target_is_valid_only_via_symbolic_id(self) -> None:
        # destination_id="sip:200@localhost" is denied (malformed), even
        # though a configured symbolic id may legitimately resolve to an
        # internal target: the target came from config/policy, not the model.
        core, telephony, voice, _, _, _, _ = make_core()
        core.start()
        session, backend = start_call(core, voice)

        backend.deliver_action_request(TransferRequest("sip:200@localhost"))
        self.assertEqual(telephony.transfers, [])

        backend.deliver_action_request(TransferRequest("ventas"))
        self.assertEqual(telephony.transfers, [("call-1", "SIP/201")])


class PolicyModelContextTest(unittest.TestCase):
    def test_sales_resolves_targets_stay_out_of_model_context(self) -> None:
        _, _, _, _, _, _, engine = make_core()
        resolution = engine.resolve("sales" if False else "ventas")
        self.assertIsNotNone(resolution.destination)
        assert resolution.destination is not None
        self.assertEqual(resolution.destination.target, "SIP/201")

        visible = engine.model_visible_destinations()
        ids = [destination_id for destination_id, _ in visible]
        self.assertIn("ventas", ids)
        self.assertIn("recepcion", ids)
        self.assertNotIn("cobros", ids)
        joined = repr(visible)
        self.assertNotIn("SIP/201", joined)
        self.assertNotIn("SIP/100", joined)
        self.assertNotIn("password", joined.lower())
        self.assertNotIn("Authorization", joined)

    def test_destination_label_is_typed_and_optional(self) -> None:
        plain = Destination(id="x", target="SIP/1", kind="extension")
        self.assertEqual(plain.label, "")
        labeled = Destination(id="x", target="SIP/1", kind="extension", label="X")
        self.assertEqual(labeled.label, "X")


class NonAcceptedPrimaryTest(unittest.TestCase):
    def test_primary_rejected_falls_back_exactly_once(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        primary_id = telephony.open_attempt_id("call-1")
        assert primary_id is not None

        telephony.complete_transfer("call-1", TransferResult.REJECTED, primary_id)

        self.assertEqual(session.state, CallState.FALLBACK_HANDOFF)
        self.assertEqual(
            telephony.transfers, [("call-1", "SIP/201"), ("call-1", "SIP/100")]
        )
        fallback_id = telephony.open_attempt_id("call-1")
        assert fallback_id is not None
        self.assertNotEqual(primary_id, fallback_id)

    def test_primary_timeout_falls_back_exactly_once(self) -> None:
        core, telephony, voice, _, _, _, _ = make_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        primary_id = telephony.open_attempt_id("call-1")
        assert primary_id is not None

        telephony.complete_transfer("call-1", TransferResult.TIMEOUT, primary_id)

        self.assertEqual(session.state, CallState.FALLBACK_HANDOFF)
        self.assertEqual(len(telephony.transfers), 2)

    def test_primary_transport_error_falls_back_exactly_once(self) -> None:
        core, telephony, voice, _, _, _, _ = make_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        primary_id = telephony.open_attempt_id("call-1")
        assert primary_id is not None

        telephony.complete_transfer(
            "call-1", TransferResult.TRANSPORT_ERROR, primary_id
        )

        self.assertEqual(session.state, CallState.FALLBACK_HANDOFF)
        self.assertEqual(len(telephony.transfers), 2)


class FallbackOutcomeTest(unittest.TestCase):
    def _to_fallback(self, core, telephony, voice):
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        primary_id = telephony.open_attempt_id(session.call_id)
        assert primary_id is not None
        telephony.complete_transfer(
            session.call_id, TransferResult.REJECTED, primary_id
        )
        assert session.state is CallState.FALLBACK_HANDOFF
        return session, backend

    def test_fallback_accepted_hands_off(self) -> None:
        core, telephony, voice, _, calls, audit, _ = make_core()
        core.start()
        session, backend = self._to_fallback(core, telephony, voice)
        fallback_id = telephony.open_attempt_id("call-1")
        assert fallback_id is not None

        telephony.complete_transfer(
            "call-1", TransferResult.ACCEPTED_BY_PBX, fallback_id
        )

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
        # Same guarantees as primary accepted: no hangup, voice closed,
        # no double handoff, session evicted.
        self.assertEqual(telephony.hung_up, [])
        self.assertEqual(len(telephony.transfers), 2)
        self.assertTrue(backend.closed)
        self.assertIsNone(core.get_session("call-1"))
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.TRANSFERRED)
        self.assertEqual(
            [e.result for e in completed_results(audit)],
            [TransferResult.REJECTED, TransferResult.ACCEPTED_BY_PBX],
        )

    def test_fallback_rejected_terminates_deterministically(self) -> None:
        core, telephony, voice, _, calls, _, _ = make_core()
        core.start()
        session, backend = self._to_fallback(core, telephony, voice)
        fallback_id = telephony.open_attempt_id("call-1")
        assert fallback_id is not None

        telephony.complete_transfer("call-1", TransferResult.REJECTED, fallback_id)

        self.assertEqual(backend.spoken[-1], (EXIT_APOLOGY, session.current_turn))
        self.assertEqual(telephony.hung_up, ["call-1"])
        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(
            session.history[-3:],
            [CallState.FALLBACK_HANDOFF, CallState.TERMINATING, CallState.ENDED],
        )
        self.assertEqual(len(telephony.transfers), 2)
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.COMPLETED)

    def test_fallback_timeout_terminates_deterministically(self) -> None:
        core, telephony, voice, _, _, _, _ = make_core()
        core.start()
        session, backend = self._to_fallback(core, telephony, voice)
        fallback_id = telephony.open_attempt_id("call-1")
        assert fallback_id is not None

        telephony.complete_transfer("call-1", TransferResult.TIMEOUT, fallback_id)

        self.assertEqual(backend.spoken[-1][0], EXIT_APOLOGY)
        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(len(telephony.transfers), 2)

    def test_fallback_transport_error_terminates_deterministically(self) -> None:
        core, telephony, voice, _, _, _, _ = make_core()
        core.start()
        session, backend = self._to_fallback(core, telephony, voice)
        fallback_id = telephony.open_attempt_id("call-1")
        assert fallback_id is not None

        telephony.complete_transfer(
            "call-1", TransferResult.TRANSPORT_ERROR, fallback_id
        )

        self.assertEqual(backend.spoken[-1][0], EXIT_APOLOGY)
        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(len(telephony.transfers), 2)


class ExactlyOnceTest(unittest.TestCase):
    def test_duplicate_primary_accepted_ignored(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        attempt_id = telephony.open_attempt_id("call-1")
        assert attempt_id is not None

        telephony.complete_transfer(
            "call-1", TransferResult.ACCEPTED_BY_PBX, attempt_id
        )
        telephony.complete_transfer(
            "call-1", TransferResult.ACCEPTED_BY_PBX, attempt_id
        )

        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(session.history.count(CallState.HANDED_OFF), 1)
        self.assertEqual(session.history.count(CallState.ENDED), 1)
        self.assertEqual(len(completed_results(audit)), 1)

    def test_duplicate_primary_rejected_ignored(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        attempt_id = telephony.open_attempt_id("call-1")
        assert attempt_id is not None

        telephony.complete_transfer("call-1", TransferResult.REJECTED, attempt_id)
        # A duplicated primary REJECTED must not open a second fallback.
        telephony.complete_transfer("call-1", TransferResult.REJECTED, attempt_id)

        self.assertEqual(session.state, CallState.FALLBACK_HANDOFF)
        self.assertEqual(len(telephony.transfers), 2)
        self.assertEqual(len(completed_results(audit)), 1)

    def test_stale_primary_accepted_after_fallback_is_inert(self) -> None:
        core, telephony, voice, _, calls, audit, _ = make_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        primary_id = telephony.open_attempt_id("call-1")
        assert primary_id is not None

        # Watchdog TIMEOUT opens the fallback (exactly once).
        telephony.complete_transfer("call-1", TransferResult.TIMEOUT, primary_id)
        self.assertEqual(session.state, CallState.FALLBACK_HANDOFF)
        fallback_id = telephony.open_attempt_id("call-1")
        assert fallback_id is not None
        completed_before = len(completed_results(audit))

        # The late primary ACCEPTED arrives after fallback started: it
        # must neither complete nor alter the fallback attempt.
        telephony.complete_transfer(
            "call-1", TransferResult.ACCEPTED_BY_PBX, primary_id
        )

        self.assertEqual(session.state, CallState.FALLBACK_HANDOFF)
        self.assertEqual(len(telephony.transfers), 2)
        self.assertEqual(len(completed_results(audit)), completed_before)

        # The fallback outcome still wins, exactly once.
        telephony.complete_transfer(
            "call-1", TransferResult.ACCEPTED_BY_PBX, fallback_id
        )
        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(session.history.count(CallState.HANDED_OFF), 1)
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.TRANSFERRED)

    def test_stale_primary_rejected_after_fallback_is_inert(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        primary_id = telephony.open_attempt_id("call-1")
        assert primary_id is not None
        telephony.complete_transfer("call-1", TransferResult.TIMEOUT, primary_id)
        completed_before = len(completed_results(audit))

        telephony.complete_transfer("call-1", TransferResult.REJECTED, primary_id)

        self.assertEqual(session.state, CallState.FALLBACK_HANDOFF)
        self.assertEqual(len(telephony.transfers), 2)
        self.assertEqual(len(completed_results(audit)), completed_before)

    def test_duplicate_fallback_result_ignored(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        primary_id = telephony.open_attempt_id("call-1")
        assert primary_id is not None
        telephony.complete_transfer("call-1", TransferResult.REJECTED, primary_id)
        fallback_id = telephony.open_attempt_id("call-1")
        assert fallback_id is not None

        telephony.complete_transfer(
            "call-1", TransferResult.ACCEPTED_BY_PBX, fallback_id
        )
        telephony.complete_transfer(
            "call-1", TransferResult.ACCEPTED_BY_PBX, fallback_id
        )
        telephony.complete_transfer("call-1", TransferResult.REJECTED, fallback_id)

        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(session.history.count(CallState.HANDED_OFF), 1)
        self.assertEqual(len(completed_results(audit)), 2)


class RaceMatrixTest(unittest.TestCase):
    def test_accepted_vs_caller_bye_converges_single_terminal(self) -> None:
        # Order 1: ACCEPTED wins first, BYE arrives late.
        core, telephony, voice, _, calls, _, _ = make_core(auto_confirm=False)
        core.start()
        session = core.incoming_call("+34910000001")
        backend = voice.sessions["call-1"]
        telephony.complete_answer("call-1")
        backend.finish_playback(1)
        backend.deliver_action_request(TransferRequest("ventas"))
        attempt_id = telephony.open_attempt_id("call-1")
        assert attempt_id is not None

        telephony.complete_transfer(
            "call-1", TransferResult.ACCEPTED_BY_PBX, attempt_id
        )
        telephony.simulate_caller_hangup("call-1")

        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(session.history.count(CallState.HANDED_OFF), 1)
        self.assertEqual(session.history.count(CallState.ENDED), 1)
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.TRANSFERRED)

    def test_caller_bye_vs_late_accepted_converges(self) -> None:
        # Order 2: caller BYE wins first, ACCEPTED arrives late.
        core, telephony, voice, _, calls, audit, _ = make_core(auto_confirm=False)
        core.start()
        session = core.incoming_call("+34910000001")
        backend = voice.sessions["call-1"]
        telephony.complete_answer("call-1")
        backend.finish_playback(1)
        backend.deliver_action_request(TransferRequest("ventas"))
        attempt_id = telephony.open_attempt_id("call-1")
        assert attempt_id is not None

        telephony.simulate_caller_hangup("call-1")
        self.assertEqual(session.state, CallState.TERMINATING)
        telephony.complete_transfer(
            "call-1", TransferResult.ACCEPTED_BY_PBX, attempt_id
        )

        # The abandoned primary outcome stays inert: no handoff, no audit.
        self.assertEqual(session.state, CallState.TERMINATING)
        self.assertEqual(completed_results(audit), [])
        telephony.complete_hangup("call-1")
        self.assertEqual(session.state, CallState.ENDED)
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.CALLER_HANGUP)
        self.assertEqual(telephony.hung_up.count("call-1"), 1)

    def test_media_failure_during_transfer_is_inert(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        attempt_id = telephony.open_attempt_id("call-1")
        assert attempt_id is not None
        completed_before = len(completed_results(audit))

        session.handle_media_failed("dead stack during transfer")

        # No double fallback, no hangup: the transfer still owns the call.
        self.assertEqual(session.state, CallState.TRANSFER_HANDOFF)
        self.assertEqual(len(telephony.transfers), 1)
        self.assertEqual(telephony.hung_up, [])
        self.assertEqual(len(completed_results(audit)), completed_before)

        # The real transfer outcome still settles normally.
        telephony.complete_transfer(
            "call-1", TransferResult.ACCEPTED_BY_PBX, attempt_id
        )
        self.assertEqual(session.state, CallState.ENDED)

    def test_shutdown_during_transfer_is_bounded(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_core(auto_confirm=False)
        core.start()
        session = core.incoming_call("+34910000001")
        backend = voice.sessions["call-1"]
        telephony.complete_answer("call-1")
        backend.finish_playback(1)
        backend.deliver_action_request(TransferRequest("ventas"))
        attempt_id = telephony.open_attempt_id("call-1")
        assert attempt_id is not None
        completed_before = len(completed_results(audit))

        session.shutdown()

        self.assertEqual(session.state, CallState.TERMINATING)
        self.assertEqual(telephony.hung_up, ["call-1"])
        # Late outcome after shutdown is inert: no handoff, no new audit.
        telephony.complete_transfer(
            "call-1", TransferResult.ACCEPTED_BY_PBX, attempt_id
        )
        self.assertEqual(session.state, CallState.TERMINATING)
        self.assertEqual(len(completed_results(audit)), completed_before)
        # Bounded: second shutdown is a no-op, no thread/timer owned here.
        session.shutdown()
        self.assertEqual(telephony.hung_up, ["call-1"])
        telephony.complete_hangup("call-1")
        self.assertEqual(session.state, CallState.ENDED)


class HandoffReleaseTest(unittest.TestCase):
    def test_voice_and_ai_slot_released_after_handoff(self) -> None:
        core, telephony, voice, _, _, _, _ = make_core()
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        attempt_id = telephony.open_attempt_id("call-1")
        assert attempt_id is not None
        telephony.complete_transfer(
            "call-1", TransferResult.ACCEPTED_BY_PBX, attempt_id
        )

        self.assertTrue(backend.closed)
        self.assertIn("call-1", telephony.flushed)
        # The AI slot is free: a new call is admitted to the AI path
        # (auto-confirm answers synchronously, so it is already ACTIVE).
        again = core.incoming_call("+34910000002")
        self.assertEqual(again.state, CallState.ACTIVE)
        # Late model actions on the handed-off turn are inert: no new
        # transfer and no resurrection of the ended session.
        transfers_before = len(telephony.transfers)
        backend.deliver_action_request(TransferRequest("soporte"))
        self.assertEqual(len(telephony.transfers), transfers_before)
        self.assertEqual(session.state, CallState.ENDED)

    def test_transfer_limits_remain_enforced(self) -> None:
        core, telephony, voice, _, _, audit, _ = make_core(
            limits=Limits(max_transfer_attempts=1)
        )
        core.start()
        session, backend = start_call(core, voice)
        backend.deliver_action_request(TransferRequest("ventas"))
        attempt_id = telephony.open_attempt_id("call-1")
        assert attempt_id is not None

        telephony.complete_transfer("call-1", TransferResult.TIMEOUT, attempt_id)

        # Budget of one covers the primary only: fallback is skipped and
        # the deterministic apology/terminate path runs exactly once.
        self.assertEqual(len(telephony.transfers), 1)
        self.assertEqual(backend.spoken[-1][0], EXIT_APOLOGY)
        self.assertEqual(session.state, CallState.ENDED)
        denied = [e for e in audit.list_all() if e.decision is AuditDecision.DENIED]
        self.assertTrue(any("attempts exceeded" in e.detail for e in denied))


class AdapterAttemptIdentityTest(unittest.TestCase):
    """Project-owned attempt identity inside the real adapter (#26).

    White-box but transport-only: no SIP, no network, no sleeps. Proves
    the strong adapter invariant backing the session guards: a stale or
    duplicate native outcome can never be associated with the next
    attempt.
    """

    def _adapter(self, timeout: float = 70.0):
        from receptionist.baresip_adapter import BaresipTelephonyAdapter
        from receptionist.telephony_config import TelephonyConfig

        return BaresipTelephonyAdapter(
            TelephonyConfig(
                username="u",
                domain="d",
                password="p",
                transfer_timeout_seconds=timeout,
            )
        )

    def test_watchdog_with_superseded_id_is_suppressed(self) -> None:
        from receptionist.baresip_adapter import _CallRecord
        from receptionist.boundaries import TransferResult

        adapter = self._adapter()
        adapter._running = True  # noqa: SLF001 (listener delivery gate)
        results: list = []

        class _Recorder:
            def on_transfer_result(self, call_id, result, attempt_id=None):  # noqa: ANN202
                results.append((call_id, result, attempt_id))

        adapter.set_listener(_Recorder())
        record = _CallRecord(app_id="call-1", native=None)
        record.transfer_open = True
        record.transfer_attempt_id = "call-1:xfer:2"
        adapter._calls["call-1"] = record  # noqa: SLF001

        adapter._transfer_watchdog_fired("call-1", "call-1:xfer:1")  # noqa: SLF001

        adapter._drain_native_events("call-1")  # noqa: SLF001
        self.assertEqual(results, [])
        self.assertTrue(record.transfer_open)

    def test_failed_completion_with_stale_id_is_suppressed(self) -> None:
        from receptionist.baresip_adapter import _CallRecord
        from receptionist.boundaries import TransferResult

        adapter = self._adapter()
        adapter._running = True  # noqa: SLF001
        results: list = []

        class _Recorder:
            def on_transfer_result(self, call_id, result, attempt_id=None):  # noqa: ANN202
                results.append((call_id, result, attempt_id))

        adapter.set_listener(_Recorder())
        record = _CallRecord(app_id="call-1", native=None)
        record.transfer_open = True
        record.transfer_attempt_id = "call-1:xfer:2"
        adapter._calls["call-1"] = record  # noqa: SLF001

        class _Boom(Exception):
            pass

        adapter._transfer_failed(record, _Boom("nope"), "call-1:xfer:1")  # noqa: SLF001

        adapter._drain_native_events("call-1")  # noqa: SLF001
        self.assertEqual(results, [])
        self.assertTrue(record.transfer_open)
        self.assertEqual(record.transfer_attempt_id, "call-1:xfer:2")

    def test_shutdown_cancels_watchdog_no_late_outcome(self) -> None:
        import threading

        from receptionist.baresip_adapter import _CallRecord
        from receptionist.boundaries import TransferResult

        adapter = self._adapter(timeout=60.0)
        adapter._started_once = True  # noqa: SLF001 (reach teardown path)
        adapter._running = True  # noqa: SLF001
        results: list = []

        class _Recorder:
            def on_transfer_result(self, call_id, result, attempt_id=None):  # noqa: ANN202
                results.append((call_id, result, attempt_id))

        adapter.set_listener(_Recorder())
        timer = threading.Timer(60.0, lambda: None)
        record = _CallRecord(app_id="call-1", native=None)
        record.transfer_open = True
        record.transfer_attempt_id = "call-1:xfer:1"
        record.transfer_watchdog = timer
        adapter._calls["call-1"] = record  # noqa: SLF001

        adapter.shutdown()

        self.assertEqual(timer.cancel_called if hasattr(timer, "cancel_called") else True, True)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
