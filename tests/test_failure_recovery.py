"""Failure/recovery behavior at the ReceptionistCore / CallSession seam (#22).

Deterministic fault injection with FakeClock and typed provider
failures: retry budgets, stale generations, cancellation, deadlines,
no-input fallback, observability degradation, breaker admission,
capacity saturation, and config-trust loss. No sleeps, no wall clock,
no network, no real models.
"""

import unittest

from receptionist.boundaries import (
    MessageConfirmed,
    MessageTextFinal,
    ProviderFailure,
    ProviderFailureCategory,
    StartMessageCapture,
    StoreUnavailableError,
    TransferRequest,
    TransferResult,
)
from receptionist.call_session import ActiveMode, CallState, NO_INPUT_REPROMPT
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
from receptionist.policy import Destination, Limits, PolicyEngine, RetentionPolicy
from receptionist.resilience import ResilienceConfig

from fakes import (
    FakeCallIds,
    FakeClock,
    FakePolicy,
    FakeTelephony,
    FakeVoiceBackend,
    FailingAuditLog,
    FailingCallRepository,
    FailingTranscriptStore,
    RaisingConfigRepository,
)


GREETING = "Bienvenido, ¿en qué puedo ayudarle?"


def default_destinations() -> dict[str, Destination]:
    return {
        "ventas": Destination(id="ventas", target="SIP/201", kind="extension", enabled=True),
        "recepcion": Destination(
            id="recepcion", target="SIP/100", kind="extension", enabled=True
        ),
    }


def make_core(
    resilience: ResilienceConfig | None = None,
    destinations: dict[str, Destination] | None = None,
    fallback_id: str = "recepcion",
    config_values: dict | None = None,
    calls=None,
    audit=None,
    transcripts=None,
    auto_confirm: bool = True,
    knowledge=None,
):
    telephony = FakeTelephony(auto_confirm=auto_confirm)
    voice = FakeVoiceBackend()
    clock = FakeClock()
    engine = PolicyEngine(
        destinations=dict(destinations) if destinations is not None else default_destinations(),
        fallback_id=fallback_id,
        limits=Limits(),
    )
    values = {"greeting": GREETING, "language": "es"} if config_values is None else config_values
    core = ReceptionistCore(
        telephony=telephony,
        voice=voice,
        config_service=ConfigService(InMemoryConfigRepository(dict(values))),
        policy=FakePolicy(),
        clock=clock,
        policy_engine=engine,
        runtime=RuntimeStorage(
            calls=calls or InMemoryCallRepository(),
            messages=InMemoryMessageRepository(clock=clock),
            transcripts=transcripts or InMemoryTranscriptStore(),
            audit=audit or InMemoryAuditLog(),
        ),
        retention=RetentionPolicy(),
        call_ids=FakeCallIds(),
        resilience=resilience,
        knowledge=knowledge,
    )
    core.start()
    return core, telephony, voice, clock, engine


def open_turn(session, voice_session, text: str = "quiero información") -> int:
    """Drive the session to an outstanding inference. Returns the turn id."""
    voice_session.finish_playback(session.current_turn)
    voice_session.deliver_caller_speech(text)
    assert session.mode is ActiveMode.INFERENCE
    return session.current_turn


def fail(voice_session, turn: int, category: ProviderFailureCategory) -> None:
    voice_session.deliver_failure(turn, ProviderFailure(category=category, detail="test"))


class ProviderRetryTest(unittest.TestCase):
    def test_retryable_categories_open_a_new_attempt(self) -> None:
        for category in (
            ProviderFailureCategory.TIMEOUT,
            ProviderFailureCategory.UNAVAILABLE,
            ProviderFailureCategory.INVALID_OUTPUT,
            ProviderFailureCategory.RESOURCE_EXHAUSTED,
            ProviderFailureCategory.INTERNAL,
        ):
            with self.subTest(category=category):
                core, _, voice, _, _ = make_core()
                session = core.incoming_call("+34910000001")
                first = voice.sessions[session.call_id]
                turn = open_turn(session, first)
                fail(first, turn, category)
                self.assertEqual(len(voice.all_sessions), 2)
                self.assertIs(voice.sessions[session.call_id], voice.all_sessions[1])
                self.assertEqual(session.mode, ActiveMode.INFERENCE)

    def test_cancelled_never_retries_and_never_feeds_breaker(self) -> None:
        core, _, voice, _, _ = make_core()
        session = core.incoming_call("+34910000001")
        first = voice.sessions[session.call_id]
        turn = open_turn(session, first)
        fail(first, turn, ProviderFailureCategory.CANCELLED)
        self.assertEqual(len(voice.all_sessions), 1)
        self.assertFalse(core.breaker.is_open)

    def test_default_budget_one_then_fallback_once(self) -> None:
        core, telephony, voice, _, _ = make_core()
        session = core.incoming_call("+34910000001")
        first = voice.sessions[session.call_id]
        turn = open_turn(session, first)
        fail(first, turn, ProviderFailureCategory.TIMEOUT)
        second = voice.sessions[session.call_id]
        fail(second, turn, ProviderFailureCategory.TIMEOUT)
        # Terminal failure routes to the configured fallback exactly once.
        self.assertEqual(telephony.transfers, [(session.call_id, "SIP/100")])
        fail(second, turn, ProviderFailureCategory.TIMEOUT)
        self.assertEqual(len(telephony.transfers), 1)

    def test_zero_budget_goes_straight_to_fallback(self) -> None:
        core, telephony, voice, _, _ = make_core(
            resilience=ResilienceConfig(provider_retries=0)
        )
        session = core.incoming_call("+34910000001")
        first = voice.sessions[session.call_id]
        turn = open_turn(session, first)
        fail(first, turn, ProviderFailureCategory.UNAVAILABLE)
        self.assertEqual(len(voice.all_sessions), 1)
        self.assertEqual(telephony.transfers, [(session.call_id, "SIP/100")])

    def test_configured_budgets_bound_total_attempts(self) -> None:
        for budget, expected_sessions in ((0, 1), (1, 2), (2, 3), (3, 4)):
            with self.subTest(budget=budget):
                core, _, voice, _, _ = make_core(
                    resilience=ResilienceConfig(provider_retries=budget)
                )
                session = core.incoming_call("+34910000001")
                turn = open_turn(session, voice.sessions[session.call_id])
                for _ in range(budget + 1):
                    fail(voice.sessions[session.call_id], turn, ProviderFailureCategory.TIMEOUT)
                # Initial attempt + retries, never more.
                self.assertEqual(len(voice.all_sessions), expected_sessions)

    def test_turn_deadline_blocks_remaining_retry(self) -> None:
        core, telephony, voice, clock, _ = make_core(
            resilience=ResilienceConfig(provider_retries=2, turn_deadline_seconds=60.0)
        )
        session = core.incoming_call("+34910000001")
        first = voice.sessions[session.call_id]
        turn = open_turn(session, first)
        fail(first, turn, ProviderFailureCategory.TIMEOUT)
        self.assertEqual(len(voice.all_sessions), 2)
        clock.advance(61.0)
        fail(voice.sessions[session.call_id], turn, ProviderFailureCategory.TIMEOUT)
        # Budget remained, but the turn deadline expired: no new attempt.
        self.assertEqual(len(voice.all_sessions), 2)
        self.assertEqual(telephony.transfers, [(session.call_id, "SIP/100")])


class StaleGenerationTest(unittest.TestCase):
    def test_late_events_from_superseded_attempt_are_ignored(self) -> None:
        core, telephony, voice, _, _ = make_core()
        session = core.incoming_call("+34910000001")
        old = voice.sessions[session.call_id]
        turn = open_turn(session, old)
        fail(old, turn, ProviderFailureCategory.TIMEOUT)
        new = voice.sessions[session.call_id]
        self.assertIsNot(old, new)
        # Late assistant text from attempt 1: ignored, nothing spoken.
        old.deliver_response("respuesta vieja", turn)
        self.assertEqual(new.spoken, [])
        self.assertEqual(session.mode, ActiveMode.INFERENCE)
        # Late privileged action from attempt 1: no transfer opens.
        old.deliver_action_request(TransferRequest(destination_id="ventas"))
        self.assertEqual(telephony.transfers, [])
        # Late failure from attempt 1: no further attempt opens.
        old.deliver_failure(turn, ProviderFailure(ProviderFailureCategory.TIMEOUT))
        self.assertEqual(len(voice.all_sessions), 2)
        # The current attempt still works.
        new.deliver_response("respuesta nueva", turn)
        self.assertEqual(session.mode, ActiveMode.SPEAKING)

    def test_late_caller_audio_from_superseded_attempt_is_ignored(self) -> None:
        core, _, voice, _, _ = make_core()
        session = core.incoming_call("+34910000001")
        old = voice.sessions[session.call_id]
        turn = open_turn(session, old)
        fail(old, turn, ProviderFailureCategory.TIMEOUT)
        old.deliver_caller_speech("texto tardío")
        # No new turn opens from the stale attempt.
        self.assertEqual(session.current_turn, turn)
        self.assertEqual(session.mode, ActiveMode.INFERENCE)


class CircuitAdmissionTest(unittest.TestCase):
    def open_breaker(self):
        core, telephony, voice, clock, _ = make_core(
            resilience=ResilienceConfig(
                provider_retries=0,
                breaker_threshold=2,
                breaker_probe_cooldown_seconds=60.0,
            )
        )
        for caller in ("+34910000001", "+34910000002"):
            session = core.incoming_call(caller)
            turn = open_turn(session, voice.sessions[session.call_id])
            fail(voice.sessions[session.call_id], turn, ProviderFailureCategory.TIMEOUT)
            telephony.complete_transfer(session.call_id, TransferResult.ACCEPTED_BY_PBX)
        assert core.breaker.is_open
        return core, telephony, voice, clock

    def test_repeated_terminal_failures_open_the_circuit(self) -> None:
        core, _, voice, _, _ = make_core(
            resilience=ResilienceConfig(provider_retries=0, breaker_threshold=2)
        )
        first = core.incoming_call("+34910000001")
        turn = open_turn(first, voice.sessions[first.call_id])
        fail(voice.sessions[first.call_id], turn, ProviderFailureCategory.UNAVAILABLE)
        self.assertFalse(core.breaker.is_open)
        from receptionist.health import HealthStatus

        self.assertEqual(core.health.status, HealthStatus.READY)

    def test_open_circuit_bypasses_provider_and_uses_fallback(self) -> None:
        core, telephony, voice, _ = self.open_breaker()
        from receptionist.health import HealthStatus

        core.tick()
        self.assertEqual(core.health.status, HealthStatus.DEGRADED)
        session = core.incoming_call("+34910000003")
        # No provider resources opened for the new call (Invariant D).
        self.assertNotIn(session.call_id, voice.sessions)
        self.assertEqual(session.state, CallState.FALLBACK_HANDOFF)
        self.assertEqual(telephony.transfers[-1], (session.call_id, "SIP/100"))

    def test_recovery_probe_closes_and_next_call_uses_provider(self) -> None:
        core, telephony, voice, clock = self.open_breaker()
        self.assertFalse(core.report_provider_probe(True))
        self.assertTrue(core.breaker.is_open)
        clock.advance(60.0)
        self.assertTrue(core.report_provider_probe(True))
        self.assertFalse(core.breaker.is_open)
        from receptionist.health import HealthStatus

        self.assertEqual(core.health.status, HealthStatus.READY)
        session = core.incoming_call("+34910000004")
        self.assertIn(session.call_id, voice.sessions)
        self.assertEqual(
            voice.sessions[session.call_id].spoken[0][0], GREETING
        )

    def test_failed_probe_keeps_circuit_open(self) -> None:
        core, _, voice, clock = self.open_breaker()
        clock.advance(60.0)
        self.assertTrue(core.report_provider_probe(False))
        self.assertTrue(core.breaker.is_open)
        session = core.incoming_call("+34910000005")
        self.assertNotIn(session.call_id, voice.sessions)

    def test_hangup_without_provider_output_keeps_streak(self) -> None:
        core, telephony, voice, _, _ = make_core(
            resilience=ResilienceConfig(provider_retries=0, breaker_threshold=2)
        )
        failed_a = core.incoming_call("+34910000001")
        turn = open_turn(failed_a, voice.sessions[failed_a.call_id])
        fail(voice.sessions[failed_a.call_id], turn, ProviderFailureCategory.TIMEOUT)
        telephony.complete_transfer(failed_a.call_id, TransferResult.ACCEPTED_BY_PBX)
        # B hangs up before any provider output: neutral, not success.
        neutral = core.incoming_call("+34910000002")
        telephony.simulate_caller_hangup(neutral.call_id)
        failed_c = core.incoming_call("+34910000003")
        turn_c = open_turn(failed_c, voice.sessions[failed_c.call_id])
        fail(voice.sessions[failed_c.call_id], turn_c, ProviderFailureCategory.TIMEOUT)
        telephony.complete_transfer(failed_c.call_id, TransferResult.ACCEPTED_BY_PBX)
        self.assertTrue(core.breaker.is_open)

    def test_fallback_only_completion_keeps_streak(self) -> None:
        core, telephony, voice, _, _ = make_core(
            resilience=ResilienceConfig(provider_retries=0, breaker_threshold=2)
        )
        failed_a = core.incoming_call("+34910000001")
        turn = open_turn(failed_a, voice.sessions[failed_a.call_id])
        fail(voice.sessions[failed_a.call_id], turn, ProviderFailureCategory.TIMEOUT)
        # A waits on its fallback transfer, still holding the AI slot.
        saturated = core.incoming_call("+34910000002")
        self.assertNotIn(saturated.call_id, voice.sessions)
        # B never opens the provider at all: neutral, not success.
        telephony.complete_transfer(saturated.call_id, TransferResult.ACCEPTED_BY_PBX)
        telephony.complete_transfer(failed_a.call_id, TransferResult.ACCEPTED_BY_PBX)
        failed_c = core.incoming_call("+34910000003")
        turn_c = open_turn(failed_c, voice.sessions[failed_c.call_id])
        fail(voice.sessions[failed_c.call_id], turn_c, ProviderFailureCategory.TIMEOUT)
        telephony.complete_transfer(failed_c.call_id, TransferResult.ACCEPTED_BY_PBX)
        self.assertTrue(core.breaker.is_open)

    def test_provider_output_then_clean_end_resets_streak(self) -> None:
        core, telephony, voice, _, _ = make_core(
            resilience=ResilienceConfig(provider_retries=0, breaker_threshold=2)
        )
        failed_a = core.incoming_call("+34910000001")
        turn = open_turn(failed_a, voice.sessions[failed_a.call_id])
        fail(voice.sessions[failed_a.call_id], turn, ProviderFailureCategory.TIMEOUT)
        telephony.complete_transfer(failed_a.call_id, TransferResult.ACCEPTED_BY_PBX)
        # B produces real provider output and ends cleanly: success evidence.
        healthy = core.incoming_call("+34910000002")
        healthy_voice = voice.sessions[healthy.call_id]
        turn_b = open_turn(healthy, healthy_voice)
        healthy_voice.deliver_response("información útil", turn_b)
        telephony.simulate_caller_hangup(healthy.call_id)
        failed_c = core.incoming_call("+34910000003")
        turn_c = open_turn(failed_c, voice.sessions[failed_c.call_id])
        fail(voice.sessions[failed_c.call_id], turn_c, ProviderFailureCategory.TIMEOUT)
        telephony.complete_transfer(failed_c.call_id, TransferResult.ACCEPTED_BY_PBX)
        self.assertFalse(core.breaker.is_open)

    def test_concurrent_neutral_end_keeps_streak(self) -> None:
        core, telephony, voice, _, _ = make_core(
            resilience=ResilienceConfig(
                provider_retries=0, breaker_threshold=2, max_ai_sessions=2
            )
        )
        concurrent = core.incoming_call("+34910000009")
        failed_a = core.incoming_call("+34910000001")
        turn = open_turn(failed_a, voice.sessions[failed_a.call_id])
        fail(voice.sessions[failed_a.call_id], turn, ProviderFailureCategory.TIMEOUT)
        telephony.complete_transfer(failed_a.call_id, TransferResult.ACCEPTED_BY_PBX)
        # The concurrent call ends with no provider output: neutral.
        telephony.simulate_caller_hangup(concurrent.call_id)
        failed_c = core.incoming_call("+34910000003")
        turn_c = open_turn(failed_c, voice.sessions[failed_c.call_id])
        fail(voice.sessions[failed_c.call_id], turn_c, ProviderFailureCategory.TIMEOUT)
        telephony.complete_transfer(failed_c.call_id, TransferResult.ACCEPTED_BY_PBX)
        self.assertTrue(core.breaker.is_open)

    def test_non_provider_trouble_never_feeds_the_breaker(self) -> None:
        core, telephony, voice, _, _ = make_core(
            resilience=ResilienceConfig(provider_retries=0, breaker_threshold=1)
        )
        # Caller hangup during an outstanding inference.
        session = core.incoming_call("+34910000001")
        current = voice.sessions[session.call_id]
        turn = open_turn(session, current)
        telephony.simulate_caller_hangup(session.call_id)
        current.deliver_failure(turn, ProviderFailure(ProviderFailureCategory.TIMEOUT))
        self.assertFalse(core.breaker.is_open)
        # Barge-in speech over the assistant is deferred, not provider sickness.
        session2 = core.incoming_call("+34910000002")
        second = voice.sessions[session2.call_id]
        second.deliver_response("hola", session2.current_turn)
        second.deliver_caller_speech("interrumpo")
        self.assertFalse(core.breaker.is_open)
        # Rejected transfer and unknown destination: policy, not provider.
        second.finish_playback(session2.current_turn)
        second.deliver_action_request(TransferRequest(destination_id="SIP/999"))
        second.deliver_action_request(TransferRequest(destination_id="nadie"))
        self.assertEqual(telephony.transfers, [])
        self.assertFalse(core.breaker.is_open)


class CapacitySaturationTest(unittest.TestCase):
    def test_second_call_falls_back_without_provider_and_slot_returns(self) -> None:
        core, telephony, voice, _, _ = make_core()
        first = core.incoming_call("+34910000001")
        self.assertIn(first.call_id, voice.sessions)
        self.assertEqual(core.capacity.active, 1)
        second = core.incoming_call("+34910000002")
        # Saturated: no VoiceBackend session opens (Invariant E holds).
        self.assertNotIn(second.call_id, voice.sessions)
        self.assertEqual(core.capacity.active, 1)
        self.assertEqual(second.state, CallState.FALLBACK_HANDOFF)
        telephony.complete_transfer(second.call_id, TransferResult.ACCEPTED_BY_PBX)
        self.assertEqual(core.capacity.active, 1)
        telephony.simulate_caller_hangup(first.call_id)
        self.assertEqual(core.capacity.active, 0)
        third = core.incoming_call("+34910000003")
        self.assertIn(third.call_id, voice.sessions)

    def test_slot_released_on_handoff(self) -> None:
        core, telephony, voice, _, _ = make_core()
        first = core.incoming_call("+34910000001")
        session_voice = voice.sessions[first.call_id]
        session_voice.finish_playback(first.current_turn)
        session_voice.deliver_action_request(TransferRequest(destination_id="ventas"))
        saturated = core.incoming_call("+34910000002")
        self.assertNotIn(saturated.call_id, voice.sessions)
        telephony.complete_transfer(first.call_id, TransferResult.ACCEPTED_BY_PBX)
        self.assertEqual(core.capacity.active, 0)
        telephony.complete_transfer(saturated.call_id, TransferResult.ACCEPTED_BY_PBX)
        self.assertEqual(core.capacity.active, 0)
        third = core.incoming_call("+34910000003")
        self.assertIn(third.call_id, voice.sessions)

    def test_slot_released_on_failed_fallback_exit(self) -> None:
        core, telephony, voice, _, _ = make_core(
            resilience=ResilienceConfig(provider_retries=0)
        )
        first = core.incoming_call("+34910000001")
        turn = open_turn(first, voice.sessions[first.call_id])
        fail(voice.sessions[first.call_id], turn, ProviderFailureCategory.TIMEOUT)
        telephony.complete_transfer(first.call_id, TransferResult.REJECTED)
        # Fallback itself failed: apology once, then terminate once.
        from receptionist.call_session import EXIT_APOLOGY

        spoken = [text for text, _ in voice.sessions[first.call_id].spoken]
        self.assertEqual(spoken.count(EXIT_APOLOGY), 1)
        self.assertEqual(telephony.hung_up, [first.call_id])
        self.assertEqual(first.state, CallState.ENDED)
        self.assertEqual(core.capacity.active, 0)


class ConfigTrustTest(unittest.TestCase):
    def test_unreadable_config_blocks_startup(self) -> None:
        telephony = FakeTelephony()
        voice = FakeVoiceBackend()
        clock = FakeClock()
        core = ReceptionistCore(
            telephony=telephony,
            voice=voice,
            config_service=ConfigService(
                RaisingConfigRepository(RuntimeError("config.db locked"))
            ),
            policy=FakePolicy(),
            clock=clock,
            policy_engine=PolicyEngine(
                destinations=default_destinations(),
                fallback_id="recepcion",
                limits=Limits(),
            ),
            runtime=RuntimeStorage.create(clock=clock),
            retention=RetentionPolicy(),
            call_ids=FakeCallIds(),
        )
        core.start()
        self.assertEqual(core.health.status, HealthStatus.NOT_READY)
        session = core.incoming_call("+34910000001")
        self.assertNotIn(session.call_id, voice.sessions)
        self.assertEqual(telephony.rejected, [session.call_id])

    def test_mid_call_config_loss_keeps_session_but_blocks_new_ones(self) -> None:
        values = {"greeting": GREETING, "language": "es"}

        class MutableConfigRepository:
            def __init__(self) -> None:
                self.broken = False

            def get(self, key: str):
                if self.broken:
                    raise RuntimeError("config.db lost")
                return values.get(key)

        telephony = FakeTelephony()
        voice = FakeVoiceBackend()
        clock = FakeClock()
        repository = MutableConfigRepository()
        core = ReceptionistCore(
            telephony=telephony,
            voice=voice,
            config_service=ConfigService(repository),
            policy=FakePolicy(),
            clock=clock,
            policy_engine=PolicyEngine(
                destinations=default_destinations(),
                fallback_id="recepcion",
                limits=Limits(),
            ),
            runtime=RuntimeStorage.create(clock=clock),
            retention=RetentionPolicy(),
            call_ids=FakeCallIds(),
        )
        core.start()
        self.assertEqual(core.health.status, HealthStatus.READY)
        live = core.incoming_call("+34910000001")
        repository.broken = True
        # The live call keeps its immutable session config: transfer works.
        live_voice = voice.sessions[live.call_id]
        live_voice.finish_playback(live.current_turn)
        live_voice.deliver_action_request(TransferRequest(destination_id="ventas"))
        self.assertEqual(telephony.transfers, [(live.call_id, "SIP/201")])
        # New sessions are refused while config authority is down.
        queued = core.incoming_call("+34910000002")
        self.assertNotIn(queued.call_id, voice.sessions)
        self.assertEqual(telephony.rejected, [queued.call_id])
        # Health mirrors the outage instead of silently staying READY.
        self.assertEqual(core.health.status, HealthStatus.NOT_READY)
        # Trust restored on a later admission lifts exactly this outage.
        telephony.simulate_caller_hangup(live.call_id)
        repository.broken = False
        revived = core.incoming_call("+34910000003")
        self.assertIn(revived.call_id, voice.sessions)
        self.assertEqual(core.health.status, HealthStatus.READY)


class ObservabilityDegradationTest(unittest.TestCase):
    def test_history_write_failure_keeps_call_and_marks_degraded(self) -> None:
        failing_calls = FailingCallRepository()
        failing_calls.fail_save = StoreUnavailableError("history disk full")
        core, telephony, voice, _, _ = make_core(calls=failing_calls)
        session = core.incoming_call("+34910000001")
        session_voice = voice.sessions[session.call_id]
        session_voice.finish_playback(session.current_turn)
        # Policy still authorizes: the transfer is placed (Invariant F).
        session_voice.deliver_action_request(TransferRequest(destination_id="ventas"))
        self.assertEqual(telephony.transfers, [(session.call_id, "SIP/201")])
        telephony.complete_transfer(session.call_id, TransferResult.ACCEPTED_BY_PBX)
        self.assertEqual(session.state, CallState.ENDED)
        self.assertTrue(session.observability_failed)
        self.assertEqual(core.health.status, HealthStatus.DEGRADED)

    def test_audit_write_failure_changes_no_policy_decision(self) -> None:
        failing_audit = FailingAuditLog()
        failing_audit.fail_record = StoreUnavailableError("audit disk full")
        core, telephony, voice, _, _ = make_core(audit=failing_audit)
        session = core.incoming_call("+34910000001")
        session_voice = voice.sessions[session.call_id]
        session_voice.finish_playback(session.current_turn)
        # Allowed stays allowed, denied stays denied (Invariant F).
        session_voice.deliver_action_request(TransferRequest(destination_id="ventas"))
        self.assertEqual(telephony.transfers, [(session.call_id, "SIP/201")])
        session_voice.deliver_action_request(TransferRequest(destination_id="nadie"))
        self.assertEqual(len(telephony.transfers), 1)
        telephony.complete_transfer(session.call_id, TransferResult.ACCEPTED_BY_PBX)
        self.assertTrue(session.observability_failed)
        self.assertEqual(core.health.status, HealthStatus.DEGRADED)

    def test_transcript_sidecar_failure_never_blocks_audio(self) -> None:
        failing_transcripts = FailingTranscriptStore()
        failing_transcripts.fail_append = RuntimeError("transcript sidecar down")
        core, telephony, voice, _, _ = make_core(
            config_values={"greeting": GREETING, "language": "es", "transcripts_enabled": "true"},
            transcripts=failing_transcripts,
        )
        session = core.incoming_call("+34910000001")
        session_voice = voice.sessions[session.call_id]
        # Greeting audio still plays despite the sidecar failure.
        self.assertEqual(session_voice.spoken[0][0], GREETING)
        turn = open_turn(session, session_voice)
        session_voice.deliver_response("información útil", turn)
        self.assertEqual(session.mode, ActiveMode.SPEAKING)
        self.assertEqual(telephony.transfers, [])
        self.assertFalse(core.breaker.is_open)
        self.assertTrue(session.observability_failed)


class KnowledgeAndMessageIsolationTest(unittest.TestCase):
    def test_knowledge_failure_feeds_no_breaker_and_blocks_nothing(self) -> None:
        import os
        import tempfile

        from receptionist.knowledge import FileKnowledgeSource, LocalKnowledgeService, KeywordRetriever

        with tempfile.TemporaryDirectory() as tmp:
            knowledge = LocalKnowledgeService(
                sources=[
                    FileKnowledgeSource(
                        source_id="docs", path=os.path.join(tmp, "falta.md")
                    )
                ],
                retriever=KeywordRetriever(),
            )
            core, telephony, voice, _, _ = make_core(knowledge=knowledge)
            result = core.query_knowledge("¿Cuál es el horario?")
            from receptionist.boundaries import KnowledgeStatus

            self.assertEqual(result.status, KnowledgeStatus.FAILURE)
            self.assertFalse(core.breaker.is_open)
            session = core.incoming_call("+34910000001")
            session_voice = voice.sessions[session.call_id]
            session_voice.finish_playback(session.current_turn)
            session_voice.deliver_action_request(TransferRequest(destination_id="ventas"))
            self.assertEqual(telephony.transfers, [(session.call_id, "SIP/201")])

    def test_provider_failure_during_capture_aborts_without_save_or_ack(self) -> None:
        core, telephony, voice, _, _ = make_core()
        session = core.incoming_call("+34910000001")
        session_voice = voice.sessions[session.call_id]
        session_voice.finish_playback(session.current_turn)
        session_voice.deliver_action_request(StartMessageCapture())
        self.assertEqual(session.mode, ActiveMode.MESSAGE_CAPTURE)
        session_voice.deliver_failure(
            session.current_turn, ProviderFailure(ProviderFailureCategory.TIMEOUT)
        )
        # No usable final text can arrive: no save, no ACK, PBX fallback.
        from receptionist.call_session import MESSAGE_SAVED_ACK

        spoken = [text for text, _ in session_voice.spoken]
        self.assertNotIn(MESSAGE_SAVED_ACK, spoken)
        self.assertEqual(telephony.transfers, [(session.call_id, "SIP/100")])
        self.assertTrue(session.provider_terminal_failure)

    def test_policy_engine_alone_maps_destinations_during_recovery(self) -> None:
        # Invariant G: even mid-retry, a typed request resolves through
        # PolicyEngine and nothing else.
        core, telephony, voice, _, _ = make_core()
        session = core.incoming_call("+34910000001")
        first = voice.sessions[session.call_id]
        turn = open_turn(session, first)
        fail(first, turn, ProviderFailureCategory.TIMEOUT)
        second = voice.sessions[session.call_id]
        second.deliver_action_request(TransferRequest(destination_id="ventas"))
        self.assertEqual(telephony.transfers, [(session.call_id, "SIP/201")])


class PhaseAwareFailureTest(unittest.TestCase):
    """Provider failures outside INFERENCE are terminal for the AI path,
    never dead air: GREETING/SPEAKING/LISTENING fall back, CAPTURE aborts
    without save or ACK. CANCELLED stays neutral; stale stays inert."""

    def test_failure_during_greeting_falls_back(self) -> None:
        core, telephony, voice, _, _ = make_core(
            resilience=ResilienceConfig(provider_retries=0)
        )
        session = core.incoming_call("+34910000001")
        self.assertEqual(session.mode, ActiveMode.GREETING)
        voice.sessions[session.call_id].deliver_failure(
            session.current_turn, ProviderFailure(ProviderFailureCategory.UNAVAILABLE)
        )
        self.assertEqual(telephony.transfers, [(session.call_id, "SIP/100")])
        self.assertTrue(session.provider_terminal_failure)

    def test_failure_during_speaking_falls_back(self) -> None:
        core, telephony, voice, _, _ = make_core(
            resilience=ResilienceConfig(provider_retries=0)
        )
        session = core.incoming_call("+34910000001")
        session_voice = voice.sessions[session.call_id]
        turn = open_turn(session, session_voice)
        session_voice.deliver_response("respuesta", turn)
        self.assertEqual(session.mode, ActiveMode.SPEAKING)
        session_voice.deliver_failure(
            turn, ProviderFailure(ProviderFailureCategory.TIMEOUT)
        )
        self.assertEqual(telephony.transfers, [(session.call_id, "SIP/100")])
        self.assertTrue(session.provider_terminal_failure)

    def test_failure_during_listening_falls_back(self) -> None:
        core, telephony, voice, _, _ = make_core(
            resilience=ResilienceConfig(provider_retries=0)
        )
        session = core.incoming_call("+34910000001")
        session_voice = voice.sessions[session.call_id]
        turn = open_turn(session, session_voice)
        session_voice.deliver_response("respuesta", turn)
        session_voice.finish_playback(turn)
        self.assertEqual(session.mode, ActiveMode.LISTENING)
        session_voice.deliver_failure(
            turn, ProviderFailure(ProviderFailureCategory.INTERNAL)
        )
        self.assertEqual(telephony.transfers, [(session.call_id, "SIP/100")])

    def test_cancelled_outside_inference_stays_neutral(self) -> None:
        core, telephony, voice, _, _ = make_core()
        session = core.incoming_call("+34910000001")
        session_voice = voice.sessions[session.call_id]
        turn = open_turn(session, session_voice)
        session_voice.deliver_response("respuesta", turn)
        session_voice.deliver_failure(
            turn, ProviderFailure(ProviderFailureCategory.CANCELLED)
        )
        self.assertEqual(session.mode, ActiveMode.SPEAKING)
        self.assertEqual(session.state, CallState.ACTIVE)
        self.assertEqual(telephony.transfers, [])
        self.assertFalse(core.breaker.is_open)

    def test_stale_failure_outside_inference_stays_inert(self) -> None:
        core, telephony, voice, _, _ = make_core()
        session = core.incoming_call("+34910000001")
        old = voice.sessions[session.call_id]
        turn = open_turn(session, old)
        fail(old, turn, ProviderFailureCategory.TIMEOUT)
        new = voice.sessions[session.call_id]
        new.deliver_response("respuesta", turn)
        self.assertEqual(session.mode, ActiveMode.SPEAKING)
        old.deliver_failure(turn, ProviderFailure(ProviderFailureCategory.TIMEOUT))
        self.assertEqual(telephony.transfers, [])
        self.assertEqual(session.mode, ActiveMode.SPEAKING)


class BrokenVoiceTerminationTest(unittest.TestCase):
    def test_failed_apology_still_terminates_exactly_once(self) -> None:
        core, telephony, voice, _, _ = make_core(
            resilience=ResilienceConfig(provider_retries=0)
        )
        session = core.incoming_call("+34910000001")
        session_voice = voice.sessions[session.call_id]
        turn = open_turn(session, session_voice)
        session_voice.fail_speak = RuntimeError("tts backend dead")
        fail(session_voice, turn, ProviderFailureCategory.TIMEOUT)
        telephony.complete_transfer(session.call_id, TransferResult.REJECTED)
        telephony.complete_transfer(session.call_id, TransferResult.TIMEOUT)
        # Termination never depends on the broken voice path.
        self.assertEqual(telephony.hung_up, [session.call_id])
        self.assertEqual(session.state, CallState.ENDED)


class ProviderFailureContractTest(unittest.TestCase):
    def test_raw_category_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            ProviderFailure(category="timeout")  # type: ignore[arg-type]

    def test_unnormalized_failure_never_reaches_retry_or_breaker(self) -> None:
        core, telephony, voice, _, _ = make_core()
        session = core.incoming_call("+34910000001")
        session_voice = voice.sessions[session.call_id]
        turn = open_turn(session, session_voice)
        session_voice.deliver_failure(turn, _Unnormalized(category="timeout"))
        self.assertEqual(len(voice.all_sessions), 1)
        self.assertEqual(telephony.transfers, [])
        self.assertFalse(session.provider_terminal_failure)
        self.assertFalse(core.breaker.is_open)


class _Unnormalized:
    """Backend-shaped garbage: not a ProviderFailure at all."""

    def __init__(self, category: str) -> None:
        self.category = category


class TimingInteractionTest(unittest.TestCase):
    def test_hard_deadline_during_retry_terminates_once(self) -> None:
        core, telephony, voice, clock, _ = make_core()
        session = core.incoming_call("+34910000001")
        first = voice.sessions[session.call_id]
        turn = open_turn(session, first)
        fail(first, turn, ProviderFailureCategory.TIMEOUT)
        retried = voice.sessions[session.call_id]
        clock.advance(600.0)
        core.tick()
        self.assertEqual(telephony.hung_up, [session.call_id])
        # Auto-confirmed hangup completes teardown synchronously.
        self.assertEqual(session.state, CallState.ENDED)
        # The late failure starts nothing and terminates nothing again.
        retried.deliver_failure(turn, ProviderFailure(ProviderFailureCategory.TIMEOUT))
        self.assertEqual(len(voice.all_sessions), 2)
        self.assertEqual(telephony.hung_up, [session.call_id])

    def test_second_no_input_falls_back_then_failure_ends_once(self) -> None:
        core, telephony, voice, clock, _ = make_core(
            resilience=ResilienceConfig(no_input_seconds=30.0)
        )
        session = core.incoming_call("+34910000001")
        session_voice = voice.sessions[session.call_id]
        session_voice.finish_playback(session.current_turn)
        self.assertEqual(session.mode, ActiveMode.LISTENING)
        clock.advance(30.0)
        core.tick()
        self.assertIn(NO_INPUT_REPROMPT, [text for text, _ in session_voice.spoken])
        self.assertEqual(telephony.transfers, [])
        clock.advance(30.0)
        core.tick()
        self.assertEqual(telephony.transfers, [(session.call_id, "SIP/100")])
        telephony.complete_transfer(session.call_id, TransferResult.REJECTED)
        telephony.complete_transfer(session.call_id, TransferResult.REJECTED)
        from receptionist.call_session import EXIT_APOLOGY

        spoken = [text for text, _ in session_voice.spoken]
        self.assertEqual(spoken.count(EXIT_APOLOGY), 1)
        self.assertEqual(telephony.hung_up, [session.call_id])
        self.assertEqual(session.state, CallState.ENDED)


if __name__ == "__main__":
    unittest.main()
