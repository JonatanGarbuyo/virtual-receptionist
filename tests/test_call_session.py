"""Behavior tests at the ReceptionistCore / CallSession seam.

Every test drives the session through project-owned interfaces with
deterministic fakes (no network, no real models, no wall-clock sleeps)
and asserts externally observable behavior: lifecycle states, active
modes, spoken audio, persisted summaries, and health.
"""

import unittest

from receptionist.boundaries import CallOutcome
from receptionist.call_session import ActiveMode, CallSession, CallState
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

from fakes import FakeCallIds, FakeClock, FakePolicy, FakeTelephony, FakeVoiceBackend


GREETING = "Bienvenido, ¿en qué puedo ayudarle?"


def make_core(
    config_values: dict | None = None,
    policy: FakePolicy | None = None,
    clock: FakeClock | None = None,
    auto_confirm: bool = True,
) -> tuple[ReceptionistCore, FakeTelephony, FakeVoiceBackend, FakeClock, InMemoryCallRepository]:
    values = {"greeting": GREETING, "language": "es"} if config_values is None else config_values
    telephony = FakeTelephony(auto_confirm=auto_confirm)
    voice = FakeVoiceBackend()
    clock = clock or FakeClock()
    calls = InMemoryCallRepository()
    core = ReceptionistCore(
        telephony=telephony,
        voice=voice,
        config_service=ConfigService(InMemoryConfigRepository(dict(values))),
        policy=policy or FakePolicy(),
        clock=clock,
        # No transfers in this suite: empty engine, audit discarded.
        policy_engine=PolicyEngine(destinations={}, fallback_id="none", limits=Limits()),
        runtime=RuntimeStorage(
            calls=calls,
            messages=InMemoryMessageRepository(clock=clock),
            transcripts=InMemoryTranscriptStore(),
            audit=InMemoryAuditLog(),
        ),
        retention=RetentionPolicy(),
        call_ids=FakeCallIds(),
    )
    return core, telephony, voice, clock, calls


class CoreStartupTest(unittest.TestCase):
    def test_core_starts_in_starting(self) -> None:
        core, _, _, _, _ = make_core()
        self.assertEqual(core.health.status, HealthStatus.STARTING)

    def test_start_with_greeting_config_becomes_ready(self) -> None:
        core, _, _, _, _ = make_core()
        core.start()
        self.assertEqual(core.health.status, HealthStatus.READY)

    def test_start_without_greeting_config_is_not_ready(self) -> None:
        core, _, _, _, _ = make_core(config_values={"language": "es"})
        core.start()
        self.assertEqual(core.health.status, HealthStatus.NOT_READY)


class AcceptedInboundCallTest(unittest.TestCase):
    def test_accepted_call_reaches_active_greeting_and_speaks_configured_greeting(self) -> None:
        core, telephony, voice, _, _ = make_core()
        core.start()
        session = core.incoming_call("+34910000001")

        self.assertIsInstance(session, CallSession)
        self.assertEqual(session.call_id, "call-1")
        self.assertEqual(session.state, CallState.ACTIVE)
        self.assertEqual(session.mode, ActiveMode.GREETING)
        self.assertEqual(session.history, [CallState.INCOMING, CallState.ANSWERING, CallState.ACTIVE])
        self.assertEqual(telephony.answered, ["call-1"])
        self.assertEqual(voice.sessions["call-1"].spoken, [(GREETING, 1)])

    def test_answering_is_observable_before_confirmation(self) -> None:
        core, telephony, voice, _, _ = make_core(auto_confirm=False)
        core.start()
        session = core.incoming_call("+34910000001")

        self.assertEqual(session.state, CallState.ANSWERING)
        self.assertEqual(telephony.answered, ["call-1"])
        self.assertEqual(voice.sessions["call-1"].spoken, [])

        telephony.complete_answer("call-1")

        self.assertEqual(session.state, CallState.ACTIVE)
        self.assertEqual(session.mode, ActiveMode.GREETING)
        self.assertEqual(voice.sessions["call-1"].spoken, [(GREETING, 1)])

    def test_greeting_comes_from_configuration_boundary(self) -> None:
        core, _, voice, _, _ = make_core(config_values={"greeting": "Hola, dígame", "language": "es"})
        core.start()
        session = core.incoming_call("+34910000001")

        self.assertEqual(voice.sessions[session.call_id].spoken, [("Hola, dígame", 1)])


class ConversationLoopTest(unittest.TestCase):
    def start_call(self) -> tuple:
        core, telephony, voice, clock, calls = make_core()
        core.start()
        session = core.incoming_call("+34910000001")
        return core, telephony, voice, clock, calls, session

    def test_greeting_playback_moves_to_listening(self) -> None:
        _, _, voice, _, _, session = self.start_call()

        voice.sessions["call-1"].finish_playback(1)

        self.assertEqual(session.mode, ActiveMode.LISTENING)

    def test_caller_utterance_runs_inference_speaking_listening(self) -> None:
        _, _, voice, _, _, session = self.start_call()
        backend = voice.sessions["call-1"]
        backend.finish_playback(1)

        backend.deliver_caller_speech("Quisiera hablar con ventas")

        self.assertEqual(session.mode, ActiveMode.INFERENCE)
        self.assertEqual(session.current_turn, 2)

        backend.deliver_response("Le comunico con ventas.", 2)

        self.assertEqual(session.mode, ActiveMode.SPEAKING)
        self.assertEqual(
            backend.spoken,
            [(GREETING, 1), ("Le comunico con ventas.", 2)],
        )

        backend.finish_playback(2)

        self.assertEqual(session.mode, ActiveMode.LISTENING)

    def test_second_utterance_starts_a_new_turn(self) -> None:
        _, _, voice, _, _, session = self.start_call()
        backend = voice.sessions["call-1"]
        backend.finish_playback(1)
        backend.deliver_caller_speech("Hola")
        backend.deliver_response("Dígame.", 2)
        backend.finish_playback(2)

        backend.deliver_caller_speech("Gracias, adiós")

        self.assertEqual(session.current_turn, 3)
        self.assertEqual(session.mode, ActiveMode.INFERENCE)

    def test_caller_speech_while_greeting_or_speaking_is_ignored(self) -> None:
        # Barge-in is deferred to a later ticket: speech outside
        # LISTENING/INFERENCE must not start a turn.
        _, _, voice, _, _, session = self.start_call()
        backend = voice.sessions["call-1"]

        backend.deliver_caller_speech("Hola, ¿me oye?")

        self.assertEqual(session.mode, ActiveMode.GREETING)
        self.assertEqual(session.current_turn, 1)
        self.assertEqual(backend.spoken, [(GREETING, 1)])


class EndCallTest(unittest.TestCase):
    def test_end_call_terminates_and_persists_completed_summary(self) -> None:
        core, telephony, voice, clock, calls = make_core()
        core.start()
        session = core.incoming_call("+34910000001")
        backend = voice.sessions["call-1"]
        backend.finish_playback(1)
        backend.deliver_caller_speech("Solo era una prueba")
        backend.deliver_response("Gracias por llamar.", 2)
        backend.finish_playback(2)
        clock.advance(42.5)

        session.end_call()

        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(
            session.history,
            [CallState.INCOMING, CallState.ANSWERING, CallState.ACTIVE,
             CallState.TERMINATING, CallState.ENDED],
        )
        self.assertEqual(telephony.hung_up, ["call-1"])
        summary = calls.get("call-1")
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.caller_id, "+34910000001")
        self.assertEqual(summary.outcome, CallOutcome.COMPLETED)
        self.assertEqual(summary.started_at, 0.0)
        self.assertEqual(summary.ended_at, 42.5)
        self.assertEqual(summary.turn_count, 1)
        self.assertTrue(backend.closed)

    def test_terminating_is_observable_before_hangup_confirmed(self) -> None:
        core, telephony, _, _, _ = make_core(auto_confirm=False)
        core.start()
        session = core.incoming_call("+34910000001")
        telephony.complete_answer("call-1")

        session.end_call()

        self.assertEqual(session.state, CallState.TERMINATING)
        self.assertEqual(telephony.hung_up, ["call-1"])

        telephony.complete_hangup("call-1")

        self.assertEqual(session.state, CallState.ENDED)

    def test_end_call_after_ended_is_a_noop(self) -> None:
        core, telephony, _, _, calls = make_core()
        core.start()
        session = core.incoming_call("+34910000001")

        session.end_call()
        session.end_call()

        self.assertEqual(telephony.hung_up, ["call-1"])
        self.assertEqual(len(calls.list_all()), 1)


class PolicyRejectTest(unittest.TestCase):
    def test_denied_call_is_rejected_without_answering(self) -> None:
        policy = FakePolicy(allow=False)
        core, telephony, voice, _, calls = make_core(policy=policy)
        core.start()
        session = core.incoming_call("+34910000002")

        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(session.history, [CallState.INCOMING, CallState.ENDED])
        self.assertEqual(policy.decisions, ["+34910000002"])
        self.assertEqual(telephony.rejected, ["call-1"])
        self.assertEqual(telephony.answered, [])
        self.assertNotIn("call-1", voice.sessions)
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.REJECTED)
        self.assertEqual(summary.turn_count, 0)


class ReadinessGateTest(unittest.TestCase):
    def test_call_before_start_opens_no_voice_and_never_reaches_active(self) -> None:
        core, telephony, voice, _, calls = make_core()
        # no start(): health is STARTING
        session = core.incoming_call("+34910000001")

        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(session.history, [CallState.INCOMING, CallState.ENDED])
        self.assertEqual(telephony.answered, [])
        self.assertEqual(telephony.rejected, ["call-1"])
        self.assertNotIn("call-1", voice.sessions)
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.REJECTED)

    def test_call_when_not_ready_opens_no_voice_and_never_reaches_active(self) -> None:
        core, telephony, voice, _, calls = make_core(config_values={"language": "es"})
        core.start()
        self.assertEqual(core.health.status, HealthStatus.NOT_READY)
        session = core.incoming_call("+34910000001")

        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(session.history, [CallState.INCOMING, CallState.ENDED])
        self.assertEqual(telephony.answered, [])
        self.assertEqual(telephony.rejected, ["call-1"])
        self.assertNotIn("call-1", voice.sessions)
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.REJECTED)

    def test_ready_call_opens_voice_and_reaches_active_greeting(self) -> None:
        core, telephony, voice, _, _ = make_core()
        core.start()
        self.assertEqual(core.health.status, HealthStatus.READY)
        session = core.incoming_call("+34910000001")

        self.assertEqual(session.state, CallState.ACTIVE)
        self.assertEqual(session.mode, ActiveMode.GREETING)
        self.assertIn("call-1", voice.sessions)
        self.assertEqual(telephony.answered, ["call-1"])


class LateEventsTest(unittest.TestCase):
    def start_active_call(self) -> tuple:
        core, telephony, voice, _, calls = make_core()
        core.start()
        session = core.incoming_call("+34910000001")
        backend = voice.sessions["call-1"]
        backend.finish_playback(1)
        return core, telephony, backend, calls, session

    def test_stale_response_after_new_utterance_is_ignored(self) -> None:
        _, _, backend, _, session = self.start_active_call()
        backend.deliver_caller_speech("first question")
        backend.deliver_caller_speech("second question")
        self.assertEqual(session.current_turn, 3)

        backend.deliver_response("stale answer to first", 2)

        self.assertEqual(session.mode, ActiveMode.INFERENCE)
        self.assertEqual(backend.spoken, [(GREETING, 1)])

        backend.deliver_response("fresh answer to second", 3)

        self.assertEqual(session.mode, ActiveMode.SPEAKING)
        self.assertEqual(
            backend.spoken,
            [(GREETING, 1), ("fresh answer to second", 3)],
        )

    def test_stale_playback_finished_does_not_regress_mode(self) -> None:
        _, _, backend, _, session = self.start_active_call()
        backend.deliver_caller_speech("first question")
        backend.deliver_response("answer", 2)
        backend.finish_playback(2)
        self.assertEqual(session.mode, ActiveMode.LISTENING)
        backend.deliver_caller_speech("follow-up")

        backend.finish_playback(2)  # late event from the obsolete turn

        self.assertEqual(session.mode, ActiveMode.INFERENCE)
        self.assertEqual(session.current_turn, 3)

    def test_caller_hangup_during_inference_ends_call_and_late_response_is_ignored(self) -> None:
        _, telephony, backend, calls, session = self.start_active_call()
        backend.deliver_caller_speech("are you still there?")

        telephony.simulate_caller_hangup("call-1")

        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(
            session.history,
            [CallState.INCOMING, CallState.ANSWERING, CallState.ACTIVE,
             CallState.TERMINATING, CallState.ENDED],
        )
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.CALLER_HANGUP)

        backend.deliver_response("too late", 2)  # must not resurrect the turn

        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(session.mode, ActiveMode.INFERENCE)
        self.assertEqual(backend.spoken, [(GREETING, 1)])

    def test_transcript_after_ended_is_ignored(self) -> None:
        _, _, backend, calls, session = self.start_active_call()

        session.end_call()
        backend.deliver_caller_speech("hello?")

        self.assertEqual(session.state, CallState.ENDED)
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.turn_count, 0)

    def test_events_for_unknown_calls_are_ignored(self) -> None:
        core, telephony, voice, _, _ = make_core(auto_confirm=False)
        core.start()
        session = core.incoming_call("+34910000001")

        telephony.simulate_caller_hangup("call-99")
        telephony.complete_answer("call-99")
        telephony.complete_hangup("call-99")

        self.assertEqual(session.state, CallState.ANSWERING)


class HealthSemanticsTest(unittest.TestCase):
    def test_degraded_and_recovery(self) -> None:
        core, _, _, _, _ = make_core()
        core.start()

        core.report_degraded("voice backend unavailable")

        self.assertEqual(core.health.status, HealthStatus.DEGRADED)
        self.assertEqual(core.health.detail, "voice backend unavailable")

        core.recover()

        self.assertEqual(core.health.status, HealthStatus.READY)

    def test_recover_only_leaves_degraded(self) -> None:
        ready_core, _, _, _, _ = make_core()
        ready_core.start()
        ready_core.recover()
        self.assertEqual(ready_core.health.status, HealthStatus.READY)

        unready_core, _, _, _, _ = make_core(config_values={"language": "es"})
        unready_core.start()
        unready_core.recover()
        self.assertEqual(unready_core.health.status, HealthStatus.NOT_READY)

    def test_not_ready_names_missing_configuration(self) -> None:
        core, _, _, _, _ = make_core(config_values={"language": "es"})
        core.start()

        self.assertEqual(core.health.status, HealthStatus.NOT_READY)
        self.assertIn("greeting", core.health.detail)

    def test_blank_required_values_count_as_missing(self) -> None:
        blank_greeting, _, _, _, _ = make_core(
            config_values={"greeting": "   ", "language": "es"}
        )
        blank_greeting.start()
        self.assertEqual(blank_greeting.health.status, HealthStatus.NOT_READY)
        self.assertIn("greeting", blank_greeting.health.detail)

        blank_language, _, _, _, _ = make_core(
            config_values={"greeting": GREETING, "language": ""}
        )
        blank_language.start()
        self.assertEqual(blank_language.health.status, HealthStatus.NOT_READY)
        self.assertIn("language", blank_language.health.detail)


class SessionEvictionTest(unittest.TestCase):
    def test_ended_session_is_evicted_from_core(self) -> None:
        core, _, _, _, _ = make_core()
        core.start()
        session = core.incoming_call("+34910000001")

        self.assertIs(core.get_session("call-1"), session)

        session.end_call()

        self.assertEqual(session.state, CallState.ENDED)
        self.assertIsNone(core.get_session("call-1"))

    def test_rejected_session_is_evicted_but_summary_persists(self) -> None:
        core, _, _, _, calls = make_core(policy=FakePolicy(allow=False))
        core.start()
        session = core.incoming_call("+34910000002")

        self.assertEqual(session.state, CallState.ENDED)
        self.assertIsNone(core.get_session("call-1"))
        summary = calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.REJECTED)

    def test_caller_hangup_evicted_after_completion(self) -> None:
        core, telephony, voice, _, _ = make_core()
        core.start()
        core.incoming_call("+34910000001")
        voice.sessions["call-1"].finish_playback(1)

        telephony.simulate_caller_hangup("call-1")

        self.assertIsNone(core.get_session("call-1"))


if __name__ == "__main__":
    unittest.main()
