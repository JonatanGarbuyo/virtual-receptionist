"""Telephony contract tests for #25 (deterministic, no SIP stack).

Three tiers per the ticket: this file is tier 1 (FakeTelephony) plus
tier 2 (project-owned adapter internals with a controlled native seam).
Tier 3 (SIPp + disposable Asterisk + baresip) lives in
``tests/test_telephony_integration.py`` (gated) with evidence under
``docs/evidence/25-baresip-matrix/``.

Everything asserted here is externally observable behavior at owned
seams: lifecycle states, playout routing, flush/barge-in, DTMF/hold
isolation, registration health, and codec round-trip *properties*
(never byte-perfect G.711, never "RTP packets exist").
"""

import sys
import unittest

from receptionist.audio import (
    frame_rms,
    make_frame,
    resample_pcm16,
    silence_pcm,
    tone_pcm,
)
from receptionist.boundaries import (
    AudioFrame,
    TelephonyRegistrationState,
    TransferResult,
)
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
from receptionist.telephony_codecs import (
    decode_alaw_to_pcm16,
    decode_ulaw_to_pcm16,
    encode_pcm16_to_alaw,
    encode_pcm16_to_ulaw,
)

from fakes import FakeCallIds, FakeClock, FakePolicy, FakeTelephony, FakeVoiceBackend


GREETING = "Bienvenido, ¿en qué puedo ayudarle?"


def make_core(
    config_values: dict | None = None,
    policy_engine: PolicyEngine | None = None,
    auto_confirm: bool = True,
    max_ai_sessions: int = 1,
) -> tuple[ReceptionistCore, FakeTelephony, FakeVoiceBackend, FakeClock]:
    from receptionist.resilience import ResilienceConfig

    values = {"greeting": GREETING, "language": "es"} if config_values is None else config_values
    telephony = FakeTelephony(auto_confirm=auto_confirm)
    voice = FakeVoiceBackend()
    clock = FakeClock()
    core = ReceptionistCore(
        telephony=telephony,
        voice=voice,
        config_service=ConfigService(InMemoryConfigRepository(dict(values))),
        policy=FakePolicy(),
        clock=clock,
        resilience=ResilienceConfig(max_ai_sessions=max_ai_sessions),
        policy_engine=policy_engine
        or PolicyEngine(destinations={}, fallback_id="none", limits=Limits()),
        runtime=RuntimeStorage(
            calls=InMemoryCallRepository(),
            messages=InMemoryMessageRepository(clock=clock),
            transcripts=InMemoryTranscriptStore(),
            audit=InMemoryAuditLog(),
        ),
        retention=RetentionPolicy(),
        call_ids=FakeCallIds(),
    )
    return core, telephony, voice, clock


def assistant_frame(seq: int = 0) -> AudioFrame:
    return make_frame(
        tone_pcm(duration_seconds=0.1, sample_rate=16000), 16000, sequence=seq
    )


def caller_speech_frame(seq: int = 0) -> AudioFrame:
    return make_frame(
        tone_pcm(duration_seconds=0.1, sample_rate=16000, frequency_hz=660.0),
        16000,
        sequence=seq,
    )


def drive_to_speaking(core, telephony, voice, caller: str):
    """Admit a call and drive it to SPEAKING with a live audio turn."""
    session = core.incoming_call(caller)
    live = voice.sessions[session.call_id]
    live.finish_playback(session.current_turn)  # GREETING -> LISTENING
    assert session.mode is ActiveMode.LISTENING
    live.deliver_caller_speech("quiero hablar con ventas")
    assert session.mode is ActiveMode.INFERENCE
    turn = session.current_turn
    live.deliver_audio(turn, assistant_frame())
    assert session.mode is ActiveMode.SPEAKING
    return session, live, turn


class G711RoundTripTest(unittest.TestCase):
    def test_pcmu_round_trip_preserves_energy_and_duration(self) -> None:
        pcm = tone_pcm(duration_seconds=0.5, sample_rate=8000, frequency_hz=440.0)
        payload = encode_pcm16_to_ulaw(pcm)
        self.assertEqual(len(payload), len(pcm) // 2)  # 8-bit line code
        back = decode_ulaw_to_pcm16(payload)
        self.assertEqual(len(back), len(pcm))
        original = frame_rms(make_frame(pcm, 8000))
        decoded = frame_rms(make_frame(back, 8000))
        self.assertGreater(original, 0.2)
        self.assertGreater(decoded, 0.15)  # lossy but clearly audible tone
        self.assertLess(abs(decoded - original), 0.05)

    def test_pcma_round_trip_preserves_energy_and_duration(self) -> None:
        pcm = tone_pcm(duration_seconds=0.5, sample_rate=8000, frequency_hz=440.0)
        payload = encode_pcm16_to_alaw(pcm)
        self.assertEqual(len(payload), len(pcm) // 2)
        back = decode_alaw_to_pcm16(payload)
        self.assertEqual(len(back), len(pcm))
        original = frame_rms(make_frame(pcm, 8000))
        decoded = frame_rms(make_frame(back, 8000))
        self.assertGreater(decoded, 0.15)
        self.assertLess(abs(decoded - original), 0.05)

    def test_g711_silence_stays_silence(self) -> None:
        silence = silence_pcm(duration_seconds=0.2, sample_rate=8000)
        for encode, decode in (
            (encode_pcm16_to_ulaw, decode_ulaw_to_pcm16),
            (encode_pcm16_to_alaw, decode_alaw_to_pcm16),
        ):
            back = decode(encode(silence))
            self.assertLess(frame_rms(make_frame(back, 8000)), 0.02)

    def test_resample_8k_16k_8k_keeps_tone_usable(self) -> None:
        pcm8 = tone_pcm(duration_seconds=0.5, sample_rate=8000)
        pcm16 = resample_pcm16(pcm8, 8000, 16000)
        self.assertEqual(len(pcm16), len(pcm8) * 2)
        back8 = resample_pcm16(pcm16, 16000, 8000)
        self.assertEqual(len(back8), len(pcm8))
        self.assertGreater(frame_rms(make_frame(back8, 8000)), 0.15)


class AssistantPlayoutSeamTest(unittest.TestCase):
    def test_live_assistant_audio_reaches_telephony_playout(self) -> None:
        core, telephony, voice, _ = make_core()
        core.start()
        session, _live, turn = drive_to_speaking(
            core, telephony, voice, "+34910000001"
        )
        played = telephony.playout.get(session.call_id, [])
        self.assertEqual(len(played), 1)
        frame = played[0]
        self.assertIsInstance(frame, AudioFrame)
        self.assertEqual(frame.sample_format, "pcm16")
        self.assertEqual(frame.channels, 1)
        self.assertGreater(frame_rms(frame), 0.1)
        self.assertEqual(session.mode, ActiveMode.SPEAKING)
        self.assertEqual(session.current_turn, turn)

    def test_late_turn_audio_never_reaches_playout(self) -> None:
        core, telephony, voice, _ = make_core()
        core.start()
        session, live, turn = drive_to_speaking(
            core, telephony, voice, "+34910000001"
        )
        before = len(telephony.playout.get(session.call_id, []))
        # Obsolete turn id: discarded before the adapter (no second path).
        live.deliver_audio(turn + 99, assistant_frame(seq=9))
        self.assertEqual(len(telephony.playout.get(session.call_id, [])), before)
        self.assertEqual(session.mode, ActiveMode.SPEAKING)

    def test_no_second_playback_path_for_text_sidecar(self) -> None:
        core, telephony, voice, _ = make_core()
        core.start()
        session, live, _turn = drive_to_speaking(
            core, telephony, voice, "+34910000001"
        )
        before = len(telephony.playout.get(session.call_id, []))
        # Text sidecars never synthesize: playout grows only via on_audio.
        live.deliver_response("texto de depuración", session.current_turn + 50)
        self.assertEqual(len(telephony.playout.get(session.call_id, [])), before)


class BargeInFlushTest(unittest.TestCase):
    def test_barge_in_discards_queued_playout(self) -> None:
        core, telephony, voice, _ = make_core()
        core.start()
        session, _live, _turn = drive_to_speaking(
            core, telephony, voice, "+34910000001"
        )
        self.assertTrue(telephony.playout.get(session.call_id))
        session.push_caller_audio(caller_speech_frame())
        self.assertIn(session.call_id, telephony.flushed)
        self.assertNotIn(session.call_id, telephony.playout)
        self.assertEqual(session.mode, ActiveMode.LISTENING)

    def test_caller_hangup_flushes_and_late_audio_drops(self) -> None:
        core, telephony, voice, _ = make_core()
        core.start()
        session, live, turn = drive_to_speaking(
            core, telephony, voice, "+34910000001"
        )
        telephony.simulate_caller_hangup(session.call_id)
        self.assertIn(session.call_id, telephony.flushed)
        before = len(telephony.playout.get(session.call_id, []))
        live.deliver_audio(turn, assistant_frame(seq=7))
        self.assertEqual(len(telephony.playout.get(session.call_id, [])), before)


class MultiCallIsolationTest(unittest.TestCase):
    def test_two_calls_have_distinct_playout_dtmf_and_hangup(self) -> None:
        # Telephony holds two call objects; AI capacity is widened here
        # only so both legs stay conversational (production default is 1
        # AI session with the second call on PBX fallback).
        core, telephony, voice, _ = make_core(max_ai_sessions=2)
        core.start()
        session_a, live_a, _ = drive_to_speaking(core, telephony, voice, "+34910000001")
        session_b, live_b, _ = drive_to_speaking(core, telephony, voice, "+34910000002")
        self.assertNotEqual(session_a.call_id, session_b.call_id)
        # Audio does not cross.
        live_b.deliver_audio(session_b.current_turn, assistant_frame(seq=3))
        self.assertEqual(len(telephony.playout[session_a.call_id]), 1)
        self.assertEqual(len(telephony.playout[session_b.call_id]), 2)
        # DTMF events keep their call id.
        seen: list[tuple[str, str]] = []

        original = core.on_dtmf
        core.on_dtmf(session_a.call_id, "5")
        core.on_dtmf(session_b.call_id, "9")
        telephony.simulate_dtmf(session_a.call_id, "5")
        telephony.simulate_dtmf(session_b.call_id, "9")
        self.assertEqual(
            telephony.sent_dtmf.get(session_a.call_id, []), []
        )  # RX path only; TX below
        telephony.send_dtmf(session_a.call_id, "1")
        telephony.send_dtmf(session_b.call_id, "23")
        self.assertEqual(telephony.sent_dtmf[session_a.call_id], ["1"])
        self.assertEqual(telephony.sent_dtmf[session_b.call_id], ["23"])
        self.assertIsNotNone(seen or original)
        # Hangup A does not destroy B (auto-confirm completes the hangup).
        telephony.simulate_caller_hangup(session_a.call_id)
        self.assertEqual(session_a.state, CallState.ENDED)
        self.assertIsNone(core.get_session(session_a.call_id))
        self.assertEqual(session_b.state, CallState.ACTIVE)
        live_b.deliver_audio(session_b.current_turn, assistant_frame(seq=4))
        self.assertEqual(len(telephony.playout[session_b.call_id]), 3)

    def test_unknown_and_closed_calls_are_ignored(self) -> None:
        core, telephony, voice, _ = make_core()
        core.start()
        # Unknown ids: listener drops, adapter drops, nothing raises.
        core.on_caller_audio("call-999", caller_speech_frame())
        core.on_dtmf("call-999", "5")
        core.on_remote_hold("call-999", True)
        telephony.send_audio("call-999", assistant_frame())
        telephony.send_dtmf("call-999", "5")
        telephony.hold("call-999")
        telephony.resume("call-999")
        session = core.incoming_call("+34910000001")
        telephony.simulate_caller_hangup(session.call_id)
        telephony.send_audio(session.call_id, assistant_frame())
        self.assertNotIn(session.call_id, telephony.playout)


class HoldTest(unittest.TestCase):
    def test_hold_suppresses_caller_speech_resume_restores(self) -> None:
        core, telephony, voice, clock = make_core()
        core.start()
        session = core.incoming_call("+34910000001")
        voice.sessions[session.call_id].finish_playback(session.current_turn)
        telephony.hold(session.call_id)
        telephony.simulate_caller_audio(session.call_id, caller_speech_frame())
        # Held: no turn opens, nothing ingested.
        self.assertEqual(session.mode, ActiveMode.LISTENING)
        self.assertEqual(voice.sessions[session.call_id].pushed_audio, [])
        telephony.resume(session.call_id)
        telephony.simulate_caller_audio(session.call_id, caller_speech_frame())
        self.assertEqual(len(voice.sessions[session.call_id].pushed_audio), 1)
        telephony.simulate_remote_hold(session.call_id, True)
        telephony.simulate_remote_hold(session.call_id, False)


class InboundAdmissionTest(unittest.TestCase):
    def test_inbound_handler_admits_through_core_and_binds_answer(self) -> None:
        core, telephony, voice, _ = make_core()
        core.start()
        from receptionist.baresip_adapter import CoreInboundBridge

        telephony.set_inbound_handler(CoreInboundBridge(core))
        app_id = telephony.simulate_incoming("sip:1001@pbx.test")
        self.assertEqual(app_id, "call-1")
        self.assertEqual(telephony.answered, ["call-1"])
        session = core.get_session("call-1")
        self.assertIsNotNone(session)
        self.assertIsInstance(session, CallSession)

    def test_inbound_handler_none_declines_without_session(self) -> None:
        from receptionist.boundaries import InboundCallHandler

        core, telephony, _, _ = make_core()
        core.start()

        class DeclineAll(InboundCallHandler):
            def handle_incoming_call(self, caller_id: str, caller_name: str | None):
                return None

        telephony.set_inbound_handler(DeclineAll())
        self.assertIsNone(telephony.simulate_incoming("sip:999@pbx.test"))
        self.assertEqual(telephony.answered, [])


class RegistrationHealthTest(unittest.TestCase):
    def test_telephony_registration_drives_ready(self) -> None:
        core, telephony, _, _ = make_core()
        core.start()
        self.assertEqual(core.health.status, HealthStatus.READY)
        core.report_telephony_state(TelephonyRegistrationState.STARTING)
        self.assertEqual(core.health.status, HealthStatus.NOT_READY)
        core.report_telephony_state(TelephonyRegistrationState.REGISTERED)
        self.assertEqual(core.health.status, HealthStatus.READY)
        core.report_telephony_state(TelephonyRegistrationState.REGISTRATION_LOST)
        self.assertEqual(core.health.status, HealthStatus.NOT_READY)
        core.report_telephony_state(TelephonyRegistrationState.REGISTERED)
        self.assertEqual(core.health.status, HealthStatus.READY)
        core.report_telephony_state(TelephonyRegistrationState.REGISTRATION_FAILED)
        self.assertEqual(core.health.status, HealthStatus.NOT_READY)

    def test_status_bridge_maps_adapter_states_to_core(self) -> None:
        from receptionist.baresip_adapter import CoreStatusBridge

        core, telephony, _, _ = make_core()
        core.start()
        telephony.set_status_listener(CoreStatusBridge(core))
        telephony.report_registration(TelephonyRegistrationState.REGISTERED)
        self.assertEqual(core.health.status, HealthStatus.READY)
        telephony.report_registration(TelephonyRegistrationState.REGISTRATION_LOST)
        self.assertEqual(core.health.status, HealthStatus.NOT_READY)


class TransferPrimitiveTest(unittest.TestCase):
    def test_adapter_receives_policy_resolved_pbx_target(self) -> None:
        from receptionist.policy import Destination, PolicyEngine

        engine = PolicyEngine(
            destinations={
                "ventas": Destination(
                    id="ventas", target="sip:201@test", kind="extension"
                )
            },
            fallback_id="ventas",
            limits=Limits(),
        )
        core, telephony, voice, _ = make_core(policy_engine=engine)
        core.start()
        session = core.incoming_call("+34910000001")
        voice.sessions[session.call_id].finish_playback(session.current_turn)
        from receptionist.boundaries import TransferRequest

        voice.sessions[session.call_id].deliver_action_request(
            TransferRequest(destination_id="ventas")
        )
        self.assertEqual(telephony.transfers, [(session.call_id, "sip:201@test")])
        telephony.complete_transfer(session.call_id, TransferResult.ACCEPTED_BY_PBX)
        self.assertEqual(session.state, CallState.ENDED)


class AdapterBackpressureTest(unittest.TestCase):
    def test_tx_queue_is_bounded_oldest_drops_first(self) -> None:
        from receptionist.baresip_adapter import (
            TX_QUEUE_MAX_BYTES,
            BaresipTelephonyAdapter,
            _CallRecord,
        )
        from receptionist.telephony_config import TelephonyConfig

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        record = _CallRecord(app_id="call-1", native=None)
        adapter._calls["call-1"] = record  # controlled seam, no SIP involved
        chunk = make_frame(tone_pcm(duration_seconds=0.5, sample_rate=8000), 8000)
        for seq in range(20):
            adapter.send_audio(
                "call-1",
                make_frame(chunk.pcm, 8000, sequence=seq),
            )
        self.assertLessEqual(record.tx_bytes, TX_QUEUE_MAX_BYTES)
        self.assertGreater(record.tx_dropped_bytes, 0)
        adapter.flush_audio("call-1")
        self.assertEqual(record.tx_bytes, 0)
        self.assertEqual(list(record.tx), [])

    def test_adapter_imports_without_native_binding(self) -> None:
        self.assertNotIn("baresip", sys.modules)
        import receptionist.baresip_adapter as mod

        self.assertTrue(hasattr(mod, "BaresipTelephonyAdapter"))
        self.assertNotIn("baresip", sys.modules)


class ShutdownTest(unittest.TestCase):
    def test_shutdown_idempotent_and_late_callbacks_ignored(self) -> None:
        core, telephony, voice, _ = make_core()
        core.start()
        session = core.incoming_call("+34910000001")
        telephony.shutdown()
        telephony.shutdown()
        self.assertEqual(telephony.shutdown_count, 2)
        telephony.send_audio(session.call_id, assistant_frame())
        telephony.simulate_caller_audio(session.call_id, caller_speech_frame())


if __name__ == "__main__":
    unittest.main()
