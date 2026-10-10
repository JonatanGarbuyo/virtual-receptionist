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

from fakes import (
    FakeCallIds,
    FakeClock,
    FakePolicy,
    FakeTelephony,
    FakeVoiceBackend,
    FakeVoiceSession,
)


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


class _DTMFRecorder:
    """TelephonyListener fan-out: records DTMF, forwards everything."""

    def __init__(self, core, observed: list) -> None:
        self._core = core
        self._observed = observed

    def on_answered(self, call_id: str) -> None:
        self._core.on_answered(call_id)

    def on_caller_hangup(self, call_id: str) -> None:
        self._core.on_caller_hangup(call_id)

    def on_hangup_completed(self, call_id: str) -> None:
        self._core.on_hangup_completed(call_id)

    def on_transfer_result(self, call_id: str, result) -> None:
        self._core.on_transfer_result(call_id, result)

    def on_caller_audio(self, call_id: str, frame) -> None:
        self._core.on_caller_audio(call_id, frame)

    def on_dtmf(self, call_id: str, digit: str) -> None:
        self._observed.append((call_id, digit))
        self._core.on_dtmf(call_id, digit)

    def on_remote_hold(self, call_id: str, held: bool) -> None:
        self._core.on_remote_hold(call_id, held)


class _TransferRecorder:
    """Records transfer outcomes delivered by the adapter."""

    def __init__(self, results: list) -> None:
        self._results = results

    def on_transfer_result(self, call_id: str, result) -> None:
        self._results.append((call_id, result))


class _AnsweredRecorder:
    """Records answered events delivered by the adapter."""

    def __init__(self, answered: list) -> None:
        self._answered = answered

    def on_answered(self, call_id: str) -> None:
        self._answered.append(call_id)


class _AudioRecorder:
    """Records caller-audio frames delivered by the adapter."""

    def __init__(self, delivered: list) -> None:
        self._delivered = delivered

    def on_caller_audio(self, call_id: str, frame) -> None:
        self._delivered.append((call_id, frame))


class _DtmfRecorder:
    """Records DTMF digits delivered by the adapter."""

    def __init__(self, delivered: list) -> None:
        self._delivered = delivered

    def on_dtmf(self, call_id: str, digit: str) -> None:
        self._delivered.append((call_id, digit))


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


class G711FixedVectorTest(unittest.TestCase):
    """Codec conformance against an independent reference table.

    The vectors below were generated once from an independent G.711
    implementation (stdlib `audioop`, since removed in newer Pythons,
    hence frozen here) — never from the module under test. A broken
    encoder/decoder (e.g. an inverted A-law sign bit) fails these even
    when a self-consistent round-trip would pass.
    """

    # sample -> [expected PCMU code, expected PCMA code]
    ENCODE_VECTORS = {
        0: [255, 213],
        1: [255, 213],
        -1: [127, 85],
        127: [239, 210],
        -128: [111, 82],
        255: [231, 218],
        -256: [103, 90],
        256: [231, 197],
        1000: [206, 250],
        -1000: [78, 122],
        8192: [159, 181],
        -8192: [31, 10],
        16383: [143, 186],
        -16384: [15, 58],
        30000: [130, 168],
        -30000: [2, 40],
        32767: [128, 170],
        -32768: [0, 42],
    }

    # code -> expected decoded sample, per law
    DECODE_ULAW = {0x00: -32124, 0x55: -716, 0x7F: 0, 0x80: 32124, 0xD5: 716, 0xFF: 0}
    DECODE_ALAW = {0x00: -5504, 0x55: -8, 0x7F: -848, 0x80: 5504, 0xD5: 8, 0xFF: 848}

    def test_pcmu_encode_matches_reference_vectors(self) -> None:
        from receptionist.telephony_codecs import linear_to_ulaw

        for sample, (expected_ulaw, _) in self.ENCODE_VECTORS.items():
            self.assertEqual(
                linear_to_ulaw(sample), expected_ulaw, f"pcmu({sample})"
            )

    def test_pcma_encode_matches_reference_vectors(self) -> None:
        from receptionist.telephony_codecs import linear_to_alaw

        for sample, (_, expected_alaw) in self.ENCODE_VECTORS.items():
            self.assertEqual(
                linear_to_alaw(sample), expected_alaw, f"pcma({sample})"
            )

    def test_pcmu_decode_matches_reference_vectors(self) -> None:
        from receptionist.telephony_codecs import ulaw_to_linear

        for code, expected in self.DECODE_ULAW.items():
            self.assertEqual(ulaw_to_linear(code), expected, f"ulaw({code:#x})")

    def test_pcma_decode_matches_reference_vectors(self) -> None:
        from receptionist.telephony_codecs import alaw_to_linear

        for code, expected in self.DECODE_ALAW.items():
            self.assertEqual(alaw_to_linear(code), expected, f"alaw({code:#x})")

    def test_alaw_sign_convention_is_inverted_from_mulaw(self) -> None:
        # A-law stores positive samples with the MSB set (0xd5 decodes
        # to +8); μ-law silence is 0xff decoding to 0. A sign-bit mixup
        # negates the whole signal yet preserves energy, so this pins
        # decode polarity explicitly rather than via RMS.
        from receptionist.telephony_codecs import (
            alaw_to_linear,
            linear_to_alaw,
            linear_to_ulaw,
            ulaw_to_linear,
        )

        self.assertEqual(linear_to_alaw(0), 0xD5)
        self.assertEqual(linear_to_ulaw(0), 0xFF)
        self.assertGreater(alaw_to_linear(0xD5), 0)
        self.assertLess(alaw_to_linear(0x55), 0)
        self.assertEqual(ulaw_to_linear(0xFF), 0)


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

    def test_playback_finished_waits_for_rtp_drain(self) -> None:
        """Generation end with queued playout keeps barge-in armed: the
        session stays SPEAKING until the adapter drains, so caller
        speech over the tail still barges (and flushes) instead of
        opening a mixed turn over stale audio."""
        core, telephony, voice, _ = make_core()
        core.start()
        session, live, turn = drive_to_speaking(
            core, telephony, voice, "+34910000001"
        )
        self.assertGreater(telephony.playout_pending_bytes(session.call_id), 0)
        live.finish_playback(turn)
        # Still draining: barge-in stays armed, no turn opens yet.
        self.assertEqual(session.mode, ActiveMode.SPEAKING)
        session.push_caller_audio(caller_speech_frame())
        self.assertIn(session.call_id, telephony.flushed)
        self.assertEqual(session.mode, ActiveMode.LISTENING)

    def test_drained_playback_returns_to_listening(self) -> None:
        core, telephony, voice, _ = make_core()
        core.start()
        session, live, turn = drive_to_speaking(
            core, telephony, voice, "+34910000001"
        )
        live.finish_playback(turn)
        self.assertEqual(session.mode, ActiveMode.SPEAKING)
        telephony.drain_playout(session.call_id, 10**9)
        self.assertEqual(telephony.playout_pending_bytes(session.call_id), 0)
        # Next media event observes the drained queue and listens.
        session.push_caller_audio(
            make_frame(silence_pcm(duration_seconds=0.1, sample_rate=16000), 16000)
        )
        self.assertEqual(session.mode, ActiveMode.LISTENING)
        self.assertNotIn(session.call_id, telephony.flushed)

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
        # DTMF RX keeps its call id end to end: record what the core
        # observes per call through a wrapping listener check.
        observed: list[tuple[str, str]] = []
        telephony.set_listener(_DTMFRecorder(core, observed))
        telephony.simulate_dtmf(session_a.call_id, "5")
        telephony.simulate_dtmf(session_b.call_id, "9")
        self.assertEqual(observed, [(session_a.call_id, "5"), (session_b.call_id, "9")])
        # DTMF TX records per call, independently.
        self.assertEqual(
            telephony.sent_dtmf.get(session_a.call_id, []), []
        )  # RX path only; TX below
        telephony.send_dtmf(session_a.call_id, "1")
        telephony.send_dtmf(session_b.call_id, "23")
        self.assertEqual(telephony.sent_dtmf[session_a.call_id], ["1"])
        self.assertEqual(telephony.sent_dtmf[session_b.call_id], ["23"])
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

    def test_remote_hold_gates_media_and_resume_flushes(self) -> None:
        core, telephony, voice, _ = make_core()
        core.start()
        session = core.incoming_call("+34910000001")
        voice.sessions[session.call_id].finish_playback(session.current_turn)
        telephony.simulate_remote_hold(session.call_id, True)
        telephony.simulate_caller_audio(session.call_id, caller_speech_frame())
        self.assertEqual(voice.sessions[session.call_id].pushed_audio, [])
        telephony.simulate_remote_hold(session.call_id, False)
        self.assertIn(session.call_id, telephony.flushed)
        telephony.simulate_caller_audio(session.call_id, caller_speech_frame())
        self.assertEqual(len(voice.sessions[session.call_id].pushed_audio), 1)

    def test_local_resume_discards_held_playout(self) -> None:
        core, telephony, voice, _ = make_core()
        core.start()
        session, _live, _turn = drive_to_speaking(
            core, telephony, voice, "+34910000001"
        )
        self.assertTrue(telephony.playout.get(session.call_id))
        telephony.hold(session.call_id)
        telephony.resume(session.call_id)
        self.assertIn(session.call_id, telephony.flushed)
        self.assertNotIn(session.call_id, telephony.playout)


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

    def test_media_loss_degrades_and_recovery_restores(self) -> None:
        from receptionist.baresip_adapter import CoreStatusBridge

        core, telephony, _, _ = make_core()
        core.start()
        telephony.set_status_listener(CoreStatusBridge(core))
        telephony.report_registration(TelephonyRegistrationState.REGISTERED)
        self.assertEqual(core.health.status, HealthStatus.READY)
        telephony.report_media(False, "call-1")
        self.assertEqual(core.health.status, HealthStatus.DEGRADED)
        telephony.report_media(True, "call-1")
        self.assertEqual(core.health.status, HealthStatus.READY)
        # Unknown calls and repeated reports never break the mapping.
        telephony.report_media(False, "call-999")
        self.assertEqual(core.health.status, HealthStatus.DEGRADED)
        telephony.report_registration(TelephonyRegistrationState.REGISTERED)
        self.assertEqual(core.health.status, HealthStatus.READY)


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
        # White-box below: project-owned internals, no SIP involved.
        adapter._calls["call-1"] = record  # noqa: SLF001
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

    def test_callback_delivery_is_isolated_per_call(self) -> None:
        import threading

        from receptionist.baresip_adapter import BaresipTelephonyAdapter
        from receptionist.telephony_config import TelephonyConfig

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        adapter._running = True  # noqa: SLF001 (listener delivery gate)
        delivered: list = []
        adapter.set_listener(_DtmfRecorder(delivered))
        # A stuck delivery on call A must never block call B: per-call
        # serializer, proven with a barrier (timeout is fail-safe only).
        stuck = adapter._cb_lock_for("call-a")  # noqa: SLF001
        stuck.acquire()
        done = threading.Event()

        def deliver_b() -> None:
            adapter._deliver("call-b", "on_dtmf", "5")  # noqa: SLF001
            done.set()

        worker = threading.Thread(target=deliver_b, daemon=True)
        worker.start()
        try:
            self.assertTrue(done.wait(timeout=5))
        finally:
            stuck.release()
        worker.join(timeout=5)
        self.assertEqual(delivered, [("call-b", "5")])

    def test_adapter_imports_without_native_binding(self) -> None:
        self.assertNotIn("baresip", sys.modules)
        import receptionist.baresip_adapter as mod

        self.assertTrue(hasattr(mod, "BaresipTelephonyAdapter"))
        self.assertNotIn("baresip", sys.modules)

    def test_remote_hold_gates_pump_and_resume_flushes(self) -> None:
        from types import SimpleNamespace

        from receptionist.baresip_adapter import BaresipTelephonyAdapter, _CallRecord
        from receptionist.telephony_config import TelephonyConfig

        written: list = []

        class _AudioStub:
            def info(self):  # noqa: ANN202
                return SimpleNamespace(
                    tx_sample_rate=8000, rx_sample_rate=8000
                )

            def write(self, pcm) -> int:  # noqa: ANN001, ANN202
                written.append(bytes(pcm))
                return len(pcm)

            def read(self, max_bytes: int) -> bytes:  # noqa: ANN202
                return tone_pcm(duration_seconds=0.02, sample_rate=8000)[:max_bytes]

            def flush_tx(self) -> None:
                written.clear()

        class _NativeStub:
            audio = _AudioStub()

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        delivered: list = []
        adapter.set_listener(_AudioRecorder(delivered))
        adapter._running = True  # noqa: SLF001 (listener delivery gate)
        record = _CallRecord(app_id="call-1", native=_NativeStub())
        adapter._calls["call-1"] = record  # noqa: SLF001
        # Healthy: TX drains, RX enqueues; dispatch delivers.
        adapter.send_audio("call-1", make_frame(
            tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
        adapter._pump_tx(record)  # noqa: SLF001
        self.assertTrue(written)
        adapter._drain_rx(record)  # noqa: SLF001
        adapter._dispatch_once("call-1")  # noqa: SLF001
        self.assertTrue(delivered)
        # Remote hold: neither direction flows.
        written.clear()
        delivered.clear()
        record.remote_hold = True
        adapter.send_audio("call-1", make_frame(
            tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
        adapter._pump_tx(record)  # noqa: SLF001
        self.assertEqual(written, [])
        adapter._drain_rx(record)  # noqa: SLF001
        adapter._dispatch_once("call-1")  # noqa: SLF001
        self.assertEqual(delivered, [])
        # Resume flushes stale media; flow restores after.
        adapter._on_native_remote_hold("call-1", False)  # noqa: SLF001
        self.assertFalse(record.remote_hold)
        adapter._drain_rx(record)  # noqa: SLF001
        adapter._dispatch_once("call-1")  # noqa: SLF001
        self.assertTrue(delivered)

    def test_blocked_listener_cannot_stall_other_call_media(self) -> None:
        """M1: a call blocked in on_caller_audio must not stall call B.

        Call A's listener blocks on a barrier (barge-in cancelling
        STT/LLM output, which can take seconds). Driving the production
        pump path (``_pump_once``) must still pump TX and enqueue RX for
        both calls and deliver B's audio; A's frame waits in its FIFO
        and is delivered in order once unblocked. Barrier/Event waits
        carry timeouts as fail-safes only, never as the mechanism.
        """
        import threading
        from types import SimpleNamespace

        from receptionist.baresip_adapter import BaresipTelephonyAdapter, _CallRecord
        from receptionist.telephony_config import TelephonyConfig

        written_b: list = []
        delivered_a: list = []
        delivered_b: list = []
        unblock_a = threading.Event()
        a_done = threading.Event()
        b_done = threading.Event()

        class _AudioStub:
            def __init__(self, written: list | None = None) -> None:
                self._written = written

            def info(self):  # noqa: ANN202
                return SimpleNamespace(
                    tx_sample_rate=8000, rx_sample_rate=8000
                )

            def write(self, pcm) -> int:  # noqa: ANN001, ANN202
                if self._written is not None:
                    self._written.append(bytes(pcm))
                return len(pcm)

            def read(self, max_bytes: int) -> bytes:  # noqa: ANN202
                return tone_pcm(duration_seconds=0.02, sample_rate=8000)[:max_bytes]

        class _BlockingListener:
            def on_caller_audio(self, call_id: str, frame) -> None:  # noqa: ANN001, ANN202
                if call_id == "call-a":
                    # Simulate blocking barge-in cancellation.
                    self._entered_a.set()
                    unblock_a.wait(timeout=10)
                    delivered_a.append((call_id, frame))
                    a_done.set()
                else:
                    delivered_b.append((call_id, frame))
                    b_done.set()

            def __init__(self) -> None:
                self._entered_a = threading.Event()

        class _NativeStub:
            def __init__(self, written: list | None = None) -> None:
                self.audio = _AudioStub(written)

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        listener = _BlockingListener()
        adapter.set_listener(listener)
        adapter._running = True  # noqa: SLF001 (listener delivery gate)
        record_a = _CallRecord(app_id="call-a", native=_NativeStub())
        record_b = _CallRecord(app_id="call-b", native=_NativeStub(written_b))
        adapter._calls["call-a"] = record_a  # noqa: SLF001
        adapter._calls["call-b"] = record_b  # noqa: SLF001
        adapter.send_audio("call-b", make_frame(
            tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
        # Production dispatch path for A (its own dispatcher thread,
        # as started at establishment); the pump itself runs below on
        # this thread.
        adapter._ensure_rx_dispatch("call-a")  # noqa: SLF001
        adapter._ensure_rx_dispatch("call-b")  # noqa: SLF001
        try:
            # One production pump pass: TX for B plus RX enqueue for
            # both calls. It must return promptly even though A's
            # delivery is about to block.
            adapter._pump_once()  # noqa: SLF001
            # B's TX was pumped by the shared pump.
            self.assertTrue(written_b)
            # B's audio is delivered while A stays blocked.
            self.assertTrue(b_done.wait(timeout=5))
            self.assertTrue(listener._entered_a.wait(timeout=5))
            self.assertEqual(delivered_a, [])
            # Release A: its queued frame is delivered in order.
            unblock_a.set()
            self.assertTrue(a_done.wait(timeout=5))
            self.assertEqual(len(delivered_a), 1)
            self.assertEqual(delivered_a[0][0], "call-a")
        finally:
            unblock_a.set()
            adapter._calls.pop("call-a", None)  # noqa: SLF001
            adapter._calls.pop("call-b", None)  # noqa: SLF001
            for record in (record_a, record_b):
                thread = record.rx_thread
                if thread is not None:
                    thread.join(timeout=5)

    def test_transfer_watchdog_reports_timeout_when_open(self) -> None:
        from receptionist.baresip_adapter import BaresipTelephonyAdapter, _CallRecord
        from receptionist.boundaries import TransferResult
        from receptionist.telephony_config import TelephonyConfig

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        adapter._running = True  # noqa: SLF001 (listener delivery gate)
        results: list = []
        adapter.set_listener(_TransferRecorder(results))
        record = _CallRecord(app_id="call-9", native=None)
        record.transfer_open = True
        adapter._calls["call-9"] = record  # noqa: SLF001
        adapter._transfer_watchdog_fired("call-9")  # noqa: SLF001
        # Watchdog outcomes route through the per-call dispatcher (never
        # inline on the timer thread): drain deterministically.
        adapter._drain_native_events("call-9")  # noqa: SLF001
        self.assertEqual(results, [("call-9", TransferResult.TIMEOUT)])
        self.assertFalse(record.transfer_open)
        # Second firing is a no-op (already closed out).
        adapter._transfer_watchdog_fired("call-9")  # noqa: SLF001
        self.assertEqual(results, [("call-9", TransferResult.TIMEOUT)])

    def test_lifecycle_events_dispatch_off_calling_thread_per_call(self) -> None:
        """R2M2: stack lifecycle callbacks must not run core inline.

        Remote BYE on call A enters a listener whose cancellation
        blocks; while A is blocked, call B's BYE (enqueued through the
        production stack-event seam) must be processed without waiting
        for A. Stopping the adapter drops pending events and stops
        dispatchers with no callbacks after STOPPED. Barrier/Event
        waits carry timeouts as fail-safes only.
        """
        import threading

        from receptionist.baresip_adapter import BaresipTelephonyAdapter, _CallRecord
        from receptionist.telephony_config import TelephonyConfig

        hung_a: list = []
        hung_b: list = []
        unblock_a = threading.Event()
        a_done = threading.Event()
        b_done = threading.Event()

        class _HangupListener:
            def on_caller_hangup(self, call_id: str) -> None:  # noqa: ANN202
                if call_id == "call-a":
                    self._entered_a.set()
                    unblock_a.wait(timeout=10)
                    hung_a.append(call_id)
                    a_done.set()
                else:
                    hung_b.append(call_id)
                    b_done.set()

            def __init__(self) -> None:
                self._entered_a = threading.Event()

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        listener = _HangupListener()
        adapter.set_listener(listener)
        adapter._running = True  # noqa: SLF001
        adapter._calls["call-a"] = _CallRecord(app_id="call-a")  # noqa: SLF001
        adapter._calls["call-b"] = _CallRecord(app_id="call-b")  # noqa: SLF001
        adapter._ensure_rx_dispatch("call-a")  # noqa: SLF001
        adapter._ensure_rx_dispatch("call-b")  # noqa: SLF001
        threads = [
            adapter._calls["call-a"].rx_thread,  # noqa: SLF001
            adapter._calls["call-b"].rx_thread,  # noqa: SLF001
        ]
        try:
            # Production seam below the vendor-typed stack callbacks:
            # plain data in, prompt return, core work on dispatchers.
            adapter._enqueue_native_event("call-a", ("closed", "OK"))  # noqa: SLF001
            adapter._enqueue_native_event("call-b", ("closed", "OK"))  # noqa: SLF001
            # B is fully processed while A stays blocked in the core.
            self.assertTrue(b_done.wait(timeout=5))
            self.assertTrue(listener._entered_a.wait(timeout=5))
            self.assertEqual(hung_a, [])
            self.assertEqual(hung_b, ["call-b"])
            # Release A: its queued close is delivered exactly once.
            unblock_a.set()
            self.assertTrue(a_done.wait(timeout=5))
            self.assertEqual(hung_a, ["call-a"])
            # Stop: pending events drop, dispatchers exit, and nothing
            # is delivered after STOPPED.
            adapter._running = False  # noqa: SLF001
            adapter._enqueue_native_event("call-b", ("closed", "OK"))  # noqa: SLF001
            self.assertEqual(hung_b, ["call-b"])
        finally:
            unblock_a.set()
            adapter._running = False  # noqa: SLF001
            for thread in threads:
                if thread is not None:
                    thread.join(timeout=5)
            for thread in threads:
                self.assertFalse(thread is not None and thread.is_alive())

    def test_terminal_delivery_does_not_resurrect_locks(self) -> None:
        """Lifecycle leak note: closing a call must leave no permanent
        per-call serializer behind, while late outcomes for removed
        calls are still delivered (core ignores unknown ids)."""
        from receptionist.baresip_adapter import BaresipTelephonyAdapter, _CallRecord
        from receptionist.boundaries import TransferResult
        from receptionist.telephony_config import TelephonyConfig

        done: list = []
        outcomes: list = []

        class _CloseRecorder:
            def on_hangup_completed(self, call_id: str) -> None:  # noqa: ANN202
                done.append(call_id)

            def on_transfer_result(self, call_id: str, result) -> None:  # noqa: ANN202
                outcomes.append((call_id, result))

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        adapter.set_listener(_CloseRecorder())
        adapter._running = True  # noqa: SLF001
        record = _CallRecord(app_id="call-x")
        record.local_close = True
        adapter._calls["call-x"] = record  # noqa: SLF001
        adapter._on_native_closed("call-x", "OK")  # noqa: SLF001
        self.assertEqual(done, ["call-x"])
        self.assertNotIn("call-x", adapter._calls)  # noqa: SLF001
        self.assertNotIn("call-x", adapter._cb_locks)  # noqa: SLF001
        # Late outcome for the removed call is still delivered, and
        # still creates no permanent entry.
        adapter._emit_transfer_result("call-x", TransferResult.TIMEOUT)  # noqa: SLF001
        self.assertEqual(outcomes, [("call-x", TransferResult.TIMEOUT)])
        self.assertNotIn("call-x", adapter._cb_locks)  # noqa: SLF001

    def test_declined_leg_never_emits_answered(self) -> None:
        from receptionist.baresip_adapter import BaresipTelephonyAdapter, _CallRecord
        from receptionist.telephony_config import TelephonyConfig

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        adapter._running = True  # noqa: SLF001 (listener delivery gate)
        answered: list = []
        adapter.set_listener(_AnsweredRecorder(answered))
        class _AudioStub:
            def info(self):  # noqa: ANN202
                from types import SimpleNamespace

                return SimpleNamespace(tx_sample_rate=8000, rx_sample_rate=8000)

        class _NativeStub:
            audio = _AudioStub()

        record = _CallRecord(app_id="declined-1", native=_NativeStub())
        record.decline_release = True
        adapter._calls["declined-1"] = record  # noqa: SLF001
        adapter._on_native_established("declined-1")  # noqa: SLF001
        self.assertEqual(answered, [])
        self.assertTrue(record.established)


class MediaErrorStreakTest(unittest.TestCase):
    """M3: persistent local media failure must surface exactly once."""

    def _adapter(self):
        from receptionist.baresip_adapter import BaresipTelephonyAdapter
        from receptionist.telephony_config import TelephonyConfig

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        adapter._running = True  # noqa: SLF001 (listener delivery gate)
        return adapter

    def test_persistent_tx_failure_reports_once_and_signals(self) -> None:
        from types import SimpleNamespace

        from receptionist.baresip_adapter import (
            MEDIA_ERROR_STREAK_LIMIT,
            _CallRecord,
        )

        states: list = []
        signals: list = []

        class _StatusStub:
            def on_registration_state(self, state, detail="") -> None:  # noqa: ANN001, ANN202
                pass

            def on_media_state(self, healthy, call_id="", detail="") -> None:  # noqa: ANN001, ANN202
                states.append((healthy, call_id))

        class _SignalRecorder:
            def on_media_failed(self, call_id, detail="") -> None:  # noqa: ANN001, ANN202
                signals.append((call_id, detail))

        class _AudioStub:
            def info(self):  # noqa: ANN202
                return SimpleNamespace(
                    tx_sample_rate=8000, rx_sample_rate=8000
                )

            def write(self, pcm) -> int:  # noqa: ANN001, ANN202
                raise RuntimeError("aumem write broken")

            def read(self, max_bytes: int) -> bytes:  # noqa: ANN202
                return b""

        class _NativeStub:
            audio = _AudioStub()

        adapter = self._adapter()
        adapter.set_status_listener(_StatusStub())
        adapter.set_listener(_SignalRecorder())
        record = _CallRecord(app_id="call-tx", native=_NativeStub())
        adapter._calls["call-tx"] = record  # noqa: SLF001
        adapter.send_audio("call-tx", make_frame(
            tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
        for _ in range(MEDIA_ERROR_STREAK_LIMIT + 10):
            adapter._pump_tx(record)  # noqa: SLF001
        # The trip only enqueues (the pump never runs listener code):
        # drain the per-call dispatcher deterministically.
        adapter._drain_native_events("call-tx")  # noqa: SLF001
        # Exactly one unhealthy transition (no per-frame spam) naming
        # the call, and exactly one application signal: the adapter
        # never owns the business outcome, so the leg stays up for the
        # application to fall back or terminate.
        unhealthy = [s for s in states if s[0] is False]
        self.assertEqual(len(unhealthy), 1)
        self.assertEqual(unhealthy[0][1], "call-tx")
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0][0], "call-tx")
        self.assertIn("tx", signals[0][1])
        self.assertFalse(record.local_close)
        self.assertEqual(record.tx_errors, MEDIA_ERROR_STREAK_LIMIT + 10)

    def test_persistent_rx_failure_reports_once_and_signals(self) -> None:
        from types import SimpleNamespace

        from receptionist.baresip_adapter import (
            MEDIA_ERROR_STREAK_LIMIT,
            _CallRecord,
        )

        states: list = []
        signals: list = []

        class _StatusStub:
            def on_registration_state(self, state, detail="") -> None:  # noqa: ANN001, ANN202
                pass

            def on_media_state(self, healthy, call_id="", detail="") -> None:  # noqa: ANN001, ANN202
                states.append((healthy, call_id))

        class _SignalRecorder:
            def on_media_failed(self, call_id, detail="") -> None:  # noqa: ANN001, ANN202
                signals.append((call_id, detail))

        class _AudioStub:
            def info(self):  # noqa: ANN202
                return SimpleNamespace(
                    tx_sample_rate=8000, rx_sample_rate=8000
                )

            def write(self, pcm) -> int:  # noqa: ANN001, ANN202
                return len(pcm)

            def read(self, max_bytes: int) -> bytes:  # noqa: ANN202
                raise RuntimeError("aumem read broken")

        class _NativeStub:
            audio = _AudioStub()

        adapter = self._adapter()
        adapter.set_status_listener(_StatusStub())
        adapter.set_listener(_SignalRecorder())
        record = _CallRecord(app_id="call-rx", native=_NativeStub())
        adapter._calls["call-rx"] = record  # noqa: SLF001
        for _ in range(MEDIA_ERROR_STREAK_LIMIT + 10):
            adapter._drain_rx(record)  # noqa: SLF001
        adapter._drain_native_events("call-rx")  # noqa: SLF001
        unhealthy = [s for s in states if s[0] is False]
        self.assertEqual(len(unhealthy), 1)
        self.assertEqual(unhealthy[0][1], "call-rx")
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0][0], "call-rx")
        self.assertIn("rx", signals[0][1])
        self.assertFalse(record.local_close)

    def test_media_success_resets_streak_and_recovers(self) -> None:
        from types import SimpleNamespace

        from receptionist.baresip_adapter import (
            MEDIA_ERROR_STREAK_LIMIT,
            _CallRecord,
        )

        states: list = []
        failing = True

        class _StatusStub:
            def on_registration_state(self, state, detail="") -> None:  # noqa: ANN001, ANN202
                pass

            def on_media_state(self, healthy, call_id="", detail="") -> None:  # noqa: ANN001, ANN202
                states.append((healthy, call_id))

        class _AudioStub:
            def info(self):  # noqa: ANN202
                return SimpleNamespace(
                    tx_sample_rate=8000, rx_sample_rate=8000
                )

            def write(self, pcm) -> int:  # noqa: ANN001, ANN202
                if failing:
                    raise RuntimeError("transient write failure")
                return len(pcm)

            def read(self, max_bytes: int) -> bytes:  # noqa: ANN202
                return b""

        class _NativeStub:
            audio = _AudioStub()

        adapter = self._adapter()
        adapter.set_status_listener(_StatusStub())
        record = _CallRecord(app_id="call-flap", native=_NativeStub())
        adapter._calls["call-flap"] = record  # noqa: SLF001
        # Blips below the limit never surface ...
        for _ in range(MEDIA_ERROR_STREAK_LIMIT - 1):
            adapter.send_audio("call-flap", make_frame(
                tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
            adapter._pump_tx(record)  # noqa: SLF001
        self.assertEqual(
            [s for s in states if s[0] is False], []
        )
        # ... and one success forgets the whole streak ...
        failing = False
        adapter.send_audio("call-flap", make_frame(
            tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
        adapter._pump_tx(record)  # noqa: SLF001
        self.assertEqual(record.tx_error_streak, 0)
        # ... so a fresh full streak is required to trip.
        failing = True
        for _ in range(MEDIA_ERROR_STREAK_LIMIT - 1):
            adapter.send_audio("call-flap", make_frame(
                tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
            adapter._pump_tx(record)  # noqa: SLF001
        self.assertEqual(
            [s for s in states if s[0] is False], []
        )
        adapter.send_audio("call-flap", make_frame(
            tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
        adapter._pump_tx(record)  # noqa: SLF001
        unhealthy = [s for s in states if s[0] is False]
        self.assertEqual(len(unhealthy), 1)


class MediaFailureCrossLayerTest(unittest.TestCase):
    """R2M1: an adapter media-error streak must drive the application
    failure path, not strand the session.

    Live ACTIVE session + real adapter TX streak -> exactly one media
    failure signal -> session leaves ACTIVE through the intended
    failure path (fallback attempt, else deterministic exit) ->
    telephony leg closes -> session evicted from core -> AI slot
    released -> repeated errors and late close do not double-handle.
    """

    def test_media_streak_terminates_evicts_and_releases_slot(self) -> None:
        from types import SimpleNamespace

        from receptionist.alerting import (
            CODE_TELEPHONY_MEDIA_LOST,
            HealthComponent,
        )
        from receptionist.baresip_adapter import (
            MEDIA_ERROR_STREAK_LIMIT,
            BaresipTelephonyAdapter,
            _CallRecord,
            wire_baresip_core,
        )
        from receptionist.call_session import CallState
        from receptionist.resilience import ResilienceConfig
        from receptionist.telephony_config import TelephonyConfig

        voice = FakeVoiceBackend()
        clock = FakeClock()
        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        core = ReceptionistCore(
            telephony=adapter,
            voice=voice,
            config_service=ConfigService(InMemoryConfigRepository(dict(
                {"greeting": GREETING, "language": "es"}))),
            policy=FakePolicy(),
            clock=clock,
            resilience=ResilienceConfig(max_ai_sessions=1),
            policy_engine=PolicyEngine(
                destinations={}, fallback_id="none", limits=Limits()),
            runtime=RuntimeStorage(
                calls=InMemoryCallRepository(),
                messages=InMemoryMessageRepository(clock=clock),
                transcripts=InMemoryTranscriptStore(),
                audit=InMemoryAuditLog(),
            ),
            retention=RetentionPolicy(),
            call_ids=FakeCallIds(),
        )
        wire_baresip_core(core, adapter)
        adapter._running = True  # noqa: SLF001
        core.start()
        # The unit seam has no SIP stack: attest the registered state
        # the real matrix proves on the wire, so admission serves.
        from receptionist.boundaries import TelephonyRegistrationState

        core.report_telephony_state(TelephonyRegistrationState.REGISTERED)
        session = core.incoming_call("+34910000001")
        call_id = session.call_id

        class _FailingAudio:
            def info(self):  # noqa: ANN202
                return SimpleNamespace(
                    tx_sample_rate=8000, rx_sample_rate=8000
                )

            def write(self, pcm) -> int:  # noqa: ANN001, ANN202
                raise RuntimeError("aumem write broken")

            def read(self, max_bytes: int) -> bytes:  # noqa: ANN202
                return b""

        class _FailingNative:
            audio = _FailingAudio()

        record = _CallRecord(app_id=call_id, native=_FailingNative())
        adapter._calls[call_id] = record  # noqa: SLF001
        core.on_answered(call_id)
        self.assertEqual(session.state, CallState.ACTIVE)

        signals: list = []
        orig_signal = core.on_media_failed
        core.on_media_failed = lambda cid, detail="": (  # noqa: E731
            signals.append(cid), orig_signal(cid, detail))
        hangups: list = []
        orig_hangup = adapter.hangup
        adapter.hangup = lambda cid: (  # noqa: E731
            hangups.append(cid), orig_hangup(cid))

        adapter.send_audio(call_id, make_frame(
            tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
        try:
            for _ in range(MEDIA_ERROR_STREAK_LIMIT):
                adapter._pump_tx(record)  # noqa: SLF001
            # The streak trip enqueues the application signal; drain
            # the per-call dispatcher deterministically (no threads,
            # no sleeps) before asserting the cross-layer outcome.
            adapter._drain_native_events(call_id)  # noqa: SLF001
            # Exactly one signal; the session left ACTIVE through the
            # failure path (no fallback configured -> exit + hangup).
            self.assertEqual(signals, [call_id])
            self.assertEqual(session.state, CallState.TERMINATING)
            self.assertEqual(hangups, [call_id])
            # The voice output was cancelled and the voice session is
            # still open (close happens at the terminal transition).
            live = voice.sessions[call_id]
            self.assertTrue(live.cancels)
            self.assertFalse(live.closed)
            # Media health degraded alongside the failure path, once.
            media_lost = [
                c for c in core.monitor.active_conditions()
                if c.component is HealthComponent.TELEPHONY
                and c.code == CODE_TELEPHONY_MEDIA_LOST
            ]
            self.assertEqual(len(media_lost), 1)
            adapter._on_native_closed(call_id, "OK")  # noqa: SLF001
            self.assertEqual(session.state, CallState.ENDED)
            self.assertIsNone(core.get_session(call_id))
            # Terminal transition closed the voice session.
            self.assertTrue(voice.sessions[call_id].closed)
            # Repeated errors and a late duplicate signal/close do not
            # double-handle: still one hangup, still one emission.
            for _ in range(10):
                adapter.send_audio(call_id, make_frame(
                    tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
                adapter._pump_tx(record)  # noqa: SLF001
            orig_signal(call_id, "late duplicate")
            adapter._on_native_closed(call_id, "OK")  # noqa: SLF001
            self.assertEqual(hangups, [call_id])
            self.assertEqual(signals, [call_id])
            # The AI slot is free again: a new call is AI-admitted.
            session2 = core.incoming_call("+34910000002")
            core.on_answered(session2.call_id)
            self.assertEqual(session2.state, CallState.ACTIVE)
        finally:
            core.on_media_failed = orig_signal
            adapter.hangup = orig_hangup


def _wired_adapter_core(max_ai_sessions: int = 1):
    """Real adapter + core wiring for cross-layer dispatch tests.

    No SIP stack: records are bound manually and stack callbacks are
    driven through the production enqueue/dispatch seam. Returns
    ``(core, adapter, voice, clock)`` with the adapter running and the
    core READY-admitted.
    """
    from receptionist.baresip_adapter import (
        BaresipTelephonyAdapter,
        wire_baresip_core,
    )
    from receptionist.resilience import ResilienceConfig
    from receptionist.telephony_config import TelephonyConfig

    voice = FakeVoiceBackend()
    clock = FakeClock()
    adapter = BaresipTelephonyAdapter(
        TelephonyConfig(username="u", domain="d", password="p")
    )
    core = ReceptionistCore(
        telephony=adapter,
        voice=voice,
        config_service=ConfigService(InMemoryConfigRepository(dict(
            {"greeting": GREETING, "language": "es"}))),
        policy=FakePolicy(),
        clock=clock,
        resilience=ResilienceConfig(max_ai_sessions=max_ai_sessions),
        policy_engine=PolicyEngine(
            destinations={}, fallback_id="none", limits=Limits()),
        runtime=RuntimeStorage(
            calls=InMemoryCallRepository(),
            messages=InMemoryMessageRepository(clock=clock),
            transcripts=InMemoryTranscriptStore(),
            audit=InMemoryAuditLog(),
        ),
        retention=RetentionPolicy(),
        call_ids=FakeCallIds(),
    )
    wire_baresip_core(core, adapter)
    adapter._running = True  # noqa: SLF001
    core.start()
    core.report_telephony_state(TelephonyRegistrationState.REGISTERED)
    return core, adapter, voice, clock


class _StubAudio:
    """Native audio double with scripted TX/RX behaviour."""

    def __init__(self, *, fail_write=False, fail_read=False) -> None:
        from types import SimpleNamespace

        self._info = SimpleNamespace(
            tx_sample_rate=8000, rx_sample_rate=8000
        )
        self._fail_write = fail_write
        self._fail_read = fail_read
        self.written: list = []
        self.reads = 0
        self.flush_tx_calls = 0
        # Optional scripted reads modeling stale native buffers: when
        # set, reads drain this list first (then empty); when None,
        # reads return live tone like a flowing call.
        self.read_script: list | None = None

    def info(self):  # noqa: ANN202
        return self._info

    def write(self, pcm) -> int:  # noqa: ANN001, ANN202
        if self._fail_write:
            raise RuntimeError("aumem write broken")
        self.written.append(bytes(pcm))
        return len(pcm)

    def read(self, max_bytes: int) -> bytes:  # noqa: ANN202
        if self._fail_read:
            raise RuntimeError("aumem read broken")
        self.reads += 1
        if self.read_script is not None:
            return self.read_script.pop(0) if self.read_script else b""
        return tone_pcm(duration_seconds=0.02, sample_rate=8000)[:max_bytes]

    def flush_tx(self) -> None:
        self.flush_tx_calls += 1


class _StubNative:
    def __init__(self, audio) -> None:  # noqa: ANN001
        self.audio = audio


class MediaFailureRxCrossLayerTest(unittest.TestCase):
    """Test B: a persistent RX streak drives the same application
    failure path as TX (signal once, cancel voice, terminate, hangup
    once, evict, release the AI slot)."""

    def test_rx_streak_terminates_evicts_and_releases_slot(self) -> None:
        from receptionist.baresip_adapter import (
            MEDIA_ERROR_STREAK_LIMIT,
            _CallRecord,
        )
        from receptionist.call_session import CallState

        core, adapter, voice, _ = _wired_adapter_core()
        session = core.incoming_call("+34910000011")
        call_id = session.call_id
        record = _CallRecord(
            app_id=call_id, native=_StubNative(_StubAudio(fail_read=True))
        )
        adapter._calls[call_id] = record  # noqa: SLF001
        core.on_answered(call_id)
        self.assertEqual(session.state, CallState.ACTIVE)

        signals: list = []
        orig_signal = core.on_media_failed
        core.on_media_failed = lambda cid, detail="": (  # noqa: E731
            signals.append((cid, detail)), orig_signal(cid, detail))
        hangups: list = []
        orig_hangup = adapter.hangup
        adapter.hangup = lambda cid: (  # noqa: E731
            hangups.append(cid), orig_hangup(cid))
        try:
            for _ in range(MEDIA_ERROR_STREAK_LIMIT):
                adapter._drain_rx(record)  # noqa: SLF001
            adapter._drain_native_events(call_id)  # noqa: SLF001
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0][0], call_id)
            self.assertIn("rx", signals[0][1])
            self.assertEqual(session.state, CallState.TERMINATING)
            self.assertEqual(hangups, [call_id])
            live = voice.sessions[call_id]
            self.assertTrue(live.cancels)
            adapter._on_native_closed(call_id, "OK")  # noqa: SLF001
            self.assertEqual(session.state, CallState.ENDED)
            self.assertIsNone(core.get_session(call_id))
            self.assertTrue(voice.sessions[call_id].closed)
            session2 = core.incoming_call("+34910000012")
            core.on_answered(session2.call_id)
            self.assertEqual(session2.state, CallState.ACTIVE)
        finally:
            core.on_media_failed = orig_signal
            adapter.hangup = orig_hangup


class MediaFailureIsolationTest(unittest.TestCase):
    """Test C: call A fails its media streak while call B stays fully
    operational (ACTIVE, RX/TX flowing, no trip) and A's teardown
    never touches B."""

    def test_failing_call_does_not_disturb_other_call(self) -> None:
        from receptionist.baresip_adapter import (
            MEDIA_ERROR_STREAK_LIMIT,
            _CallRecord,
        )
        from receptionist.call_session import CallState

        core, adapter, _, _ = _wired_adapter_core(max_ai_sessions=2)
        session_a = core.incoming_call("+34910000021")
        session_b = core.incoming_call("+34910000022")
        record_a = _CallRecord(
            app_id=session_a.call_id,
            native=_StubNative(_StubAudio(fail_write=True)),
        )
        audio_b = _StubAudio()
        record_b = _CallRecord(
            app_id=session_b.call_id, native=_StubNative(audio_b)
        )
        adapter._calls[session_a.call_id] = record_a  # noqa: SLF001
        adapter._calls[session_b.call_id] = record_b  # noqa: SLF001
        core.on_answered(session_a.call_id)
        core.on_answered(session_b.call_id)
        self.assertEqual(session_a.state, CallState.ACTIVE)
        self.assertEqual(session_b.state, CallState.ACTIVE)

        hangups: list = []
        orig_hangup = adapter.hangup
        adapter.hangup = lambda cid: (  # noqa: E731
            hangups.append(cid), orig_hangup(cid))
        try:
            adapter.send_audio(
                session_a.call_id,
                make_frame(
                    tone_pcm(duration_seconds=0.02, sample_rate=16000),
                    16000,
                ),
            )
            adapter.send_audio(
                session_b.call_id,
                make_frame(
                    tone_pcm(duration_seconds=0.02, sample_rate=16000),
                    16000,
                ),
            )
            for _ in range(MEDIA_ERROR_STREAK_LIMIT):
                adapter._pump_tx(record_a)  # noqa: SLF001
                adapter._pump_tx(record_b)  # noqa: SLF001
                adapter._drain_rx(record_b)  # noqa: SLF001
            adapter._drain_native_events(session_a.call_id)  # noqa: SLF001
            adapter._drain_native_events(session_b.call_id)  # noqa: SLF001
            # A left ACTIVE exactly once; B never tripped.
            self.assertEqual(session_a.state, CallState.TERMINATING)
            self.assertEqual(hangups, [session_a.call_id])
            self.assertEqual(session_b.state, CallState.ACTIVE)
            self.assertEqual(record_b.tx_error_streak, 0)
            self.assertEqual(record_b.rx_error_streak, 0)
            self.assertTrue(audio_b.written)
            # A's leg closes and evicts; B is untouched.
            adapter._on_native_closed(session_a.call_id, "OK")  # noqa: SLF001
            self.assertEqual(session_a.state, CallState.ENDED)
            self.assertIsNone(core.get_session(session_a.call_id))
            self.assertEqual(session_b.state, CallState.ACTIVE)
            self.assertIsNotNone(core.get_session(session_b.call_id))
            self.assertEqual(hangups, [session_a.call_id])
        finally:
            adapter.hangup = orig_hangup


class MediaFailureHangupRaceTest(unittest.TestCase):
    """Test D: a media-error trip racing a remote BYE converges on one
    terminal outcome -- one hangup, one eviction, no zombie, and late
    events after close are inert. Both arrival orders are covered
    through the single per-call serialization (the production
    dispatcher processes both events on one thread)."""

    def _race_once(self, media_first: bool) -> None:
        from receptionist.baresip_adapter import (
            MEDIA_ERROR_STREAK_LIMIT,
            _CallRecord,
        )
        from receptionist.call_session import CallState

        core, adapter, voice, _ = _wired_adapter_core()
        session = core.incoming_call("+34910000031")
        call_id = session.call_id
        record = _CallRecord(
            app_id=call_id, native=_StubNative(_StubAudio(fail_write=True))
        )
        adapter._calls[call_id] = record  # noqa: SLF001
        core.on_answered(call_id)
        adapter.send_audio(call_id, make_frame(
            tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))

        hangups: list = []
        orig_hangup = adapter.hangup
        adapter.hangup = lambda cid: (  # noqa: E731
            hangups.append(cid), orig_hangup(cid))
        try:
            for _ in range(MEDIA_ERROR_STREAK_LIMIT):
                adapter._pump_tx(record)  # noqa: SLF001
            if media_first:
                adapter._drain_native_events(call_id)  # noqa: SLF001
                self.assertEqual(session.state, CallState.TERMINATING)
                core.on_caller_hangup(call_id)
            else:
                core.on_caller_hangup(call_id)
                self.assertEqual(session.state, CallState.TERMINATING)
                adapter._drain_native_events(call_id)  # noqa: SLF001
            # Exactly one leg teardown was requested, whichever won.
            self.assertEqual(hangups, [call_id])
            self.assertEqual(session.state, CallState.TERMINATING)
            # The close completes the single terminal outcome ...
            adapter._on_native_closed(call_id, "OK")  # noqa: SLF001
            self.assertEqual(session.state, CallState.ENDED)
            self.assertIsNone(core.get_session(call_id))
            self.assertTrue(voice.sessions[call_id].closed)
            # ... and everything late is inert: duplicate close, a
            # repeated media signal, and post-close DTMF change
            # nothing and resurrect nothing.
            adapter._on_native_closed(call_id, "OK")  # noqa: SLF001
            core.on_media_failed(call_id, "late duplicate")
            adapter._enqueue_native_event(  # noqa: SLF001
                call_id, ("dtmf", "5"))
            adapter._drain_native_events(call_id)  # noqa: SLF001
            self.assertEqual(hangups, [call_id])
            self.assertEqual(session.state, CallState.ENDED)
            self.assertNotIn(call_id, adapter._calls)  # noqa: SLF001
            self.assertNotIn(call_id, adapter._cb_locks)  # noqa: SLF001
        finally:
            adapter.hangup = orig_hangup

    def test_media_then_bye_converges(self) -> None:
        self._race_once(media_first=True)

    def test_bye_then_media_converges(self) -> None:
        self._race_once(media_first=False)


class _GatedVoiceSession(FakeVoiceSession):
    """Voice session whose output cancellation blocks until released
    (slow STT/LLM/process teardown, signalled through Events)."""

    def __init__(self, call_id, listener, entered, release) -> None:  # noqa: ANN001, ANN202
        super().__init__(call_id, listener)
        self._entered = entered
        self._release = release

    def cancel_output(self, reason) -> None:  # noqa: ANN001
        self.cancels.append(reason)
        self._entered.set()
        self._release.wait(timeout=10)


class _GatedVoiceBackend(FakeVoiceBackend):
    """Backend blocking cancellation for one configured call only."""

    def __init__(self, gated_call_id: str, entered, release) -> None:  # noqa: ANN001, ANN202
        super().__init__()
        self._gated_call_id = gated_call_id
        self._entered = entered
        self._release = release

    def open_session(self, call_id: str, listener) -> FakeVoiceSession:  # noqa: ANN001, ANN202
        if call_id == self._gated_call_id:
            session = _GatedVoiceSession(
                call_id, listener, self._entered, self._release
            )
        else:
            session = FakeVoiceSession(call_id, listener)
        self.sessions[call_id] = session
        self.all_sessions.append(session)
        return session


class SlowTeardownNonBlockingTest(unittest.TestCase):
    """Tests E+F: call A blocks inside slow application cancellation
    while call B's signaling (DTMF, hold/resume, close) and the
    registration/health path keep progressing -- nothing shared
    waits for A. Event waits carry timeouts as fail-safes only,
    never as the correctness mechanism."""

    def test_blocked_cancel_never_stalls_other_call_or_registration(
        self,
    ) -> None:
        import threading
        import time

        from receptionist.baresip_adapter import (
            BaresipTelephonyAdapter,
            _CallRecord,
            wire_baresip_core,
        )
        from receptionist.call_session import CallState
        from receptionist.health import HealthStatus
        from receptionist.resilience import ResilienceConfig
        from receptionist.telephony_config import TelephonyConfig

        entered_a = threading.Event()
        release_a = threading.Event()

        voice = FakeVoiceBackend()
        clock = FakeClock()
        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        core = ReceptionistCore(
            telephony=adapter,
            voice=voice,
            config_service=ConfigService(InMemoryConfigRepository(dict(
                {"greeting": GREETING, "language": "es"}))),
            policy=FakePolicy(),
            clock=clock,
            resilience=ResilienceConfig(max_ai_sessions=2),
            policy_engine=PolicyEngine(
                destinations={}, fallback_id="none", limits=Limits()),
            runtime=RuntimeStorage(
                calls=InMemoryCallRepository(),
                messages=InMemoryMessageRepository(clock=clock),
                transcripts=InMemoryTranscriptStore(),
                audit=InMemoryAuditLog(),
            ),
            retention=RetentionPolicy(),
            call_ids=FakeCallIds(),
        )
        # Swap in the gated backend before any session opens.
        core._voice = _GatedVoiceBackend(  # noqa: SLF001
            "call-1", entered_a, release_a)
        wire_baresip_core(core, adapter)
        adapter._running = True  # noqa: SLF001
        core.start()
        core.report_telephony_state(TelephonyRegistrationState.REGISTERED)

        session_a = core.incoming_call("+34910000041")
        session_b = core.incoming_call("+34910000042")
        adapter._calls[session_a.call_id] = _CallRecord(  # noqa: SLF001
            app_id=session_a.call_id,
            native=_StubNative(_StubAudio()),
        )
        adapter._calls[session_b.call_id] = _CallRecord(  # noqa: SLF001
            app_id=session_b.call_id,
            native=_StubNative(_StubAudio()),
        )
        core.on_answered(session_a.call_id)
        core.on_answered(session_b.call_id)
        self.assertEqual(session_a.state, CallState.ACTIVE)
        self.assertEqual(session_b.state, CallState.ACTIVE)

        observed: list = []
        b_step: dict = {}
        for _name in ("dtmf", "hold:True", "hold:False"):
            b_step[_name] = threading.Event()
        orig_dtmf = core.on_dtmf
        orig_hold = core.on_remote_hold

        def _dtmf(call_id: str, digit: str) -> None:
            observed.append(("dtmf", call_id, digit))
            if call_id == session_b.call_id:
                b_step["dtmf"].set()
            orig_dtmf(call_id, digit)

        def _hold(call_id: str, held: bool) -> None:
            observed.append(("hold", call_id, held))
            if call_id == session_b.call_id:
                b_step[f"hold:{held}"].set()
            orig_hold(call_id, held)

        core.on_dtmf = _dtmf  # type: ignore[method-assign]
        core.on_remote_hold = _hold  # type: ignore[method-assign]
        hangups: list = []
        orig_hangup = adapter.hangup
        adapter.hangup = lambda cid: (  # noqa: E731
            hangups.append(cid), orig_hangup(cid))
        try:
            # Remote BYE on A enters the slow cancel path on A's own
            # dispatcher (the enqueue returns immediately, exactly as
            # the stack callback would).
            adapter._enqueue_native_event(  # noqa: SLF001
                session_a.call_id, ("closed", "remote BYE"))
            self.assertTrue(entered_a.wait(timeout=5))
            self.assertNotIn(session_a.call_id, hangups)
            # While A is blocked: B's full signaling sequence plus a
            # registration/health update must all progress. Paced so
            # each hold is processed before the next is enqueued
            # (coalescing only merges unprocessed holds).
            adapter._enqueue_native_event(  # noqa: SLF001
                session_b.call_id, ("dtmf", "5"))
            self.assertTrue(b_step["dtmf"].wait(timeout=5))
            adapter._enqueue_native_event(  # noqa: SLF001
                session_b.call_id, ("hold", True))
            self.assertTrue(b_step["hold:True"].wait(timeout=5))
            adapter._enqueue_native_event(  # noqa: SLF001
                session_b.call_id, ("hold", False))
            self.assertTrue(b_step["hold:False"].wait(timeout=5))
            adapter._enqueue_native_event(  # noqa: SLF001
                session_b.call_id, ("closed", "remote BYE"))
            core.report_telephony_state(TelephonyRegistrationState.REGISTERED)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if (
                    ("dtmf", session_b.call_id, "5") in observed
                    and ("hold", session_b.call_id, True) in observed
                    and ("hold", session_b.call_id, False) in observed
                    and session_b.call_id in hangups
                    and core.health.status is HealthStatus.READY
                ):
                    break
            self.assertIn(("dtmf", session_b.call_id, "5"), observed)
            self.assertIn(("hold", session_b.call_id, True), observed)
            self.assertIn(("hold", session_b.call_id, False), observed)
            self.assertIn(session_b.call_id, hangups)
            # B ran the full two-step re-entrantly on its own
            # dispatcher (caller hangup -> hangup -> completed) while A
            # stayed blocked: ended and evicted without waiting for A.
            self.assertEqual(session_b.state, CallState.ENDED)
            self.assertIsNone(core.get_session(session_b.call_id))
            self.assertIs(core.health.status, HealthStatus.READY)
            # A is still blocked: nothing above waited for it.
            self.assertNotIn(session_a.call_id, hangups)
            # Release A: its close completes exactly once; both legs
            # then finish and evict.
            release_a.set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if session_a.call_id in hangups:
                    break
            self.assertIn(session_a.call_id, hangups)
            adapter._on_native_closed(  # noqa: SLF001
                session_a.call_id, "OK")
            self.assertEqual(session_a.state, CallState.ENDED)
            self.assertIsNone(core.get_session(session_a.call_id))
            self.assertEqual(hangups.count(session_a.call_id), 1)
            self.assertEqual(hangups.count(session_b.call_id), 1)
        finally:
            release_a.set()
            core.on_dtmf = orig_dtmf  # type: ignore[method-assign]
            core.on_remote_hold = orig_hold  # type: ignore[method-assign]
            adapter.hangup = orig_hangup


class SignalingOrderTest(unittest.TestCase):
    """Test G: one call's lifecycle keeps causal order on its
    dispatcher (answered -> hold -> DTMF -> resume -> hangup), exactly
    as enqueued, even though delivery runs on another thread."""

    def test_lifecycle_order_is_fifo_per_call(self) -> None:
        import threading

        from receptionist.baresip_adapter import (
            BaresipTelephonyAdapter,
            _CallRecord,
        )
        from receptionist.telephony_config import TelephonyConfig

        observed: list = []
        terminal = threading.Event()

        class _OrderRecorder:
            def on_answered(self, call_id: str) -> None:  # noqa: ANN202
                observed.append("answered")
                seen["answered"].set()

            def on_remote_hold(self, call_id: str, held: bool) -> None:  # noqa: ANN001, ANN202
                observed.append(f"hold:{held}")
                seen[f"hold:{held}"].set()

            def on_dtmf(self, call_id: str, digit: str) -> None:  # noqa: ANN001, ANN202
                observed.append(f"dtmf:{digit}")
                seen[f"dtmf:{digit}"].set()

            def on_caller_hangup(self, call_id: str) -> None:  # noqa: ANN202
                observed.append("caller_hangup")
                terminal.set()

        seen: dict = {}
        for _name in ("answered", "hold:True", "dtmf:5", "hold:False"):
            seen[_name] = threading.Event()

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        adapter.set_listener(_OrderRecorder())
        adapter._running = True  # noqa: SLF001
        adapter._calls["call-o"] = _CallRecord(  # noqa: SLF001
            app_id="call-o", native=_StubNative(_StubAudio()))
        try:
            # Paced through the live dispatcher: each step is only
            # enqueued after the previous one was delivered, so the
            # test observes dispatch order without any timing
            # assumption (coalescing only merges unprocessed holds).
            adapter._enqueue_native_event("call-o", ("established",))  # noqa: SLF001
            self.assertTrue(seen["answered"].wait(timeout=5))
            adapter._enqueue_native_event(  # noqa: SLF001
                "call-o", ("hold", True))
            self.assertTrue(seen["hold:True"].wait(timeout=5))
            adapter._enqueue_native_event(  # noqa: SLF001
                "call-o", ("dtmf", "5"))
            self.assertTrue(seen["dtmf:5"].wait(timeout=5))
            adapter._enqueue_native_event(  # noqa: SLF001
                "call-o", ("hold", False))
            self.assertTrue(seen["hold:False"].wait(timeout=5))
            adapter._enqueue_native_event(  # noqa: SLF001
                "call-o", ("closed", "remote BYE"))
            self.assertTrue(terminal.wait(timeout=5))
            self.assertEqual(
                observed,
                ["answered", "hold:True", "dtmf:5", "hold:False",
                 "caller_hangup"],
            )
        finally:
            adapter._calls.pop("call-o", None)  # noqa: SLF001
            thread = adapter._rx_threads.get("call-o")  # noqa: SLF001
            if thread is not None:
                thread.join(timeout=5)
            self.assertFalse(thread is not None and thread.is_alive())


class TerminalBackpressureTest(unittest.TestCase):
    """Test H: a flooded DTMF backlog stays bounded while every
    hold/resume snapshot (full FIFO, never coalesced) and the
    terminal close are never dropped -- the close arrives after the
    kept backlog, in causal order."""

    def test_terminal_close_survives_nonterminal_flood(self) -> None:
        from receptionist.baresip_adapter import (
            EV_DISPATCH_MAX_NONTERMINAL,
            BaresipTelephonyAdapter,
            _CallRecord,
        )
        from receptionist.telephony_config import TelephonyConfig

        observed: list = []

        class _FloodRecorder:
            def on_dtmf(self, call_id: str, digit: str) -> None:  # noqa: ANN001, ANN202
                observed.append(("dtmf", digit))

            def on_remote_hold(self, call_id: str, held: bool) -> None:  # noqa: ANN001, ANN202
                observed.append(("hold", held))

            def on_caller_hangup(self, call_id: str) -> None:  # noqa: ANN202
                observed.append(("hangup",))

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        adapter.set_listener(_FloodRecorder())
        adapter._running = True  # noqa: SLF001
        record = _CallRecord(app_id="call-f")
        # Deterministic queue pressure without dispatcher threads: the
        # production enqueue/backpressure unit runs inline; delivery
        # is drained explicitly below.
        record.rx_dispatch_started = True
        adapter._calls["call-f"] = record  # noqa: SLF001
        try:
            for index in range(200):
                adapter._enqueue_native_event(  # noqa: SLF001
                    "call-f", ("dtmf", str(index % 10)))
            # One hold before and one after the flood: full FIFO keeps
            # both, in order, around the bounded DTMF backlog.
            adapter._enqueue_native_event("call-f", ("hold", True))  # noqa: SLF001
            for index in range(200, 400):
                adapter._enqueue_native_event(  # noqa: SLF001
                    "call-f", ("dtmf", str(index % 10)))
            adapter._enqueue_native_event(  # noqa: SLF001
                "call-f", ("hold", False))
            pending_dtmf = [
                event for event in record.ev_dispatch
                if event and event[0] == "dtmf"
            ]
            pending_holds = [
                event for event in record.ev_dispatch
                if event and event[0] == "hold"
            ]
            self.assertLessEqual(len(pending_dtmf), EV_DISPATCH_MAX_NONTERMINAL)
            # Both holds survived, in causal order.
            self.assertEqual(pending_holds, [("hold", True), ("hold", False)])
            adapter._enqueue_native_event("call-f", ("closed", "OK"))  # noqa: SLF001
            drained = adapter._drain_native_events("call-f")  # noqa: SLF001
            self.assertGreater(drained, 0)
            # Terminal close arrived exactly once, last.
            self.assertEqual(observed[-1], ("hangup",))
            self.assertEqual(observed.count(("hangup",)), 1)
            # Causal order survives: hold True was enqueued before
            # the second flood, so FIFO delivers it first (older DTMF
            # ahead of it was evicted), then the 128 kept DTMF, then
            # hold False, then the terminal hangup.
            self.assertEqual(observed[0], ("hold", True))
            self.assertEqual(observed[EV_DISPATCH_MAX_NONTERMINAL + 1],
                             ("hold", False))
            self.assertEqual(
                [entry for entry in observed if entry[0] == "hold"],
                [("hold", True), ("hold", False)],
            )
            # DTMF kept the newest backlog in order (oldest dropped).
            kept = [entry[1] for entry in observed if entry[0] == "dtmf"]
            self.assertEqual(len(kept), EV_DISPATCH_MAX_NONTERMINAL)
            self.assertEqual(
                kept, [str(index % 10) for index in range(272, 400)])
        finally:
            adapter._calls.pop("call-f", None)  # noqa: SLF001


class ShutdownBoundedTest(unittest.TestCase):
    """Test I: shutdown with a callback blocked inside slow
    application work, queued signaling, and pending RX audio returns
    boundedly, leaves no call/lock state behind, and dispatcher
    threads exit once released."""

    def test_shutdown_returns_bounded_with_blocked_dispatcher(self) -> None:
        import threading

        from receptionist.baresip_adapter import (
            BaresipTelephonyAdapter,
            _CallRecord,
        )
        from receptionist.telephony_config import TelephonyConfig

        entered = threading.Event()
        release = threading.Event()

        class _BlockingCloseListener:
            def on_caller_hangup(self, call_id: str) -> None:  # noqa: ANN202
                entered.set()
                release.wait(timeout=30)

            def on_dtmf(self, call_id: str, digit: str) -> None:  # noqa: ANN001, ANN202
                pass

            def on_remote_hold(self, call_id: str, held: bool) -> None:  # noqa: ANN001, ANN202
                pass

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        adapter.set_listener(_BlockingCloseListener())
        adapter._running = True  # noqa: SLF001
        adapter._started_once = True  # noqa: SLF001 (started adapter)
        record = _CallRecord(app_id="call-s")
        adapter._calls["call-s"] = record  # noqa: SLF001
        try:
            adapter._ensure_rx_dispatch("call-s")  # noqa: SLF001
            thread = record.rx_thread
            adapter._enqueue_native_event(  # noqa: SLF001
                "call-s", ("closed", "remote BYE"))
            self.assertTrue(entered.wait(timeout=5))
            # More signaling + RX audio queue behind the blocked close.
            adapter._enqueue_native_event("call-s", ("dtmf", "5"))  # noqa: SLF001
            adapter._enqueue_native_event(  # noqa: SLF001
                "call-s", ("hold", True))
            record.rx_dispatch.append(assistant_frame())
            record.rx_ready.set()
            # Shutdown must return boundedly even though application
            # work is still blocked (internal joins all carry
            # timeouts); the 30 s listener gate is only a fail-safe
            # so the test can never hang the suite.
            done = threading.Event()
            worker = threading.Thread(
                target=lambda: (adapter.shutdown(), done.set()),
                name="test-shutdown",
                daemon=True,
            )
            worker.start()
            self.assertTrue(done.wait(timeout=25))
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(adapter._calls, {})  # noqa: SLF001
            self.assertEqual(adapter._cb_locks, {})  # noqa: SLF001
        finally:
            release.set()
            if thread is not None:
                thread.join(timeout=5)
            self.assertFalse(thread is not None and thread.is_alive())
            self.assertEqual(adapter._rx_threads, {})  # noqa: SLF001
            adapter._running = False  # noqa: SLF001


class DispatcherLeakTest(unittest.TestCase):
    """Callback/event lifecycle: N calls open and close leaving no
    per-call state behind (records, serializers, dispatcher threads,
    queued events)."""

    def test_open_close_cycles_leave_no_per_call_state(self) -> None:
        from receptionist.baresip_adapter import (
            BaresipTelephonyAdapter,
            _CallRecord,
        )
        from receptionist.telephony_config import TelephonyConfig

        closed: list = []

        class _LeakRecorder:
            def on_answered(self, call_id: str) -> None:  # noqa: ANN202
                pass

            def on_remote_hold(self, call_id: str, held: bool) -> None:  # noqa: ANN001, ANN202
                pass

            def on_dtmf(self, call_id: str, digit: str) -> None:  # noqa: ANN001, ANN202
                pass

            def on_caller_hangup(self, call_id: str) -> None:  # noqa: ANN202
                closed.append(call_id)

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        adapter.set_listener(_LeakRecorder())
        adapter._running = True  # noqa: SLF001
        try:
            for index in range(25):
                call_id = f"call-leak-{index}"
                adapter._calls[call_id] = _CallRecord(  # noqa: SLF001
                    app_id=call_id)
                adapter._enqueue_native_event(  # noqa: SLF001
                    call_id, ("established",))
                adapter._enqueue_native_event(  # noqa: SLF001
                    call_id, ("hold", True))
                adapter._enqueue_native_event(  # noqa: SLF001
                    call_id, ("dtmf", "5"))
                adapter._enqueue_native_event(  # noqa: SLF001
                    call_id, ("hold", False))
                adapter._enqueue_native_event(  # noqa: SLF001
                    call_id, ("closed", "remote BYE"))
                # Deterministic drain (a live dispatcher thread may
                # also be draining: every event is popped exactly
                # once, so delivery stays exactly-once either way).
                adapter._drain_native_events(call_id)  # noqa: SLF001
                # Remote close parks the record for the core's hangup
                # (production two-step); completing it drops the leg.
                adapter.hangup(call_id)
                self.assertNotIn(call_id, adapter._calls)  # noqa: SLF001
            threads = list(adapter._rx_threads.values())  # noqa: SLF001
            for thread in threads:
                thread.join(timeout=5)
            self.assertEqual(len(closed), 25)
            self.assertEqual(adapter._calls, {})  # noqa: SLF001
            self.assertEqual(adapter._cb_locks, {})  # noqa: SLF001
            self.assertEqual(adapter._rx_threads, {})  # noqa: SLF001
        finally:
            adapter._running = False  # noqa: SLF001
            for thread in list(adapter._rx_threads.values()):  # noqa: SLF001
                thread.join(timeout=5)


class RemoteHoldGateTest(unittest.TestCase):
    """Remote-HOLD media gate: the stack callback gates TX/RX
    synchronously (lightweight flag + queue mutation, never core/AI
    work), discards pre-HOLD queued caller audio, resume flushes
    stale media, holds stay full FIFO, and sibling calls are
    unaffected. Event waits are fail-safes only."""

    def _gated_adapter(self, *call_ids: str):
        import threading

        from receptionist.baresip_adapter import (
            BaresipTelephonyAdapter,
            _CallRecord,
        )
        from receptionist.telephony_config import TelephonyConfig

        delivered: list = []
        blocked: dict = {}
        audio: dict = {}

        class _GateRecorder:
            def on_caller_audio(self, call_id: str, frame) -> None:  # noqa: ANN001, ANN202
                gate = blocked.get(call_id)
                if gate is not None:
                    gate["entered"].set()
                    gate["release"].wait(timeout=10)
                delivered.append((call_id, frame))

            def on_remote_hold(self, call_id: str, held: bool) -> None:  # noqa: ANN001, ANN202
                delivered.append((call_id, f"hold:{held}"))

            def on_dtmf(self, call_id: str, digit: str) -> None:  # noqa: ANN001, ANN202
                delivered.append((call_id, f"dtmf:{digit}"))

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        adapter.set_listener(_GateRecorder())
        adapter._running = True  # noqa: SLF001
        for call_id in call_ids:
            stub = _StubAudio()
            audio[call_id] = stub
            adapter._calls[call_id] = _CallRecord(  # noqa: SLF001
                app_id=call_id, native=_StubNative(stub))
        return adapter, delivered, blocked, audio

    def test_hold_entry_gates_pump_and_discards_queued_rx(self) -> None:
        adapter, delivered, _, audio = self._gated_adapter("call-h")
        try:
            adapter.send_audio("call-h", make_frame(
                tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
            adapter._pump_once()  # noqa: SLF001
            self.assertEqual(len(audio["call-h"].written), 1)
            reads_before = audio["call-h"].reads
            self.assertGreater(reads_before, 0)
            # One caller frame is already queued for delivery ...
            record = adapter._calls["call-h"]  # noqa: SLF001
            self.assertEqual(len(record.rx_dispatch), 1)
            # ... then the stack reports HOLD: the gate engages
            # synchronously and the queued pre-HOLD audio is gone.
            self.assertTrue(adapter._gate_remote_hold("call-h", True))  # noqa: SLF001
            adapter._enqueue_native_event("call-h", ("hold", True))  # noqa: SLF001
            self.assertTrue(record.remote_hold)
            self.assertEqual(list(record.rx_dispatch), [])
            # The pump moves nothing more for the held call.
            adapter._pump_once()  # noqa: SLF001
            self.assertEqual(len(audio["call-h"].written), 1)
            self.assertEqual(audio["call-h"].reads, reads_before)
            # Draining delivers the hold notification, never caller
            # audio: nothing queued before the gate may cross it.
            adapter._drain_native_events("call-h")  # noqa: SLF001
            adapter._dispatch_once("call-h")  # noqa: SLF001
            self.assertEqual(delivered, [("call-h", "hold:True")])
        finally:
            adapter._calls.pop("call-h", None)  # noqa: SLF001
            adapter._running = False  # noqa: SLF001

    def test_blocked_dispatcher_hold_still_gates_and_b_flows(self) -> None:
        import threading

        adapter, delivered, blocked, audio = self._gated_adapter(
            "call-a", "call-b")
        blocked["call-a"] = {
            "entered": threading.Event(), "release": threading.Event(),
        }
        try:
            adapter._ensure_rx_dispatch("call-a")  # noqa: SLF001
            adapter._ensure_rx_dispatch("call-b")  # noqa: SLF001
            # A frame for A enters delivery and blocks A's dispatcher
            # (slow application work) ...
            record_a = adapter._calls["call-a"]  # noqa: SLF001
            record_a.rx_dispatch.append(assistant_frame())
            record_a.rx_ready.set()
            self.assertTrue(blocked["call-a"]["entered"].wait(timeout=5))
            # ... while A is blocked, the stack reports HOLD: the sync
            # gate returns immediately and stops A's media.
            self.assertTrue(adapter._gate_remote_hold("call-a", True))  # noqa: SLF001
            adapter._enqueue_native_event("call-a", ("hold", True))  # noqa: SLF001
            self.assertTrue(record_a.remote_hold)
            adapter.send_audio("call-a", make_frame(
                tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
            adapter.send_audio("call-b", make_frame(
                tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
            adapter._pump_once()  # noqa: SLF001
            # A: zero TX/RX movement after HOLD. B: fully flowing.
            self.assertEqual(audio["call-a"].written, [])
            reads_a = audio["call-a"].reads
            adapter._pump_once()  # noqa: SLF001
            self.assertEqual(audio["call-a"].written, [])
            self.assertEqual(audio["call-a"].reads, reads_a)
            self.assertEqual(len(audio["call-b"].written), 1)
            self.assertGreater(audio["call-b"].reads, 0)
            # B's signaling is delivered while A stays blocked+held.
            adapter._enqueue_native_event("call-b", ("dtmf", "5"))  # noqa: SLF001
            adapter._drain_native_events("call-b")  # noqa: SLF001
            self.assertIn(("call-b", "dtmf:5"), delivered)
            # Release A: only the pre-HOLD in-flight frame plus the
            # hold notification were ever delivered for A.
            blocked["call-a"]["release"].set()
            adapter._drain_native_events("call-a")  # noqa: SLF001
            adapter._dispatch_once("call-a")  # noqa: SLF001
            audio_a = [entry for entry in delivered if entry[0] == "call-a"]
            self.assertEqual(len(audio_a), 2)
            self.assertIn(("call-a", "hold:True"), audio_a)
        finally:
            blocked["call-a"]["release"].set()
            adapter._calls.pop("call-a", None)  # noqa: SLF001
            adapter._calls.pop("call-b", None)  # noqa: SLF001
            adapter._running = False  # noqa: SLF001
            for thread in list(adapter._rx_threads.values()):  # noqa: SLF001
                thread.join(timeout=5)

    def test_unpaced_hold_dtmf_resume_keeps_causal_order(self) -> None:
        adapter, delivered, _, _ = self._gated_adapter("call-q")
        try:
            record = adapter._calls["call-q"]  # noqa: SLF001
            # No dispatcher threads: the production enqueue unit runs
            # inline; the full-FIFO queue must preserve causality even
            # when nothing is processed between enqueues.
            record.rx_dispatch_started = True
            adapter._enqueue_native_event("call-q", ("hold", True))  # noqa: SLF001
            adapter._enqueue_native_event("call-q", ("dtmf", "5"))  # noqa: SLF001
            adapter._enqueue_native_event(  # noqa: SLF001
                "call-q", ("hold", False))
            adapter._drain_native_events("call-q")  # noqa: SLF001
            self.assertEqual(
                delivered,
                [("call-q", "hold:True"), ("call-q", "dtmf:5"),
                 ("call-q", "hold:False")],
            )
        finally:
            adapter._calls.pop("call-q", None)  # noqa: SLF001
            adapter._running = False  # noqa: SLF001

    def test_resume_flushes_stale_and_fresh_media_flows(self) -> None:
        adapter, delivered, _, audio = self._gated_adapter("call-r")
        try:
            adapter.send_audio("call-r", make_frame(
                tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
            adapter._pump_once()  # noqa: SLF001
            self.assertEqual(len(audio["call-r"].written), 1)
            # HOLD with media in flight: queued RX is discarded at the
            # gate; TX pumped before the gate already went out.
            self.assertTrue(adapter._gate_remote_hold("call-r", True))  # noqa: SLF001
            adapter._enqueue_native_event("call-r", ("hold", True))  # noqa: SLF001
            adapter.send_audio("call-r", make_frame(
                tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
            adapter.send_audio("call-r", make_frame(
                tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
            adapter._pump_once()  # noqa: SLF001
            self.assertEqual(len(audio["call-r"].written), 1)
            record = adapter._calls["call-r"]  # noqa: SLF001
            self.assertGreater(record.tx_bytes, 0)
            # RESUME is accepted at the stack gate but the gate stays
            # CLOSED there (native cleanup is dispatcher work, never
            # SIP-loop work).
            self.assertTrue(adapter._gate_remote_hold("call-r", False))  # noqa: SLF001
            adapter._enqueue_native_event(  # noqa: SLF001
                "call-r", ("hold", False))
            self.assertTrue(record.remote_hold)
            self.assertGreater(record.tx_bytes, 0)
            # The dispatcher runs native cleanup first (queued backlog
            # dropped, flush_tx, bounded stale RX drain), and only
            # then reopens the gate and notifies.
            adapter._drain_native_events("call-r")  # noqa: SLF001
            self.assertFalse(record.remote_hold)
            self.assertEqual(record.tx_bytes, 0)
            self.assertEqual(list(record.rx_dispatch), [])
            self.assertEqual(audio["call-r"].flush_tx_calls, 1)
            # Fresh media flows again both ways.
            adapter.send_audio("call-r", make_frame(
                tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
            adapter._pump_once()  # noqa: SLF001
            self.assertEqual(len(audio["call-r"].written), 2)
            adapter._dispatch_once("call-r")  # noqa: SLF001
            signals = [entry[1] for entry in delivered
                       if entry[0] == "call-r"
                       and isinstance(entry[1], str)]
            self.assertEqual(signals, ["hold:True", "hold:False"])
            audio_entries = [
                entry for entry in delivered
                if entry[0] == "call-r" and not isinstance(entry[1], str)
            ]
            self.assertEqual(len(audio_entries), 1)
        finally:
            adapter._calls.pop("call-r", None)  # noqa: SLF001
            adapter._running = False  # noqa: SLF001


class ResumeNativeCleanupTest(unittest.TestCase):
    """RESUME invariant regression: the remote-hold gate stays closed
    until native stale media is cleaned off the SIP loop.

    Stack RESUME only enqueues; the per-call dispatcher runs native
    cleanup (``flush_tx`` + bounded stale RX drain) while the gate is
    still closed, and only then reopens the gate and delivers
    ``hold(False)``. A pump racing the pending resume moves zero
    TX/RX. Event waits are fail-safes only."""

    def test_gate_stays_closed_through_native_cleanup(self) -> None:
        import threading

        from receptionist.baresip_adapter import (
            BaresipTelephonyAdapter,
            _CallRecord,
        )
        from receptionist.telephony_config import TelephonyConfig

        delivered: list = []
        entered = threading.Event()
        release = threading.Event()
        resumed = threading.Event()

        class _ResumeRecorder:
            def on_caller_audio(self, call_id: str, frame) -> None:  # noqa: ANN001, ANN202
                delivered.append((call_id, "audio"))

            def on_dtmf(self, call_id: str, digit: str) -> None:  # noqa: ANN001, ANN202
                if digit == "0":
                    # Park the dispatcher here: everything behind this
                    # event (hold, DTMF, resume) stays queued while the
                    # test drives the pump against the closed gate.
                    entered.set()
                    release.wait(timeout=10)
                delivered.append((call_id, f"dtmf:{digit}"))

            def on_remote_hold(self, call_id: str, held: bool) -> None:  # noqa: ANN001, ANN202
                delivered.append((call_id, f"hold:{held}"))
                if not held:
                    resumed.set()

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        adapter.set_listener(_ResumeRecorder())
        adapter._running = True  # noqa: SLF001
        stub = _StubAudio()
        # Stale native buffers: three chunks the dispatcher cleanup
        # must drain (then empty, so the bounded drain terminates).
        stub.read_script = [b"\x01" * 640, b"\x02" * 640, b"\x03" * 640]
        record = _CallRecord(app_id="call-n", native=_StubNative(stub))
        adapter._calls["call-n"] = record  # noqa: SLF001
        try:
            adapter._ensure_rx_dispatch("call-n")  # noqa: SLF001
            thread = record.rx_thread
            # Live dispatcher, then HOLD with stale app-side TX.
            adapter._enqueue_native_event("call-n", ("dtmf", "0"))  # noqa: SLF001
            self.assertTrue(entered.wait(timeout=5))
            self.assertTrue(adapter._gate_remote_hold("call-n", True))  # noqa: SLF001
            adapter._enqueue_native_event("call-n", ("hold", True))  # noqa: SLF001
            adapter.send_audio("call-n", make_frame(
                tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
            self.assertGreater(record.tx_bytes, 0)
            # DTMF between hold and resume (causal order must survive).
            adapter._enqueue_native_event("call-n", ("dtmf", "5"))  # noqa: SLF001
            # Stack RESUME while the dispatcher is still parked: the
            # gate must stay closed (native cleanup pending).
            self.assertTrue(adapter._gate_remote_hold("call-n", False))  # noqa: SLF001
            adapter._enqueue_native_event(  # noqa: SLF001
                "call-n", ("hold", False))
            self.assertTrue(record.remote_hold)
            # A pump racing the pending resume moves zero TX/RX: no
            # writes, no reads, nothing newly deliverable.
            writes_before = len(stub.written)
            reads_before = stub.reads
            delivered_before = len(delivered)
            adapter._pump_once()  # noqa: SLF001
            adapter._pump_once()  # noqa: SLF001
            self.assertEqual(len(stub.written), writes_before)
            self.assertEqual(stub.reads, reads_before)
            self.assertEqual(len(delivered), delivered_before)
            self.assertTrue(record.remote_hold)
            self.assertEqual(stub.flush_tx_calls, 0)
            # Release the dispatcher: native cleanup runs first
            # (flush_tx + stale RX drained), and only then the gate
            # reopens and hold(False) is delivered.
            release.set()
            self.assertTrue(resumed.wait(timeout=5))
            self.assertEqual(stub.flush_tx_calls, 1)
            self.assertEqual(stub.read_script, [])
            self.assertFalse(record.remote_hold)
            self.assertEqual(record.tx_bytes, 0)
            self.assertEqual(
                delivered,
                [("call-n", "dtmf:0"), ("call-n", "hold:True"),
                 ("call-n", "dtmf:5"), ("call-n", "hold:False")],
            )
            # Fresh media flows again both ways after the cleanup.
            stub.read_script = None
            adapter.send_audio("call-n", make_frame(
                tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
            adapter._pump_once()  # noqa: SLF001
            self.assertEqual(len(stub.written), writes_before + 1)
            adapter._dispatch_once("call-n")  # noqa: SLF001
            self.assertEqual(delivered[-1], ("call-n", "audio"))
        finally:
            release.set()
            adapter._calls.pop("call-n", None)  # noqa: SLF001
            adapter._running = False  # noqa: SLF001
            if thread is not None:
                thread.join(timeout=5)
            self.assertFalse(thread is not None and thread.is_alive())


class RefreshSequentialTest(unittest.TestCase):
    """B1: refresh is two sequential single-leg dances, never one
    concurrent race. unregister() is awaited first (binding
    deterministically gone), then register() (binding
    deterministically back); fresh requires both legs confirmed.
    A scripted UA plus a real loop thread drives the production
    seam; Event-free settle bounds each case by ~1 s."""

    def _loop_adapter(self, ua):
        import asyncio
        import threading

        from receptionist.baresip_adapter import BaresipTelephonyAdapter
        from receptionist.telephony_config import TelephonyConfig

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        adapter._loop = loop  # noqa: SLF001
        adapter._running = True  # noqa: SLF001
        adapter._ua = ua  # noqa: SLF001
        adapter._reg_state = TelephonyRegistrationState.REGISTERED  # noqa: SLF001
        return adapter, loop, thread

    def _close_loop(self, loop, thread) -> None:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)

    class _ScriptedUA:
        """Fake stack UA: records primitive order, confirms each leg
        with one REGISTER_OK like the binding (one opaque event per
        dance leg), or raises scripted failures."""

        def __init__(self, adapter, fail_unregister=None, fail_register=None) -> None:  # noqa: ANN001, ANN202
            self._adapter = adapter
            self._fail_unregister = fail_unregister
            self._fail_register = fail_register
            self.calls: list = []

        async def unregister(self) -> None:
            self.calls.append("unregister")
            if self._fail_unregister is not None:
                raise self._fail_unregister
            with self._adapter._lock:  # noqa: SLF001
                self._adapter._register_ok_count += 1  # noqa: SLF001

        async def register(self) -> None:
            self.calls.append("register")
            if self._fail_register is not None:
                raise self._fail_register
            with self._adapter._lock:  # noqa: SLF001
                self._adapter._register_ok_count += 1  # noqa: SLF001

    def test_refresh_unregisters_first_then_registers(self) -> None:
        ua_holder: dict = {}
        adapter, loop, thread = self._loop_adapter(None)  # type: ignore[arg-type]
        try:
            ua = self._ScriptedUA(adapter)
            ua_holder["ua"] = ua
            adapter._ua = ua  # noqa: SLF001
            state, fresh = adapter.refresh_registration(timeout=10)
            # Sequential, unregister first: no concurrent legs means
            # server scheduling cannot reorder the outcome.
            self.assertEqual(ua.calls, ["unregister", "register"])
            self.assertTrue(fresh)
            self.assertIs(state, TelephonyRegistrationState.REGISTERED)
        finally:
            self._close_loop(loop, thread)

    def test_refresh_register_failure_is_not_fresh(self) -> None:
        class _RegFailed(Exception):
            pass

        adapter, loop, thread = self._loop_adapter(None)  # type: ignore[arg-type]
        try:
            ua = self._ScriptedUA(
                adapter, fail_register=_RegFailed(" registrar rejected"))
            adapter._ua = ua  # noqa: SLF001
            _, fresh = adapter.refresh_registration(timeout=10)
            # The unregister leg ran (and confirmed deletion); the
            # register leg never confirmed a live binding.
            self.assertEqual(ua.calls, ["unregister", "register"])
            self.assertFalse(fresh)
        finally:
            self._close_loop(loop, thread)

    def test_refresh_unregister_failure_skips_register(self) -> None:
        class _UnregFailed(Exception):
            pass

        adapter, loop, thread = self._loop_adapter(None)  # type: ignore[arg-type]
        try:
            ua = self._ScriptedUA(
                adapter, fail_unregister=_UnregFailed("no answer"))
            adapter._ua = ua  # noqa: SLF001
            _, fresh = adapter.refresh_registration(timeout=10)
            self.assertEqual(ua.calls, ["unregister"])
            self.assertFalse(fresh)
        finally:
            self._close_loop(loop, thread)


def _named_error(name: str) -> Exception:
    """Vendor-named error double without importing the native binding
    (the adapter branches on ``type(error).__name__`` only)."""
    return type(name, (Exception,), {})()


class DeclineFallbackTest(unittest.TestCase):
    """M1: a 486 decline is never silently abandoned. Stale-handle
    races retry with backoff; every other failure falls through to
    answer+BYE release, which pops the record even when the 486 did
    take effect. Deterministic: the shared coroutine runs under
    ``asyncio.run`` with stub natives (no loop thread, no sleeps)."""

    def _adapter(self):
        from receptionist.baresip_adapter import BaresipTelephonyAdapter
        from receptionist.telephony_config import TelephonyConfig

        adapter = BaresipTelephonyAdapter(
            TelephonyConfig(username="u", domain="d", password="p")
        )
        adapter._running = True  # noqa: SLF001
        adapter._REJECT_RETRY_DELAYS = ()  # noqa: SLF001 (no timing in tests)
        return adapter

    def _loop_adapter(self):
        """Adapter with a real loop thread so the fire-and-forget
        answer+BYE release executes (its answer-failure path pops
        immediately, never the 15 s safety wait)."""
        import asyncio
        import threading

        adapter = self._adapter()
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        adapter._loop = loop  # noqa: SLF001
        return adapter, loop, thread

    def _close_loop(self, loop, thread) -> None:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)

    def _await_pop(self, adapter, call_id: str, timeout: float = 5.0) -> None:
        """Bounded pop wait: the release coroutine pops synchronously
        right after answer() settles, so this is a fail-safe join,
        never the mechanism (answer invocation is event-observed)."""
        import time

        deadline = time.monotonic() + timeout
        while call_id in adapter._calls:  # noqa: SLF001
            if time.monotonic() >= deadline:
                break
            time.sleep(0.005)
        self.assertNotIn(call_id, adapter._calls)  # noqa: SLF001

    class _DeclineNative:
        def __init__(self, reject_error=None, answer_error=None) -> None:  # noqa: ANN001, ANN202
            self._reject_error = reject_error
            self._answer_error = answer_error
            self.reject_calls = 0
            self.answer_calls = 0

        def reject(self):  # noqa: ANN202
            self.reject_calls += 1
            if self._reject_error is not None:
                raise self._reject_error
            return None

        def answer(self):  # noqa: ANN202
            self.answer_calls += 1
            if self._answer_error is not None:
                raise self._answer_error
            return None

        def hangup(self):  # noqa: ANN202
            return None

    def _record(self, adapter, native, call_id="call-d"):
        from receptionist.baresip_adapter import _CallRecord

        record = _CallRecord(app_id=call_id, native=native)
        adapter._calls[call_id] = record  # noqa: SLF001
        return record

    def test_non_stale_errors_fall_through_to_release(self) -> None:
        import asyncio
        import threading

        for name in (
            "CommandQueueFull",
            "CommandTimeout",
            "RuntimeDead",
            "BaresipError",
        ):
            adapter, loop, thread = self._loop_adapter()
            try:
                answered = threading.Event()
                stale = _named_error("StaleHandleError")
                native = self._DeclineNative(
                    reject_error=_named_error(name),
                    answer_error=stale)
                orig_answer = native.answer

                def _answer_and_signal():  # noqa: ANN202
                    try:
                        return orig_answer()
                    finally:
                        answered.set()

                native.answer = _answer_and_signal  # type: ignore[method-assign]
                record = self._record(adapter, native)
                asyncio.run(adapter._decline_with_fallback(  # noqa: SLF001
                    record, native.reject))
                # No silent abandon: the fallback answered (the 486
                # path failed with a non-stale error) ...
                self.assertEqual(native.reject_calls, 1)
                self.assertTrue(answered.wait(timeout=5))
                self.assertEqual(native.answer_calls, 1)
                # ... saw the gone leg, and popped the record + lock.
                self._await_pop(adapter, "call-d")
                self.assertNotIn("call-d", adapter._cb_locks)  # noqa: SLF001
            finally:
                self._close_loop(loop, thread)

    def test_stale_exhaustion_falls_through_to_release(self) -> None:
        import asyncio
        import threading

        adapter, loop, thread = self._loop_adapter()
        try:
            answered = threading.Event()
            stale = _named_error("StaleHandleError")
            native = self._DeclineNative(
                reject_error=stale, answer_error=stale)
            orig_answer = native.answer

            def _answer_and_signal():  # noqa: ANN202
                try:
                    return orig_answer()
                finally:
                    answered.set()

            native.answer = _answer_and_signal  # type: ignore[method-assign]
            record = self._record(adapter, native)
            asyncio.run(adapter._decline_with_fallback(  # noqa: SLF001
                record, native.reject))
            self.assertEqual(native.reject_calls, 1)
            self.assertTrue(answered.wait(timeout=5))
            self.assertEqual(native.answer_calls, 1)
            self._await_pop(adapter, "call-d")
        finally:
            self._close_loop(loop, thread)

    def test_successful_reject_never_falls_back(self) -> None:
        import asyncio

        adapter = self._adapter()
        native = self._DeclineNative()
        record = self._record(adapter, native)
        asyncio.run(adapter._decline_with_fallback(  # noqa: SLF001
            record, native.reject))
        self.assertEqual(native.reject_calls, 1)
        self.assertEqual(native.answer_calls, 0)
        # Normal close routing still owns the leg.
        self.assertIn("call-d", adapter._calls)  # noqa: SLF001

    def test_reject_without_stack_drops_record(self) -> None:
        adapter = self._adapter()
        native = self._DeclineNative()
        self._record(adapter, native)
        # No loop thread: no close will ever arrive, so reject() must
        # drop the record instead of leaking the admission slot.
        adapter.reject("call-d")
        self.assertNotIn("call-d", adapter._calls)  # noqa: SLF001
        self.assertNotIn("call-d", adapter._cb_locks)  # noqa: SLF001

    def test_decline_native_without_stack_retains_nothing(self) -> None:
        adapter = self._adapter()
        native = self._DeclineNative()
        adapter._decline_native(native)  # noqa: SLF001
        self.assertEqual(adapter._calls, {})  # noqa: SLF001
        self.assertEqual(adapter._cb_locks, {})  # noqa: SLF001


class TelephonyConfigTest(unittest.TestCase):
    def test_strict_integers_and_typed_listen(self) -> None:
        from receptionist.telephony_config import telephony_config_from_mapping

        values = {
            "sip.username": "700",
            "sip.domain": "pbx.test",
            "sip.password": "s3cret",
            "sip.reg_interval": "120",
            "sip.max_calls": "2",
            "sip.listen": "127.0.0.1:5070",
            "sip.codecs": "pcma",
        }
        cfg = telephony_config_from_mapping(values.get)
        self.assertEqual(cfg.reg_interval, 120)
        self.assertEqual(cfg.sip_listen, "127.0.0.1:5070")
        self.assertEqual(cfg.audio_codecs, ("pcma",))
        self.assertNotIn("s3cret", repr(cfg))
        self.assertNotIn("s3cret", cfg.redacted_summary)
        with self.assertRaises(ValueError):
            telephony_config_from_mapping({**values, "sip.max_calls": "x"}.get)
        with self.assertRaises(ValueError):
            telephony_config_from_mapping({"sip.username": "700"}.get)

    def test_factory_builds_adapter_from_config_service(self) -> None:
        from receptionist.baresip_adapter import (
            BaresipTelephonyAdapter,
            baresip_adapter_from_config,
        )
        from receptionist.config import ConfigService, InMemoryConfigRepository

        service = ConfigService(
            InMemoryConfigRepository(
                {
                    "sip.username": "700",
                    "sip.domain": "pbx.test",
                    "sip.password": "s3cret",
                    "sip.codecs": "pcmu,pcma",
                }
            )
        )
        adapter = baresip_adapter_from_config(service)
        self.assertIsInstance(adapter, BaresipTelephonyAdapter)
        self.assertIn("sip:700@pbx.test", adapter.config_summary)

    def test_fake_refresh_counts_and_keeps_state(self) -> None:
        from receptionist.boundaries import TelephonyRegistrationState

        core, telephony, _, _ = make_core()
        core.start()
        telephony.report_registration(TelephonyRegistrationState.REGISTERED)
        state, fresh = telephony.refresh_registration()
        self.assertEqual(state, TelephonyRegistrationState.REGISTERED)
        self.assertTrue(fresh)
        self.assertEqual(telephony.refreshes, 1)


class ShutdownTest(unittest.TestCase):
    def test_shutdown_idempotent_and_late_callbacks_ignored(self) -> None:
        core, telephony, voice, _ = make_core()
        core.start()
        session = core.incoming_call("+34910000001")
        live = voice.sessions[session.call_id]
        live.finish_playback(session.current_turn)
        telephony.shutdown()
        telephony.shutdown()
        self.assertEqual(telephony.shutdown_count, 2)
        self.assertTrue(telephony.shut_down)
        # Late playout for a dead call is dropped, never queued.
        telephony.send_audio(session.call_id, assistant_frame())
        self.assertNotIn(session.call_id, telephony.playout)
        # Late caller audio never reaches the voice backend.
        pushed_before = len(live.pushed_audio)
        telephony.simulate_caller_audio(session.call_id, caller_speech_frame())
        self.assertEqual(len(live.pushed_audio), pushed_before)
        # Late hangup completions are contained, never raise.
        telephony.complete_hangup(session.call_id)


if __name__ == "__main__":
    unittest.main()
