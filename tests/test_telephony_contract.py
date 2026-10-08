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
        self.assertEqual(results, [("call-9", TransferResult.TIMEOUT)])
        self.assertFalse(record.transfer_open)
        # Second firing is a no-op (already closed out).
        adapter._transfer_watchdog_fired("call-9")  # noqa: SLF001
        self.assertEqual(results, [("call-9", TransferResult.TIMEOUT)])


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

    def test_persistent_tx_failure_reports_once_and_hangs_up(self) -> None:
        from types import SimpleNamespace

        from receptionist.baresip_adapter import (
            MEDIA_ERROR_STREAK_LIMIT,
            _CallRecord,
        )

        states: list = []

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
                raise RuntimeError("aumem write broken")

            def read(self, max_bytes: int) -> bytes:  # noqa: ANN202
                return b""

        class _NativeStub:
            audio = _AudioStub()

        adapter = self._adapter()
        adapter.set_status_listener(_StatusStub())
        record = _CallRecord(app_id="call-tx", native=_NativeStub())
        adapter._calls["call-tx"] = record  # noqa: SLF001
        adapter.send_audio("call-tx", make_frame(
            tone_pcm(duration_seconds=0.02, sample_rate=16000), 16000))
        for _ in range(MEDIA_ERROR_STREAK_LIMIT + 10):
            adapter._pump_tx(record)  # noqa: SLF001
        # Exactly one unhealthy transition (no per-frame spam) naming
        # the call, and the dead leg is released.
        unhealthy = [s for s in states if s[0] is False]
        self.assertEqual(len(unhealthy), 1)
        self.assertEqual(unhealthy[0][1], "call-tx")
        self.assertTrue(record.local_close)
        self.assertEqual(record.tx_errors, MEDIA_ERROR_STREAK_LIMIT + 10)

    def test_persistent_rx_failure_reports_once_and_hangs_up(self) -> None:
        from types import SimpleNamespace

        from receptionist.baresip_adapter import (
            MEDIA_ERROR_STREAK_LIMIT,
            _CallRecord,
        )

        states: list = []

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
                return len(pcm)

            def read(self, max_bytes: int) -> bytes:  # noqa: ANN202
                raise RuntimeError("aumem read broken")

        class _NativeStub:
            audio = _AudioStub()

        adapter = self._adapter()
        adapter.set_status_listener(_StatusStub())
        record = _CallRecord(app_id="call-rx", native=_NativeStub())
        adapter._calls["call-rx"] = record  # noqa: SLF001
        for _ in range(MEDIA_ERROR_STREAK_LIMIT + 10):
            adapter._drain_rx(record)  # noqa: SLF001
        unhealthy = [s for s in states if s[0] is False]
        self.assertEqual(len(unhealthy), 1)
        self.assertEqual(unhealthy[0][1], "call-rx")
        self.assertTrue(record.local_close)

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
