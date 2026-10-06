"""Real-model integration for the cascaded backend (#24).

Gated: never runs in the PR suite. Runs only with REAL_CASCADED=1 plus
an installed model pack (MODEL_ROOT + MANIFEST) and, for the full
turn, an operator-provided Spanish speech fixture (REAL_AUDIO_FIXTURE,
16 kHz mono WAV or raw PCM16). Otherwise the tests skip with an
explicit reason instead of failing or faking.

Run:  REAL_CASCADED=1 MODEL_ROOT=/models MANIFEST=/models/pinned.json \\
          REAL_AUDIO_FIXTURE=/path/to/es-16k.wav \\
          PYTHONPATH=src:tests python3 -m unittest tests.test_cascaded_real -v
"""

from __future__ import annotations

import os
import unittest

REAL_ENABLED = os.environ.get("REAL_CASCADED") == "1"
MODEL_ROOT = os.environ.get("MODEL_ROOT", "")
MANIFEST = os.environ.get("MANIFEST", "")
FIXTURE = os.environ.get("REAL_AUDIO_FIXTURE", "")


def _reasons() -> list[str]:
    reasons = []
    if not REAL_ENABLED:
        reasons.append("REAL_CASCADED != 1")
    if not MODEL_ROOT or not os.path.isdir(MODEL_ROOT):
        reasons.append("MODEL_ROOT missing")
    if not MANIFEST or not os.path.isfile(MANIFEST):
        reasons.append("MANIFEST missing")
    return reasons


@unittest.skipIf(_reasons(), f"real-model gate closed: {'; '.join(_reasons())}")
class RealModelIntegrityTest(unittest.TestCase):
    def test_pack_verifies_clean(self) -> None:
        from receptionist.voice_manifest import load_manifest_file, verify_manifest

        manifest = load_manifest_file(MANIFEST)
        # The gate asserts the installed pack only; runtime warmup is
        # exercised by the turn test below against real servers.
        self.assertEqual(verify_manifest(MODEL_ROOT, manifest), [])


@unittest.skipIf(
    _reasons() or not (FIXTURE and os.path.isfile(FIXTURE)),
    "real-model turn needs REAL_AUDIO_FIXTURE Spanish speech",
)
class RealModelTurnTest(unittest.TestCase):
    def test_one_real_spanish_turn_end_to_end(self) -> None:
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
        from real_turn import read_pcm_16k_mono  # noqa: E402

        from receptionist.audio import resample_pcm16, split_pcm  # noqa: E402
        from receptionist.boundaries import ProviderFailure  # noqa: E402
        from receptionist.cascaded import VoiceProfile  # noqa: E402
        from receptionist.local_runtimes import build_cascaded_backend  # noqa: E402
        from receptionist.voice_manifest import (  # noqa: E402
            load_manifest_file,
            verify_manifest,
        )

        manifest = load_manifest_file(MANIFEST)
        self.assertEqual(verify_manifest(MODEL_ROOT, manifest), [])
        profile = VoiceProfile(
            profile_id=manifest.profile_id,
            model_root=MODEL_ROOT,
            manifest_path=MANIFEST,
            stt_executable=os.environ.get("STT_EXE", "whisper-server"),
            llm_executable=os.environ.get("LLM_EXE", "llama-server"),
            tts_voice=os.environ.get("TTS_VOICE", "es-female-1"),
        )
        backend = build_cascaded_backend(profile)
        backend.start()
        try:
            self.assertEqual(backend.warm(), [])
            self.assertTrue(backend.ready)

            events: dict[str, list] = {
                "sidecars": [],
                "responses": [],
                "audios": [],
                "actions": [],
                "failures": [],
            }

            class Listener:
                def on_transcript(self, text: str) -> None:
                    pass

                def on_transcript_sidecar(self, text: str) -> None:
                    events["sidecars"].append(text)

                def on_response(self, turn_id: int, text: str) -> None:
                    events["responses"].append(text)

                def on_audio(self, turn_id: int, frame) -> None:
                    events["audios"].append(frame)

                def on_playback_finished(self, turn_id: int) -> None:
                    pass

                def on_action_request(self, action: object) -> None:
                    events["actions"].append(action)

                def on_provider_failure(
                    self, turn_id: int, failure: ProviderFailure
                ) -> None:
                    events["failures"].append(failure)

            pcm, rate = read_pcm_16k_mono(FIXTURE)
            pcm16 = resample_pcm16(pcm, rate, 16000)
            session = backend.open_session("real-1", Listener())
            for frame in split_pcm(pcm16, 16000):
                session.push_audio(frame)
            session.commit_turn(1)
            self.assertTrue(session.wait_until_idle(timeout=300.0))

            # Properties, never exact neural text (no flaky assertions).
            self.assertEqual(events["failures"], [])
            self.assertTrue(events["sidecars"])
            self.assertTrue(events["responses"] or events["actions"])
            self.assertGreater(
                sum(len(frame.pcm) for frame in events["audios"]), 0
            )
            self.assertIsNotNone(session.last_timings)
            first_ms = session.last_timings.eou_to_first_audio_ms
            assert first_ms is not None
            print(f"\nreal turn: eou_to_first_audio_ms={first_ms:.1f}")
        finally:
            backend.shutdown()


@unittest.skipIf(_reasons(), f"real-model gate closed: {'; '.join(_reasons())}")
class RealSlotRecycleTest(unittest.TestCase):
    """MAJOR-1 proof against pinned llama-server: aborting an
    in-flight generation frees the single slot (restart), so the next
    request completes without waiting out the abandoned tail."""

    def test_cancel_recycles_llm_slot_on_real_server(self) -> None:
        import threading
        import time

        from receptionist.cascaded import CancelToken, VoiceProfile  # noqa: E402
        from receptionist.local_runtimes import build_cascaded_backend  # noqa: E402
        from receptionist.voice_manifest import (  # noqa: E402
            load_manifest_file,
            verify_manifest,
        )

        manifest = load_manifest_file(MANIFEST)
        self.assertEqual(verify_manifest(MODEL_ROOT, manifest), [])
        profile = VoiceProfile(
            profile_id=manifest.profile_id,
            model_root=MODEL_ROOT,
            manifest_path=MANIFEST,
            stt_executable=os.environ.get("STT_EXE", "whisper-server"),
            llm_executable=os.environ.get("LLM_EXE", "llama-server"),
            tts_voice=os.environ.get("TTS_VOICE", "es-female-1"),
        )
        backend = build_cascaded_backend(profile)
        backend.start()
        try:
            self.assertEqual(backend.warm(), [])
            adapter = backend._llm
            token = CancelToken()
            outcome: dict = {}

            def long_gen() -> None:
                try:
                    adapter.generate(
                        "Cuenta del 1 al 300 escribiendo cada numero en "
                        "una linea separada. " * 4,
                        token,
                    )
                    outcome["done"] = True
                except Exception as error:
                    outcome["error"] = type(error).__name__

            worker = threading.Thread(target=long_gen, daemon=True)
            worker.start()
            deadline = time.monotonic() + 60.0
            while adapter._inflight == 0 and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertGreater(adapter._inflight, 0)
            # Session order: token first (immediate output-stop), then
            # recycle (frees the single slot).
            token.set()
            recycled = adapter.abort_inflight()
            self.assertTrue(recycled)
            worker.join(timeout=30.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(outcome.get("error"), "CancelledError")
            restarts = adapter._server.restarts
            t0 = time.monotonic()
            text = adapter.generate(
                '{"spoken_text": "Hola.", "action": null}', CancelToken()
            )
            short_ms = (time.monotonic() - t0) * 1000.0
            self.assertTrue(text.strip())
            print(
                f"\nreal slot: restarts={restarts} "
                f"short_gen_ms={short_ms:.0f}"
            )
            # Slot freed by recycle: far below the 60 s client timeout
            # the abandoned tail would otherwise force.
            self.assertLess(short_ms, 30000.0)
        finally:
            backend.shutdown()


if __name__ == "__main__":
    unittest.main()
