"""Run one REAL local cascaded turn on CPU and report evidence (#24).

Offline: needs installed runtimes + pinned manifest + model pack.
Fails closed with a clear reason when anything is missing — it never
fabricates a turn. Prints a sanitized evidence document (ids, sizes,
timings; transcript summarized, never full prompts).

Run:  PYTHONPATH=src python3 tools/real_turn.py --model-root /models \\
          --manifest /models/pinned.json --audio-fixture /path/to/es-16k.wav
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def fail(reason: str) -> int:
    print(json.dumps({"ok": False, "reason": reason}, indent=2))
    return 3


def read_pcm_16k_mono(path: str) -> tuple[bytes, int]:
    """Accept raw PCM16 mono or .wav (converted via stdlib wave)."""
    with open(path, "rb") as handle:
        magic = handle.read(4)
    if magic == b"RIFF":
        import wave

        with wave.open(path, "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("fixture must be mono 16-bit WAV")
            return wav.readframes(wav.getnframes()), wav.getframerate()
    with open(path, "rb") as handle:
        return handle.read(), 16000


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one real cascaded turn.")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-fixture", required=True)
    parser.add_argument("--stt-exe", default="whisper-cli")
    parser.add_argument("--llm-exe", default="llama-cli")
    parser.add_argument("--tts-voice-dir", default="")
    parser.add_argument("--profile-id", default="cascaded-cpu-baseline-v1")
    args = parser.parse_args()

    from receptionist.audio import resample_pcm16, split_pcm
    from receptionist.boundaries import ProviderFailure
    from receptionist.cascaded import CascadedVoiceBackend, VoiceProfile
    from receptionist.local_runtimes import (
        LlamaSubprocessLLM,
        SherpaOnnxTTS,
        WhisperSubprocessSTT,
    )
    from receptionist.voice_manifest import load_manifest_file, sha256_file

    try:
        manifest = load_manifest_file(args.manifest)
    except Exception as error:
        return fail(f"manifest unreadable: {type(error).__name__}")

    from receptionist.voice_manifest import verify_manifest

    problems = verify_manifest(args.model_root, manifest)
    if problems:
        return fail(f"integrity failed: {[f'{p.component}:{p.reason}' for p in problems]}")

    by_component = {c.component: c for c in manifest.components}
    try:
        import os as _os

        stt_path = _os.path.join(args.model_root, by_component["stt"].filename)
        llm_path = _os.path.join(args.model_root, by_component["llm"].filename)
        stt = WhisperSubprocessSTT(executable=args.stt_exe, model_path=stt_path)
        llm = LlamaSubprocessLLM(executable=args.llm_exe, model_path=llm_path)
        tts = SherpaOnnxTTS(
            model_dir=args.tts_voice_dir
            or _os.path.join(args.model_root, "tts"),
        )
    except Exception as error:
        return fail(f"adapter construction failed: {type(error).__name__}")

    try:
        pcm, rate = read_pcm_16k_mono(args.audio_fixture)
    except Exception as error:
        return fail(f"fixture unreadable: {type(error).__name__}")
    pcm16 = resample_pcm16(pcm, rate, 16000)

    profile = VoiceProfile(
        profile_id=args.profile_id,
        model_root=args.model_root,
        manifest_path=args.manifest,
    )
    backend = CascadedVoiceBackend(
        profile=profile, stt=stt, llm=llm, tts=tts, knowledge_lookup=None
    )
    backend.start()
    warm_problems = backend.warm()
    if warm_problems:
        backend.shutdown()
        return fail(f"warmup failed: {warm_problems}")

    events: dict = {
        "responses": [],
        "audios": [],
        "actions": [],
        "failures": [],
        "playbacks": [],
    }

    class Listener:
        def on_transcript(self, text: str) -> None:
            pass

        def on_response(self, turn_id: int, text: str) -> None:
            events["responses"].append((turn_id, text))

        def on_audio(self, turn_id: int, frame) -> None:
            events["audios"].append((turn_id, len(frame.pcm), frame.sample_rate))

        def on_playback_finished(self, turn_id: int) -> None:
            events["playbacks"].append(turn_id)

        def on_action_request(self, action: object) -> None:
            events["actions"].append(type(action).__name__)

        def on_provider_failure(self, turn_id: int, failure: ProviderFailure) -> None:
            events["failures"].append((turn_id, failure.category.value))

    session = backend.open_session("real-turn-1", Listener())
    for frame in split_pcm(pcm16, 16000, call_id="real-turn-1", turn_id=1):
        session.push_audio(frame)
    started = time.monotonic()
    session.commit_turn(1)
    total_ms = (time.monotonic() - started) * 1000.0
    timings = session.last_timings
    backend.shutdown()

    try:
        import resource

        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        peak_kb = -1
    try:
        with open("/proc/meminfo") as handle:
            mem_total = next(
                line for line in handle if line.startswith("MemTotal")
            ).strip()
    except Exception:
        mem_total = "unknown"

    transcript_summary = ""
    if events["responses"]:
        text = events["responses"][0][1]
        transcript_summary = f"{len(text.split())} words / {len(text)} chars"

    evidence = {
        "ok": not events["failures"] and bool(events["audios"]),
        "cpu_arch": platform.machine(),
        "cpu_count": os.cpu_count(),
        "mem_total": mem_total,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "profile_id": manifest.profile_id,
        "model_ids": [c.model_id for c in manifest.components],
        "sha256": {c.component: sha256_file(f"{args.model_root}/{c.filename}") for c in manifest.components},
        "input_fixture": os.path.basename(args.audio_fixture),
        "input_seconds": round(len(pcm16) / 2 / 16000, 2),
        "llm_result": (
            events["actions"][0]
            if events["actions"]
            else ("spoken" if events["responses"] else "none")
        ),
        "transcript_summary": transcript_summary,
        "output_audio_bytes": sum(n for _, n, _ in events["audios"]),
        "output_audio_frames": len(events["audios"]),
        "eou_to_first_audio_ms": (
            round(timings.eou_to_first_audio_ms, 1)
            if timings and timings.eou_to_first_audio_ms is not None
            else None
        ),
        "total_turn_ms": round(total_ms, 1),
        "latency_band": timings.latency_band() if timings else "unknown",
        "peak_rss_kb": peak_kb,
        "failures": events["failures"],
    }
    print(json.dumps(evidence, indent=2))
    return 0 if evidence["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
