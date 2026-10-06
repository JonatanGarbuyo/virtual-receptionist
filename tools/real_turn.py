"""Run one REAL local cascaded turn on CPU and report evidence (#24).

Offline: needs installed runtimes + pinned manifest + model pack.
Fails closed with a clear reason when anything is missing — it never
fabricates a turn. Prints a sanitized evidence document (ids, sizes,
timings, versions; transcript summarized, never full prompts).

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


def _rss_kb(pid) -> int:
    """Resident set size from /proc (Linux-only evidence runner)."""
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return -1


def _measure_tree(backend) -> dict[str, int]:
    """Composite RSS while every runtime is resident: this Python
    process (sherpa TTS lives in-process) plus both server processes.
    Sampled after the turn, servers still warm."""
    whisper_pid = backend._stt._server.pid
    llama_pid = backend._llm._server.pid
    own = _rss_kb("self")
    whisper = _rss_kb(whisper_pid) if whisper_pid else -1
    llama = _rss_kb(llama_pid) if llama_pid else -1
    parts = [value for value in (own, whisper, llama) if value > 0]
    return {
        "rss_kb_self": own,
        "rss_kb_whisper_server": whisper,
        "rss_kb_llama_server": llama,
        "rss_kb_total_estimated": sum(parts) if parts else -1,
    }


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
    parser.add_argument("--stt-exe", default="whisper-server")
    parser.add_argument("--llm-exe", default="llama-server")
    parser.add_argument("--tts-voice", default="es-female-1")
    parser.add_argument("--llm-threads", type=int, default=4)
    parser.add_argument("--profile-id", default="cascaded-cpu-baseline-v1")
    args = parser.parse_args()

    from receptionist.audio import resample_pcm16, split_pcm
    from receptionist.boundaries import ProviderFailure
    from receptionist.cascaded import VoiceProfile
    from receptionist.local_runtimes import build_cascaded_backend
    from receptionist.voice_manifest import (
        load_manifest_file,
        resolve_trusted_path,
        sha256_file,
        verify_manifest,
    )

    try:
        manifest = load_manifest_file(args.manifest)
    except Exception as error:
        return fail(f"manifest unreadable: {type(error).__name__}")

    problems = verify_manifest(args.model_root, manifest)
    if problems:
        return fail(f"integrity failed: {[f'{p.component}:{p.reason}' for p in problems]}")

    try:
        fixture_sha = sha256_file(args.audio_fixture)
    except OSError as error:
        return fail(f"fixture unreadable: {type(error).__name__}")

    profile = VoiceProfile(
        profile_id=args.profile_id,
        model_root=args.model_root,
        manifest_path=args.manifest,
        stt_executable=args.stt_exe,
        llm_executable=args.llm_exe,
        tts_voice=args.tts_voice,
        llm_threads=args.llm_threads,
    )
    try:
        backend = build_cascaded_backend(profile)
    except Exception as error:
        return fail(f"backend composition failed: {type(error).__name__}")

    try:
        pcm, rate = read_pcm_16k_mono(args.audio_fixture)
    except Exception as error:
        backend.shutdown()
        return fail(f"fixture unreadable: {type(error).__name__}")
    pcm16 = resample_pcm16(pcm, rate, 16000)

    backend.start()
    warm_problems = backend.warm()
    versions = {
        "stt": backend._stt.version_info(),
        "llm": backend._llm.version_info(),
        "tts": backend._tts.version_info(),
    }
    if warm_problems or not backend.ready:
        ready, detail = backend.check_ready()
        backend.shutdown()
        return fail(f"warmup failed: {warm_problems or [detail]}")

    events: dict = {
        "sidecars": [],
        "responses": [],
        "audios": [],
        "actions": [],
        "failures": [],
        "playbacks": [],
    }

    class Listener:
        def on_transcript(self, text: str) -> None:
            pass

        def on_transcript_sidecar(self, text: str) -> None:
            events["sidecars"].append(text)

        def on_response(self, turn_id: int, text: str) -> None:
            events["responses"].append((turn_id, text))

        def on_audio(self, turn_id: int, frame) -> None:
            events["audios"].append((turn_id, len(frame.pcm), frame.sample_rate))

        def on_playback_finished(self, turn_id: int) -> None:
            events["playbacks"].append(turn_id)

        def on_action_request(self, action: object) -> None:
            detail = getattr(action, "destination_id", "")
            events["actions"].append(
                f"{type(action).__name__}:{detail}" if detail else type(action).__name__
            )

        def on_provider_failure(self, turn_id: int, failure: ProviderFailure) -> None:
            events["failures"].append((turn_id, failure.category.value))

    session = backend.open_session("real-turn-1", Listener())
    for frame in split_pcm(pcm16, 16000, call_id="real-turn-1", turn_id=1):
        session.push_audio(frame)
    started = time.monotonic()
    session.commit_turn(1)
    drained = session.wait_until_idle(timeout=300.0)
    total_ms = (time.monotonic() - started) * 1000.0
    timings = session.last_timings
    memory = _measure_tree(backend)
    backend.shutdown()

    stt_text = events["sidecars"][0] if events["sidecars"] else ""

    try:
        with open("/proc/meminfo") as handle:
            mem_total = next(
                line for line in handle if line.startswith("MemTotal")
            ).strip()
    except Exception:
        mem_total = "unknown"

    stt_text = events["sidecars"][0] if events["sidecars"] else ""
    out_bytes = sum(n for _, n, _ in events["audios"])
    out_rate = events["audios"][0][2] if events["audios"] else 16000

    def _stage_ms(start: float, end: float) -> float | None:
        if start > 0.0 and end >= start:
            return round((end - start) * 1000.0, 1)
        return None

    stages = (
        {
            "stt_ms": _stage_ms(timings.stt_started_at, timings.stt_finished_at),
            "llm_ms": _stage_ms(timings.llm_started_at, timings.llm_finished_at),
            "tts_to_first_audio_ms": _stage_ms(
                timings.tts_started_at, timings.first_audio_at
            ),
        }
        if timings
        else {}
    )
    evidence = {
        "ok": bool(drained) and not events["failures"] and bool(events["audios"]),
        "cpu_arch": platform.machine(),
        "cpu_count": os.cpu_count(),
        "mem_total": mem_total,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_only": {
            "whisper_flag": "--no-gpu",
            "llama_flag": "-ngl 0",
            "llama_offline_flag": "--offline",
            "sherpa_provider": "cpu",
        },
        "runtime_versions": versions,
        "profile_id": manifest.profile_id,
        "model_ids": [c.model_id for c in manifest.components],
        "artifact_sha256": {
            filename: sha256_file(resolve_trusted_path(args.model_root, filename))
            for component in manifest.components
            for filename in (component.files or (component.filename,))
        },
        "input_fixture": os.path.basename(args.audio_fixture),
        "input_fixture_sha256": fixture_sha,
        "input_seconds": round(len(pcm16) / 2 / 16000, 2),
        "stt_transcript_summary": (
            f"{len(stt_text.split())} words / {len(stt_text)} chars" if stt_text else "none"
        ),
        "llm_result": (
            events["actions"][0]
            if events["actions"]
            else ("spoken" if events["responses"] else "none")
        ),
        "output_audio_bytes": out_bytes,
        "output_audio_frames": len(events["audios"]),
        "output_audio_seconds": round(out_bytes / 2 / (out_rate or 16000), 2),
        "eou_to_first_audio_ms": (
            round(timings.eou_to_first_audio_ms, 1)
            if timings and timings.eou_to_first_audio_ms is not None
            else None
        ),
        "stage_ms": stages,
        "total_turn_ms": round(total_ms, 1),
        "latency_band": timings.latency_band() if timings else "unknown",
        **memory,
        "failures": events["failures"],
    }
    print(json.dumps(evidence, indent=2))
    return 0 if evidence["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
