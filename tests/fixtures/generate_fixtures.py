"""Generate privacy-safe synthetic audio fixtures (#24).

Every fixture is synthesized locally by this script: no customer
recordings, no redistributed speech. Real-speech STT validation uses
an operator-provided fixture (see tests/test_cascaded_real.py), never
anything committed here.

Run:  PYTHONPATH=src python3 tests/fixtures/generate_fixtures.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from receptionist.audio import silence_pcm, tone_pcm

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    fixtures = {
        # Speech-energy probe: deterministic 440 Hz tone, 16 kHz mono.
        "caller-tone-16k.pcm": tone_pcm(
            duration_seconds=1.0, sample_rate=16000, frequency_hz=440.0
        ),
        # Argentine-Spanish-ish cadence probe: two-tone stand-in used
        # only for pipeline/contract shaping, never as STT ground truth.
        "caller-cadence-16k.pcm": tone_pcm(
            duration_seconds=1.0, sample_rate=16000, frequency_hz=330.0
        )
        + tone_pcm(duration_seconds=0.5, sample_rate=16000, frequency_hz=392.0),
        # Telephony-rate probe: 8 kHz input exercising session resample.
        "caller-tone-8k.pcm": tone_pcm(
            duration_seconds=1.0, sample_rate=8000, frequency_hz=440.0
        ),
        # No-input probe: digital silence.
        "silence-16k.pcm": silence_pcm(duration_seconds=1.0, sample_rate=16000),
    }
    for name, payload in fixtures.items():
        path = os.path.join(HERE, name)
        with open(path, "wb") as handle:
            handle.write(payload)
        print(f"wrote {name} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()
