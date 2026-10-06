"""Project-owned PCM helpers for the voice seam (#24).

Everything here is generic signed PCM16 mono: no codecs, no vendors,
no model concepts. G.711 conversion belongs to the telephony/media
adapter (#25). Audio is transient: nothing here writes audio to disk.
"""

from __future__ import annotations

import math
import struct

from receptionist.boundaries import (
    AUDIO_CHANNELS_MONO,
    AUDIO_SAMPLE_FORMAT_PCM16,
    AudioFrame,
)


def make_frame(
    pcm: bytes,
    sample_rate: int,
    *,
    call_id: str = "",
    turn_id: int = 0,
    sequence: int = 0,
    timestamp: float = 0.0,
) -> AudioFrame:
    """Build a validated mono PCM16 frame for the seam."""
    return AudioFrame(
        pcm=bytes(pcm),
        sample_rate=sample_rate,
        channels=AUDIO_CHANNELS_MONO,
        sample_format=AUDIO_SAMPLE_FORMAT_PCM16,
        call_id=call_id,
        turn_id=turn_id,
        sequence=sequence,
        timestamp=timestamp,
    )


def split_pcm(
    pcm: bytes,
    sample_rate: int,
    *,
    samples_per_chunk: int = 1600,
    call_id: str = "",
    turn_id: int = 0,
    timestamp: float = 0.0,
) -> list[AudioFrame]:
    """Split raw PCM16 bytes into sequentially numbered frames.

    Chunking is a transport detail: the application never depends on a
    vendor's native chunk size, because adapters normalize behind this
    helper before crossing the seam.
    """
    if len(pcm) % 2 != 0:
        raise ValueError("pcm16 needs an even byte count")
    if samples_per_chunk <= 0:
        raise ValueError("samples_per_chunk must be > 0")
    stride = samples_per_chunk * 2
    frames = []
    for index in range(0, len(pcm), stride):
        frames.append(
            make_frame(
                pcm[index : index + stride],
                sample_rate,
                call_id=call_id,
                turn_id=turn_id,
                sequence=len(frames),
                timestamp=timestamp,
            )
        )
    if not frames:
        raise ValueError("split_pcm needs non-empty PCM bytes")
    return frames


def frame_rms(frame: AudioFrame) -> float:
    """Root-mean-square amplitude in [0, 1]: speech/silence gate input."""
    count = len(frame.pcm) // 2
    if count == 0:
        return 0.0
    values = struct.unpack(f"<{count}h", frame.pcm)
    energy = sum(sample * sample for sample in values) / count
    return math.sqrt(energy) / 32768.0


#: RMS below this counts as silence for barge-in gating: pushed frames
#: that carry no speech energy must never cancel assistant output.
SILENCE_RMS_THRESHOLD = 0.02


def is_speech(frame: AudioFrame, *, threshold: float = SILENCE_RMS_THRESHOLD) -> bool:
    """Whether a caller frame carries speech energy (barge-in gate)."""
    return frame_rms(frame) >= threshold


def tone_pcm(
    *,
    duration_seconds: float = 0.5,
    sample_rate: int = 16000,
    frequency_hz: float = 440.0,
    amplitude: float = 0.4,
) -> bytes:
    """Deterministic synthetic tone: privacy-safe test fixture source.

    Never a customer recording. Used for pipeline/contract tests and
    as the warmup/silence probe signal shape.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")
    count = int(duration_seconds * sample_rate)
    peak = int(32767 * amplitude)
    samples = [
        int(peak * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate))
        for index in range(count)
    ]
    return struct.pack(f"<{count}h", *samples)


def silence_pcm(
    *, duration_seconds: float = 0.5, sample_rate: int = 16000
) -> bytes:
    """Deterministic digital silence: the no-input probe signal."""
    count = int(duration_seconds * sample_rate)
    if count <= 0:
        raise ValueError("duration_seconds must be > 0")
    return bytes(count * 2)


def pcm_duration_seconds(pcm: bytes, sample_rate: int) -> float:
    """Duration of PCM16 mono bytes at the given rate."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    return (len(pcm) // 2) / float(sample_rate)


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolated mono PCM16 resampling (stdlib only).

    The session normalizes caller audio (e.g. 8 kHz telephony PCM) to
    the STT input rate behind the seam, so adapters never branch on
    vendor or telephony rates.
    """
    if len(pcm) % 2 != 0:
        raise ValueError("pcm16 needs an even byte count")
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be > 0")
    if src_rate == dst_rate:
        return bytes(pcm)
    src_count = len(pcm) // 2
    if src_count == 0:
        return b""
    src = struct.unpack(f"<{src_count}h", pcm)
    dst_count = max(1, round(src_count * dst_rate / src_rate))
    out = [0] * dst_count
    for index in range(dst_count):
        if dst_count == 1:
            position = 0.0
        else:
            position = index * (src_count - 1) / (dst_count - 1)
        before = int(position)
        frac = position - before
        after = min(before + 1, src_count - 1)
        out[index] = int(round(src[before] * (1.0 - frac) + src[after] * frac))
    return struct.pack(f"<{dst_count}h", *out)
