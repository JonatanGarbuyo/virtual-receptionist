"""Project-owned G.711 codec helpers for the telephony adapter (#25).

Stdlib only. The production media path (baresip/libbaresip through
``baresip-python`` with the ``aumem`` driver) already converts line codec
<-> PCM16 inside the stack: the adapter only resamples and normalizes.
These helpers exist so tests can *prove* media properties instead of
asserting "RTP packets exist":

- encode app PCM fixtures to PCMU/PCMA payloads,
- decode them back and assert energy/frequency/duration properties,
- document that G.711 is lossy (no byte-perfect assertions).

ITU-T G.711 μ-law / A-law, signed 16-bit linear PCM <-> 8-bit codes,
following the canonical Sun/public-domain segment-table algorithms
(validated sample-for-sample against an independent reference; see
``tests/test_telephony_contract.py`` for the fixed vector table).
"""

from __future__ import annotations

import struct

#: Segment boundaries shared by both laws (Sun tables).
_SEGMENTS = (0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF)


def _segment(value: int) -> int:
    for index, edge in enumerate(_SEGMENTS):
        if value <= edge:
            return index
    return len(_SEGMENTS)


def linear_to_ulaw(sample: int) -> int:
    """Encode one 16-bit linear sample to 8-bit μ-law."""
    sample = max(-32768, min(32767, int(sample)))
    if sample < 0:
        sample = 0x84 - sample
        mask = 0x7F
    else:
        sample += 0x84
        mask = 0xFF
    segment = _segment(sample)
    if segment >= 8:
        return 0x7F ^ mask
    return (((segment << 4) | ((sample >> (segment + 3)) & 0x0F)) ^ mask) & 0xFF


def ulaw_to_linear(code: int) -> int:
    """Decode one 8-bit μ-law code to a 16-bit linear sample."""
    code = (~int(code)) & 0xFF
    sign = code & 0x80
    exponent = (code >> 4) & 0x07
    mantissa = code & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    sample -= 0x84
    return -sample if sign else sample


def linear_to_alaw(sample: int) -> int:
    """Encode one 16-bit linear sample to 8-bit A-law.

    Note the inverted sign convention: a positive sample encodes with
    the MSB *set* (the `0xD5` mask), unlike μ-law.
    """
    sample = max(-32768, min(32767, int(sample)))
    if sample >= 0:
        mask = 0xD5
    else:
        mask = 0x55
        sample = -sample - 1
    segment = _segment(sample)
    if segment >= 8:
        return 0x7F ^ mask
    value = segment << 4
    if segment < 2:
        value |= (sample >> 4) & 0x0F
    else:
        value |= (sample >> (segment + 3)) & 0x0F
    return (value ^ mask) & 0xFF


def alaw_to_linear(code: int) -> int:
    """Decode one 8-bit A-law code to a 16-bit linear sample.

    After the `0x55` toggle a set sign bit means *positive* (inverse of
    μ-law); getting this backwards negates the whole signal.
    """
    code = int(code) ^ 0x55
    sign = code & 0x80
    exponent = (code >> 4) & 0x07
    mantissa = code & 0x0F
    if exponent == 0:
        sample = (mantissa << 4) + 8
    else:
        sample = ((mantissa << 4) + 0x108) << (exponent - 1)
    return sample if sign else -sample


def encode_pcm16_to_ulaw(pcm: bytes) -> bytes:
    """Encode mono PCM16-LE bytes to a PCMU payload."""
    if len(pcm) % 2 != 0:
        raise ValueError("pcm16 needs an even byte count")
    count = len(pcm) // 2
    samples = struct.unpack(f"<{count}h", pcm)
    return bytes(linear_to_ulaw(sample) for sample in samples)


def decode_ulaw_to_pcm16(payload: bytes) -> bytes:
    """Decode a PCMU payload to mono PCM16-LE bytes."""
    return struct.pack(f"<{len(payload)}h", *(ulaw_to_linear(b) for b in payload))


def encode_pcm16_to_alaw(pcm: bytes) -> bytes:
    """Encode mono PCM16-LE bytes to a PCMA payload."""
    if len(pcm) % 2 != 0:
        raise ValueError("pcm16 needs an even byte count")
    count = len(pcm) // 2
    samples = struct.unpack(f"<{count}h", pcm)
    return bytes(linear_to_alaw(sample) for sample in samples)


def decode_alaw_to_pcm16(payload: bytes) -> bytes:
    """Decode a PCMA payload to mono PCM16-LE bytes."""
    return struct.pack(f"<{len(payload)}h", *(alaw_to_linear(b) for b in payload))
