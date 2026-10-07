"""Project-owned G.711 codec helpers for the telephony adapter (#25).

Stdlib only. The production media path (baresip/libbaresip through
``baresip-python`` with the ``aumem`` driver) already converts line codec
<-> PCM16 inside the stack: the adapter only resamples and normalizes.
These helpers exist so tests can *prove* media properties instead of
asserting "RTP packets exist":

- encode app PCM fixtures to PCMU/PCMA payloads,
- decode them back and assert energy/frequency/duration properties,
- document that G.711 is lossy (no byte-perfect assertions).

ITU-T G.711 μ-law / A-law, signed 16-bit linear PCM <-> 8-bit codes.
"""

from __future__ import annotations

_MULAW_BIAS = 0x84
_MULAW_CLIP = 32635


def linear_to_ulaw(sample: int) -> int:
    """Encode one 16-bit linear sample to 8-bit μ-law."""
    sample = max(-32768, min(32767, int(sample)))
    sign = 0x80 if sample < 0 else 0x00
    if sample < 0:
        sample = -sample
    if sample > _MULAW_CLIP:
        sample = _MULAW_CLIP
    sample += _MULAW_BIAS
    exponent = 7
    for exp in range(7, -1, -1):
        if sample & (1 << (exp + 7)):
            exponent = exp
            break
    else:
        exponent = 0
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def ulaw_to_linear(code: int) -> int:
    """Decode one 8-bit μ-law code to a 16-bit linear sample."""
    code = (~int(code)) & 0xFF
    sign = code & 0x80
    exponent = (code >> 4) & 0x07
    mantissa = code & 0x0F
    sample = ((mantissa << 3) + _MULAW_BIAS) << exponent
    sample -= _MULAW_BIAS
    return -sample if sign else sample


def linear_to_alaw(sample: int) -> int:
    """Encode one 16-bit linear sample to 8-bit A-law."""
    sample = max(-32768, min(32767, int(sample)))
    sign = 0x80 if sample < 0 else 0x00
    if sample < 0:
        sample = -sample
    if sample > 32767:
        sample = 32767
    if sample >= 256:
        exponent = 7
        for exp in range(7, 0, -1):
            if sample & (1 << (exp + 7)):
                exponent = exp
                break
        mantissa = (sample >> (exponent + 3)) & 0x0F
        code = (exponent << 4) | mantissa
    else:
        code = (sample >> 4) & 0x0F
    return (sign | code ^ 0x55) & 0xFF


def alaw_to_linear(code: int) -> int:
    """Decode one 8-bit A-law code to a 16-bit linear sample."""
    code = int(code) ^ 0x55
    sign = code & 0x80
    exponent = (code >> 4) & 0x07
    mantissa = code & 0x0F
    if exponent == 0:
        sample = (mantissa << 4) + 8
    else:
        sample = ((mantissa << 4) + 0x108) << (exponent - 1)
    return -sample if sign else sample


def encode_pcm16_to_ulaw(pcm: bytes) -> bytes:
    """Encode mono PCM16-LE bytes to a PCMU payload."""
    if len(pcm) % 2 != 0:
        raise ValueError("pcm16 needs an even byte count")
    count = len(pcm) // 2
    import struct

    samples = struct.unpack(f"<{count}h", pcm)
    return bytes(linear_to_ulaw(s) for s in samples)


def decode_ulaw_to_pcm16(payload: bytes) -> bytes:
    """Decode a PCMU payload to mono PCM16-LE bytes."""
    import struct

    samples = [ulaw_to_linear(b) for b in payload]
    return struct.pack(f"<{len(samples)}h", *samples)


def encode_pcm16_to_alaw(pcm: bytes) -> bytes:
    """Encode mono PCM16-LE bytes to a PCMA payload."""
    if len(pcm) % 2 != 0:
        raise ValueError("pcm16 needs an even byte count")
    count = len(pcm) // 2
    import struct

    samples = struct.unpack(f"<{count}h", pcm)
    return bytes(linear_to_alaw(s) for s in samples)


def decode_alaw_to_pcm16(payload: bytes) -> bytes:
    """Decode a PCMA payload to mono PCM16-LE bytes."""
    import struct

    samples = [alaw_to_linear(b) for b in payload]
    return struct.pack(f"<{len(samples)}h", *samples)
