"""Deterministic doubles for the cascaded voice backend tests.

Test-only: synchronous scripted STT/LLM/TTS adapters plus structured
document helpers. Production code never imports this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from receptionist.boundaries import ProviderFailureCategory
from receptionist.cascaded import (
    AdapterError,
    CancelToken,
    CancelledError,
    STTResult,
    TTS_SAMPLE_RATE,
)



@dataclass
class ScriptedFailure:
    category: ProviderFailureCategory
    detail: str = "scripted failure"


class FakeSTTAdapter:
    """Deterministic STT double: scripted transcripts/failures."""

    component = "stt"

    def __init__(
        self,
        transcripts: list[str] | None = None,
        failures: list[ScriptedFailure] | None = None,
    ) -> None:
        self.transcripts = list(transcripts or [])
        self.failures = list(failures or [])
        self.calls: list[dict] = []
        self.warmups = 0
        self.closed = False

    def transcribe(self, pcm: bytes, sample_rate: int, cancel: CancelToken) -> STTResult:
        cancel.throw_if_cancelled()
        self.calls.append({"bytes": len(pcm), "sample_rate": sample_rate})
        if self.failures:
            failure = self.failures.pop(0)
            raise AdapterError(failure.category, failure.detail)
        if self.transcripts:
            return STTResult(text=self.transcripts.pop(0))
        return STTResult(text="")

    def warmup(self) -> None:
        self.warmups += 1

    def close(self) -> None:
        self.closed = True


class FakeLLMAdapter:
    """Deterministic LLM double: scripted raw structured documents."""

    component = "llm"

    def __init__(
        self,
        documents: list[str] | None = None,
        failures: list[ScriptedFailure] | None = None,
    ) -> None:
        self.documents = list(documents or [])
        self.failures = list(failures or [])
        self.prompts: list[str] = []
        self.warmups = 0
        self.closed = False

    def generate(self, prompt: str, cancel: CancelToken) -> str:
        cancel.throw_if_cancelled()
        self.prompts.append(prompt)
        if self.failures:
            failure = self.failures.pop(0)
            raise AdapterError(failure.category, failure.detail)
        if self.documents:
            return self.documents.pop(0)
        return json.dumps({"spoken_text": "De acuerdo.", "action": None})

    def warmup(self) -> None:
        self.warmups += 1

    def close(self) -> None:
        self.closed = True


class FakeTTSAdapter:
    """Deterministic TTS double: scripted PCM chunks per synthesis.

    ``fail_after_chunks`` emits N chunks then raises; ``fail_before``
    raises before any audio. ``on_chunk_hook`` runs before each chunk
    so tests can interleave barge-in deterministically.
    """

    component = "tts"

    def __init__(
        self,
        chunks: list[bytes] | None = None,
        failures: list[ScriptedFailure] | None = None,
        fail_after_chunks: int | None = None,
        fail_before: ScriptedFailure | None = None,
        on_chunk_hook: Callable[[], None] | None = None,
        sample_rate: int = TTS_SAMPLE_RATE,
    ) -> None:
        self.chunks = list(chunks or [])
        self.failures = list(failures or [])
        self.fail_after_chunks = fail_after_chunks
        self.fail_before = fail_before
        self.on_chunk_hook = on_chunk_hook
        self.sample_rate = sample_rate
        self.texts: list[str] = []
        self.warmups = 0
        self.closed = False

    def synthesize(
        self,
        text: str,
        cancel: CancelToken,
        on_chunk: Callable[[bytes, int], None],
    ) -> int:
        cancel.throw_if_cancelled()
        self.texts.append(text)
        if self.fail_before is not None:
            failure = self.fail_before
            self.fail_before = None
            raise AdapterError(failure.category, failure.detail)
        if self.failures:
            failure = self.failures.pop(0)
            raise AdapterError(failure.category, failure.detail)
        total = 0
        emitted = 0
        pending = list(self.chunks) if self.chunks else [_default_tts_bytes()]
        for piece in pending:
            if self.on_chunk_hook is not None:
                self.on_chunk_hook()
            cancel.throw_if_cancelled()
            on_chunk(piece, self.sample_rate)
            total += len(piece)
            emitted += 1
            if (
                self.fail_after_chunks is not None
                and emitted >= self.fail_after_chunks
            ):
                self.fail_after_chunks = None
                raise AdapterError(
                    ProviderFailureCategory.INTERNAL, "tts failed mid-stream"
                )
        return total

    def warmup(self) -> None:
        self.warmups += 1

    def close(self) -> None:
        self.closed = True


def _default_tts_bytes() -> bytes:
    from receptionist.audio import tone_pcm

    return tone_pcm(duration_seconds=0.2)


class FailingAdapter:
    """Warmup-failure double for readiness tests."""

    def __init__(
        self, component: str, category: ProviderFailureCategory
    ) -> None:
        self.component = component
        self._category = category
        self.closed = False

    def transcribe(self, pcm: bytes, sample_rate: int, cancel: CancelToken) -> STTResult:
        raise AdapterError(self._category, "failing adapter")

    def generate(self, prompt: str, cancel: CancelToken) -> str:
        raise AdapterError(self._category, "failing adapter")

    def synthesize(
        self,
        text: str,
        cancel: CancelToken,
        on_chunk: Callable[[bytes, int], None],
    ) -> int:
        raise AdapterError(self._category, "failing adapter")

    def warmup(self) -> None:
        raise AdapterError(self._category, "warmup failed")

    def close(self) -> None:
        self.closed = True


def spoken_document(text: str) -> str:
    """Test helper: spoken-only structured document."""
    return json.dumps({"spoken_text": text, "action": None})


def transfer_document(text: str, destination_id: str) -> str:
    """Test helper: spoken text plus a typed transfer request."""
    return json.dumps(
        {
            "spoken_text": text,
            "action": {"type": "transfer", "destination_id": destination_id},
        }
    )
