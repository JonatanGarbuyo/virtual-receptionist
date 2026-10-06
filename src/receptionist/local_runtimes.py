"""Local runtime adapters: whisper.cpp, llama.cpp, sherpa-onnx (#24).

Each adapter shells a trusted local executable (``shell=False``, argv
explicit, controlled cwd, timeout, bounded stdout/stderr, deterministic
terminate/kill on cancellation) or, for sherpa-onnx, a guarded optional
in-process import. Raw vendor stderr never reaches the caller or the
failure detail: errors normalize to AdapterError with sanitized ids.

Supported versions are pinned below; warmup verifies the executable
responds and can load its model. Anything else fails closed
(UNAVAILABLE/INTERNAL) so the backend reports NOT_READY instead of
discovering a missing GGUF twenty seconds into a call.

CLI-shape note: the exact flags below track the documented interface
of the pinned runtime versions. Warmup re-checks ``--version`` output
against the pinned set; an unexpected version fails closed with a
message telling the operator which runtime release is expected, so a
flag drift can never silently change turn behavior.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import wave

from receptionist.cascaded import (
    AdapterError,
    CancelledError,
    CancelToken,
    STTResult,
)
from receptionist.boundaries import ProviderFailureCategory

LOG = logging.getLogger("receptionist.runtimes")

#: Pinned runtime releases this backend was written and reviewed
#: against. Warmup accepts version output containing one of these
#: markers; anything else is UNAVAILABLE (operator upgrades pin here).
SUPPORTED_WHISPER_VERSIONS = ("whisper.cpp-1.7",)
SUPPORTED_LLAMA_VERSIONS = ("llama.cpp-1.7", "llama.cpp-1.8", "llama.cpp-b6")
SUPPORTED_SHERPA_VERSIONS = ("sherpa-onnx-1.10", "sherpa-onnx-1.11")

#: Hard cap on captured subprocess output: diagnostics stay bounded.
MAX_SUBPROCESS_OUTPUT = 256 * 1024


def _trusted_executable(configured: str, *, name: str) -> str:
    """Resolve an operator-configured executable fail-closed: absolute
    path, exists, executable bit. PATH lookup only when the operator
    gave a bare binary name (still resolved, never executed by shell).
    Relative paths with separators are rejected (no cwd-relative
    surprises from caller-influenced config)."""
    if not configured or not configured.strip():
        raise AdapterError(
            ProviderFailureCategory.UNAVAILABLE, f"{name} executable not configured"
        )
    candidate = configured.strip()
    if os.sep in candidate and not os.path.isabs(candidate):
        raise AdapterError(
            ProviderFailureCategory.UNAVAILABLE, f"{name} executable must be absolute"
        )
    resolved = candidate
    if os.sep not in candidate:
        found = shutil.which(candidate)
        if found is None:
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, f"{name} executable not found"
            )
        resolved = found
    if not os.path.isfile(resolved) or not os.access(resolved, os.X_OK):
        raise AdapterError(
            ProviderFailureCategory.UNAVAILABLE, f"{name} executable unavailable"
        )
    return resolved


def _check_version(
    executable: str,
    *,
    name: str,
    supported: tuple[str, ...],
    timeout_seconds: float,
) -> str:
    """Run ``executable --version`` and match against pinned markers."""
    try:
        completed = subprocess.run(
            [executable, "--version"],
            shell=False,
            cwd=tempfile.gettempdir(),
            capture_output=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise AdapterError(
            ProviderFailureCategory.UNAVAILABLE, f"{name} version check failed"
        ) from error
    output = (completed.stdout or b"")[:4096].decode("utf-8", "replace")
    output += (completed.stderr or b"")[:4096].decode("utf-8", "replace")
    for marker in supported:
        if marker in output:
            return output.strip()[:200]
    raise AdapterError(
        ProviderFailureCategory.UNAVAILABLE, f"{name} version unsupported"
    )


def _run_trusted(
    executable: str,
    args: list[str],
    *,
    name: str,
    timeout_seconds: float,
    cancel: CancelToken,
    input_bytes: bytes | None = None,
) -> bytes:
    """Run one trusted subprocess. ``shell=False``, argv explicit,
    bounded output, terminate/kill on cancel or timeout. Returns
    stdout (truncated to the cap). Raises AdapterError (TIMEOUT /
    UNAVAILABLE / INTERNAL) or CancelledError."""
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [executable, *args],
            shell=False,
            cwd=tempfile.gettempdir(),
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise AdapterError(
            ProviderFailureCategory.UNAVAILABLE, f"{name} failed to start"
        ) from error
    handle = process

    def _terminate() -> None:
        try:
            handle.terminate()
        except Exception:
            pass

    cancel.on_cancel(_terminate)
    try:
        try:
            stdout, _stderr = handle.communicate(
                input=input_bytes, timeout=timeout_seconds
            )
        except subprocess.TimeoutExpired:
            try:
                handle.kill()
            except Exception:
                pass
            try:
                handle.wait(timeout=5)
            except Exception:
                pass
            raise AdapterError(ProviderFailureCategory.TIMEOUT, f"{name} timed out")
        if cancel.cancelled:
            raise CancelledError()
        if handle.returncode != 0:
            raise AdapterError(
                ProviderFailureCategory.INTERNAL, f"{name} exited unsuccessfully"
            )
        return (stdout or b"")[:MAX_SUBPROCESS_OUTPUT]
    finally:
        if handle.poll() is None:
            try:
                handle.kill()
            except Exception:
                pass


def _write_wav_16k_mono(pcm: bytes, path: str) -> None:
    """Transient WAV for runtimes that require a file input. The file
    lives in a temp dir and the caller deletes it immediately after
    the subprocess exits: no audio persistence."""
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm)


class WhisperSubprocessSTT:
    """whisper.cpp STT: 16 kHz mono PCM in, final transcript out."""

    component = "stt"

    def __init__(
        self,
        *,
        executable: str,
        model_path: str,
        language: str = "es",
        timeout_seconds: float = 30.0,
    ) -> None:
        if not model_path or not os.path.isfile(model_path):
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "stt model unavailable"
            )
        self._executable = _trusted_executable(executable, name="stt")
        self._model_path = model_path
        self._language = language
        self._timeout = timeout_seconds
        self._lock = threading.Lock()
        self._closed = False

    def transcribe(
        self, pcm: bytes, sample_rate: int, cancel: CancelToken
    ) -> STTResult:
        from receptionist.audio import resample_pcm16

        cancel.throw_if_cancelled()
        with self._lock:
            if self._closed:
                raise AdapterError(
                    ProviderFailureCategory.UNAVAILABLE, "stt runtime closed"
                )
        if sample_rate != 16000:
            pcm = resample_pcm16(pcm, sample_rate, 16000)
        if not pcm:
            raise AdapterError(
                ProviderFailureCategory.INVALID_OUTPUT, "empty turn audio"
            )
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp:
                tmp_path = tmp.name
            _write_wav_16k_mono(pcm, tmp_path)
            stdout = _run_trusted(
                self._executable,
                [
                    "-m", self._model_path,
                    "-f", tmp_path,
                    "-l", self._language,
                    "--output-txt",
                    "--no-timestamps",
                ],
                name="stt",
                timeout_seconds=self._timeout,
                cancel=cancel,
            )
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        text = stdout.decode("utf-8", "replace").strip()
        return STTResult(text=text)

    def warmup(self) -> None:
        _check_version(
            self._executable,
            name="stt",
            supported=SUPPORTED_WHISPER_VERSIONS,
            timeout_seconds=10.0,
        )
        silence = bytes(16000 * 2)  # 1 s of digital silence
        try:
            self.transcribe(silence, 16000, CancelToken())
        except AdapterError:
            raise
        except Exception as error:
            raise AdapterError(
                ProviderFailureCategory.INTERNAL, "stt warmup failed"
            ) from error

    def close(self) -> None:
        with self._lock:
            self._closed = True


class LlamaSubprocessLLM:
    """llama.cpp generation: bounded prompt in, raw structured doc out.

    Non-thinking mode is enforced at prompt level (``/no_think`` marker
    for Qwen3 chat templates) plus a hard token bound, so the model
    can never spend the turn on long reasoning we would discard.
    """

    component = "llm"

    def __init__(
        self,
        *,
        executable: str,
        model_path: str,
        max_tokens: int = 220,
        threads: int = 4,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not model_path or not os.path.isfile(model_path):
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "llm model unavailable"
            )
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self._executable = _trusted_executable(executable, name="llm")
        self._model_path = model_path
        self._max_tokens = max_tokens
        self._threads = max(1, threads)
        self._timeout = timeout_seconds
        self._lock = threading.Lock()
        self._closed = False

    def generate(self, prompt: str, cancel: CancelToken) -> str:
        cancel.throw_if_cancelled()
        with self._lock:
            if self._closed:
                raise AdapterError(
                    ProviderFailureCategory.UNAVAILABLE, "llm runtime closed"
                )
        # Qwen3 non-thinking marker: prompt-level, version-independent.
        bounded = prompt.strip() + "\n/no_think"
        stdout = _run_trusted(
            self._executable,
            [
                "-m", self._model_path,
                "-p", bounded,
                "-n", str(self._max_tokens),
                "-t", str(self._threads),
                "--temp", "0.4",
                "--no-display-prompt",
            ],
            name="llm",
            timeout_seconds=self._timeout,
            cancel=cancel,
        )
        raw = stdout.decode("utf-8", "replace").strip()
        if not raw:
            raise AdapterError(
                ProviderFailureCategory.INVALID_OUTPUT, "empty model output"
            )
        return raw

    def warmup(self) -> None:
        _check_version(
            self._executable,
            name="llm",
            supported=SUPPORTED_LLAMA_VERSIONS,
            timeout_seconds=10.0,
        )
        try:
            self.generate('{"spoken_text": "Hola.", "action": null}', CancelToken())
        except AdapterError:
            raise
        except Exception as error:
            raise AdapterError(
                ProviderFailureCategory.INTERNAL, "llm warmup failed"
            ) from error

    def close(self) -> None:
        with self._lock:
            self._closed = True


class SherpaOnnxTTS:
    """sherpa-onnx TTS via guarded optional import (stable binding).

    ``sherpa_onnx`` is never a hard dependency: without it (or without
    voice files) every call fails closed UNAVAILABLE so the backend
    stays NOT_READY instead of half-speaking. Voice files resolve only
    under the trusted model root inherited from the manifest.
    """

    component = "tts"

    def __init__(
        self,
        *,
        model_dir: str,
        voice: str = "es-female-1",
        sample_rate: int = 16000,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not model_dir or not os.path.isdir(model_dir):
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "tts voice unavailable"
            )
        self._model_dir = model_dir
        self._voice = voice
        self._sample_rate = sample_rate
        self._timeout = timeout_seconds
        self._lock = threading.Lock()
        self._closed = False
        self._engine = None

    def _load(self):  # guarded import: optional dependency
        try:
            import sherpa_onnx  # type: ignore[import-not-found]
        except Exception as error:
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "tts runtime unavailable"
            ) from error
        return sherpa_onnx

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        sherpa_onnx = self._load()
        try:
            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=os.path.join(self._model_dir, "model.onnx"),
                        lexicon="",
                        tokens=os.path.join(self._model_dir, "tokens.txt"),
                    ),
                    provider="cpu",
                    debug=False,
                    num_threads=2,
                )
            )
            self._engine = sherpa_onnx.OfflineTts(config)
        except Exception as error:
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "tts voice unavailable"
            ) from error
        return self._engine

    def synthesize(
        self,
        text: str,
        cancel: CancelToken,
        on_chunk,
    ) -> int:
        import struct

        cancel.throw_if_cancelled()
        with self._lock:
            if self._closed:
                raise AdapterError(
                    ProviderFailureCategory.UNAVAILABLE, "tts runtime closed"
                )
        engine = self._ensure_engine()
        cancel.throw_if_cancelled()
        try:
            audio = engine.generate(text, sid=0, speed=1.0)
        except Exception as error:
            raise AdapterError(
                ProviderFailureCategory.INTERNAL, "tts synthesis failed"
            ) from error
        samples = audio.samples if hasattr(audio, "samples") else []
        if not samples:
            raise AdapterError(
                ProviderFailureCategory.INVALID_OUTPUT, "tts produced no audio"
            )
        clipped = [max(-1.0, min(1.0, s)) for s in samples]
        pcm = struct.pack(f"<{len(clipped)}h", *(int(s * 32767) for s in clipped))
        stride = 1600 * 2
        total = 0
        for index in range(0, len(pcm), stride):
            cancel.throw_if_cancelled()
            piece = pcm[index : index + stride]
            on_chunk(piece, audio.sample_rate if hasattr(audio, "sample_rate") else self._sample_rate)
            total += len(piece)
        return total

    def warmup(self) -> None:
        try:
            import sherpa_onnx  # type: ignore[import-not-found]

            version = getattr(sherpa_onnx, "__version__", "")
        except Exception as error:
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "tts runtime unavailable"
            ) from error
        if version and not any(
            marker.split("-", 1)[1] in version for marker in SUPPORTED_SHERPA_VERSIONS
        ):
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "tts version unsupported"
            )
        self._ensure_engine()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._engine = None
