"""Local runtime adapters: whisper.cpp, llama.cpp, sherpa-onnx (#24).

Persistence model (MAJOR-2): STT and LLM run as **long-lived local
server processes** (`whisper-server`, `llama-server`), spawned once at
`start()` with the model resident; every turn is an HTTP request
against 127.0.0.1. Warmup proves the resident model actually serves
(silence inference / tiny completion), never mere file existence.
TTS runs its engine in-process (resident after first load) with
sentence-chunked synthesis so first audio and cancellation boundaries
are incremental.

API sources (verified against upstream docs/code, not invented):
- whisper-server: `POST /inference` multipart
  (`file`, `temperature`, `response_format=json`) → `{"text": ...}`;
  flags `-m/--model --host --port -l/--language --no-gpu`.
- llama-server: `POST /completion`
  (`prompt`, `n_predict`, `temperature`, `cache_prompt`) →
  `{"content": ..., "truncated": ...}`; `GET /health` → 200
  `{"status": "ok"}` (503 while loading); flags `--ctx-size`,
  `--reasoning off`, `--offline`, `--no-webui`, `-ngl 0` (CPU-only),
  `-np 1`, `-t`.
- sherpa-onnx Python: `OfflineTts.generate(text, GenerationConfig)`
  is monolithic per call (streaming callbacks exist only for some
  model types upstream), so incrementality here is per-sentence
  generate calls with cancel checks between them.

Security: `shell=False`, argv explicit, controlled cwd, timeouts,
capped request/response bodies, terminate/kill on close, loopback
only. Raw vendor stderr never crosses the seam: failures normalize to
AdapterError with stable ids; bounded stderr goes to DEBUG logs only.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave

from receptionist.cascaded import (
    AdapterError,
    CancelledError,
    CancelToken,
    STTResult,
)
from receptionist.boundaries import ProviderFailureCategory

LOG = logging.getLogger("receptionist.runtimes")

#: Hard cap on captured subprocess/HTTP bodies: diagnostics stay bounded.
MAX_SUBPROCESS_OUTPUT = 256 * 1024
MAX_HTTP_BODY = 4 * 1024 * 1024

#: Loopback hosts a runtime server may bind. Never 0.0.0.0, never remote.
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

_WHISPER_VERSION_RE = re.compile(
    r"whisper\.cpp version:\s*(\d+)\.(\d+)\.(\d+)"
)
_LLAMA_VERSION_RE = re.compile(r"version:\s*(\d+)\.(\d+)\.(\d+)")


def parse_whisper_version(output: str) -> tuple[int, int, int] | None:
    """Parse upstream `whisper --version` text, e.g.
    ``whisper.cpp version: 1.7.4``. Returns None when unparseable."""
    match = _WHISPER_VERSION_RE.search(output or "")
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def parse_llama_version(output: str) -> tuple[int, int, int] | None:
    """Parse upstream `--version` text, e.g.
    ``version: 0.6.0 (build 5828, commit e2f6b73e)`` (stderr).
    Returns None when unparseable."""
    match = _LLAMA_VERSION_RE.search(output or "")
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def format_version(version: tuple[int, int, int] | None) -> str:
    return ".".join(str(part) for part in version) if version else "unknown"


def sherpa_version() -> str:
    """Best-effort sherpa-onnx version: package metadata, then module
    attribute, else ``unknown`` (recorded as such; engine load is the
    real proof). Upstream defines no `__version__` guarantee."""
    try:
        from importlib.metadata import version as package_version

        return package_version("sherpa-onnx")
    except Exception:
        pass
    try:
        import sherpa_onnx  # type: ignore[import-not-found]

        return str(getattr(sherpa_onnx, "__version__", "") or "unknown")
    except Exception:
        return "unknown"


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


def _check_loopback(host: str, *, name: str) -> None:
    if host not in LOOPBACK_HOSTS:
        raise AdapterError(
            ProviderFailureCategory.UNAVAILABLE,
            f"{name} server must stay on loopback",
        )


def query_version(executable: str, *, name: str, timeout_seconds: float = 10.0) -> str:
    """Run `executable --version`, return the raw combined output
    (capped). Raises AdapterError when the binary cannot answer."""
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
    output = (completed.stdout or b"") + b"\n" + (completed.stderr or b"")
    return output[:4096].decode("utf-8", "replace").strip()


def _run_trusted(
    executable: str,
    args: list[str],
    *,
    name: str,
    timeout_seconds: float,
    cancel: CancelToken,
    input_bytes: bytes | None = None,
) -> bytes:
    """Run one trusted one-shot subprocess (version probes and similar).

    `shell=False`, argv explicit, stdout/stderr spooled to temp files
    (bounded read afterwards), terminate/kill on cancel or timeout.
    Returns stdout (capped). Diagnostics stay in DEBUG logs; failure
    details stay stable ids.
    """
    try:
        stdout_spool = tempfile.TemporaryFile()
        stderr_spool = tempfile.TemporaryFile()
    except OSError as error:
        raise AdapterError(
            ProviderFailureCategory.UNAVAILABLE, f"{name} cannot spool output"
        ) from error
    process: subprocess.Popen[bytes] | None = None
    try:
        try:
            process = subprocess.Popen(
                [executable, *args],
                shell=False,
                cwd=tempfile.gettempdir(),
                stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
                stdout=stdout_spool,
                stderr=stderr_spool,
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
            handle.communicate(input=input_bytes, timeout=timeout_seconds)
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
            try:
                stderr_spool.seek(0)
                tail = stderr_spool.read(2048).decode("utf-8", "replace")
            except Exception:
                tail = ""
            LOG.debug("%s failed rc=%s stderr_tail=%r", name, handle.returncode, tail[-500:])
            raise AdapterError(
                ProviderFailureCategory.INTERNAL, f"{name} exited unsuccessfully"
            )
        stdout_spool.seek(0)
        return stdout_spool.read(MAX_SUBPROCESS_OUTPUT + 1)[:MAX_SUBPROCESS_OUTPUT]
    finally:
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except Exception:
                pass
        try:
            stdout_spool.close()
        except Exception:
            pass
        try:
            stderr_spool.close()
        except Exception:
            pass


def pick_free_port(host: str = "127.0.0.1") -> int:
    """One ephemeral loopback port for a runtime server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_for_port(
    host: str, port: int, *, deadline_seconds: float = 30.0
) -> None:
    """Block (bounded) until the runtime server accepts TCP. Raises
    AdapterError TIMEOUT past the deadline. Only used at startup, never
    per turn."""
    if host not in ("127.0.0.1", "localhost"):
        raise AdapterError(
            ProviderFailureCategory.UNAVAILABLE, "runtime server must stay local"
        )
    end = time.monotonic() + deadline_seconds
    last: Exception | None = None
    while time.monotonic() < end:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError as error:
            last = error
            time.sleep(0.05)
    raise AdapterError(
        ProviderFailureCategory.TIMEOUT, "runtime server did not listen"
    ) from last


def http_get_json(
    url: str, *, timeout_seconds: float, cancel: CancelToken
) -> tuple[int, dict]:
    """GET JSON from a loopback runtime server. Capped body, parsed
    defensively (non-JSON or non-object fails closed)."""
    _check_url_loopback(url)
    cancel.throw_if_cancelled()
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            raw = response.read(MAX_HTTP_BODY + 1)
    except urllib.error.HTTPError as error:
        try:
            raw = error.read(MAX_HTTP_BODY + 1)
        except Exception:
            raw = b""
        return int(error.code), _parse_json_object(raw)
    except (OSError, ValueError) as error:
        raise AdapterError(
            ProviderFailureCategory.UNAVAILABLE, "runtime server unreachable"
        ) from error
    if cancel.cancelled:
        raise CancelledError()
    return status, _parse_json_object(raw)


def http_post_json(
    url: str,
    payload: dict,
    *,
    timeout_seconds: float,
    cancel: CancelToken,
) -> tuple[int, dict]:
    """POST JSON to a loopback runtime server. Capped bodies both ways."""
    _check_url_loopback(url)
    cancel.throw_if_cancelled()
    try:
        body = json.dumps(payload).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AdapterError(
            ProviderFailureCategory.INTERNAL, "runtime request unencodable"
        ) from error
    if len(body) > MAX_HTTP_BODY:
        raise AdapterError(
            ProviderFailureCategory.INVALID_OUTPUT, "runtime request too large"
        )
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            raw = response.read(MAX_HTTP_BODY + 1)
    except urllib.error.HTTPError as error:
        try:
            raw = error.read(MAX_HTTP_BODY + 1)
        except Exception:
            raw = b""
        return int(error.code), _parse_json_object(raw)
    except (OSError, ValueError) as error:
        raise AdapterError(
            ProviderFailureCategory.UNAVAILABLE, "runtime server unreachable"
        ) from error
    if cancel.cancelled:
        raise CancelledError()
    return status, _parse_json_object(raw)


def http_post_multipart(
    url: str,
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    *,
    timeout_seconds: float,
    cancel: CancelToken,
) -> tuple[int, dict]:
    """POST multipart/form-data (WAV upload) to a loopback server."""
    _check_url_loopback(url)
    cancel.throw_if_cancelled()
    if len(file_bytes) > MAX_HTTP_BODY:
        raise AdapterError(
            ProviderFailureCategory.INVALID_OUTPUT, "runtime request too large"
        )
    boundary = uuid.uuid4().hex
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"'
            f"\r\n\r\n{value}\r\n".encode("utf-8")
        )
    ctype = mimetypes.guess_type(filename)[0] or "audio/wav"
    chunks.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
        f'filename="{filename}"\r\nContent-Type: {ctype}\r\n\r\n'.encode("utf-8")
        + file_bytes
        + b"\r\n"
    )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(chunks)
    if len(body) > MAX_HTTP_BODY + 1024:
        raise AdapterError(
            ProviderFailureCategory.INVALID_OUTPUT, "runtime request too large"
        )
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            raw = response.read(MAX_HTTP_BODY + 1)
    except urllib.error.HTTPError as error:
        try:
            raw = error.read(MAX_HTTP_BODY + 1)
        except Exception:
            raw = b""
        return int(error.code), _parse_json_object(raw)
    except (OSError, ValueError) as error:
        raise AdapterError(
            ProviderFailureCategory.UNAVAILABLE, "runtime server unreachable"
        ) from error
    if cancel.cancelled:
        raise CancelledError()
    return status, _parse_json_object(raw)


def _check_url_loopback(url: str) -> None:
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    if host not in LOOPBACK_HOSTS:
        raise AdapterError(
            ProviderFailureCategory.UNAVAILABLE, "runtime server must stay local"
        )


def _parse_json_object(raw: bytes) -> dict:
    if not raw or len(raw) > MAX_HTTP_BODY + 1:
        return {}
    try:
        document = json.loads(raw[: MAX_HTTP_BODY + 1].decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return {}
    return document if isinstance(document, dict) else {}


def _write_wav_16k_mono(pcm: bytes, path: str) -> None:
    """Transient WAV for the STT server upload. Deleted (with any
    runtime-created siblings) immediately after the request."""
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm)


def _cleanup_tracked_wav(tmp_path: str) -> None:
    """Delete the transient WAV plus any sibling the runtime derived
    from it (defense for file-output modes we never request)."""
    import glob

    candidates = [tmp_path] + sorted(glob.glob(tmp_path + ".*"))
    for path in candidates:
        try:
            if os.path.isfile(path):
                os.unlink(path)
        except OSError:
            pass


class _ServerProcess:
    """One long-lived loopback runtime server. Spawned once, resident
    until `stop()` (terminate, then kill past a bounded grace)."""

    def __init__(self, *, name: str) -> None:
        self._name = name
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def spawn(self, argv: list[str]) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            try:
                self._process = subprocess.Popen(
                    argv,
                    shell=False,
                    cwd=tempfile.gettempdir(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as error:
                raise AdapterError(
                    ProviderFailureCategory.UNAVAILABLE,
                    f"{self._name} failed to start",
                ) from error

    def stop(self) -> None:
        with self._lock:
            process, self._process = self._process, None
        if process is None:
            return
        try:
            process.terminate()
        except Exception:
            pass
        try:
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


class WhisperServerSTT:
    """whisper.cpp STT via a resident `whisper-server` process.

    Spawned once (`-m model --host 127.0.0.1 --port P -l es --no-gpu`);
    every turn POSTs the transient WAV to `/inference`
    (`response_format=json` → `{"text": ...}`). No `--output-txt`
    anywhere: transcription travels in the HTTP body only, and the
    transient WAV (plus any runtime-derived sibling) is deleted right
    after the request — no transcript ever touches disk as a side effect.
    """

    component = "stt"

    def __init__(
        self,
        *,
        executable: str,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 0,
        language: str = "es",
        timeout_seconds: float = 30.0,
    ) -> None:
        if not model_path or not os.path.isfile(model_path):
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "stt model unavailable"
            )
        _check_loopback(host, name="stt")
        self._executable = _trusted_executable(executable, name="stt")
        self._model_path = model_path
        self._host = host
        self._port = port or pick_free_port(host)
        self._language = language
        self._timeout = timeout_seconds
        self._server = _ServerProcess(name="stt")
        self._version = "unknown"
        self._lock = threading.Lock()
        self._closed = False

    def version_info(self) -> str:
        return self._version

    def start(self) -> None:
        """Spawn the resident server (idempotent). The model loads here,
        once — never per turn."""
        with self._lock:
            if self._closed:
                raise AdapterError(
                    ProviderFailureCategory.UNAVAILABLE, "stt runtime closed"
                )
        self._server.spawn(
            [
                self._executable,
                "-m", self._model_path,
                "--host", self._host,
                "--port", str(self._port),
                "-l", self._language,
                "--no-gpu",
            ]
        )
        wait_for_port(self._host, self._port, deadline_seconds=120.0)

    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

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
        if not self._server.running:
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "stt server not running"
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
            with open(tmp_path, "rb") as handle:
                wav_bytes = handle.read()
            status, document = http_post_multipart(
                self.base_url() + "/inference",
                {"temperature": "0.0", "response_format": "json"},
                "file",
                "turn.wav",
                wav_bytes,
                timeout_seconds=self._timeout,
                cancel=cancel,
            )
        finally:
            if tmp_path:
                _cleanup_tracked_wav(tmp_path)
        if status != 200:
            raise AdapterError(
                ProviderFailureCategory.INTERNAL, "stt inference failed"
            )
        text = document.get("text", "")
        if not isinstance(text, str):
            raise AdapterError(
                ProviderFailureCategory.INVALID_OUTPUT, "stt response malformed"
            )
        return STTResult(text=text.strip())

    def warmup(self) -> None:
        raw = query_version(self._executable, name="stt")
        parsed = parse_whisper_version(raw)
        if parsed is None:
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "stt version unsupported"
            )
        self._version = format_version(parsed)
        self.start()
        # Silence inference through the resident model: proves it serves.
        self.transcribe(bytes(16000 * 2), 16000, CancelToken())

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._server.stop()


class LlamaServerLLM:
    """llama.cpp generation via a resident `llama-server` process.

    Spawned once (`-m model --ctx-size 4096 --reasoning off --offline
    --no-webui -ngl 0 -np 1`); every turn POSTs `/completion`
    (`prompt`, bounded `n_predict`, `cache_prompt:false`). Non-thinking
    is enforced server-side by `--reasoning off`, so no prompt marker
    and no discarded reasoning. CPU-only via `-ngl 0`; `--offline`
    forbids network access from the runtime itself.
    """

    component = "llm"

    def __init__(
        self,
        *,
        executable: str,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 0,
        ctx_size: int = 4096,
        max_tokens: int = 220,
        threads: int = 4,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not model_path or not os.path.isfile(model_path):
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "llm model unavailable"
            )
        if max_tokens <= 0 or ctx_size <= 0:
            raise ValueError("llm bounds must be > 0")
        _check_loopback(host, name="llm")
        self._executable = _trusted_executable(executable, name="llm")
        self._model_path = model_path
        self._host = host
        self._port = port or pick_free_port(host)
        self._ctx_size = ctx_size
        self._max_tokens = max_tokens
        self._threads = max(1, threads)
        self._timeout = timeout_seconds
        self._server = _ServerProcess(name="llm")
        self._version = "unknown"
        self._lock = threading.Lock()
        self._closed = False

    def version_info(self) -> str:
        return self._version

    def start(self) -> None:
        """Spawn the resident server (idempotent). The GGUF loads here,
        once — never per turn."""
        with self._lock:
            if self._closed:
                raise AdapterError(
                    ProviderFailureCategory.UNAVAILABLE, "llm runtime closed"
                )
        self._server.spawn(
            [
                self._executable,
                "-m", self._model_path,
                "--host", self._host,
                "--port", str(self._port),
                "--ctx-size", str(self._ctx_size),
                "-t", str(self._threads),
                "-np", "1",
                "-ngl", "0",
                "--reasoning", "off",
                "--offline",
                "--no-webui",
            ]
        )
        self._wait_healthy(deadline_seconds=300.0)

    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def _wait_healthy(self, *, deadline_seconds: float) -> None:
        end = time.monotonic() + deadline_seconds
        while time.monotonic() < end:
            try:
                status, document = http_get_json(
                    self.base_url() + "/health",
                    timeout_seconds=5.0,
                    cancel=CancelToken(),
                )
            except AdapterError:
                time.sleep(0.1)
                continue
            if status == 200 and document.get("status") == "ok":
                return
            time.sleep(0.2)
        raise AdapterError(
            ProviderFailureCategory.TIMEOUT, "llm server did not warm"
        )

    def generate(self, prompt: str, cancel: CancelToken) -> str:
        cancel.throw_if_cancelled()
        with self._lock:
            if self._closed:
                raise AdapterError(
                    ProviderFailureCategory.UNAVAILABLE, "llm runtime closed"
                )
        if not self._server.running:
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "llm server not running"
            )
        status, document = http_post_json(
            self.base_url() + "/completion",
            {
                "prompt": prompt,
                "n_predict": self._max_tokens,
                "temperature": 0.4,
                "cache_prompt": False,
                "stream": False,
            },
            timeout_seconds=self._timeout,
            cancel=cancel,
        )
        if status != 200:
            raise AdapterError(
                ProviderFailureCategory.INTERNAL, "llm completion failed"
            )
        content = document.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise AdapterError(
                ProviderFailureCategory.INVALID_OUTPUT, "empty model output"
            )
        if document.get("truncated") is True:
            raise AdapterError(
                ProviderFailureCategory.INVALID_OUTPUT, "model output truncated"
            )
        return content.strip()

    def warmup(self) -> None:
        raw = query_version(self._executable, name="llm")
        parsed = parse_llama_version(raw)
        if parsed is None:
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "llm version unsupported"
            )
        self._version = format_version(parsed)
        self.start()
        # Tiny completion through the resident model: proves it serves.
        self.generate('{"spoken_text": "Hola.", "action": null}', CancelToken())

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._server.stop()


def build_cascaded_backend(profile, knowledge_lookup=None, clock=None):
    """Compose the production cascaded backend from a canonical voice
    profile (config.db via ConfigService). Every operator-configurable
    value — model root, executables, voice + speaker, timeouts — flows
    into the adapters here; nothing in `voice.*` is dead surface.

    Raises AdapterError fail-closed when the model root or a required
    voice directory is missing.
    """
    import os as _os

    from receptionist.cascaded import CascadedVoiceBackend

    root = (profile.model_root or "").strip()
    if not root or not _os.path.isdir(root):
        raise AdapterError(
            ProviderFailureCategory.UNAVAILABLE, "model root unavailable"
        )
    by_file = {
        "stt": _os.path.join(root, "stt", "ggml-model-base.bin"),
        "llm": _os.path.join(root, "llm", "qwen3-1.7b-q4_k_m.gguf"),
    }
    voice_dir = _os.path.join(root, "tts", profile.tts_voice or "es-female-1")
    stt = WhisperServerSTT(
        executable=profile.stt_executable,
        model_path=by_file["stt"],
        host=profile.server_host,
        timeout_seconds=profile.stt_timeout_seconds,
    )
    llm = LlamaServerLLM(
        executable=profile.llm_executable,
        model_path=by_file["llm"],
        host=profile.server_host,
        timeout_seconds=profile.llm_timeout_seconds,
    )
    tts = SherpaOnnxTTS(
        model_dir=voice_dir,
        voice=profile.tts_voice,
        speaker_id=profile.tts_speaker_id,
        timeout_seconds=profile.tts_timeout_seconds,
    )
    return CascadedVoiceBackend(
        profile=profile,
        stt=stt,
        llm=llm,
        tts=tts,
        knowledge_lookup=knowledge_lookup,
        clock=clock,
    )


def split_sentences(text: str, *, max_chars: int = 400) -> list[str]:
    """Split utterance into synthesizable sentences (bounded each).

    Deterministic, punctuation-driven; never splits mid-word below the
    cap. The TTS adapter synthesizes one sentence per engine call so
    first audio and cancel boundaries stay incremental.
    """
    parts = re.split(r"(?<=[.!?…;:\n])\s+", text.strip())
    sentences: list[str] = []
    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        while len(chunk) > max_chars:
            cut = chunk.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            sentences.append(chunk[:cut].strip())
            chunk = chunk[cut:].strip()
        if chunk:
            sentences.append(chunk)
    return sentences


def _call_with_deadline(
    fn, *, budget_seconds: float, description: str
):
    """Run one blocking engine call with an effective timeout.

    The call runs on a daemon worker; past the budget the turn fails
    TIMEOUT while the orphaned worker is abandoned (never awaited).
    Daemon-only so a stuck engine can never wedge shutdown.
    """
    result: list = []
    failure: list[BaseException] = []

    def _target() -> None:
        try:
            result.append(fn())
        except BaseException as error:  # capture, never leak threads
            failure.append(error)

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=max(budget_seconds, 0.01))
    if worker.is_alive():
        raise AdapterError(ProviderFailureCategory.TIMEOUT, f"{description} timed out")
    if failure:
        error = failure[0]
        if isinstance(error, AdapterError):
            raise error
        raise AdapterError(
            ProviderFailureCategory.INTERNAL, f"{description} failed"
        ) from error
    return result[0] if result else None


class SherpaOnnxTTS:
    """sherpa-onnx TTS via guarded optional import (stable binding).

    The engine loads once and stays resident. Synthesis is
    sentence-chunked: first audio emits after the first sentence (never
    after the whole utterance), cancel is observed between sentences,
    and the profile timeout bounds the whole call — the parts of M3 the
    upstream Python API actually supports (streaming callbacks exist
    only for some model types, so they are not claimed here).

    `sherpa_onnx` is never a hard dependency: without it (or without
    the voice files) every call fails closed UNAVAILABLE. Voice files
    resolve only under `<model_root>/tts/<voice>/` as listed in the
    manifest (`model.onnx`, `tokens.txt`, optional lexicon/espeak data):
    `voice` selects the directory, `speaker_id` the voice inside it.
    """

    component = "tts"

    def __init__(
        self,
        *,
        model_dir: str,
        voice: str = "es-female-1",
        speaker_id: int = 0,
        sample_rate: int = 16000,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not model_dir or not os.path.isdir(model_dir):
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "tts voice unavailable"
            )
        if speaker_id < 0:
            raise ValueError("speaker_id must be >= 0")
        self._model_dir = model_dir
        self._voice = voice
        self._speaker_id = speaker_id
        self._sample_rate = sample_rate
        self._timeout = timeout_seconds
        self._lock = threading.Lock()
        self._closed = False
        self._engine = None
        self._version = "unknown"

    def version_info(self) -> str:
        return self._version

    def _load(self):  # guarded import: optional dependency
        try:
            import sherpa_onnx  # type: ignore[import-not-found]
        except Exception as error:
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "tts runtime unavailable"
            ) from error
        return sherpa_onnx

    def _voice_file(self, name: str) -> str:
        path = os.path.join(self._model_dir, name)
        if not os.path.isfile(path):
            raise AdapterError(
                ProviderFailureCategory.UNAVAILABLE, "tts voice unavailable"
            )
        return path

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        sherpa_onnx = self._load()
        try:
            vits = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=self._voice_file("model.onnx"),
                lexicon="",
                tokens=self._voice_file("tokens.txt"),
            )
            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=vits,
                    provider="cpu",
                    debug=False,
                    num_threads=2,
                )
            )
            self._engine = sherpa_onnx.OfflineTts(config)
        except AdapterError:
            raise
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
        sentences = split_sentences(text)
        if not sentences:
            raise AdapterError(
                ProviderFailureCategory.INVALID_OUTPUT, "tts produced no audio"
            )
        deadline = time.monotonic() + self._timeout
        total = 0
        for sentence in sentences:
            cancel.throw_if_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AdapterError(
                    ProviderFailureCategory.TIMEOUT, "tts synthesis timed out"
                )
            try:
                audio = _call_with_deadline(
                    lambda: engine.generate(sentence, sid=self._speaker_id, speed=1.0),
                    budget_seconds=remaining,
                    description="tts synthesis",
                )
            except AdapterError:
                raise
            except Exception as error:
                raise AdapterError(
                    ProviderFailureCategory.INTERNAL, "tts synthesis failed"
                ) from error
            samples = getattr(audio, "samples", []) if audio is not None else []
            if not samples:
                raise AdapterError(
                    ProviderFailureCategory.INVALID_OUTPUT, "tts produced no audio"
                )
            rate = getattr(audio, "sample_rate", self._sample_rate)
            clipped = [max(-1.0, min(1.0, s)) for s in samples]
            pcm = struct.pack(
                f"<{len(clipped)}h", *(int(s * 32767) for s in clipped)
            )
            stride = 1600 * 2
            for index in range(0, len(pcm), stride):
                cancel.throw_if_cancelled()
                piece = pcm[index : index + stride]
                on_chunk(piece, int(rate) if rate else self._sample_rate)
                total += len(piece)
        return total

    def warmup(self) -> None:
        self._version = sherpa_version()
        if self._version == "unknown":
            try:
                self._load()
            except AdapterError:
                raise
        self._ensure_engine()
        # Tiny synthesis through the resident voice: proves it speaks.
        generated: list[bool] = []

        def _discard(pcm: bytes, rate: int) -> None:
            generated.append(True)

        self.synthesize("Hola.", CancelToken(), _discard)
        if not generated:
            raise AdapterError(
                ProviderFailureCategory.INVALID_OUTPUT, "tts produced no audio"
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._engine = None
