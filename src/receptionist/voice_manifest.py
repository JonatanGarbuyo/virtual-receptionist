"""Reproducible model-pack manifest for the local cascaded backend (#24).

The manifest identifies every model artifact by logical id plus
integrity metadata (SHA-256, size, architecture/quantization,
language/voice, source/provenance, license). It never contains model
bytes, never triggers downloads, and never accepts mutable tags like
``latest.gguf`` as identity. Paths resolve only under a trusted model
root: absolute paths and ``..`` traversal fail closed, and the
filename is never executed as a shell command anywhere.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

#: Manifest schema versions this code understands.
MANIFEST_SCHEMA_VERSION = 1

#: Required components of the approved CPU baseline profile.
BASELINE_PROFILE_ID = "cascaded-cpu-baseline-v1"
REQUIRED_BASELINE_COMPONENTS = ("stt", "llm", "tts")


@dataclass(frozen=True)
class ModelComponent:
    """One model artifact in a pack. ``filename`` is relative to the
    trusted model root; ``required`` marks startup-blocking artifacts.

    ``files`` lists every runtime file actually executed for this
    component (model + sidecars such as tokens/lexicon), each resolved
    and hashed like ``filename``. Defaults to ``(filename,)`` for
    single-file components; multi-file components (TTS voices) must
    list all of them — an unlisted sidecar is an unverified sidecar.
    """

    component: str
    runtime: str
    runtime_version: str
    model_id: str
    filename: str
    sha256: str
    size: int = 0
    arch_quant: str = ""
    language: str = ""
    voice: str = ""
    source: str = ""
    license: str = ""
    required: bool = True
    files: tuple[str, ...] = ()
    file_hashes: tuple[str, ...] = ()

    def resolved_files(self) -> list[tuple[str, str]]:
        """(filename, expected sha256) for every verified file."""
        names = self.files or (self.filename,)
        if self.file_hashes:
            if len(self.file_hashes) != len(names):
                raise ValueError(
                    f"manifest component {self.component}: files/file_hashes mismatch"
                )
            return list(zip(names, [h.lower() for h in self.file_hashes]))
        if len(names) == 1:
            return [(names[0], self.sha256)]
        raise ValueError(
            f"manifest component {self.component}: multi-file entry needs file_hashes"
        )


@dataclass(frozen=True)
class ModelManifest:
    """Project-owned manifest: schema version, profile id, components."""

    schema_version: int
    profile_id: str
    components: tuple[ModelComponent, ...] = field(default_factory=tuple)


def _component_from_dict(raw: dict) -> ModelComponent:
    try:
        component = str(raw["component"])
        runtime = str(raw["runtime"])
        runtime_version = str(raw.get("runtime_version", ""))
        model_id = str(raw["model_id"])
        filename = str(raw["filename"])
        sha256 = str(raw["sha256"]).lower()
    except KeyError as error:
        raise ValueError(f"manifest component missing key: {error}") from error
    size = raw.get("size", 0)
    if not isinstance(size, int) or size < 0:
        raise ValueError(f"manifest component has invalid size: {size!r}")
    digest_len = len(sha256)
    if digest_len != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ValueError(f"manifest component has invalid sha256: {sha256!r}")
    _check_relative_filename(filename)
    raw_files = raw.get("files", [filename])
    if not isinstance(raw_files, list) or not raw_files or not all(
        isinstance(item, str) for item in raw_files
    ):
        raise ValueError("manifest component files must be a non-empty string list")
    for name in raw_files:
        _check_relative_filename(name)
    raw_hashes = raw.get("file_hashes", [])
    if raw_hashes and (
        not isinstance(raw_hashes, list)
        or len(raw_hashes) != len(raw_files)
        or not all(isinstance(item, str) for item in raw_hashes)
    ):
        raise ValueError("manifest component file_hashes must match files")
    for digest in raw_hashes:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError(f"manifest component has invalid sha256: {digest!r}")
    return ModelComponent(
        component=component,
        runtime=runtime,
        runtime_version=runtime_version,
        model_id=model_id,
        filename=filename,
        sha256=sha256,
        size=size,
        arch_quant=str(raw.get("arch_quant", "")),
        language=str(raw.get("language", "")),
        voice=str(raw.get("voice", "")),
        source=str(raw.get("source", "")),
        license=str(raw.get("license", "")),
        required=bool(raw.get("required", True)),
        files=tuple(raw_files),
        file_hashes=tuple(h.lower() for h in raw_hashes),
    )


def _check_relative_filename(filename: str) -> None:
    """Reject absolute paths, traversal, and empty names fail-closed."""
    if not filename or not filename.strip():
        raise ValueError("manifest filename must be non-empty")
    candidate = PurePosixPath(filename)
    if candidate.is_absolute():
        raise ValueError(f"manifest filename must be relative: {filename!r}")
    if ".." in candidate.parts:
        raise ValueError(f"manifest filename must not traverse: {filename!r}")
    if filename != str(candidate):
        raise ValueError(f"manifest filename is not normalized: {filename!r}")


def load_manifest(raw: dict) -> ModelManifest:
    """Parse an already-loaded manifest document. Raises ValueError."""
    try:
        schema_version = int(raw["schema_version"])
        profile_id = str(raw["profile_id"])
        raw_components = raw["components"]
    except KeyError as error:
        raise ValueError(f"manifest missing key: {error}") from error
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {schema_version!r}")
    if not profile_id.strip():
        raise ValueError("manifest profile_id must be non-empty")
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("manifest needs a non-empty components list")
    components = tuple(
        _component_from_dict(item) for item in raw_components
    )
    seen = [c.component for c in components]
    if len(set(seen)) != len(seen):
        raise ValueError(f"manifest has duplicate components: {seen!r}")
    return ModelManifest(
        schema_version=schema_version,
        profile_id=profile_id,
        components=components,
    )


def load_manifest_file(path: str) -> ModelManifest:
    """Read and parse a manifest JSON file. Raises ValueError/OSError."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("manifest document must be a JSON object")
    return load_manifest(raw)


def manifest_to_dict(manifest: ModelManifest) -> dict:
    """Serialize a manifest back to a plain JSON-compatible document."""
    return {
        "schema_version": manifest.schema_version,
        "profile_id": manifest.profile_id,
        "components": [
            {
                "component": c.component,
                "runtime": c.runtime,
                "runtime_version": c.runtime_version,
                "model_id": c.model_id,
                "filename": c.filename,
                "sha256": c.sha256,
                "size": c.size,
                "arch_quant": c.arch_quant,
                "language": c.language,
                "voice": c.voice,
                "source": c.source,
                "license": c.license,
                "required": c.required,
                "files": list(c.files or (c.filename,)),
                "file_hashes": list(c.file_hashes),
            }
            for c in manifest.components
        ],
    }


def resolve_trusted_path(model_root: str, filename: str) -> str:
    """Join a manifest filename under the trusted model root.

    The manifest/profile is operator configuration; caller text, LLM
    output, knowledge content, and transcripts can never reach this
    path: only values validated by `_check_relative_filename` resolve.
    """
    _check_relative_filename(filename)
    import os

    joined = os.path.normpath(os.path.join(model_root, filename))
    root = os.path.normpath(model_root)
    if joined != root and not joined.startswith(root + os.sep):
        raise ValueError(f"model path escapes trusted root: {filename!r}")
    return joined


def sha256_file(path: str) -> str:
    """Hex SHA-256 of a file, streamed (no whole-file buffering)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_runtime_version(raw: str) -> tuple[int, int, int] | None:
    """Strict X.Y.Z runtime version. Anything else (empty, `unknown`,
    build tags) is unparseable and never compatible."""
    if not isinstance(raw, str):
        return None
    match = _VERSION_RE.match(raw.strip())
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def versions_compatible(actual: str, pinned: str) -> bool:
    """Compat gate: same MAJOR.MINOR series with actual patch >= pinned
    patch. Anything unparseable or empty on either side is incompatible:
    a productive manifest must pin an exact version and the installed
    runtime must identifiably belong to its series."""
    actual_v = parse_runtime_version(actual)
    pinned_v = parse_runtime_version(pinned)
    if actual_v is None or pinned_v is None:
        return False
    return (
        actual_v[0] == pinned_v[0]
        and actual_v[1] == pinned_v[1]
        and actual_v[2] >= pinned_v[2]
    )


@dataclass(frozen=True)
class IntegrityProblem:
    """One failed integrity check. Sanitized identity only: component
    plus a stable reason; never raw stderr, bytes, or file content."""

    component: str
    reason: str


def verify_manifest(model_root: str, manifest: ModelManifest) -> list[IntegrityProblem]:
    """Check every required component file: exists under the trusted
    root, non-empty, size matches when declared (primary file), SHA-256
    matches. Optional components that are fully absent are skipped;
    present-but-corrupt optional files are still reported. Never raises
    for content problems (returns them); never downloads anything."""
    import os

    problems: list[IntegrityProblem] = []
    for component in manifest.components:
        try:
            entries = component.resolved_files()
        except ValueError:
            problems.append(IntegrityProblem(component.component, "invalid_entry"))
            continue
        for index, (filename, expected) in enumerate(entries):
            try:
                path = resolve_trusted_path(model_root, filename)
            except ValueError:
                problems.append(IntegrityProblem(component.component, "unsafe_path"))
                continue
            if not os.path.isfile(path):
                if component.required:
                    problems.append(
                        IntegrityProblem(component.component, "missing_file")
                    )
                continue
            try:
                actual_size = os.path.getsize(path)
            except OSError:
                problems.append(IntegrityProblem(component.component, "unreadable_file"))
                continue
            if actual_size == 0:
                problems.append(IntegrityProblem(component.component, "empty_file"))
                continue
            if index == 0 and component.size and actual_size != component.size:
                problems.append(IntegrityProblem(component.component, "size_mismatch"))
                continue
            try:
                actual_digest = sha256_file(path)
            except OSError:
                problems.append(IntegrityProblem(component.component, "unreadable_file"))
                continue
            if actual_digest != expected:
                problems.append(IntegrityProblem(component.component, "hash_mismatch"))
    return problems


def baseline_manifest(
    *,
    whisper_sha256: str,
    llama_sha256: str,
    tts_model_sha256: str,
    tts_tokens_sha256: str,
    whisper_size: int = 0,
    llama_size: int = 0,
    tts_voice: str = "es-female-1",
    whisper_runtime_version: str = "",
    llama_runtime_version: str = "",
    tts_runtime_version: str = "",
) -> ModelManifest:
    """Build the approved CPU baseline profile manifest.

    The caller supplies the artifact checksums measured at install
    time (pinned per deployment); logical ids, runtimes, and
    provenance are fixed here so the default can never silently drift
    to a different model. The TTS voice is a directory
    (``tts/<voice>/``) whose executed files — ``model.onnx`` plus
    ``tokens.txt`` — are each hashed: swapping any of them invalidates
    integrity.
    """
    return ModelManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        profile_id=BASELINE_PROFILE_ID,
        components=(
            ModelComponent(
                component="stt",
                runtime="whisper.cpp",
                runtime_version=whisper_runtime_version,
                model_id="whisper-base-multilingual",
                filename="stt/ggml-base.bin",
                sha256=whisper_sha256.lower(),
                size=whisper_size,
                arch_quant="base",
                language="multilingual",
                source="https://huggingface.co/ggerganov/whisper.cpp",
                license="MIT (OpenAI Whisper model; check upstream terms)",
                required=True,
            ),
            ModelComponent(
                component="llm",
                runtime="llama.cpp",
                runtime_version=llama_runtime_version,
                model_id="qwen3-1.7b-q4_k_m",
                filename="llm/Qwen3-1.7B-Q4_K_M.gguf",
                sha256=llama_sha256.lower(),
                size=llama_size,
                arch_quant="GGUF Q4_K_M",
                language="multilingual",
                source="https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF",
                license="Apache-2.0 (Qwen; check upstream terms)",
                required=True,
            ),
            ModelComponent(
                component="tts",
                runtime="sherpa-onnx",
                runtime_version=tts_runtime_version,
                model_id="tts-es-onnx",
                filename=f"tts/{tts_voice}/model.onnx",
                sha256=tts_model_sha256.lower(),
                arch_quant="ONNX VITS-compatible",
                language="es",
                voice=tts_voice,
                source="operator-provisioned Spanish ONNX voice",
                license="operator-provisioned (check voice license before use)",
                required=True,
                files=(
                    f"tts/{tts_voice}/model.onnx",
                    f"tts/{tts_voice}/tokens.txt",
                ),
                file_hashes=(
                    tts_model_sha256.lower(),
                    tts_tokens_sha256.lower(),
                ),
            ),
        ),
    )
