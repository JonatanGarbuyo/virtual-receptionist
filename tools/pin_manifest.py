"""Pin a deployment manifest from installed artifacts (#24).

Reads the template (logical ids, runtimes, provenance), fills
sha256/size measured from the trusted model root plus the real
runtime versions measured from the installed executables
(`--stt-exe`, `--llm-exe`; sherpa-onnx from this Python runtime),
and writes the pinned manifest. A productive pinned manifest never
carries an empty `runtime_version`. Never changes logical ids, never
downloads anything, never contacts the network.

Run:  python3 tools/pin_manifest.py --model-root /models \\
          --template models/baseline.manifest.json --out /models/pinned.json \\
          --stt-exe /usr/local/bin/whisper-server \\
          --llm-exe /usr/local/bin/llama-server
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from receptionist.voice_manifest import (  # noqa: E402
    load_manifest,
    manifest_to_dict,
    parse_runtime_version,
    resolve_trusted_path,
    sha256_file,
)


def measured_versions(stt_exe: str, llm_exe: str) -> dict[str, str]:
    """Measure real runtime versions: `--version` output of both
    server binaries plus this interpreter's sherpa-onnx metadata."""
    from receptionist.local_runtimes import (
        format_version,
        parse_llama_version,
        parse_whisper_version,
        query_version,
        sherpa_version,
    )

    versions: dict[str, str] = {}
    if stt_exe:
        raw = query_version(stt_exe, name="stt")
        parsed = parse_whisper_version(raw)
        versions["stt"] = format_version(parsed)
    if llm_exe:
        raw = query_version(llm_exe, name="llm")
        parsed = parse_llama_version(raw)
        versions["llm"] = format_version(parsed)
    versions["tts"] = sherpa_version()
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pin a model-pack manifest.")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stt-exe", default="")
    parser.add_argument("--llm-exe", default="")
    args = parser.parse_args(argv)

    with open(args.template, encoding="utf-8") as handle:
        template = json.load(handle)
    manifest = load_manifest(template)
    try:
        versions = measured_versions(args.stt_exe, args.llm_exe)
    except Exception as error:
        print(f"pin_manifest FAILED: cannot measure runtime versions: {error}")
        return 1
    components = []
    failed = False
    for component in manifest.components:
        names = list(component.files or (component.filename,))
        hashes = []
        sizes = []
        for name in names:
            path = resolve_trusted_path(args.model_root, name)
            if not os.path.isfile(path):
                print(f"missing artifact: {name}")
                failed = True
                break
            hashes.append(sha256_file(path))
            sizes.append(os.path.getsize(path))
        else:
            version = versions.get(component.component, "")
            if parse_runtime_version(version) is None:
                print(
                    f"unpinned runtime version for {component.component}: {version!r}"
                )
                failed = True
                continue
            primary_size = sizes[0] if len(names) == 1 and component.size == 0 else component.size
            components.append(
                component.__class__(
                    **{
                        **component.__dict__,
                        "sha256": hashes[0],
                        "size": primary_size or sizes[0],
                        "files": tuple(names),
                        "file_hashes": tuple(hashes),
                        "runtime_version": version,
                    }
                )
            )
            continue
        # A required component with a missing file keeps its placeholder
        # and fails the run below: pinning must never look successful.
        components.append(component)
        if component.required:
            failed = True
    if failed:
        print("pin_manifest FAILED: see problems above; manifest not written")
        return 1
    pinned = manifest.__class__(
        schema_version=manifest.schema_version,
        profile_id=manifest.profile_id,
        components=tuple(components),
    )
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(manifest_to_dict(pinned), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"pinned manifest: {args.out}")
    return 0
    pinned = manifest.__class__(
        schema_version=manifest.schema_version,
        profile_id=manifest.profile_id,
        components=tuple(components),
    )
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(manifest_to_dict(pinned), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"pinned manifest: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
