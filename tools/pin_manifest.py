"""Pin a deployment manifest from installed artifacts (#24).

Reads the template (logical ids, runtimes, provenance), fills
sha256/size/runtime versions measured from the trusted model root,
and writes the pinned manifest. Never changes logical ids, never
downloads anything, never contacts the network.

Run:  python3 tools/pin_manifest.py --model-root /models \\
          --template models/baseline.manifest.json --out /models/pinned.json
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
    resolve_trusted_path,
    sha256_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pin a model-pack manifest.")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    with open(args.template, encoding="utf-8") as handle:
        template = json.load(handle)
    manifest = load_manifest(template)
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
            primary_size = sizes[0] if len(names) == 1 and component.size == 0 else component.size
            components.append(
                component.__class__(
                    **{
                        **component.__dict__,
                        "sha256": hashes[0],
                        "size": primary_size or sizes[0],
                        "files": tuple(names),
                        "file_hashes": tuple(hashes),
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
        print("pin_manifest FAILED: required artifacts missing; manifest not written")
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


if __name__ == "__main__":
    raise SystemExit(main())
