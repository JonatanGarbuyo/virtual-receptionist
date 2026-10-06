# Real CPU turn evidence — issue #24 (`docs/evidence/24-real-turn/`)

- `evidence.json`: full sanitized output of `tools/real_turn.py`
  (`ok: true`), including the complete `artifact_sha256` map. No model
  bytes, no fixture audio, no prompts, no secrets, no absolute local
  paths.
- `pinned-manifest.json`: the exact pinned manifest the run verified
  against (relative paths + SHA-256 only).

## How it was produced

```sh
export LD_LIBRARY_PATH=<whisper.cpp>/build/bin:<llama.cpp>/build/bin
REAL_CASCADED=1 MODEL_ROOT=/models MANIFEST=/models/pinned.json \
  REAL_AUDIO_FIXTURE=/path/to/fixture.wav \
  STT_EXE=<...>/whisper-server LLM_EXE=<...>/llama-server \
  PYTHONPATH=src:tests python3 -m unittest tests.test_cascaded_real -v
PYTHONPATH=src python3 tools/real_turn.py --model-root /models \
  --manifest /models/pinned.json --audio-fixture /path/to/fixture.wav \
  --stt-exe <...>/whisper-server --llm-exe <...>/llama-server \
  --tts-voice es-female-1 --llm-threads 8
```

## Build pins

- whisper.cpp tag `v1.9.5` (`d1be6fde…`), `whisper-server` (+ same-build
  `whisper-cli` for `--version`).
- llama.cpp tag `v0.6.0` (`d8123504…`), `llama-server`.
- sherpa-onnx `1.13.8` (pip metadata).
- Voice `vits-piper-es_ES-sharvard-medium-int8`, installed as
  `tts/es-female-1/` (model + tokens + espeak-ng-data, 357 hashed files).
- Fixture: synthetic sherpa utterance of
  “Hola, quisiera hablar con ventas.”, provenance + sha in
  `tests/fixtures/real-fixture.provenance.json`.

## Slot-recycle proof (MAJOR-1, real v0.6.0)

Aborting an in-flight 300-token generation and immediately issuing a
short one: `restarts=1`, short request completed in **995 ms** (vs ~13 s
queued tail measured with socket-close only). Covered by
`RealSlotRecycleTest` (gated).

## Latency risk

EOU→first-audio ≈ 4.4–5.4 s across runs (`unhealthy` band), dominated by
LLM prompt processing on CPU (~3.2–3.4 s). Documented risk for the
benchmark follow-up; #24 requires instrumentation, which this evidences.
Memory total ≈ 2.8 GB (target normal ≤4 GB).
