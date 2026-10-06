# Model packs (#24)

This directory holds **metadata only**. No model artifacts are
committed here (or anywhere in the repo): GGUF/ONNX/bin files are
operator-provisioned per installation.

## Baseline profile: `cascaded-cpu-baseline-v1`

| Component | Runtime | Logical model |
|---|---|---|
| STT | whisper.cpp | Whisper **base multilingual** (`ggml-model-base.bin`; not `base.en`) |
| LLM | llama.cpp, non-thinking, bounded output | Qwen3-1.7B **GGUF Q4_K_M** |
| TTS | sherpa-onnx, CPU | Configurable Spanish ONNX/VITS-compatible voice |

Resource envelope: modern x86_64, 4 cores, 8 GB minimum / 16 GB
recommended, no GPU. Target RSS <= 4 GB, hard cap <= 6 GB. One
simultaneous AI call (`max_ai_sessions = 1`); models load once and
stay warm — never per turn, never duplicated per call.

## Provenance (review before downloading)

- STT model: `ggerganov/whisper.cpp` base model, MIT (OpenAI Whisper
  weights — check upstream terms). No redistribution in this repo.
- LLM: `Qwen/Qwen3-1.7B-GGUF`, file `qwen3-1.7b-q4_k_m.gguf`,
  Apache-2.0 (check upstream terms). No redistribution in this repo.
- TTS voice: operator-provisioned Spanish ONNX voice. Check the
  voice's license before installing; the manifest records it but this
  repo never auto-accepts licenses.

## Install (offline-capable, no auto-download)

1. Download the three artifacts on any machine (see sources above).
2. Lay them out under a trusted model root:
   `stt/ggml-model-base.bin`, `llm/qwen3-1.7b-q4_k_m.gguf`,
   `tts/es-voice.onnx` (+ voice sidecar files for sherpa-onnx).
3. Pin the manifest: `python3 tools/pin_manifest.py --model-root
   /var/lib/receptionist/models --template
   models/baseline.manifest.json --out
   /var/lib/receptionist/models/baseline.manifest.json`
   (fills sha256/size/runtime versions; never edits logical ids).
4. Point config.db at the root + manifest
   (`voice.model_root`, `voice.manifest`) and restart: the backend
   validates integrity, warms each runtime, and only then reports
   READY. A corrupt/missing artifact is NOT_READY, never a mid-call
   surprise.

The turn runs fully offline afterwards: no HuggingFace fetch, no
telemetry, no network. Loopback IPC only where a runtime needs it.
