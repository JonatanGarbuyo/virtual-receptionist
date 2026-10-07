# #25 evidence: baresip telephony adapter matrix

`evidence.json` is written by `tools/sip/run_matrix.py` (one file per
run). Pinned artifacts of the final code:

- `evidence-pcmu.json` — full matrix, PCMU-forced: **39/39 pass**.
- `evidence-pcma.json` — full matrix, PCMA-forced: **39/39 pass**.
- `evidence.json` — copy of the latest run (PCMU 39/39).

Each check carries its detail; the SIP timelines are redacted (no
Authorization, no secrets). The gated unittest
(`tests/test_telephony_integration.py`, `VR_RUN_SIP=1`) replays the
matrix and fails on any non-pass.

## Binding decision

**`baresip-python accepted for v0.1`** (`baresip-python==0.5.2a3`,
wheels statically bundle libre/libbaresip; CPython 3.11–3.13 per
upstream metadata, verified here on 3.12/x86_64; aarch64 per upstream
wheels, best-effort until gated here).

The binding passed the full matrix without patching and without
leaking native types past the project-owned boundary:

- REGISTER/re-registration (200 OK, refresh, bad creds, PBX down,
  loss→recovery with health transitions), headless `aumem` PCM both
  directions, PCMU + PCMA (forced per run, energy-verified post-decode,
  never "packets exist"), DTMF RX/TX (RFC4733), local hold/resume with
  SDP answers, remote hold observed on direct legs, blind REFER with
  normalized ACCEPTED/REJECTED/TIMEOUT, two simultaneous call objects
  with per-call audio/hangup isolation, idempotent shutdown joining all
  worker threads, normalized errors, thread-safe media from any thread.

No thin libbaresip shim was needed. No contract term was lowered to
fit the binding.

## Known upstream limitation (worked around, not patched)

`Call.reject()`/`hangup()` on a just-arrived inbound leg always raise
`StaleHandleError` (verified immediate and delayed, raw binding, no
adapter involved). SIP decline therefore releases via public
answer+BYE (`BaresipTelephonyAdapter.reject`, documented in code):
same observable outcome for the core (released, slot freed,
exactly-once close, no AI resources), wire shows 200+BYE instead of
486. To be reported upstream; the scenario (`uac_expect_reject.xml`)
upgrades to 486 when fixed, with no contract change.

## Interop notes (all observed, all encoded in scenarios/harness)

- Post-answer in-dialog media refresh from the stack (~90 ms,
  full codec offer); peers answer it; scenarios script it.
- Asterisk B2BUA terminates hold locally and never forwards re-INVITEs;
  a registered UA proxy-routes outbound INVITEs through its registrar
  (true direct legs need an unregistered counterparty).
- sipp correlates responses by its internally generated Call-ID:
  overriding the header breaks mapping; scenarios keep `[call_id]`,
  From-tags differ per scenario, each run is a fresh process/stack.
- Barge-in works as designed: a caller tone tail overlapping a fresh
  assistant turn cancels it (harness sequences directions apart).

## How to reproduce

See `tools/sip/README.md`. Pins: Asterisk
`1:20.6.0~dfsg+~cs6.13.40431414-2build5` (Ubuntu 24.04 universe),
`sipp 1:3.7.2-1build3`, `baresip-python==0.5.2a3`.
