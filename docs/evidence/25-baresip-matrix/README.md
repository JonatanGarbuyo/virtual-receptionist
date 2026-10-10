# #25 evidence: baresip telephony adapter matrix

`evidence.json` is written by `tools/sip/run_matrix.py` (one file per
run). Pinned artifacts of the final code:

- `evidence-pcmu.json` — full matrix, PCMU-forced: **46/46 pass**.
- `evidence-pcma.json` — full matrix, PCMA-forced: **46/46 pass**.
- `evidence.json` — copy of the latest run.

46 checks per run: clean-tree provenance gate (1), imaging/pbx/
registration (8), SIPp scenarios + events (6: answer/BYE + hangup +
cleanup, 486 decline, hold/resume, RTP-timeout fault), media both
directions + negotiated codec (7), DTMF (2), hold local/remote/direct
(5), local hangup + 2-call isolation (4), direct leg (1), counterparty
registration (1), transfers ACCEPTED/REJECTED (3), re-registration
verified + outage + media health + recovery (6), shutdown + zombies (2).

Each check carries its detail; the SIP timelines are redacted (no
Authorization, no secrets). The gated unittest
(`tests/test_telephony_integration.py`, `VR_RUN_SIP=1`) replays the
matrix and fails on any non-pass.

## Provenance (auditable two-step flow)

Each evidence file records `meta.tested_code_sha` (the exact HEAD the
matrix ran on) plus `meta.worktree_dirty` (uncommitted input paths,
always empty here). `run_matrix.py` refuses to run on a dirty input
tree (`worktree-clean` check; `--allow-dirty` override marks the
evidence unattributable instead). Evidence outputs
(`docs/evidence/`) are run products and never invalidate a run.
Flow: commit code/scenario/harness → run both matrices clean →
evidence/docs-only commit. The evidence commit's parent is the tested
code commit.

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

## Decline path (corrected during review)

An earlier revision claimed `Call.reject()`/`hangup()` on a fresh
inbound leg always raise `StaleHandleError` and released declines via
answer+BYE. Independent raw-binding probes (4/4 variants) disproved
that: native `reject()`/`hangup()` answer `486` deterministically, so
`BaresipTelephonyAdapter.reject()` attempts the native rejection first
and the scenario (`uac_expect_reject.xml`) asserts the `486` the caller
receives. Answer-then-release remains only as a fallback for a
genuinely lost race (`StaleHandleError`, e.g. a simultaneous remote
CANCEL) so the slot never leaks.

## Interop notes (all observed, all encoded in scenarios/harness)

- Post-answer in-dialog media refresh from the stack (~90 ms,
  full codec offer); peers answer it; scenarios script it.
- Re-registration refresh is unregister-old-dialog + register-new-dialog
  on the wire: the stack answers the expires-0 leg with 200 OK before
  the re-register leg completes, so `refresh_registration()` waits for
  outcome quiescence before reporting `fresh` (else a registrar query
  in between observes the contact deleted — a real race, proven on
  the wire, not a stale read).
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
