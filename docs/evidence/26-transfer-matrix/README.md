# #26 evidence: symbolic transfer → real PBX handoff matrix

`evidence.json` is written by `tools/sip/run_transfer_matrix.py` (one
file per run). Pinned artifacts of the final code:

- `evidence.json` — focused transfer matrix, PCMU: **13/13 pass**.
- `evidence-pcma.json` — focused transfer matrix, PCMA: **13/13 pass**.

13 checks per run: clean-tree provenance gate (1), imaging/pbx/
registration/readiness (7: image, container, ready, REGISTER, READY,
both counterparties), five transfer scenarios (5: M1 primary accepted,
M2 rejected→fallback accepted, M3 timeout→fallback accepted,
M4 transport-error→fallback accepted, M5 double failure exit).

Each scenario is driven THROUGH `CallSession` with a typed
`TransferRequest` carrying only a synthetic symbolic destination id
(`primaria-ok`, `primaria-inexistente`, `primaria-lenta`,
`primaria-rota`, `guardia`, `guardia-caida`). Policy/config resolves
each id to a trusted PBX-local target; the adapter only ever sees the
resolved target and echoes a project-owned attempt id on exactly one
normalized outcome. Per-scenario records carry the normalized
outcomes, the append-only outcome list, fallback attempt counts,
final `CallState`, history, transfer budget used, the audit slice,
and a sanitized SIP/event timeline (no Authorization, no secrets —
the runner redacts and fails the run on any leak).

## Outcome mechanisms (exactly as run, no hidden fabrication)

- **accepted**: real REFER to registered auto-answering 702; the PBX
  bridges caller→702 (`target_up`, session `HANDED_OFF`, fallback
  never called in M1).
- **rejected**: real REFER to unassigned `sip:999@…` / `sip:998@…`;
  the PBX answers 404 and the adapter normalizes to `REJECTED`.
- **timeout**: a single-shot stuck transfer command holds the primary
  open on a real established dialog without result; the REAL
  declarative adapter watchdog (`sip.transfer_timeout`, 8 s in the
  harness) reports `TIMEOUT` — proven at t=8.0 s in the outcome
  timeline — and the REAL fallback REFER completes on the pristine
  dialog. Real PBX stalls proved unusable for this: Asterisk
  fast-fails every tried shape (404-style NOTIFY failure, or a
  binding-level `TransferFailed` that also wedges the dialog for the
  fallback), so the stuck-command fault at the owned native seam is
  the deterministic mechanism, stated here explicitly.
- **transport-error**: a single-shot native-seam fault (`OSError` on a
  real established leg, delegating afterwards so there is no restore
  race); the REAL adapter normalizes to `TRANSPORT_ERROR` and the
  REAL fallback REFER completes. `CallSession` never fabricates the
  enum in any scenario.

## Race-A lesson (encoded in code, proven by M1)

A far-end BYE routinely reaches the adapter just ahead of the REFER
outcome it was caused by. Two cooperating fixes, both in the
telephony adapter (no business logic moved, no session contract
changed):

- transfer outcomes jump ahead of still-queued closes for the same
  call in the per-call dispatcher FIFO;
- remote closes racing an open transfer are stashed until the
  transfer settles (`ACCEPTED` wins and drops the stash; anything
  else emits the stash first so the session terminates instead of
  falling back onto a dead dialog).

Local closes always win immediately; shutdown drops records outright.

## Provenance (auditable two-step flow)

Same gate as #25: each evidence file records
`meta.tested_code_sha` plus `meta.worktree_dirty` (always empty
here). `run_transfer_matrix.py` refuses to run on a dirty input tree
(`worktree-clean` check; `--allow-dirty` override marks the evidence
unattributable instead). Evidence outputs (`docs/evidence/`) are run
products and never invalidate a run. Flow: commit code/scenario/
harness → run matrix clean → evidence/docs-only commit. The
evidence commit's parent is the tested code commit.

The gated unittest (`tests/test_transfer_integration.py`,
`VR_RUN_SIP_TRANSFER=1`) replays the matrix and fails on any
non-pass. PR CI never runs it (fast deterministic suites only).

## How to reproduce

See `tools/sip/README.md`. Pins: Asterisk
`1:20.6.0~dfsg+~cs6.13.40431414-2build5`, `baresip-python==0.5.2a3`.
Run: `python3 tools/sip/run_transfer_matrix.py --codecs pcmu` from a
clean tree (`PYTHONPATH=src:tests`).
