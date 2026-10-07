# #25 SIP integration harness (SIPp + disposable Asterisk + baresip)

Three roles, deliberately separated (spec #17):

- **SIPp** (`sipp/`, pinned `sip-tester 1:3.7.2`): deterministic protocol
  scripts against the adapter's loopback listener. Precise message
  sequences, no PBX in the middle.
- **Asterisk** (`asterisk/`, pinned Ubuntu 24.04 + Asterisk 20.6.0):
  real registrar/PBX semantics (REGISTER, routing, B2BUA legs, REFER
  handling, RTP bridging). Test-only credentials, internal routes only.
- **baresip counterparty** (`caller.py`, own process — one baresip
  Runtime per process): the far end for media/DTMF/hold/transfer
  assertions, driven by line-JSON.

## Layout

- `asterisk/Dockerfile`, `pjsip.conf`, `extensions.conf`, `rtp.conf` —
  disposable PBX. Endpoints: `700` receptionist, `701` caller,
  `702` transfer target (live UA), `201` answering dialplan extension.
  PCMU/PCMA + RFC4733, `direct_media=no`, internal test context only.
- `sipp/Dockerfile`, `scenarios/*.xml` — versioned SIPp scenarios:
  - `uac_answer_bye`: INVITE answered, remote BYE, exactly-once hangup.
  - `uac_expect_reject`: declined INVITE released promptly (200+BYE;
    see decline note below).
  - `uac_hold_resume`: re-INVITE hold/resume with SDP answers.
- `caller.py` — counterparty process (JSON protocol on stdin/stdout).
- `reg_probe.py` — negative registration probe (own process).
- `run_matrix.py` — the full matrix; writes
  `docs/evidence/25-baresip-matrix/evidence.json`.

## Run

```bash
# full PCMU matrix (~8 min, needs docker)
PYTHONPATH=src:tests python3 tools/sip/run_matrix.py --codecs pcmu
# PCMA leg
PYTHONPATH=src:tests python3 tools/sip/run_matrix.py --codecs pcma
# gated unittest wrapper (CI skips this; PR CI stays fast)
VR_RUN_SIP=1 python -m unittest tests.test_telephony_integration -v
```

Everything runs on loopback (`--network host` containers + host
processes). The adapter listens on `127.0.0.1:5070` (`sip_listen` via
`extra_config_text`); counterparties use 5071/5072/5074/5075/5076 —
every process needs its own SIP port (baresip binds at start).

## Findings encoded here (not assumptions)

- The stack emits one post-answer in-dialog media refresh; SIPp
  scenarios answer it like any peer (deterministic, versioned).
- Asterisk as B2BUA terminates hold locally (MOH) and does not forward
  re-INVITEs end to end; remote-hold observation is proven on direct
  legs (SIPp s3, unregistered-caller leg), local hold on PBX legs.
- A registered baresip UA proxy-routes outbound INVITEs through its
  registrar, so "direct" legs need an unregistered counterparty.
- `Call.reject()`/`hangup()` on a fresh inbound leg raise
  `StaleHandleError` on baresip-python 0.5.2a3; decline is released via
  public answer+BYE until upstream fixes it (see adapter docstring and
  the evidence README). No patching, no unsafe handles.
