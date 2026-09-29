# baresip-python spike

This directory validates whether the alpha `baresip-python` binding can
implement Virtual Receptionist's narrow telephony boundary without exposing the
AI/core to generic SIP control.

It is **not production code** and is isolated from the future application
package intentionally.

## What is pinned

`requirements.txt` pins `baresip-python==0.5.2a1`.

The binding is still alpha, so the spike must be rerun when upgrading it.

## Why this binding is promising

The current public API exposes the exact primitives we need:

- SIP REGISTER with automatic refresh;
- inbound INVITE callbacks and explicit answer/reject;
- headless bidirectional PCM through the `aumem` driver;
- PCMU/PCMA codec restriction through typed account config;
- DTMF receive/send;
- hold/resume;
- blind transfer;
- attended transfer for future work;
- typed call/registration failures;
- asyncio-native lifecycle;
- x86_64 and aarch64 Linux wheels.

The binding's incoming REFER policy is set to `reject` in this spike. A remote
peer must never be able to make the receptionist dial an arbitrary REFER target.

## Setup

Python 3.11+ on Linux:

```bash
python -m venv .venv
. .venv/bin/activate
pip install --pre -r requirements.txt
```

Create a normal SIP extension/account in the PBX. For the initial test, prefer
a LAN deployment with no NAT between the PBX and the spike.

Required environment variables:

```bash
export SIP_USER=700
export SIP_PASS='secret'
export SIP_DOMAIN='192.168.1.10'
```

Optional:

```bash
export SIP_REGISTRAR='192.168.1.10'
export SIP_TRANSPORT=udp
export SIP_NET_INTERFACE=192.168.1.50
export SIP_TRACE=0
```

## Gate 1 — registration

```bash
python smoke.py register
```

Expected:

```text
REGISTER_OK 700@192.168.1.10
UNREGISTER_OK
```

This validates initial REGISTER/unREGISTER. Leave the process running longer
when checking automatic refresh/re-registration behavior.

## Gate 2 — inbound + PCM

```bash
ECHO_SECONDS=5 python smoke.py inbound
```

Call the configured extension from another PBX endpoint.

The spike answers, prints negotiated PCM information, and echoes caller PCM for
five seconds. This proves the headless RTP/codec-to-PCM path without a sound
device.

Expected markers:

```text
INCOMING ...
ESTABLISHED ...
PCM_OK echoed_bytes=...
```

## Gate 3 — DTMF

To send DTMF from the receptionist as well as receive it:

```bash
TEST_DTMF_TX=5 python smoke.py inbound
```

Press digits from the caller too. The process should print `DTMF_RX`.

## Gate 4 — hold/resume

```bash
TEST_HOLD=1 HOLD_SECONDS=2 python smoke.py inbound
```

Validate both the signaling and media recovery after re-INVITE.

## Gate 5 — blind transfer

Use only an internal PBX destination created for the test:

```bash
TRANSFER_TARGET='sip:201@192.168.1.10' python smoke.py inbound
```

A successful SIP transfer closes the receptionist's original call. A failed
transfer should leave the call usable and produce `TRANSFER_FAILED`.

Production code will never accept this URI from the caller/model. The future
adapter receives a trusted `ResolvedDestination` created by Policy/Config.

## Gate 6 — concurrency / attended-transfer preparation

The runtime is configured with `max_concurrent_calls=2`, matching one caller
plus one consultation leg. The upstream project already exposes
`attended_transfer(consult_call)`; this spike deliberately does not implement
the future attended-transfer feature yet.

For #8 we still need to validate two simultaneous live call objects against the
target PBX before considering the binding proven.

## Pass/fail checklist for #8

- [ ] REGISTER / unregister
- [ ] automatic re-registration over a long-running test
- [ ] inbound INVITE / answer
- [ ] reject path
- [ ] caller hangup and local hangup
- [ ] PCMU
- [ ] PCMA
- [ ] bidirectional PCM through `aumem`
- [ ] DTMF RX
- [ ] DTMF TX
- [ ] hold/resume and audio recovery
- [ ] blind transfer success
- [ ] blind transfer failure keeps original call alive
- [ ] two simultaneous call objects
- [ ] clean SIGTERM/runtime shutdown
- [ ] registration failure behavior
- [ ] RTP failure/timeout behavior
- [ ] Linux x86_64 wheel
- [ ] Linux aarch64 wheel (packaging validation; hardware test if available)
- [ ] sustained media test without asyncio starvation

## Decision rule

Use `baresip-python` for v0.1 if the target-PBX test passes without patching
the binding and all binding-specific objects remain confined behind our own
telephony adapter.

If it fails materially, keep the same `TelephonyPort` and replace only the
implementation with a thin project-owned shim over libbaresip.
