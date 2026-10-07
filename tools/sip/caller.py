"""Harness counterparty for #25 integration: a raw baresip caller.

A separate process (baresip allows one Runtime per process; the adapter
under test owns the main process's). Driven by line-JSON commands on
stdin, emits line-JSON events on stdout. Never imports project code:
it is the fake far end, not the system under test.

Commands (one JSON object per line):
  {"op": "start", "user": "701", "password": ..., "domain": "127.0.0.1",
   "codecs": ["pcmu"], "listen": "127.0.0.1:5071"}
  {"op": "dial", "uri": "sip:700@127.0.0.1:5060", "tag": "A"}
  {"op": "tone", "tag": "A", "seconds": 3.0, "freq": 440.0}
  {"op": "read", "tag": "A", "seconds": 3.0}
  {"op": "dtmf", "tag": "A", "digits": "5"}
  {"op": "hold", "tag": "A"} / {"op": "resume", "tag": "A"}
  {"op": "state", "tag": "A"}
  {"op": "transfer_accept_next", "target_contains": "201"}
  {"op": "hangup", "tag": "A"}
  {"op": "stop"}

Events: {"event": "registered"}, {"event": "established", "tag": ...},
{"event": "closed", "tag": ..., "reason": ...},
{"event": "dtmf", "tag": ..., "digit": ...},
{"event": "remote_hold", "tag": ..., "held": ...},
{"event": "transfer_request", "tag": ..., "target": ...},
{"event": "tone_done", "tag": ..., "bytes": N},
{"event": "read_done", "tag": ..., "bytes": N, "rms": R},
{"event": "error", "op": ..., "detail": ...}
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
import sys
import threading
import time


def _tone_pcm8(seconds: float, freq: float) -> bytes:
    count = int(seconds * 8000)
    peak = int(32767 * 0.4)
    samples = [
        int(peak * math.sin(2.0 * math.pi * freq * i / 8000)) for i in range(count)
    ]
    return struct.pack(f"<{count}h", *samples)


def _rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    count = len(pcm) // 2
    values = struct.unpack(f"<{count}h", pcm)
    return math.sqrt(sum(v * v for v in values) / count) / 32768.0


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _maybe_sip_trace() -> None:
    import logging as _logging
    import os as _os

    if _os.getenv("VR_SIP_TRACE", "") != "1":
        return
    _logging.basicConfig(
        level=_logging.DEBUG, format="CB-SIP %(name)s %(message)s"
    )
    _logging.getLogger("baresip.native.sip").setLevel(_logging.DEBUG)


async def amain() -> int:
    from baresip import Account, Config, Runtime, UserAgent
    from baresip.events import Event as _Event

    runtime: Runtime | None = None
    ua: UserAgent | None = None
    calls: dict[str, object] = {}
    accept_next: dict | None = None

    def read_loop() -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except Exception:
                continue
            future = asyncio.run_coroutine_threadsafe(handle(cmd), loop)
            try:
                future.result(timeout=60)
            except Exception as error:
                emit({"event": "error", "op": cmd.get("op"), "detail": type(error).__name__})

    async def handle(cmd: dict) -> None:
        nonlocal runtime, ua, accept_next
        op = cmd.get("op")
        if op == "start":
            _maybe_sip_trace()
            config = Config(
                net_interface="127.0.0.1",
                audio_driver="aumem",
                max_concurrent_calls=4,
                sip_trace=True,
                extra_config_text=f"sip_listen {cmd.get('listen', '127.0.0.1:5071')}\n",
            )
            runtime = Runtime()
            await runtime.start(config)
            # Unregistered counterparty dials truly peer-to-peer (no
            # outbound proxy); otherwise every dialog proxy-routes
            # through the registrar even with a direct R-URI.
            use_proxy = bool(cmd.get("register", True))
            account = Account(
                user=cmd["user"],
                password=cmd.get("password", ""),
                domain=cmd["domain"],
                registrar=cmd.get("domain") if use_proxy else None,
                transport="udp",
                audio_codecs=tuple(cmd.get("codecs", ["pcmu"])),
                dtmf_mode="rtpevent",
            )
            ua = await UserAgent.create(runtime, account, transfer_policy="manual")

            def on_event(event) -> None:  # noqa: ANN001
                if event.event is _Event.CALL_TRANSFER and getattr(event, "text", ""):
                    emit(
                        {
                            "event": "transfer_request",
                            "target": event.text,
                        }
                    )

            ua.on(on_event)
            if cmd.get("register", True):
                await ua.register()
                emit({"event": "registered"})
            else:
                emit({"event": "started"})
        elif op == "arm_autoanswer":
            # Answer the next inbound INVITE(s) as tag INCOMING-<n>.
            async def _answer_incoming(call) -> None:  # noqa: ANN001
                n = len([k for k in calls if k.startswith("INCOMING-")])
                tag = f"INCOMING-{n}"
                calls[tag] = call
                call.on(make_listener(tag))
                call.on_dtmf(make_dtmf(tag))
                try:
                    await call.answer()
                except Exception as error:
                    emit({"event": "error", "op": "autoanswer",
                          "detail": type(error).__name__})
                    return
                try:
                    await call.wait_established(timeout=20)
                    emit({"event": "established", "tag": tag})
                except Exception as error:
                    emit({"event": "error", "op": "autoanswer-wait",
                          "detail": type(error).__name__})

            def _on_incoming(call) -> None:  # noqa: ANN001
                emit({"event": "incoming", "peer": str(getattr(call, "peer", ""))})
                asyncio.run_coroutine_threadsafe(_answer_incoming(call), loop)

            ua.on_incoming(_on_incoming)  # type: ignore[union-attr]
            emit({"event": "autoanswer_armed"})
        elif op == "dial":
            tag = cmd["tag"]
            call = await ua.dial(cmd["uri"])  # type: ignore[union-attr]
            calls[tag] = call
            call.on(make_listener(tag))
            call.on_dtmf(make_dtmf(tag))
            try:
                await call.wait_established(timeout=20)
            except Exception as error:
                emit({"event": "error", "op": "dial", "detail": type(error).__name__})
                return
            emit({"event": "established", "tag": tag})
        elif op == "tone":
            call = calls[cmd["tag"]]
            pcm = _tone_pcm8(float(cmd.get("seconds", 2.0)), float(cmd.get("freq", 440.0)))
            total = await asyncio.to_thread(pump_write, call, pcm)
            emit({"event": "tone_done", "tag": cmd["tag"], "bytes": total})
        elif op == "read":
            call = calls[cmd["tag"]]
            # Drain-then-measure: the RX buffer accumulates idle silence
            # between commands, so the first bytes read are stale. Discard
            # `drain` seconds, then measure `seconds` of live audio. Both
            # RMS values are reported; the tail decides.
            drain = float(cmd.get("drain", 2.0))
            seconds = float(cmd.get("seconds", 3.0))
            if drain > 0:
                await asyncio.to_thread(pump_read, call, drain)
            pcm = await asyncio.to_thread(pump_read, call, seconds)
            tail = pcm[-32000:] if len(pcm) > 32000 else pcm
            try:
                info = call.audio.info()
                rx_cap = info.rx_capacity
            except Exception:
                rx_cap = -1
            emit(
                {
                    "event": "read_done",
                    "tag": cmd["tag"],
                    "bytes": len(pcm),
                    "rms": round(_rms(pcm), 4),
                    "rms_tail": round(_rms(tail), 4),
                    "rx_capacity": rx_cap,
                }
            )
        elif op == "dtmf":
            call = calls[cmd["tag"]]
            await call.send_dtmf(cmd["digits"])
            emit({"event": "dtmf_done", "tag": cmd["tag"]})
        elif op == "hold":
            await calls[cmd["tag"]].hold()
            emit({"event": "hold_done", "tag": cmd["tag"]})
        elif op == "resume":
            await calls[cmd["tag"]].resume()
            emit({"event": "resume_done", "tag": cmd["tag"]})
        elif op == "state":
            call = calls[cmd["tag"]]
            emit(
                {
                    "event": "state",
                    "tag": cmd["tag"],
                    "state": call.state.value,
                    "remote_hold": call.remote_on_hold,
                }
            )
        elif op == "transfer_accept_next":
            accept_next = {"target_contains": cmd.get("target_contains", "")}
            emit({"event": "transfer_armed"})
        elif op == "hangup":
            call = calls.pop(cmd["tag"], None)
            if call is not None:
                try:
                    await call.hangup()
                except Exception:
                    pass
            emit({"event": "hangup_done", "tag": cmd["tag"]})
        elif op == "stop":
            for call in list(calls.values()):
                try:
                    await call.hangup()
                except Exception:
                    pass
            try:
                await ua.unregister()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                await runtime.close()  # type: ignore[union-attr]
            except Exception:
                pass
            emit({"event": "stopped"})

    import os as _os

    _debug = _os.getenv("VR_CALLER_DEBUG", "") == "1"

    def make_listener(tag: str):  # noqa: ANN001, ANN202
        def listener(event) -> None:  # noqa: ANN001
            if _debug:
                print(
                    f"CALLEE-EVT {tag} {event.event} {getattr(event, 'text', '')!r}",
                    file=sys.stderr,
                    flush=True,
                )
            if event.event is _Event.CALL_CLOSED:
                calls.pop(tag, None)
                emit({"event": "closed", "tag": tag, "reason": event.text or ""})
            elif event.event is _Event.CALL_HOLD:
                emit({"event": "remote_hold", "tag": tag, "held": True})
            elif event.event is _Event.CALL_RESUME:
                emit({"event": "remote_hold", "tag": tag, "held": False})
            elif event.event is _Event.CALL_TRANSFER and getattr(event, "text", ""):
                target = event.text
                emit({"event": "transfer_request", "tag": tag, "target": target})
                wanted = (accept_next or {}).get("target_contains", "")
                if wanted and wanted in target:
                    asyncio.run_coroutine_threadsafe(accept(call), loop)

        return listener

    async def accept(call) -> None:  # noqa: ANN001
        try:
            new_call = await call.accept_transfer()
            tag = f"xfer-{len([k for k in calls if k.startswith('xfer')])}"
            calls[tag] = new_call
            emit({"event": "transfer_accepted", "tag": tag})
        except Exception as error:
            emit({"event": "transfer_accept_failed", "detail": type(error).__name__})

    def make_dtmf(tag: str):  # noqa: ANN001, ANN202
        def on_digit(evt) -> None:  # noqa: ANN001
            emit({"event": "dtmf", "tag": tag, "digit": evt.digit})

        return on_digit

    def pump_write(call, pcm: bytes) -> int:  # noqa: ANN001
        total = 0
        step = 320  # 20 ms at 8 kHz mono16
        for offset in range(0, len(pcm), step):
            try:
                total += call.audio.write(pcm[offset : offset + step])
            except Exception:
                break
            time.sleep(0.02)
        return total

    def pump_read(call, seconds: float) -> bytes:  # noqa: ANN001
        out = bytearray()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                chunk = call.audio.read(640)
            except Exception:
                time.sleep(0.02)
                continue
            if chunk:
                out += chunk
            else:
                time.sleep(0.01)
        return bytes(out)

    loop = asyncio.get_running_loop()
    done = asyncio.Event()

    _handle_stop = handle

    async def handle(cmd: dict) -> None:  # noqa: F811
        if cmd.get("op") == "stop":
            await _handle_stop(cmd)
            loop.call_soon_threadsafe(done.set)
        else:
            await _handle_stop(cmd)

    reader = threading.Thread(target=read_loop, daemon=True)
    reader.start()
    await done.wait()
    return 0


def main() -> int:
    try:
        return asyncio.run(amain())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
