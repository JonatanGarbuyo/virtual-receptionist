"""baresip-python smoke test for Virtual Receptionist.

This is intentionally a prototype. It exercises only public baresip-python APIs
and never becomes the application's policy layer.

Examples:

  SIP_USER=700 SIP_PASS=secret SIP_DOMAIN=192.168.1.10 \
    python smoke.py register

  SIP_USER=700 SIP_PASS=secret SIP_DOMAIN=192.168.1.10 \
    ECHO_SECONDS=5 TEST_HOLD=1 python smoke.py inbound

  SIP_USER=700 SIP_PASS=secret SIP_DOMAIN=192.168.1.10 \
    TRANSFER_TARGET=sip:201@192.168.1.10 python smoke.py inbound
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal

from baresip import (
    Account,
    AudioNotActive,
    AudioRestarted,
    CallState,
    Config,
    Event,
    Runtime,
    UserAgent,
)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def make_account() -> Account:
    return Account(
        user=os.environ["SIP_USER"],
        password=os.getenv("SIP_PASS", ""),
        domain=os.environ["SIP_DOMAIN"],
        registrar=os.getenv("SIP_REGISTRAR") or None,
        transport=os.getenv("SIP_TRANSPORT", "udp"),
        # Deliberately restrict the v0.1 spike to ordinary telephony codecs.
        audio_codecs=("pcmu", "pcma"),
        dtmf_mode=os.getenv("SIP_DTMF_MODE", "auto"),
    )


def make_config() -> Config:
    return Config(
        net_interface=os.getenv("SIP_NET_INTERFACE") or None,
        audio_driver="aumem",
        max_concurrent_calls=2,
        rtp_timeout=int(os.getenv("RTP_TIMEOUT", "15")),
        sip_trace=env_bool("SIP_TRACE"),
    )


async def register_only() -> None:
    runtime = Runtime()
    await runtime.start(make_config())
    try:
        account = make_account()
        # Reject peer-originated REFER automatically. The receptionist must
        # never let a remote caller instruct it to place an arbitrary call.
        ua = await UserAgent.create(runtime, account, transfer_policy="reject")
        await ua.register()
        print(f"REGISTER_OK {account.user}@{account.domain}", flush=True)

        stay = float(os.getenv("REGISTER_STAY_SECONDS", "5"))
        await asyncio.sleep(stay)

        await ua.unregister()
        print("UNREGISTER_OK", flush=True)
    finally:
        await runtime.close()


async def echo_pcm(call, seconds: float, closed: asyncio.Future) -> int:
    """Echo far-end PCM back for a bounded interval.

    This is not product behavior. It proves the no-sound-card PCM path.
    """
    if seconds <= 0:
        return 0

    deadline = asyncio.get_running_loop().time() + seconds
    total = 0
    while asyncio.get_running_loop().time() < deadline and not closed.done():
        try:
            pcm = call.audio.read(4096)
            if pcm:
                total += call.audio.write(pcm)
            else:
                await asyncio.sleep(0.01)
        except AudioRestarted:
            # Hold/re-INVITE may rebuild the streams; retry against the new
            # epoch exactly as the public API documents.
            await asyncio.sleep(0)
        except AudioNotActive:
            break
    return total


async def serve_one_inbound() -> None:
    runtime = Runtime()
    await runtime.start(make_config())
    try:
        account = make_account()
        ua = await UserAgent.create(runtime, account, transfer_policy="reject")
        await ua.register()
        print(f"REGISTER_OK {account.user}@{account.domain}", flush=True)

        incoming: asyncio.Queue = asyncio.Queue()
        ua.on_incoming(incoming.put_nowait)

        incoming_timeout = float(os.getenv("INCOMING_TIMEOUT", "120"))
        print(f"WAITING_INBOUND timeout={incoming_timeout:g}s", flush=True)
        call = await asyncio.wait_for(incoming.get(), incoming_timeout)
        print(f"INCOMING peer={call.peer}", flush=True)

        closed = asyncio.get_running_loop().create_future()

        def on_event(event) -> None:
            if event.event is Event.CALL_CLOSED and not closed.done():
                closed.set_result(event)

        def on_dtmf(event) -> None:
            print(
                f"DTMF_RX digit={event.digit} duration_ms={event.duration_ms}",
                flush=True,
            )

        call.on(on_event)
        call.on_dtmf(on_dtmf)

        await call.answer()
        await call.wait_established()
        info = call.audio.info()
        print(
            "ESTABLISHED "
            f"rx={info.rx_sample_rate}Hz/{info.rx_channels}ch "
            f"tx={info.tx_sample_rate}Hz/{info.tx_channels}ch",
            flush=True,
        )

        tx_digits = os.getenv("TEST_DTMF_TX")
        if tx_digits:
            await call.send_dtmf(tx_digits)
            print(f"DTMF_TX digits={tx_digits}", flush=True)

        echoed = await echo_pcm(
            call,
            float(os.getenv("ECHO_SECONDS", "3")),
            closed,
        )
        print(f"PCM_OK echoed_bytes={echoed}", flush=True)

        if env_bool("TEST_HOLD") and call.state is not CallState.CLOSED:
            await call.hold()
            print("HOLD_SENT", flush=True)
            await asyncio.sleep(float(os.getenv("HOLD_SECONDS", "1")))
            await call.resume()
            print("RESUME_SENT", flush=True)

        transfer_target = os.getenv("TRANSFER_TARGET")
        if transfer_target and call.state is not CallState.CLOSED:
            # TRANSFER_TARGET is an operator-provided smoke-test value.
            # Production code receives this only from ResolvedDestination.
            await call.hold()
            print(f"TRANSFER_START target={transfer_target}", flush=True)
            try:
                await call.transfer(transfer_target)
            except Exception as exc:
                print(
                    f"TRANSFER_FAILED type={type(exc).__name__} detail={exc}",
                    flush=True,
                )
                # A failed transfer leaves the original call established.
                if call.state is not CallState.CLOSED:
                    try:
                        await call.resume()
                    except Exception:
                        pass
            else:
                print("TRANSFER_ACCEPTED_AND_CALL_CLOSED", flush=True)

        if call.state is not CallState.CLOSED:
            max_call_seconds = float(os.getenv("SMOKE_CALL_SECONDS", "15"))
            try:
                await asyncio.wait_for(asyncio.shield(closed), max_call_seconds)
                print("CALLER_HANGUP", flush=True)
            except TimeoutError:
                await call.hangup()
                print("LOCAL_HANGUP", flush=True)

        await ua.unregister()
        print("UNREGISTER_OK", flush=True)
    finally:
        await runtime.close()


async def amain(mode: str) -> None:
    if mode == "register":
        await register_only()
    elif mode == "inbound":
        await serve_one_inbound()
    else:
        raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("register", "inbound"))
    args = parser.parse_args()

    # Make Ctrl-C deterministic instead of interrupting native teardown.
    stop = asyncio.Event()

    async def runner() -> None:
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGTERM, stop.set)
        except NotImplementedError:
            pass

        task = asyncio.create_task(amain(args.mode))
        stop_task = asyncio.create_task(stop.wait())
        done, _ = await asyncio.wait(
            {task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_task in done and not task.done():
            task.cancel()
        stop_task.cancel()
        await task

    asyncio.run(runner())


if __name__ == "__main__":
    main()
