"""Synthetic receptionist call (ticket #18 tracer bullet).

Runs one deterministic inbound session from INCOMING to ENDED through
project-owned interfaces only: fake telephony, fake voice backend, fake
clock, in-memory configuration/policy, and in-memory persistence.
No network, no real models, no wall-clock sleeps.

Run:  PYTHONPATH=src python3 examples/synthetic_call.py
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "tests")

from fakes import FakeClock, FakePolicy, FakeTelephony, FakeVoiceBackend
from receptionist.config import ConfigService, InMemoryConfigRepository
from receptionist.core import ReceptionistCore
from receptionist.persistence import InMemoryCallRepository


def log(step: str, session) -> None:
    mode = session.mode.value if session.mode is not None else "-"
    print(f"{step:45s} state={session.state.value:12s} mode={mode:10s} turn={session.current_turn}")


def main() -> None:
    telephony = FakeTelephony()
    voice = FakeVoiceBackend()
    clock = FakeClock()
    calls = InMemoryCallRepository()
    core = ReceptionistCore(
        telephony=telephony,
        voice=voice,
        config_service=ConfigService(
            InMemoryConfigRepository(
                {"greeting": "Bienvenido, ¿en qué puedo ayudarle?", "language": "es"}
            )
        ),
        policy=FakePolicy(),
        calls=calls,
        clock=clock,
    )
    core.start()
    print(f"health: {core.health.status.value} ({core.health.detail})")

    session = core.incoming_call("+34910000001")
    backend = voice.sessions[session.call_id]
    log("incoming -> answered", session)
    print(f"  telephony answered: {telephony.answered}")
    print(f"  greeting spoken: {backend.spoken}")

    backend.finish_playback(1)
    log("greeting playback done", session)

    clock.advance(3.0)
    backend.deliver_caller_speech("Quisiera hablar con ventas, por favor.")
    log("caller utterance -> inference", session)

    backend.deliver_response("Por supuesto, le comunico con ventas.", session.current_turn)
    log("backend response -> speaking", session)
    print(f"  spoken so far: {backend.spoken}")

    backend.finish_playback(session.current_turn)
    log("response playback done", session)

    clock.advance(39.5)
    session.end_call()
    log("local hangup -> ended", session)

    summary = calls.get(session.call_id)
    print(f"telephony hangup: {telephony.hung_up}")
    print(f"summary: {summary}")
    print(f"history: {[s.value for s in session.history]}")
    print(f"health: {core.health.status.value}")


if __name__ == "__main__":
    main()
