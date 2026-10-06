"""One synthetic cascaded turn through project-owned seams (#24).

Deterministic: fake STT/LLM/TTS doubles drive the REAL coordinator
and CallSession over generic PCM. No models, no network, no audio
files. Real-model evidence comes from tools/real_turn.py (gated).

Run:  PYTHONPATH=src python3 examples/cascaded_call.py
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "tests")

from fakes import FakeCallIds, FakeClock, FakePolicy, FakeTelephony
from receptionist.audio import make_frame, tone_pcm
from receptionist.cascaded import CascadedVoiceBackend, VoiceProfile
from cascaded_fakes import (
    FakeLLMAdapter,
    FakeSTTAdapter,
    FakeTTSAdapter,
    transfer_document,
)
from receptionist.config import ConfigService, InMemoryConfigRepository
from receptionist.core import ReceptionistCore
from receptionist.ids import UuidCallIds
from receptionist.persistence import (
    InMemoryAuditLog,
    InMemoryCallRepository,
    InMemoryMessageRepository,
    InMemoryTranscriptStore,
    RuntimeStorage,
)
from receptionist.policy import Destination, Limits, PolicyEngine, RetentionPolicy


def main() -> None:
    clock = FakeClock()
    backend = CascadedVoiceBackend(
        profile=VoiceProfile(profile_id="demo", require_manifest=False),
        stt=FakeSTTAdapter(["quiero hablar con ventas por favor"]),
        llm=FakeLLMAdapter(
            [transfer_document("Por supuesto, le comunico con ventas.", "ventas")]
        ),
        tts=FakeTTSAdapter(),
        clock=clock,
    )
    backend.start()
    assert backend.warm() == [] and backend.ready

    telephony = FakeTelephony()
    core = ReceptionistCore(
        telephony=telephony,
        voice=backend,
        config_service=ConfigService(
            InMemoryConfigRepository(
                {"greeting": "Bienvenido, ¿en qué puedo ayudarle?", "language": "es"}
            )
        ),
        policy=FakePolicy(),
        clock=clock,
        policy_engine=PolicyEngine(
            destinations={
                "ventas": Destination(id="ventas", target="SIP/201", kind="extension"),
                "recepcion": Destination(
                    id="recepcion", target="SIP/200", kind="extension"
                ),
            },
            fallback_id="recepcion",
            limits=Limits(),
        ),
        runtime=RuntimeStorage(
            calls=InMemoryCallRepository(),
            messages=InMemoryMessageRepository(clock=clock),
            transcripts=InMemoryTranscriptStore(),
            audit=InMemoryAuditLog(),
        ),
        retention=RetentionPolicy(),
        call_ids=UuidCallIds(),
    )
    core.start()
    print(f"health: {core.health.status.value} ({core.health.detail})")

    session = core.incoming_call("+34910000001")
    print(f"answered: state={session.state.value} mode={session.mode.value}")
    session.drain_voice()

    pcm = tone_pcm(duration_seconds=0.6)
    session.push_caller_audio(make_frame(pcm, 16000, call_id=session.call_id))
    session.commit_caller_turn()
    session.drain_voice()
    print(f"after turn: state={session.state.value} mode={session.mode}")
    print(f"telephony transfers: {telephony.transfers}")
    voice = session.voice_session
    timings = getattr(voice, "last_timings", None)
    print(f"turn timings: {timings}")


if __name__ == "__main__":
    main()
