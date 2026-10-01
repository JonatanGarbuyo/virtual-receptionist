"""Behavior tests for confirmed caller messages (#20).

Same seam (ReceptionistCore / CallSession), same deterministic fakes.
Message-taking is a caller-visible flow: final usable text, structured
draft, readback/confirmation, persistence, and success acknowledgement
only after the commit succeeds.
"""

import unittest

from receptionist.boundaries import (
    AuditDecision,
    AuditEvent,
    CallOutcome,
    CallSummary,
    MessageConfirmed,
    MessageDraft,
    MessageRejected,
    MessageTextFinal,
    StartMessageCapture,
    StoreUnavailableError,
    TransferResult,
    TranscriptEntry,
    TransientStoreError,
)
from receptionist.call_session import ActiveMode, CallState
from receptionist.config import ConfigService, InMemoryConfigRepository
from receptionist.core import ReceptionistCore
from receptionist.persistence import (
    InMemoryCallRepository,
    InMemoryMessageRepository,
    InMemoryTranscriptStore,
    InMemoryAuditLog,
    RuntimeStorage,
)
from receptionist.policy import Destination, Limits, PolicyEngine, RetentionPolicy

from fakes import FakeClock, FakePolicy, FakeTelephony, FakeVoiceBackend, FailingMessageRepository

# Independent literal pinning call_session.EXIT_APOLOGY.
EXIT_APOLOGY = "Lo siento, no fue posible comunicarle. La llamada terminará."


DAY = 86400.0


def summary(
    call_id: str,
    caller_id: str = "+34910000001",
    started_at: float = 1000.0,
    **fields,
) -> CallSummary:
    return CallSummary(
        call_id=call_id,
        caller_id=caller_id,
        started_at=started_at,
        ended_at=started_at + 10.0,
        outcome=CallOutcome.COMPLETED,
        turn_count=1,
        **fields,
    )


class CallSummaryFieldsTest(unittest.TestCase):
    def test_extended_fields_default_to_none(self) -> None:
        record = summary("call-1")
        self.assertIsNone(record.caller_name)
        self.assertIsNone(record.handoff_destination_id)
        self.assertIsNone(record.message_id)
        self.assertIsNone(record.failure_category)

    def test_extended_fields_are_stored(self) -> None:
        repo = InMemoryCallRepository()
        repo.save(
            summary(
                "call-1",
                caller_name="García",
                handoff_destination_id="ventas",
                message_id="msg-1",
                failure_category="transfer_failed",
            )
        )
        fetched = repo.get("call-1")
        assert fetched is not None
        self.assertEqual(fetched.caller_name, "García")
        self.assertEqual(fetched.handoff_destination_id, "ventas")
        self.assertEqual(fetched.message_id, "msg-1")
        self.assertEqual(fetched.failure_category, "transfer_failed")


class MessageRepositoryTest(unittest.TestCase):
    def test_save_assigns_deterministic_ids(self) -> None:
        repo = InMemoryMessageRepository()
        first = repo.save(
            MessageDraft(call_id="call-1", caller_id="+3491", caller_name=None, text="hola")
        )
        second = repo.save(
            MessageDraft(call_id="call-2", caller_id="+3492", caller_name="Luz", text="chau")
        )
        self.assertEqual(first.id, "msg-1")
        self.assertEqual(second.id, "msg-2")
        self.assertEqual(first.text, "hola")
        self.assertEqual(first.created_at, second.created_at)
        self.assertEqual(repo.get("msg-1"), first)
        self.assertEqual(len(repo.list_all()), 2)

    def test_created_at_comes_from_injected_clock(self) -> None:
        clock = FakeClock(start=5000.0)
        repo = InMemoryMessageRepository(clock=clock)
        clock.advance(12.0)
        record = repo.save(
            MessageDraft(call_id="call-1", caller_id="+3491", caller_name=None, text="hola")
        )
        self.assertEqual(record.created_at, 5012.0)


class CallHistoryQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = InMemoryCallRepository()
        self.repo.save(summary("call-1", caller_id="+34910000001", started_at=1000.0))
        self.repo.save(summary("call-2", caller_id="+34910000002", started_at=2000.0))
        self.repo.save(summary("call-3", caller_id="+34910000001", started_at=3000.0))

    def test_find_by_caller(self) -> None:
        found = self.repo.find_by_caller("+34910000001")
        self.assertEqual([s.call_id for s in found], ["call-1", "call-3"])
        self.assertEqual(self.repo.find_by_caller("+34000000000"), [])

    def test_find_in_range_is_inclusive(self) -> None:
        found = self.repo.find_in_range(1000.0, 2000.0)
        self.assertEqual([s.call_id for s in found], ["call-1", "call-2"])
        self.assertEqual(self.repo.find_in_range(1500.0, 2500.0)[0].call_id, "call-2")
        self.assertEqual(self.repo.find_in_range(4000.0, 5000.0), [])


class RetentionTest(unittest.TestCase):
    def test_prune_keeps_exact_boundary_and_drops_expired_plus_one(self) -> None:
        now = 10_000_000.0
        clock = FakeClock(start=now - 365 * DAY)
        messages = InMemoryMessageRepository(clock=clock)
        messages.save(
            MessageDraft(call_id="c1", caller_id="+1", caller_name=None, text="kept"),
        )
        self.assertEqual(messages.prune_before(now - 365 * DAY), 0)
        self.assertEqual(len(messages.list_all()), 1)
        self.assertEqual(messages.prune_before(now - 365 * DAY + 1), 1)
        self.assertEqual(messages.list_all(), [])

    def test_repositories_prune_before_cutoff(self) -> None:

        calls = InMemoryCallRepository()
        calls.save(summary("old", started_at=100.0))
        calls.save(summary("new", started_at=9000.0))
        self.assertEqual(calls.prune_before(1000.0), 1)
        self.assertEqual([s.call_id for s in calls.list_all()], ["new"])

        audit = InMemoryAuditLog()
        audit.record(
            AuditEvent(
                timestamp=100.0,
                call_id="c",
                action="transfer",
                destination="d",
                decision=AuditDecision.DENIED,
            )
        )
        audit.record(
            AuditEvent(
                timestamp=9000.0,
                call_id="c",
                action="transfer",
                destination="d",
                decision=AuditDecision.DENIED,
            )
        )
        self.assertEqual(audit.prune_before(1000.0), 1)
        self.assertEqual(len(audit.list_all()), 1)

        transcripts = InMemoryTranscriptStore()
        transcripts.append(TranscriptEntry(call_id="c", timestamp=100.0, speaker="caller", text="a"))
        transcripts.append(TranscriptEntry(call_id="c", timestamp=9000.0, speaker="caller", text="b"))
        self.assertEqual(transcripts.prune_before(1000.0), 1)
        self.assertEqual(
            [e.text for e in transcripts.entries_for("c")], ["b"]
        )

    def test_runtime_storage_prune_is_idempotent(self) -> None:
        storage = RuntimeStorage.create()
        storage.calls.save(summary("old", started_at=100.0))
        policy = RetentionPolicy(history_days=180)
        now = 100.0 + 181 * DAY

        first = storage.prune_expired(policy, now)
        second = storage.prune_expired(policy, now)

        self.assertEqual(first["calls"], 1)
        self.assertEqual(second, {"messages": 0, "calls": 0, "transcripts": 0, "audit": 0})

    def test_retention_defaults_match_spec(self) -> None:
        policy = RetentionPolicy()
        self.assertEqual(
            (policy.messages_days, policy.history_days, policy.transcripts_days, policy.audit_days),
            (365, 180, 7, 30),
        )


READBACK_TEMPLATE = "Le confirmo su mensaje: {text}. ¿Es correcto?"
SUCCESS_ACK = "Su mensaje ha sido guardado. Gracias."


def make_message_core(
    limits=None,
    destinations=None,
    fallback_id: str = "recepcion",
    config_values: dict | None = None,
    auto_confirm: bool = True,
    messages=None,
) -> tuple:
    if destinations is None:
        destinations = {
            "ventas": Destination(
                id="ventas", target="SIP/201", kind="extension", enabled=True
            ),
            "recepcion": Destination(
                id="recepcion", target="SIP/100", kind="extension", enabled=True
            ),
        }

    telephony = FakeTelephony(auto_confirm=auto_confirm)
    voice = FakeVoiceBackend()
    clock = FakeClock()
    runtime = RuntimeStorage.create(clock=clock)
    if messages is not None:
        runtime.messages = messages
    engine = PolicyEngine(
        destinations=dict(destinations) if destinations is not None else {},
        fallback_id=fallback_id,
        limits=limits or Limits(),
    )
    values = (
        {"greeting": "Bienvenido, ¿en qué puedo ayudarle?", "language": "es"}
        if config_values is None
        else config_values
    )
    core = ReceptionistCore(
        telephony=telephony,
        voice=voice,
        config_service=ConfigService(InMemoryConfigRepository(dict(values))),
        policy=FakePolicy(),
        clock=clock,
        policy_engine=engine,
        runtime=runtime,
        retention=RetentionPolicy(),
    )
    return core, telephony, voice, clock, runtime


def start_listening_call(core, voice, caller_id="+34910000001", caller_name=None):
    session = core.incoming_call(caller_id, caller_name=caller_name)
    backend = voice.sessions[session.call_id]
    backend.finish_playback(1)
    return session, backend


class MessageCaptureFlowTest(unittest.TestCase):
    def test_full_capture_confirm_persist_acknowledge(self) -> None:

        core, telephony, voice, _, runtime = make_message_core()
        core.start()
        session, backend = start_listening_call(
            core, voice, caller_id="+34910000001", caller_name="García"
        )

        backend.deliver_action_request(StartMessageCapture())
        self.assertEqual(session.mode, ActiveMode.MESSAGE_CAPTURE)

        backend.deliver_action_request(MessageTextFinal("llamo por la factura 42"))
        self.assertEqual(
            backend.spoken[-1],
            (READBACK_TEMPLATE.format(text="llamo por la factura 42"), session.current_turn),
        )

        backend.deliver_action_request(MessageConfirmed())

        records = runtime.messages.list_all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].id, "msg-1")
        self.assertEqual(records[0].call_id, "call-1")
        self.assertEqual(records[0].caller_id, "+34910000001")
        self.assertEqual(records[0].caller_name, "García")
        self.assertEqual(records[0].text, "llamo por la factura 42")
        self.assertEqual(backend.spoken[-1][0], SUCCESS_ACK)
        self.assertEqual(session.mode, ActiveMode.LISTENING)

        session.end_call()
        summary = runtime.calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.message_id, "msg-1")
        self.assertEqual(summary.caller_name, "García")

    def test_empty_final_text_is_ignored(self) -> None:

        core, telephony, voice, _, runtime = make_message_core()
        core.start()
        session, backend = start_listening_call(core, voice)
        backend.deliver_action_request(StartMessageCapture())
        spoken_before = list(backend.spoken)

        for empty in ("", "   "):
            backend.deliver_action_request(MessageTextFinal(empty))

        self.assertEqual(session.mode, ActiveMode.MESSAGE_CAPTURE)
        self.assertEqual(backend.spoken, spoken_before)
        self.assertEqual(runtime.messages.list_all(), [])

    def test_final_text_outside_capture_is_ignored(self) -> None:

        core, _, voice, _, runtime = make_message_core()
        core.start()
        session, backend = start_listening_call(core, voice)

        backend.deliver_action_request(MessageTextFinal("hola"))
        backend.deliver_action_request(MessageConfirmed())

        self.assertEqual(session.mode, ActiveMode.LISTENING)
        self.assertEqual(runtime.messages.list_all(), [])

    def test_rejected_draft_discards_without_persisting(self) -> None:

        core, _, voice, _, runtime = make_message_core()
        core.start()
        session, backend = start_listening_call(core, voice)
        backend.deliver_action_request(StartMessageCapture())
        backend.deliver_action_request(MessageTextFinal("texto a descartar"))

        backend.deliver_action_request(MessageRejected())

        self.assertEqual(session.mode, ActiveMode.LISTENING)
        self.assertEqual(runtime.messages.list_all(), [])
        self.assertNotIn(SUCCESS_ACK, [text for text, _ in backend.spoken])

    def test_correction_replaces_draft_before_confirm(self) -> None:

        core, _, voice, _, runtime = make_message_core()
        core.start()
        _, backend = start_listening_call(core, voice)
        backend.deliver_action_request(StartMessageCapture())
        backend.deliver_action_request(MessageTextFinal("texto viejo"))

        backend.deliver_action_request(MessageTextFinal("texto nuevo"))
        backend.deliver_action_request(MessageConfirmed())

        records = runtime.messages.list_all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].text, "texto nuevo")
        readbacks = [text for text, _ in backend.spoken if text.startswith("Le confirmo")]
        self.assertEqual(len(readbacks), 2)

    def test_transcripts_disabled_by_default(self) -> None:

        core, _, voice, _, runtime = make_message_core()
        core.start()
        session, backend = start_listening_call(core, voice)
        backend.deliver_caller_speech("una consulta")
        backend.deliver_response("una respuesta.", session.current_turn)
        backend.deliver_action_request(StartMessageCapture())
        backend.deliver_action_request(MessageTextFinal("un mensaje"))
        backend.deliver_action_request(MessageConfirmed())
        session.end_call()

        self.assertEqual(runtime.transcripts.entries_for("call-1"), [])


class MessagePersistFailureTest(unittest.TestCase):
    def drive_to_confirmation(self, core, voice):

        session, backend = start_listening_call(core, voice)
        backend.deliver_action_request(StartMessageCapture())
        backend.deliver_action_request(MessageTextFinal("mensaje importante"))
        return session, backend

    def confirm(self, backend):

        backend.deliver_action_request(MessageConfirmed())

    def test_transient_failure_retries_then_succeeds(self) -> None:

        repo = FailingMessageRepository()
        repo.fail_script.append(TransientStoreError("timeout"))
        core, _, voice, _, runtime = make_message_core(messages=repo)
        core.start()
        session, backend = self.drive_to_confirmation(core, voice)

        self.confirm(backend)

        self.assertEqual(repo.saves, 2)
        self.assertEqual(len(runtime.messages.list_all()), 1)
        self.assertEqual(backend.spoken[-1][0], SUCCESS_ACK)
        self.assertEqual(session.mode, ActiveMode.LISTENING)

    def test_retry_exhausted_routes_to_fallback(self) -> None:

        repo = FailingMessageRepository()
        repo.fail_script.extend([TransientStoreError("t1"), TransientStoreError("t2")])
        core, telephony, voice, _, runtime = make_message_core(messages=repo)
        core.start()
        session, backend = self.drive_to_confirmation(core, voice)

        self.confirm(backend)

        self.assertEqual(repo.saves, 2)
        self.assertEqual(runtime.messages.list_all(), [])
        self.assertNotIn(SUCCESS_ACK, [text for text, _ in backend.spoken])
        self.assertEqual(session.state, CallState.FALLBACK_HANDOFF)
        self.assertEqual(
            telephony.transfers, [("call-1", "SIP/100")]
        )

    def test_unavailable_store_skips_retry_straight_to_fallback(self) -> None:

        repo = FailingMessageRepository()
        repo.fail_script.append(StoreUnavailableError("runtime.db down"))
        core, telephony, voice, _, runtime = make_message_core(messages=repo)
        core.start()
        session, backend = self.drive_to_confirmation(core, voice)

        self.confirm(backend)

        self.assertEqual(repo.saves, 1)
        self.assertEqual(runtime.messages.list_all(), [])
        self.assertEqual(session.state, CallState.FALLBACK_HANDOFF)

    def test_no_success_acknowledgement_before_commit(self) -> None:

        repo = FailingMessageRepository()
        spoken_during_save: list = []
        core, _, voice, _, _ = make_message_core(messages=repo)
        core.start()
        _, backend = self.drive_to_confirmation(core, voice)
        repo.on_save = lambda: spoken_during_save.append(list(backend.spoken))

        self.confirm(backend)

        for snapshot in spoken_during_save:
            self.assertNotIn(SUCCESS_ACK, [text for text, _ in snapshot])

    def test_failed_persist_invents_no_id_and_links_nothing(self) -> None:

        repo = FailingMessageRepository()
        repo.fail_script.append(StoreUnavailableError("down"))
        core, telephony, voice, _, runtime = make_message_core(messages=repo)
        core.start()
        session, backend = self.drive_to_confirmation(core, voice)
        self.confirm(backend)
        telephony.complete_transfer("call-1", TransferResult.ACCEPTED_BY_PBX)

        session.end_call()
        summary = runtime.calls.get("call-1")
        assert summary is not None
        self.assertIsNone(summary.message_id)
        self.assertEqual(runtime.messages.list_all(), [])

    def test_fallback_accepted_after_persist_failure_hands_off(self) -> None:

        repo = FailingMessageRepository()
        repo.fail_script.append(StoreUnavailableError("down"))
        core, telephony, voice, _, runtime = make_message_core(messages=repo)
        core.start()
        session, backend = self.drive_to_confirmation(core, voice)
        self.confirm(backend)

        telephony.complete_transfer("call-1", TransferResult.ACCEPTED_BY_PBX)

        self.assertEqual(session.state, CallState.ENDED)
        summary = runtime.calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.TRANSFERRED)
        self.assertIsNone(summary.message_id)
        self.assertEqual(runtime.messages.list_all(), [])

    def test_fallback_failed_after_persist_failure_uses_exit(self) -> None:

        repo = FailingMessageRepository()
        repo.fail_script.append(StoreUnavailableError("down"))
        core, telephony, voice, _, runtime = make_message_core(messages=repo)
        core.start()
        session, backend = start_listening_call(core, voice)
        backend.deliver_action_request(StartMessageCapture())
        backend.deliver_action_request(MessageTextFinal("mensaje"))
        self.confirm(backend)
        telephony.complete_transfer("call-1", TransferResult.TRANSPORT_ERROR)

        self.assertEqual(backend.spoken[-1][0], EXIT_APOLOGY)
        self.assertEqual(session.state, CallState.ENDED)
        summary = runtime.calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.failure_category, "message_persist_failed")


class CaptureInterruptionTest(unittest.TestCase):
    def test_hangup_during_capture_discards_draft(self) -> None:
        core, telephony, voice, _, runtime = make_message_core()
        core.start()
        session, backend = start_listening_call(core, voice)
        backend.deliver_action_request(StartMessageCapture())
        backend.deliver_action_request(MessageTextFinal("texto a medio camino"))

        telephony.simulate_caller_hangup("call-1")

        self.assertEqual(session.state, CallState.ENDED)
        summary = runtime.calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.CALLER_HANGUP)
        self.assertIsNone(summary.message_id)
        self.assertEqual(runtime.messages.list_all(), [])
        self.assertNotIn(SUCCESS_ACK, [text for text, _ in backend.spoken])

    def test_hangup_during_confirmation_ignores_late_confirm(self) -> None:
        core, telephony, voice, _, runtime = make_message_core(auto_confirm=False)
        core.start()
        session, backend = start_listening_call(core, voice)
        telephony.complete_answer("call-1")
        backend.finish_playback(1)
        backend.deliver_action_request(StartMessageCapture())
        backend.deliver_action_request(MessageTextFinal("texto"))

        telephony.simulate_caller_hangup("call-1")
        self.assertEqual(session.state, CallState.TERMINATING)
        backend.deliver_action_request(MessageConfirmed())

        self.assertEqual(session.state, CallState.TERMINATING)
        telephony.complete_hangup("call-1")
        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(runtime.messages.list_all(), [])

    def test_call_deadline_during_capture_ends_without_persist(self) -> None:
        core, _, voice, clock, runtime = make_message_core()
        core.start()
        session, backend = start_listening_call(core, voice)
        backend.deliver_action_request(StartMessageCapture())
        backend.deliver_action_request(MessageTextFinal("texto"))

        clock.advance(601.0)
        core.tick()

        self.assertEqual(session.state, CallState.ENDED)
        self.assertEqual(runtime.messages.list_all(), [])

    def test_duplicate_confirmation_persists_once(self) -> None:
        core, _, voice, _, runtime = make_message_core()
        core.start()
        session, backend = start_listening_call(core, voice)
        backend.deliver_action_request(StartMessageCapture())
        backend.deliver_action_request(MessageTextFinal("texto"))

        backend.deliver_action_request(MessageConfirmed())
        backend.deliver_action_request(MessageConfirmed())

        self.assertEqual(len(runtime.messages.list_all()), 1)
        acks = [text for text, _ in backend.spoken if text == SUCCESS_ACK]
        self.assertEqual(len(acks), 1)
        self.assertEqual(session.mode, ActiveMode.LISTENING)


class CaptureExpiryTest(unittest.TestCase):
    def test_ninety_second_default_returns_to_listening(self) -> None:
        core, _, voice, clock, runtime = make_message_core()
        core.start()
        session, backend = start_listening_call(core, voice)
        backend.deliver_action_request(StartMessageCapture())

        clock.advance(91.0)
        core.tick()

        self.assertEqual(session.state, CallState.ACTIVE)
        self.assertEqual(session.mode, ActiveMode.LISTENING)
        self.assertEqual(runtime.messages.list_all(), [])

        backend.deliver_action_request(StartMessageCapture())
        backend.deliver_action_request(MessageTextFinal("texto tras expirar"))
        backend.deliver_action_request(MessageConfirmed())
        self.assertEqual(len(runtime.messages.list_all()), 1)

    def test_configurable_capture_deadline(self) -> None:
        from receptionist.policy import Limits

        core, _, voice, clock, runtime = make_message_core(
            limits=Limits(message_capture_seconds=10.0)
        )
        core.start()
        session, backend = start_listening_call(core, voice)
        backend.deliver_action_request(StartMessageCapture())

        clock.advance(9.0)
        backend.deliver_caller_speech("ruido")
        self.assertEqual(session.mode, ActiveMode.MESSAGE_CAPTURE)

        clock.advance(2.0)
        backend.deliver_caller_speech("más ruido")

        self.assertEqual(session.mode, ActiveMode.LISTENING)
        self.assertEqual(runtime.messages.list_all(), [])

    def test_second_start_while_capturing_is_ignored(self) -> None:
        core, _, voice, clock, _ = make_message_core()
        core.start()
        session, backend = start_listening_call(core, voice)
        backend.deliver_action_request(StartMessageCapture())

        clock.advance(5.0)
        backend.deliver_action_request(StartMessageCapture())

        self.assertEqual(session.mode, ActiveMode.MESSAGE_CAPTURE)


class TransferFailureCategoryTest(unittest.TestCase):
    def test_transfer_fallback_marks_failure_category(self) -> None:
        core, telephony, voice, _, runtime = make_message_core()
        core.start()
        session, backend = start_listening_call(core, voice)
        from receptionist.boundaries import TransferRequest

        backend.deliver_action_request(TransferRequest("ventas"))
        telephony.complete_transfer("call-1", TransferResult.TIMEOUT)
        telephony.complete_transfer("call-1", TransferResult.ACCEPTED_BY_PBX)

        summary = runtime.calls.get("call-1")
        assert summary is not None
        self.assertEqual(summary.outcome, CallOutcome.TRANSFERRED)
        self.assertEqual(summary.failure_category, "transfer_failed")
        self.assertEqual(summary.handoff_destination_id, "recepcion")


class NoAudioPersistenceTest(unittest.TestCase):
    def test_runtime_stores_hold_no_audio_blobs_or_files(self) -> None:
        import glob
        import os

        core, telephony, voice, _, runtime = make_message_core()
        core.start()
        session, backend = start_listening_call(core, voice)
        backend.deliver_caller_speech("una consulta con contenido hablado")
        backend.deliver_response("una respuesta hablada.", session.current_turn)
        backend.deliver_action_request(StartMessageCapture())
        backend.deliver_action_request(MessageTextFinal("mensaje hablado"))
        backend.deliver_action_request(MessageConfirmed())
        session.end_call()

        for record in runtime.messages.list_all():
            for value in record.__dict__.values():
                self.assertNotIsInstance(value, (bytes, bytearray))
        for summary in runtime.calls.list_all():
            for value in summary.__dict__.values():
                self.assertNotIsInstance(value, (bytes, bytearray))
        for entry in runtime.transcripts.entries_for("call-1"):
            self.assertNotIsInstance(entry.text, (bytes, bytearray))
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertEqual(glob.glob(os.path.join(repo_root, "**", "*.wav"), recursive=True), [])
        self.assertEqual(glob.glob(os.path.join(repo_root, "**", "*.pcm"), recursive=True), [])


class PrivacyBoundaryTest(unittest.TestCase):
    def test_sensitive_speech_lives_only_in_authorized_stores(self) -> None:
        secret = "tarjeta 1234-5678"
        core, _, voice, _, runtime = make_message_core()
        core.start()
        session, backend = start_listening_call(core, voice)
        backend.deliver_caller_speech(f"mi {secret} por favor")
        backend.deliver_action_request(StartMessageCapture())
        backend.deliver_action_request(MessageTextFinal(f"pago con {secret}"))
        backend.deliver_action_request(MessageConfirmed())
        session.end_call()

        message_texts = [m.text for m in runtime.messages.list_all()]
        self.assertTrue(any(secret in text for text in message_texts))
        for event in runtime.audit.list_all():
            self.assertNotIn(secret, event.destination)
            self.assertNotIn(secret, event.detail)
        for summary in runtime.calls.list_all():
            blob = " ".join(
                str(value) for value in summary.__dict__.values() if value is not None
            )
            self.assertNotIn(secret, blob)
        self.assertEqual(runtime.transcripts.entries_for("call-1"), [])


class CallerIdHandlingTest(unittest.TestCase):
    def test_caller_id_does_not_change_policy_treatment(self) -> None:
        from receptionist.boundaries import TransferRequest

        for caller_id, caller_name in [("+34910000001", "García"), ("unknown", None)]:
            with self.subTest(caller_id=caller_id):
                core, telephony, voice, _, runtime = make_message_core()
                core.start()
                session, backend = start_listening_call(
                    core, voice, caller_id=caller_id, caller_name=caller_name
                )
                backend.deliver_action_request(TransferRequest("ventas"))

                self.assertEqual(session.state, CallState.TRANSFER_HANDOFF)
                self.assertEqual(telephony.transfers, [("call-1", "SIP/201")])
                session.end_call()
                summary = runtime.calls.get("call-1")
                assert summary is not None
                self.assertEqual(summary.caller_id, caller_id)
                self.assertEqual(summary.caller_name, caller_name)


class OperatorQueryFlowTest(unittest.TestCase):
    def test_lookup_by_time_and_caller_id(self) -> None:
        core, _, voice, clock, runtime = make_message_core()
        core.start()
        s1, _ = start_listening_call(core, voice, caller_id="+34910000001")
        s1.end_call()
        clock.advance(1000.0)
        _, _ = start_listening_call(core, voice, caller_id="+34910000002")
        core.get_session("call-2").end_call()
        clock.advance(1000.0)
        _, _ = start_listening_call(core, voice, caller_id="+34910000001")
        core.get_session("call-3").end_call()

        by_caller = runtime.calls.find_by_caller("+34910000001")
        self.assertEqual([s.call_id for s in by_caller], ["call-1", "call-3"])
        in_range = runtime.calls.find_in_range(500.0, 1500.0)
        self.assertEqual([s.call_id for s in in_range], ["call-2"])


class ConfigRuntimeSeparationTest(unittest.TestCase):
    def test_caller_data_never_touches_configuration(self) -> None:
        from receptionist.config import ConfigService, InMemoryConfigRepository
        from receptionist.core import ReceptionistCore
        from receptionist.persistence import RuntimeStorage
        from receptionist.policy import Limits, PolicyEngine, RetentionPolicy
        from fakes import FakePolicy, FakeTelephony, FakeVoiceBackend

        config_repo = InMemoryConfigRepository(
            {"greeting": "Hola", "language": "es"}
        )
        config_keys = ("greeting", "language", "transcripts_enabled")
        before = {key: config_repo.get(key) for key in config_keys}
        clock = FakeClock()
        voice = FakeVoiceBackend()
        core = ReceptionistCore(
            telephony=FakeTelephony(),
            voice=voice,
            config_service=ConfigService(config_repo),
            policy=FakePolicy(),
            clock=clock,
            policy_engine=PolicyEngine(destinations={}, fallback_id="none", limits=Limits()),
            runtime=RuntimeStorage.create(clock=clock),
            retention=RetentionPolicy(),
        )
        core.start()
        session, backend = start_listening_call(core, voice, "+34910000999")
        backend.deliver_caller_speech("datos sensibles del caller")
        backend.deliver_action_request(StartMessageCapture())
        backend.deliver_action_request(MessageTextFinal("mensaje sensible"))
        backend.deliver_action_request(MessageConfirmed())
        session.end_call()

        self.assertEqual(
            {key: config_repo.get(key) for key in config_keys}, before
        )


class RetentionFlowTest(unittest.TestCase):
    def test_prune_via_core_is_idempotent(self) -> None:
        core, _, voice, clock, runtime = make_message_core()
        core.start()
        session, backend = start_listening_call(core, voice)
        backend.deliver_action_request(StartMessageCapture())
        backend.deliver_action_request(MessageTextFinal("texto"))
        backend.deliver_action_request(MessageConfirmed())
        session.end_call()

        clock.advance(400 * 86400.0)
        first = core.prune_expired()
        second = core.prune_expired()

        self.assertEqual(first["messages"], 1)
        self.assertEqual(first["calls"], 1)
        self.assertEqual(second, {"messages": 0, "calls": 0, "transcripts": 0, "audit": 0})
        self.assertEqual(runtime.messages.list_all(), [])
        self.assertEqual(runtime.calls.list_all(), [])


class TranscriptsEnabledTest(unittest.TestCase):
    ENABLED_CONFIG = {
        "greeting": "Bienvenido, ¿en qué puedo ayudarle?",
        "language": "es",
        "transcripts_enabled": "true",
    }

    def test_enabled_transcripts_record_text_only(self) -> None:
        core, _, voice, _, runtime = make_message_core(
            config_values=dict(self.ENABLED_CONFIG)
        )
        core.start()
        session, backend = start_listening_call(core, voice)
        backend.deliver_caller_speech("una consulta")
        backend.deliver_response("una respuesta.", session.current_turn)
        backend.finish_playback(session.current_turn)
        backend.deliver_action_request(StartMessageCapture())
        backend.deliver_action_request(MessageTextFinal("un mensaje"))
        backend.deliver_action_request(MessageConfirmed())

        entries = runtime.transcripts.entries_for("call-1")
        self.assertEqual(
            [(e.speaker, e.text) for e in entries],
            [
                ("assistant", "Bienvenido, ¿en qué puedo ayudarle?"),
                ("caller", "una consulta"),
                ("assistant", "una respuesta."),
                ("assistant", READBACK_TEMPLATE.format(text="un mensaje")),
                ("assistant", SUCCESS_ACK),
            ],
        )
        for entry in entries:
            self.assertIsInstance(entry.call_id, str)
            self.assertIsInstance(entry.timestamp, float)
            self.assertIsInstance(entry.speaker, str)
            self.assertIsInstance(entry.text, str)

    def test_routing_decisions_ignore_transcript_content(self) -> None:
        from receptionist.boundaries import TransferRequest

        core, telephony, voice, _, _ = make_message_core(
            config_values=dict(self.ENABLED_CONFIG)
        )
        core.start()
        _, backend = start_listening_call(core, voice)

        backend.deliver_action_request(TransferRequest("nadie"))

        self.assertEqual(telephony.transfers, [])
        self.assertEqual(telephony.answered, ["call-1"])


if __name__ == "__main__":
    unittest.main()
