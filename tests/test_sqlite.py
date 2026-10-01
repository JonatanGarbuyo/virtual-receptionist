"""Durable SQLite stores behind the repository boundaries (#20).

config.db and runtime.db are physically separate files holding the same
contracts the in-memory stand-ins serve. stdlib sqlite3 only. Tests use
temporary directories; nothing here touches the network or the wall clock.
"""

import os
import sqlite3
import tempfile
import unittest

from receptionist.boundaries import (
    AuditDecision,
    AuditEvent,
    CallOutcome,
    CallSummary,
    MessageDraft,
    StoreUnavailableError,
    TranscriptEntry,
    TransientStoreError,
    TransferResult,
)
from receptionist.sqlite_storage import (
    SQLiteAuditLog,
    SQLiteCallRepository,
    SQLiteConfigRepository,
    SQLiteMessageRepository,
    SQLiteTranscriptStore,
)

from fakes import FakeClock


def connect(path):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


class SabotagedConnection(sqlite3.Connection):
    """Fails the next execute() with a scripted sqlite3 error, once."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_with = None

    def execute(self, *args, **kwargs):
        if self.fail_with is not None:
            error, self.fail_with = self.fail_with, None
            raise error
        return super().execute(*args, **kwargs)


def sabotaged_connect(path):
    return sqlite3.connect(path, factory=SabotagedConnection)


class ConfigDatabaseTest(unittest.TestCase):
    def test_config_roundtrip_survives_reopen(self) -> None:
        from receptionist.config import ConfigService

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.db")
            conn = connect(path)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO config(key, value) VALUES (?, ?)", ("greeting", "Hola")
            )
            conn.execute(
                "INSERT INTO config(key, value) VALUES (?, ?)", ("language", "es")
            )
            conn.commit()
            service = ConfigService(SQLiteConfigRepository(conn))
            self.assertEqual(service.get_greeting(), "Hola")
            self.assertEqual(service.missing_required(), [])
            conn.close()

            reopened = connect(path)
            reread = ConfigService(SQLiteConfigRepository(reopened))
            self.assertEqual(reread.get_greeting(), "Hola")
            self.assertEqual(reread.get_language(), "es")
            reopened.close()

    def test_config_and_runtime_are_separate_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.db")
            runtime_path = os.path.join(tmp, "runtime.db")
            config_conn = connect(config_path)
            config_conn.execute(
                "CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            config_conn.execute(
                "INSERT INTO config(key, value) VALUES (?, ?)", ("greeting", "Hola")
            )
            config_conn.commit()

            runtime_conn = connect(runtime_path)
            SQLiteCallRepository(runtime_conn)
            tables = [
                row[0]
                for row in runtime_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            ]
            self.assertNotIn("config", tables)
            config_conn.close()
            runtime_conn.close()
            self.assertNotEqual(config_path, runtime_path)


class MessageDatabaseTest(unittest.TestCase):
    def test_message_survives_reopen_with_correct_timestamp(self) -> None:
        clock = FakeClock(start=1_800_000_000.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "runtime.db")
            conn = connect(path)
            repo = SQLiteMessageRepository(conn, clock=clock)
            clock.advance(12.0)
            saved = repo.save(
                MessageDraft(call_id="call-1", caller_id="+3491", caller_name="Luz", text="hola")
            )
            self.assertEqual(saved.id, "msg-1")
            self.assertEqual(saved.created_at, 1_800_000_012.0)
            conn.close()

            reopened = connect(path)
            reread = SQLiteMessageRepository(reopened, clock=clock)
            fetched = reread.get("msg-1")
            assert fetched is not None
            self.assertEqual(fetched.text, "hola")
            self.assertEqual(fetched.caller_name, "Luz")
            self.assertEqual(fetched.created_at, 1_800_000_012.0)
            third = reread.save(
                MessageDraft(call_id="call-2", caller_id="+3492", caller_name=None, text="chau")
            )
            self.assertEqual(third.id, "msg-2")
            reopened.close()

    def test_failed_save_commits_nothing(self) -> None:
        clock = FakeClock(start=1_800_000_000.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "runtime.db")
            conn = sabotaged_connect(path)
            repo = SQLiteMessageRepository(conn, clock=clock)
            conn.fail_with = sqlite3.OperationalError("database is locked")

            with self.assertRaises(TransientStoreError):
                repo.save(
                    MessageDraft(call_id="c", caller_id="+1", caller_name=None, text="x")
                )
            self.assertEqual(repo.list_all(), [])


class ErrorNormalizationTest(unittest.TestCase):
    def test_operational_error_is_transient(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn = sabotaged_connect(os.path.join(tmp, "runtime.db"))
            repo = SQLiteMessageRepository(conn, clock=FakeClock())
            conn.fail_with = sqlite3.OperationalError("locked")

            with self.assertRaises(TransientStoreError):
                repo.save(
                    MessageDraft(call_id="c", caller_id="+1", caller_name=None, text="x")
                )

    def test_integrity_error_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn = sabotaged_connect(os.path.join(tmp, "runtime.db"))
            repo = SQLiteMessageRepository(conn, clock=FakeClock())
            conn.fail_with = sqlite3.IntegrityError("constraint failed")

            with self.assertRaises(StoreUnavailableError):
                repo.save(
                    MessageDraft(call_id="c", caller_id="+1", caller_name=None, text="x")
                )

    def test_closed_connection_is_unavailable_not_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "runtime.db")
            conn = connect(path)
            repo = SQLiteMessageRepository(conn, clock=FakeClock())
            conn.close()

            with self.assertRaises(StoreUnavailableError) as caught:
                repo.save(
                    MessageDraft(call_id="c", caller_id="+1", caller_name=None, text="x")
                )
            self.assertNotIsInstance(caught.exception, sqlite3.Error)


class RuntimeTablesTest(unittest.TestCase):
    def test_calls_queries_and_prune(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = SQLiteCallRepository(connect(os.path.join(tmp, "runtime.db")))
            repo.save(
                CallSummary(
                    call_id="call-1",
                    caller_id="+34910000001",
                    started_at=1000.0,
                    ended_at=1010.0,
                    outcome=CallOutcome.TRANSFERRED,
                    turn_count=3,
                    caller_name="García",
                    handoff_destination_id="ventas",
                    message_id=None,
                    failure_category="transfer_failed",
                )
            )
            repo.save(
                CallSummary(
                    call_id="call-2",
                    caller_id="+34910000002",
                    started_at=2000.0,
                    ended_at=2010.0,
                    outcome=CallOutcome.COMPLETED,
                    turn_count=1,
                )
            )

            self.assertEqual(
                [s.call_id for s in repo.find_by_caller("+34910000001")], ["call-1"]
            )
            self.assertEqual(
                [s.call_id for s in repo.find_in_range(1000.0, 2000.0)],
                ["call-1", "call-2"],
            )
            fetched = repo.get("call-1")
            assert fetched is not None
            self.assertEqual(fetched.caller_name, "García")
            self.assertEqual(fetched.handoff_destination_id, "ventas")
            self.assertEqual(fetched.failure_category, "transfer_failed")
            self.assertEqual(repo.prune_before(1500.0), 1)
            self.assertEqual([s.call_id for s in repo.list_all()], ["call-2"])

    def test_transcripts_roundtrip_and_prune(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteTranscriptStore(connect(os.path.join(tmp, "runtime.db")))
            store.append(
                TranscriptEntry(call_id="c", timestamp=100.0, speaker="caller", text="a")
            )
            store.append(
                TranscriptEntry(call_id="c", timestamp=9000.0, speaker="caller", text="b")
            )
            self.assertEqual([e.text for e in store.entries_for("c")], ["a", "b"])
            self.assertEqual(store.prune_before(1000.0), 1)
            self.assertEqual([e.text for e in store.entries_for("c")], ["b"])

    def test_audit_roundtrip_and_prune(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = SQLiteAuditLog(connect(os.path.join(tmp, "runtime.db")))
            log.record(
                AuditEvent(
                    timestamp=100.0,
                    call_id="c",
                    action="transfer",
                    destination="ventas",
                    decision=AuditDecision.ALLOWED,
                    result=TransferResult.ACCEPTED_BY_PBX,
                    detail="target SIP/201",
                )
            )
            events = log.list_all()
            self.assertEqual(len(events), 1)
            self.assertIs(events[0].result, TransferResult.ACCEPTED_BY_PBX)
            self.assertEqual(log.prune_before(1000.0), 1)
            self.assertEqual(log.list_all(), [])

    def test_runtime_prune_never_touches_config(self) -> None:
        from receptionist.config import ConfigService

        with tempfile.TemporaryDirectory() as tmp:
            config_conn = connect(os.path.join(tmp, "config.db"))
            config_conn.execute(
                "CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            config_conn.execute(
                "INSERT INTO config(key, value) VALUES (?, ?)", ("greeting", "Hola")
            )
            config_conn.commit()

            runtime_conn = connect(os.path.join(tmp, "runtime.db"))
            calls = SQLiteCallRepository(runtime_conn)
            calls.save(
                CallSummary(
                    call_id="old",
                    caller_id="+1",
                    started_at=100.0,
                    ended_at=110.0,
                    outcome=CallOutcome.COMPLETED,
                    turn_count=1,
                )
            )
            self.assertEqual(calls.prune_before(1000.0), 1)

            service = ConfigService(SQLiteConfigRepository(config_conn))
            self.assertEqual(service.get_greeting(), "Hola")
            config_conn.close()
            runtime_conn.close()


class SessionOnSqliteTest(unittest.TestCase):
    def test_full_message_flow_persists_across_reopen(self) -> None:
        from receptionist.boundaries import (
            MessageConfirmed,
            MessageTextFinal,
            StartMessageCapture,
        )
        from receptionist.call_session import ActiveMode
        from receptionist.config import ConfigService, InMemoryConfigRepository
        from receptionist.core import ReceptionistCore
        from receptionist.persistence import RuntimeStorage
        from receptionist.policy import Limits, PolicyEngine, RetentionPolicy
        from fakes import FakeCallIds, FakeClock, FakePolicy, FakeTelephony, FakeVoiceBackend

        with tempfile.TemporaryDirectory() as tmp:
            runtime_conn = connect(os.path.join(tmp, "runtime.db"))
            clock = FakeClock()
            voice = FakeVoiceBackend()
            runtime = RuntimeStorage(
                calls=SQLiteCallRepository(runtime_conn),
                messages=SQLiteMessageRepository(runtime_conn, clock=clock),
                transcripts=SQLiteTranscriptStore(runtime_conn),
                audit=SQLiteAuditLog(runtime_conn),
            )
            core = ReceptionistCore(
                telephony=FakeTelephony(),
                voice=voice,
                config_service=ConfigService(
                    InMemoryConfigRepository({"greeting": "Hola", "language": "es"})
                ),
                policy=FakePolicy(),
                clock=clock,
                policy_engine=PolicyEngine(
                    destinations={}, fallback_id="none", limits=Limits()
                ),
                runtime=runtime,
                retention=RetentionPolicy(),
                call_ids=FakeCallIds(),
            )
            core.start()
            session = core.incoming_call("+34910000001", caller_name="García")
            backend = voice.sessions[session.call_id]
            backend.finish_playback(1)
            backend.deliver_action_request(StartMessageCapture())
            backend.deliver_action_request(MessageTextFinal("texto durable"))
            backend.deliver_action_request(MessageConfirmed())
            self.assertEqual(session.mode, ActiveMode.LISTENING)
            session.end_call()
            runtime_conn.close()

            reopened = connect(os.path.join(tmp, "runtime.db"))
            messages = SQLiteMessageRepository(reopened, clock=clock)
            calls = SQLiteCallRepository(reopened)
            records = messages.list_all()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].text, "texto durable")
            summary = calls.get("call-1")
            assert summary is not None
            self.assertEqual(summary.message_id, records[0].id)
            self.assertEqual(summary.caller_name, "García")
            reopened.close()


def build_core(runtime_conn, clock, voice, call_ids, destinations=None):
    from receptionist.config import ConfigService, InMemoryConfigRepository
    from receptionist.core import ReceptionistCore
    from receptionist.persistence import RuntimeStorage
    from receptionist.policy import Limits, PolicyEngine, RetentionPolicy
    from fakes import FakePolicy, FakeTelephony

    return ReceptionistCore(
        telephony=FakeTelephony(),
        voice=voice,
        config_service=ConfigService(
            InMemoryConfigRepository({"greeting": "Hola", "language": "es"})
        ),
        policy=FakePolicy(),
        clock=clock,
        policy_engine=PolicyEngine(
            destinations=dict(destinations or {}),
            fallback_id="none",
            limits=Limits(),
        ),
        runtime=RuntimeStorage(
            calls=SQLiteCallRepository(runtime_conn),
            messages=SQLiteMessageRepository(runtime_conn, clock=clock),
            transcripts=SQLiteTranscriptStore(runtime_conn),
            audit=SQLiteAuditLog(runtime_conn),
        ),
        retention=RetentionPolicy(),
        call_ids=call_ids,
    )


class CallIdUniquenessTest(unittest.TestCase):
    def test_deterministic_generator_for_tests(self) -> None:
        from fakes import FakeCallIds

        ids = FakeCallIds()
        self.assertEqual(ids.next_id(), "call-1")
        self.assertEqual(ids.next_id(), "call-2")

    def test_call_ids_unique_across_restarts(self) -> None:
        from receptionist.boundaries import (
            MessageConfirmed,
            MessageTextFinal,
            StartMessageCapture,
        )
        from receptionist.ids import UuidCallIds
        from fakes import FakeClock, FakeVoiceBackend

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "runtime.db")
            clock = FakeClock()
            first_ids: list[str] = []

            for n in range(2):
                conn = connect(path)
                voice = FakeVoiceBackend()
                core = build_core(conn, clock, voice, UuidCallIds())
                core.start()
                session = core.incoming_call(f"+3491000000{n}")
                backend = voice.sessions[session.call_id]
                backend.finish_playback(1)
                backend.deliver_action_request(StartMessageCapture())
                backend.deliver_action_request(MessageTextFinal(f"mensaje {n}"))
                backend.deliver_action_request(MessageConfirmed())
                session.end_call()
                first_ids.append(session.call_id)
                conn.close()

            self.assertNotEqual(first_ids[0], first_ids[1])
            check = connect(path)
            calls = SQLiteCallRepository(check)
            summaries = calls.list_all()
            self.assertEqual(len(summaries), 2)
            stored_ids = sorted(s.call_id for s in summaries)
            self.assertEqual(stored_ids, sorted(first_ids))
            messages = SQLiteMessageRepository(check, clock=clock)
            records = messages.list_all()
            self.assertEqual(len(records), 2)
            for record in records:
                self.assertIn(record.call_id, stored_ids)
                linked = calls.get(record.call_id)
                assert linked is not None
                self.assertEqual(linked.message_id, record.id)
            check.close()


if __name__ == "__main__":
    unittest.main()
