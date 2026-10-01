"""Durable SQLite stores behind the project-owned repository boundaries.

config.db side: SQLiteConfigRepository (read through ConfigService).
runtime.db side: SQLiteCallRepository, SQLiteMessageRepository,
SQLiteTranscriptStore, SQLiteAuditLog. Same file path means the same
store; different paths are physically separate stores. stdlib sqlite3
only. Schema migrations and upgrade orchestration belong to #29.

Error normalization: every sqlite3.Error raised inside these adapters is
mapped to the persistence boundary hierarchy, so the core only ever sees
normalized errors and never raw driver exceptions:

- sqlite3.OperationalError (busy/locked, I/O) -> TransientStoreError.
  Retryable: the failed statement rolled back, nothing was committed.
- any other sqlite3.Error -> StoreUnavailableError.
  Not retried here: route to PBX fallback.
"""

from __future__ import annotations

import sqlite3

from receptionist.boundaries import (
    AuditDecision,
    AuditEvent,
    CallOutcome,
    CallSummary,
    Clock,
    FaqEntry,
    KnowledgeChunk,
    KnowledgeError,
    MessageDraft,
    MessageRecord,
    StoreUnavailableError,
    TranscriptEntry,
    TransientStoreError,
    TransferResult,
)


def _normalize(error: sqlite3.Error) -> Exception:
    if isinstance(error, sqlite3.OperationalError):
        return TransientStoreError(str(error))
    return StoreUnavailableError(str(error))


class SQLiteConfigRepository:
    """Read-only config.db access. Seeding happens outside this boundary."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        try:
            with self._conn:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS config("
                    "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
        except sqlite3.Error as error:
            raise _normalize(error) from error

    def get(self, key: str) -> str | None:
        try:
            row = self._conn.execute(
                "SELECT value FROM config WHERE key = ?", (key,)
            ).fetchone()
        except sqlite3.Error as error:
            raise _normalize(error) from error
        return row[0] if row is not None else None


class SQLiteCallRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        try:
            with self._conn:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS calls("
                    "call_id TEXT PRIMARY KEY, caller_id TEXT NOT NULL, "
                    "caller_name TEXT, started_at REAL NOT NULL, "
                    "ended_at REAL NOT NULL, outcome TEXT NOT NULL, "
                    "turn_count INTEGER NOT NULL, handoff_destination_id TEXT, "
                    "message_id TEXT, failure_category TEXT)"
                )
        except sqlite3.Error as error:
            raise _normalize(error) from error

    def save(self, summary: CallSummary) -> None:
        # Plain INSERT: a call_id is immutable history identity. A collision
        # raises (normalized to StoreUnavailableError), never replaces.
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO calls(call_id, caller_id, caller_name, "
                    "started_at, ended_at, outcome, turn_count, "
                    "handoff_destination_id, message_id, failure_category) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        summary.call_id,
                        summary.caller_id,
                        summary.caller_name,
                        summary.started_at,
                        summary.ended_at,
                        summary.outcome.value,
                        summary.turn_count,
                        summary.handoff_destination_id,
                        summary.message_id,
                        summary.failure_category,
                    ),
                )
        except sqlite3.Error as error:
            raise _normalize(error) from error

    def get(self, call_id: str) -> CallSummary | None:
        try:
            row = self._conn.execute(
                "SELECT call_id, caller_id, caller_name, started_at, ended_at, "
                "outcome, turn_count, handoff_destination_id, message_id, "
                "failure_category FROM calls WHERE call_id = ?",
                (call_id,),
            ).fetchone()
        except sqlite3.Error as error:
            raise _normalize(error) from error
        return _decode_call(row) if row is not None else None

    def list_all(self) -> list[CallSummary]:
        try:
            rows = self._conn.execute(
                "SELECT call_id, caller_id, caller_name, started_at, ended_at, "
                "outcome, turn_count, handoff_destination_id, message_id, "
                "failure_category FROM calls"
            ).fetchall()
        except sqlite3.Error as error:
            raise _normalize(error) from error
        return [_decode_call(row) for row in rows]

    def find_by_caller(self, caller_id: str) -> list[CallSummary]:
        try:
            rows = self._conn.execute(
                "SELECT call_id, caller_id, caller_name, started_at, ended_at, "
                "outcome, turn_count, handoff_destination_id, message_id, "
                "failure_category FROM calls WHERE caller_id = ?",
                (caller_id,),
            ).fetchall()
        except sqlite3.Error as error:
            raise _normalize(error) from error
        return [_decode_call(row) for row in rows]

    def find_in_range(self, start: float, end: float) -> list[CallSummary]:
        try:
            rows = self._conn.execute(
                "SELECT call_id, caller_id, caller_name, started_at, ended_at, "
                "outcome, turn_count, handoff_destination_id, message_id, "
                "failure_category FROM calls "
                "WHERE started_at >= ? AND started_at <= ?",
                (start, end),
            ).fetchall()
        except sqlite3.Error as error:
            raise _normalize(error) from error
        return [_decode_call(row) for row in rows]

    def prune_before(self, cutoff: float) -> int:
        try:
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM calls WHERE started_at < ?", (cutoff,)
                )
                return cursor.rowcount
        except sqlite3.Error as error:
            raise _normalize(error) from error


def _decode_call(row: tuple) -> CallSummary:
    return CallSummary(
        call_id=row[0],
        caller_id=row[1],
        caller_name=row[2],
        started_at=row[3],
        ended_at=row[4],
        outcome=CallOutcome(row[5]),
        turn_count=row[6],
        handoff_destination_id=row[7],
        message_id=row[8],
        failure_category=row[9],
    )


class SQLiteMessageRepository:
    """Confirmed messages. One transaction per save: a retryable failure
    implies nothing was committed, so bounded retries cannot duplicate."""

    def __init__(self, conn: sqlite3.Connection, clock: Clock) -> None:
        self._conn = conn
        self._clock = clock
        try:
            with self._conn:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS messages("
                    "seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "id TEXT UNIQUE NOT NULL, call_id TEXT NOT NULL, "
                    "caller_id TEXT NOT NULL, caller_name TEXT, "
                    "text TEXT NOT NULL, created_at REAL NOT NULL)"
                )
        except sqlite3.Error as error:
            raise _normalize(error) from error

    def save(self, draft: MessageDraft) -> MessageRecord:
        created_at = self._clock.now()
        try:
            with self._conn:
                cursor = self._conn.execute(
                    "INSERT INTO messages(id, call_id, caller_id, caller_name, "
                    "text, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        f"msg-pending",
                        draft.call_id,
                        draft.caller_id,
                        draft.caller_name,
                        draft.text,
                        created_at,
                    ),
                )
                message_id = f"msg-{cursor.lastrowid}"
                self._conn.execute(
                    "UPDATE messages SET id = ? WHERE seq = ?",
                    (message_id, cursor.lastrowid),
                )
        except sqlite3.Error as error:
            raise _normalize(error) from error
        return MessageRecord(
            id=message_id,
            call_id=draft.call_id,
            caller_id=draft.caller_id,
            caller_name=draft.caller_name,
            text=draft.text,
            created_at=created_at,
        )

    def get(self, message_id: str) -> MessageRecord | None:
        try:
            row = self._conn.execute(
                "SELECT id, call_id, caller_id, caller_name, text, created_at "
                "FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
        except sqlite3.Error as error:
            raise _normalize(error) from error
        return _decode_message(row) if row is not None else None

    def list_all(self) -> list[MessageRecord]:
        try:
            rows = self._conn.execute(
                "SELECT id, call_id, caller_id, caller_name, text, created_at "
                "FROM messages"
            ).fetchall()
        except sqlite3.Error as error:
            raise _normalize(error) from error
        return [_decode_message(row) for row in rows]

    def prune_before(self, cutoff: float) -> int:
        try:
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM messages WHERE created_at < ?", (cutoff,)
                )
                return cursor.rowcount
        except sqlite3.Error as error:
            raise _normalize(error) from error


def _decode_message(row: tuple) -> MessageRecord:
    return MessageRecord(
        id=row[0],
        call_id=row[1],
        caller_id=row[2],
        caller_name=row[3],
        text=row[4],
        created_at=row[5],
    )


class SQLiteTranscriptStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        try:
            with self._conn:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS transcripts("
                    "call_id TEXT NOT NULL, timestamp REAL NOT NULL, "
                    "speaker TEXT NOT NULL, text TEXT NOT NULL)"
                )
        except sqlite3.Error as error:
            raise _normalize(error) from error

    def append(self, entry: TranscriptEntry) -> None:
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO transcripts(call_id, timestamp, speaker, text) "
                    "VALUES (?, ?, ?, ?)",
                    (entry.call_id, entry.timestamp, entry.speaker, entry.text),
                )
        except sqlite3.Error as error:
            raise _normalize(error) from error

    def entries_for(self, call_id: str) -> list[TranscriptEntry]:
        try:
            rows = self._conn.execute(
                "SELECT call_id, timestamp, speaker, text FROM transcripts "
                "WHERE call_id = ?",
                (call_id,),
            ).fetchall()
        except sqlite3.Error as error:
            raise _normalize(error) from error
        return [
            TranscriptEntry(call_id=row[0], timestamp=row[1], speaker=row[2], text=row[3])
            for row in rows
        ]

    def prune_before(self, cutoff: float) -> int:
        try:
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM transcripts WHERE timestamp < ?", (cutoff,)
                )
                return cursor.rowcount
        except sqlite3.Error as error:
            raise _normalize(error) from error


class SQLiteAuditLog:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        try:
            with self._conn:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS audit("
                    "call_id TEXT NOT NULL, timestamp REAL NOT NULL, "
                    "action TEXT NOT NULL, destination TEXT NOT NULL, "
                    "decision TEXT NOT NULL, result TEXT, detail TEXT NOT NULL)"
                )
        except sqlite3.Error as error:
            raise _normalize(error) from error

    def record(self, event: AuditEvent) -> None:
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO audit(call_id, timestamp, action, destination, "
                    "decision, result, detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.call_id,
                        event.timestamp,
                        event.action,
                        event.destination,
                        event.decision.value,
                        event.result.value if event.result is not None else None,
                        event.detail,
                    ),
                )
        except sqlite3.Error as error:
            raise _normalize(error) from error

    def list_all(self) -> list[AuditEvent]:
        try:
            rows = self._conn.execute(
                "SELECT call_id, timestamp, action, destination, decision, "
                "result, detail FROM audit"
            ).fetchall()
        except sqlite3.Error as error:
            raise _normalize(error) from error
        return [_decode_audit(row) for row in rows]

    def prune_before(self, cutoff: float) -> int:
        try:
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM audit WHERE timestamp < ?", (cutoff,)
                )
                return cursor.rowcount
        except sqlite3.Error as error:
            raise _normalize(error) from error


def _decode_audit(row: tuple) -> AuditEvent:
    return AuditEvent(
        call_id=row[0],
        timestamp=row[1],
        action=row[2],
        destination=row[3],
        decision=AuditDecision(row[4]),
        result=TransferResult(row[5]) if row[5] is not None else None,
        detail=row[6],
    )


class SQLiteFaqSource:
    """Structured FAQ knowledge in SQLite. Operator-written config data:
    ``save`` is an explicit operator write (replace by id), while reads
    serve enabled entries only as provenance-preserving chunks.

    Lives in a knowledge/config-side database file chosen by the operator
    (e.g. knowledge.db or config.db), never in runtime.db: caller history
    and retention pruning must not touch knowledge. Chunks carry
    question + answer so retrieval matches either; the question is also
    the chunk title. Failures raise KnowledgeError, never raw sqlite3.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        try:
            with self._conn:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS faq("
                    "id TEXT PRIMARY KEY, question TEXT NOT NULL, "
                    "answer TEXT NOT NULL, keywords TEXT NOT NULL DEFAULT '', "
                    "enabled INTEGER NOT NULL DEFAULT 1)"
                )
        except sqlite3.Error as error:
            raise KnowledgeError(f"faq store unavailable: {error}") from error

    def save(self, entry: FaqEntry) -> None:
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO faq(id, question, answer, keywords, "
                    "enabled) VALUES (?, ?, ?, ?, ?)",
                    (
                        entry.id,
                        entry.question,
                        entry.answer,
                        entry.keywords,
                        1 if entry.enabled else 0,
                    ),
                )
        except sqlite3.Error as error:
            raise KnowledgeError(f"faq store unavailable: {error}") from error

    def chunks(self) -> list[KnowledgeChunk]:
        try:
            rows = self._conn.execute(
                "SELECT id, question, answer, keywords FROM faq WHERE enabled != 0"
            ).fetchall()
        except sqlite3.Error as error:
            raise KnowledgeError(f"faq store unavailable: {error}") from error
        return [
            KnowledgeChunk(
                source_id="faq",
                chunk_id=f"faq:{row[0]}",
                text=f"{row[1]}\n{row[2]}\n{row[3]}",
                title=row[1],
                origin=f"faq:{row[0]}",
            )
            for row in rows
        ]

    @property
    def source_id(self) -> str:
        return "faq"

    def close(self) -> None:
        self._conn.close()
