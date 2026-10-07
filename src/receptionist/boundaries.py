"""Project-owned contracts for the ReceptionistCore / CallSession seam.

Production code depends only on these protocols, never on concrete
adapters. Tests provide deterministic fakes behind the same protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Clock(Protocol):
    """Controllable time source. Production uses wall clock; tests use a fake."""

    def now(self) -> float: ...


class CallIdGenerator(Protocol):
    """Issues call ids unique across process restarts.

    Production uses random UUIDs; tests use a deterministic counter.
    The core never invents ids itself and knows nothing about SQLite.
    """

    def next_id(self) -> str: ...


class TransferResult(Enum):
    """Normalized handoff outcome. Only these four values may cross the seam."""

    ACCEPTED_BY_PBX = "accepted_by_pbx"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    TRANSPORT_ERROR = "transport_error"


class TelephonyListener(Protocol):
    """Normalized telephony events into the application.

    Threading: the adapter may invoke these from SIP/media threads. The
    core dispatches them onto its own serialization; listener
    implementations must be thread-safe and must never run LLM/STT/TTS,
    SQLite, or network I/O inline. Callbacks are lightweight: they record
    and return.
    """

    def on_answered(self, call_id: str) -> None: ...
    def on_caller_hangup(self, call_id: str) -> None: ...
    def on_hangup_completed(self, call_id: str) -> None: ...
    def on_transfer_result(self, call_id: str, result: TransferResult) -> None: ...
    def on_caller_audio(self, call_id: str, frame: AudioFrame) -> None: ...
    def on_dtmf(self, call_id: str, digit: str) -> None: ...
    def on_remote_hold(self, call_id: str, held: bool) -> None: ...


class InboundCallHandler(Protocol):
    """Synchronous inbound-call admission seam (project-owned).

    The adapter invokes this from its SIP thread when a native INVITE
    arrives, holding the native call object internally in a pending
    slot. The handler (composition layer) admits the call through
    ``ReceptionistCore.incoming_call(...)`` and returns the application
    ``call_id`` so the adapter can bind native <-> application id.

    Re-entrancy contract: ``incoming_call`` synchronously calls back
    into the adapter (``answer``/``reject``/``blind_transfer``) for the
    returned id while this handler is still on the stack. The adapter
    therefore binds a pending native call to the id demanded by that
    re-entrant call instead of requiring the binding to pre-exist.

    Returns the application call id to bind, or ``None`` to decline at
    the SIP level (the adapter rejects the native call). Never receives
    or returns native handles, pointers, SIP dialogs, or binding
    objects: only untrusted caller metadata in and an opaque
    application id out.
    """

    def handle_incoming_call(
        self, caller_id: str, caller_name: str | None
    ) -> str | None: ...


class TelephonyRegistrationState(Enum):
    """Explicit telephony lifecycle. The service is READY only when the
    state is REGISTERED; any other state keeps or moves health away
    from READY through the project-owned status hook."""

    STARTING = "starting"
    REGISTERED = "registered"
    REGISTRATION_FAILED = "registration_failed"
    REGISTRATION_LOST = "registration_lost"
    STOPPING = "stopping"
    STOPPED = "stopped"


class TelephonyStatusListener(Protocol):
    """Project-owned registration/health hook. States only, never SIP
    internals, never credentials, never auth headers."""

    def on_registration_state(
        self, state: TelephonyRegistrationState, detail: str = ""
    ) -> None: ...


class TelephonyAdapter(Protocol):
    """Call-control + media boundary. No dial-by-URI capability exists
    on this contract: the only outbound primitive is ``blind_transfer``
    to a ``pbx_target`` already resolved/trusted by the application.

    Execution contract (all methods): synchronous, non-blocking, never
    raise vendor/native exceptions to the core. SIP failures surface as
    normalized listener events (``on_transfer_result`` with
    ``TransferResult``; hangup completion via ``on_hangup_completed``).
    Every method is safe to call from listener callbacks (no re-entrant
    deadlock) and ``send_audio``/``flush_audio`` are safe from any
    thread, including media threads.
    """

    def set_listener(self, listener: TelephonyListener) -> None: ...
    def set_inbound_handler(self, handler: InboundCallHandler | None) -> None: ...
    def set_status_listener(self, listener: TelephonyStatusListener | None) -> None: ...
    def start(self) -> None: ...
    def shutdown(self) -> None: ...
    def answer(self, call_id: str) -> None: ...
    def reject(self, call_id: str) -> None: ...
    def hangup(self, call_id: str) -> None: ...
    def blind_transfer(self, call_id: str, pbx_target: str) -> None: ...
    def send_audio(self, call_id: str, frame: AudioFrame) -> None: ...
    def flush_audio(self, call_id: str) -> None: ...
    def send_dtmf(self, call_id: str, digits: str) -> None: ...
    def hold(self, call_id: str) -> None: ...
    def resume(self, call_id: str) -> None: ...


@dataclass(frozen=True)
class TransferRequest:
    """The only transfer shape the model/requester may produce.

    A symbolic destination id resolved by policy; never a number or URI.
    """

    destination_id: str


@dataclass(frozen=True)
class StartMessageCapture:
    """Typed request to open message capture. Carries no content."""


@dataclass(frozen=True)
class MessageTextFinal:
    """Final usable message text from the voice backend (STT final)."""

    text: str


@dataclass(frozen=True)
class MessageConfirmed:
    """The caller accepted the readback. Persist only after this."""


@dataclass(frozen=True)
class MessageRejected:
    """The caller rejected the draft. Discard it."""


#: Baseline sample format for every AudioFrame crossing the seam.
#: Signed 16-bit little-endian PCM. Telephony line codecs are converted
#: to/from this form by the telephony/media adapter (#25); this
#: contract never assumes any line codec.
AUDIO_SAMPLE_FORMAT_PCM16 = "pcm16"

#: Baseline channel count: mono. Multi-channel audio is mixed down
#: before crossing the seam.
AUDIO_CHANNELS_MONO = 1

#: Hard cap on one AudioFrame's payload: transport chunks are small
#: (100 ms at 16 kHz is 3.2 KiB); anything larger is a resource bug,
#: never a legitimate turn fragment.
MAX_AUDIO_FRAME_BYTES = 1024 * 1024


@dataclass(frozen=True)
class AudioFrame:
    """One chunk of generic caller or assistant PCM.

    Project-owned media type: no codec, container, vendor, or runtime
    concepts appear here. Raw PCM bytes only, never base64, never a
    file path. Frames are transient: neither the session nor the
    backend persists them.

    ``call_id``/``turn_id`` identify ownership when known (empty/zero
    when the producer cannot know yet); ``sequence`` orders frames
    inside one turn; ``timestamp`` comes from the session clock.
    """

    pcm: bytes
    sample_rate: int
    channels: int = AUDIO_CHANNELS_MONO
    sample_format: str = AUDIO_SAMPLE_FORMAT_PCM16
    call_id: str = ""
    turn_id: int = 0
    sequence: int = 0
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.pcm, (bytes, bytearray)) or len(self.pcm) == 0:
            raise ValueError("audio frame needs non-empty PCM bytes")
        if self.sample_rate <= 0:
            raise ValueError(f"audio frame needs a sample rate, got {self.sample_rate!r}")
        if self.channels != AUDIO_CHANNELS_MONO:
            raise ValueError(f"audio frame must be mono, got {self.channels!r}")
        if self.sample_format != AUDIO_SAMPLE_FORMAT_PCM16:
            raise ValueError(
                f"audio frame must be {AUDIO_SAMPLE_FORMAT_PCM16}, "
                f"got {self.sample_format!r}"
            )
        if len(self.pcm) % 2 != 0:
            raise ValueError("pcm16 audio frame needs an even byte count")
        if len(self.pcm) > MAX_AUDIO_FRAME_BYTES:
            raise ValueError(
                f"audio frame exceeds {MAX_AUDIO_FRAME_BYTES} bytes: "
                f"{len(self.pcm)}"
            )
        if self.sequence < 0:
            raise ValueError(f"audio frame sequence must be >= 0, got {self.sequence!r}")


class CancelReason(Enum):
    """Typed app-owned cancellation reasons for assistant output.

    The backend stops synthesis/playback, discards undelivered audio,
    and invalidates the corresponding generation/attempt. Late output
    from the cancelled turn must never cross the seam afterwards.
    """

    BARGE_IN = "barge_in"
    CALLER_HANGUP = "caller_hangup"
    TURN_TIMEOUT = "turn_timeout"
    TRANSFER_HANDOFF = "transfer_handoff"
    FALLBACK_HANDOFF = "fallback_handoff"
    CALL_LIMIT = "call_limit"
    SHUTDOWN = "shutdown"


class VoiceListener(Protocol):
    """Events flowing from the voice backend into one call session.

    Attempt identity travels separately: the session hands the backend a
    per-attempt listener, so late events from a superseded attempt never
    reach these methods as current. `on_provider_failure` carries the
    normalized taxonomy below, never vendor exceptions or strings.

    `on_audio` carries assistant PCM (TTS output) as a separate event:
    audio is never hidden inside `on_response`, which stays an optional
    assistant-text sidecar (transcript/debugging/deterministic tests).
    `on_transcript_sidecar` carries observational caller text (e.g. the
    primary STT result of a cascaded turn): unlike `on_transcript`, it
    never opens a turn and never authorizes anything.

    Threading contract: backend worker threads may invoke these methods
    concurrently with media-input threads. Sessions serialize them
    internally; listener implementations behind other backends must be
    thread-safe if their backend is asynchronous.
    """

    def on_transcript(self, text: str) -> None: ...
    def on_transcript_sidecar(self, text: str) -> None: ...
    def on_response(self, turn_id: int, text: str) -> None: ...
    def on_audio(self, turn_id: int, frame: AudioFrame) -> None: ...
    def on_playback_finished(self, turn_id: int) -> None: ...
    def on_action_request(self, action: object) -> None: ...
    def on_provider_failure(self, turn_id: int, failure: ProviderFailure) -> None: ...


class ProviderFailureCategory(Enum):
    """Project-owned failure taxonomy for conversational providers.

    Future STT/LLM/TTS runtimes translate their errors into exactly
    these categories. The core only ever reasons about these values,
    never vendor exceptions, payloads, or strings.
    """

    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    INVALID_OUTPUT = "invalid_output"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


@dataclass(frozen=True)
class ProviderFailure:
    """One normalized provider failure. Carries only the category plus a
    short safe detail string: never prompts, transcripts, audio, secrets,
    auth headers, or provider payloads."""

    category: ProviderFailureCategory
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.category, ProviderFailureCategory):
            raise ValueError(f"provider failure needs a category, got {self.category!r}")


class VoiceSession(Protocol):
    """One backend voice stream for one call.

    `push_audio` buffers transient caller PCM (never persisted);
    `commit_turn` marks the app-owned end-of-user-turn and starts the
    STT -> LLM -> TTS pipeline for the buffered audio; `cancel_output`
    stops assistant output for the given app-owned reason and discards
    undelivered audio; `speak` plays a fixed application text (greeting,
    reprompt, apology) through the same output path; `close` cancels
    all session work, frees buffers/processes, and is idempotent.

    Execution contract: `push_audio`, `commit_turn`, `cancel_output`,
    `speak`, and `close` are thread-safe. `commit_turn`/`speak` enqueue
    pipeline work on the session worker and return promptly; listener
    events arrive asynchronously on that worker. Media input, hangup,
    and barge-in can therefore be processed while a turn is in flight:
    inference never blocks a media-adapter thread.

    Cancellation is two-phase: output-stop is immediate (token +
    generation), then runtimes with a request actually in flight are
    recycled synchronously so single-slot servers (`-np 1`) cannot trap
    the next turn behind an abandoned tail. That recycle runs only on
    real preemption and stays bounded (seconds); idle cancels never
    block.
    """

    def push_audio(self, frame: AudioFrame) -> None: ...
    def commit_turn(self, turn_id: int) -> None: ...
    def cancel_output(self, reason: CancelReason) -> None: ...
    def speak(self, text: str, turn_id: int) -> None: ...
    def close(self) -> None: ...


class VoiceBackend(Protocol):
    def open_session(self, call_id: str, listener: VoiceListener) -> VoiceSession: ...


class ConfigRepository(Protocol):
    """Key/value configuration source behind the service boundary."""

    def get(self, key: str) -> str | None: ...


class Policy(Protocol):
    """Admission policy: answer the inbound call or reject it."""

    def should_answer(self, caller_id: str) -> bool: ...


class CallOutcome(Enum):
    COMPLETED = "completed"
    CALLER_HANGUP = "caller_hangup"
    REJECTED = "rejected"
    TRANSFERRED = "transferred"


class StoreUnavailableError(Exception):
    """The runtime store cannot serve requests right now. Not retryable here."""


class TransientStoreError(Exception):
    """A possibly-transient store failure. Bounded retries may follow."""


@dataclass(frozen=True)
class MessageDraft:
    """Unpersisted caller message. The id is assigned by the store on save."""

    call_id: str
    caller_id: str
    caller_name: str | None
    text: str


@dataclass(frozen=True)
class MessageRecord:
    """Persisted confirmed message: text/structure only, never audio."""

    id: str
    call_id: str
    caller_id: str
    caller_name: str | None
    text: str
    created_at: float


class MessageRepository(Protocol):
    """Runtime persistence boundary for confirmed messages.

    Atomicity contract: save() either commits the record and returns it,
    or raises before committing anything. A TransientStoreError therefore
    implies no record was stored, which is what makes the session's bounded
    retry safe against duplicates. Stores must only raise
    TransientStoreError (retryable) or StoreUnavailableError (fallback).
    """

    def save(self, draft: MessageDraft) -> MessageRecord: ...
    def get(self, message_id: str) -> MessageRecord | None: ...
    def list_all(self) -> list[MessageRecord]: ...
    def prune_before(self, cutoff: float) -> int: ...


@dataclass(frozen=True)
class TranscriptEntry:
    """One observational transcript line: text only, never authority."""

    call_id: str
    timestamp: float
    speaker: str
    text: str


class TranscriptStore(Protocol):
    """Append-only text transcript boundary. Disabled by default."""

    def append(self, entry: TranscriptEntry) -> None: ...
    def entries_for(self, call_id: str) -> list[TranscriptEntry]: ...
    def prune_before(self, cutoff: float) -> int: ...


class AuditDecision(Enum):
    REQUESTED = "requested"
    ALLOWED = "allowed"
    DENIED = "denied"
    COMPLETED = "completed"


@dataclass(frozen=True)
class AuditEvent:
    """One structured privileged-action record. No transcript content."""

    timestamp: float
    call_id: str
    action: str
    destination: str
    decision: AuditDecision
    result: TransferResult | None = None
    detail: str = ""


class AuditLog(Protocol):
    """Append-only audit boundary for privileged actions."""

    def record(self, event: AuditEvent) -> None: ...
    def list_all(self) -> list[AuditEvent]: ...
    def prune_before(self, cutoff: float) -> int: ...


@dataclass(frozen=True)
class CallSummary:
    call_id: str
    caller_id: str
    started_at: float
    ended_at: float
    outcome: CallOutcome
    turn_count: int
    caller_name: str | None = None
    handoff_destination_id: str | None = None
    message_id: str | None = None
    failure_category: str | None = None


class CallRepository(Protocol):
    """Runtime persistence boundary for call summaries."""

    def save(self, summary: CallSummary) -> None: ...
    def get(self, call_id: str) -> CallSummary | None: ...
    def list_all(self) -> list[CallSummary]: ...
    def find_by_caller(self, caller_id: str) -> list[CallSummary]: ...
    def find_in_range(self, start: float, end: float) -> list[CallSummary]: ...
    def prune_before(self, cutoff: float) -> int: ...


class KnowledgeError(Exception):
    """Normalized knowledge failure. Raw sqlite3/filesystem errors never
    cross the knowledge boundary; sources wrap them in this type."""


@dataclass(frozen=True)
class KnowledgeQuery:
    """Untrusted question text from the caller/conversation. Never authority,
    never a filesystem path, never configuration."""

    text: str


@dataclass(frozen=True)
class KnowledgeChunk:
    """One retrievable unit with stable provenance. Informational context
    only: carrying text here never authorizes a privileged action.

    ``text`` is the servable authoritative content. ``search_text`` holds
    optional match-only terms (e.g. FAQ keywords) consulted by retrievers
    but never presented as content; empty means match against ``text``.
    """

    source_id: str
    chunk_id: str
    text: str
    title: str = ""
    origin: str = ""
    search_text: str = ""


@dataclass(frozen=True)
class FaqEntry:
    """One operator-authored structured entry. Disabled entries are stored
    but never served."""

    id: str
    question: str
    answer: str
    keywords: str = ""
    enabled: bool = True


class KnowledgeStatus(Enum):
    """FOUND: relevant chunks. NO_RESULT: normal empty outcome, not a
    failure. FAILURE: normalized service failure, details in error."""

    FOUND = "found"
    NO_RESULT = "no_result"
    FAILURE = "failure"


@dataclass(frozen=True)
class KnowledgeResult:
    """The only shape a knowledge lookup returns. No-result and failure
    are values, never exceptions."""

    status: KnowledgeStatus
    chunks: tuple[KnowledgeChunk, ...] = ()
    error: str = ""

    @classmethod
    def found(cls, chunks: tuple[KnowledgeChunk, ...]) -> KnowledgeResult:
        return cls(status=KnowledgeStatus.FOUND, chunks=tuple(chunks))

    @classmethod
    def no_result(cls) -> KnowledgeResult:
        return cls(status=KnowledgeStatus.NO_RESULT)

    @classmethod
    def failure(cls, error: str) -> KnowledgeResult:
        return cls(status=KnowledgeStatus.FAILURE, error=error)


class KnowledgeSource(Protocol):
    """Where chunks come from. Implementations read local data only;
    the source list always comes from trusted configuration, never from
    caller or model input."""

    @property
    def source_id(self) -> str: ...

    def chunks(self) -> list[KnowledgeChunk]:
        """All servable chunks. Raises KnowledgeError when unreadable."""
        ...


class KnowledgeRetriever(Protocol):
    """How relevant chunks are selected. Deterministic, local only: no
    embeddings, no vector DB, no network. Replaceable by semantic RAG
    later behind this same protocol."""

    def retrieve(
        self, query: str, chunks: list[KnowledgeChunk], limit: int
    ) -> list[KnowledgeChunk]: ...


class KnowledgeService(Protocol):
    """Stable application-facing knowledge contract. The core depends
    only on this, never on SQLite, files, or retriever details. Returns
    informational context with provenance; never acts, transfers,
    resolves destinations, or touches configuration."""

    def query(self, query: KnowledgeQuery) -> KnowledgeResult: ...


class KnowledgeSourceKind(Enum):
    """The closed set of v0.1 source implementations."""

    FAQ = "faq"
    FILE = "file"


@dataclass(frozen=True)
class KnowledgeSourceDeclaration:
    """One operator-declared knowledge source: the canonical config
    record. The query/caller never controls these values; only trusted
    configuration does. ``locator`` is the FAQ database path (FAQ kind)
    or the document path (FILE kind). Disabled declarations are stored
    but never built into live sources."""

    source_id: str
    kind: KnowledgeSourceKind
    enabled: bool = True
    locator: str = ""


class KnowledgeSourceRepository(Protocol):
    """Configuration-side persistence boundary for source declarations.
    Lives in config.db, behind ConfigService, like every other config
    domain. Stores declarations/enablement only, never document content."""

    def save(self, declaration: KnowledgeSourceDeclaration) -> None: ...
    def list_all(self) -> list[KnowledgeSourceDeclaration]: ...
