"""ReceptionistCore: admission, per-call sessions, and service health."""

from __future__ import annotations

from collections.abc import Callable

from receptionist.boundaries import (
    CallIdGenerator,
    Clock,
    KnowledgeQuery,
    KnowledgeResult,
    KnowledgeService,
    Policy,
    TelephonyAdapter,
    TransferResult,
    VoiceBackend,
)
from receptionist.call_session import CallSession, CallState
from receptionist.config import ConfigService
from receptionist.health import HealthStatus, ServiceHealth
from receptionist.persistence import RuntimeStorage
from receptionist.policy import PolicyEngine, RetentionPolicy


class ReceptionistCore:
    """Owns the call lifecycle across sessions; one instance per service."""

    def __init__(
        self,
        *,
        telephony: TelephonyAdapter,
        voice: VoiceBackend,
        config_service: ConfigService,
        policy: Policy,
        clock: Clock,
        policy_engine: PolicyEngine,
        runtime: RuntimeStorage,
        retention: RetentionPolicy,
        call_ids: CallIdGenerator,
        knowledge: KnowledgeService | None = None,
    ) -> None:
        self._telephony = telephony
        self._voice = voice
        self._config = config_service
        self._policy = policy
        self._clock = clock
        self._policy_engine = policy_engine
        self._runtime = runtime
        self._retention = retention
        self._call_ids = call_ids
        self._knowledge = knowledge
        self.health = ServiceHealth()
        self._sessions: dict[str, CallSession] = {}
        self._last_prune: float | None = None
        telephony.set_listener(self)

    def start(self) -> ServiceHealth:
        """Validate required configuration, then expose readiness."""
        missing = self._config.missing_required()
        if missing:
            self.health.mark_not_ready(f"missing required configuration: {', '.join(missing)}")
        else:
            self.health.mark_ready("configuration loaded")
        return self.health

    def report_degraded(self, detail: str) -> None:
        self.health.mark_degraded(detail)

    def recover(self) -> None:
        self.health.recover()

    # -- admission -----------------------------------------------------------

    def incoming_call(self, caller_id: str, caller_name: str | None = None) -> CallSession:
        """Synthetic inbound entry point: readiness, then policy, then answer."""
        call_id = self._call_ids.next_id()
        greeting = self._config.get_greeting() or ""
        session = CallSession(
            call_id=call_id,
            caller_id=caller_id,
            caller_name=caller_name,
            telephony=self._telephony,
            voice=self._voice,
            greeting=greeting,
            clock=self._clock,
            policy_engine=self._policy_engine,
            runtime=self._runtime,
            transcripts_enabled=self._config.transcripts_enabled(),
        )
        self._sessions[call_id] = session
        # No AI session may start while STARTING/NOT_READY. Refusing here
        # acquires no voice resources (lazy open in request_answer); exact
        # real-adapter fallback behavior belongs to a later ticket.
        # DEGRADED handling is left to its ticket as well.
        admitted = self.health.status not in (HealthStatus.STARTING, HealthStatus.NOT_READY)
        if admitted and self._policy.should_answer(caller_id):
            session.request_answer()
        else:
            session.reject()
            del self._sessions[call_id]
        return session

    def get_session(self, call_id: str) -> CallSession | None:
        return self._sessions.get(call_id)

    def tick(self) -> None:
        """Enforce time-based deadlines across sessions, deterministically.

        The production runtime calls this periodically; under test the fake
        clock advances first. No background timers live in the core.
        Retention cleanup rides along on a deterministic throttle so expired
        records are pruned without anyone remembering to call for it.
        """
        for call_id in list(self._sessions):
            session = self._sessions.get(call_id)
            if session is not None:
                session.check_timeouts()
        now = self._clock.now()
        if (
            self._last_prune is None
            or now - self._last_prune >= self._retention.prune_interval_seconds
        ):
            self._runtime.prune_expired(self._retention, now)
            self._last_prune = now

    def prune_expired(self) -> dict[str, int]:
        """Run bounded-retention cleanup over runtime storage. Idempotent."""
        return self._runtime.prune_expired(self._retention, self._clock.now())

    def query_knowledge(self, text: str) -> KnowledgeResult:
        """Deterministic knowledge seam for the conversational path.

        Returns informational chunks with provenance, an explicit NO_RESULT
        when nothing relevant exists, or a normalized FAILURE when the
        service is down. Without a configured service the answer is
        NO_RESULT (no company knowledge), never a failure. Retrieved text
        is never parsed into actions: transfers still require a typed
        TransferRequest resolved by PolicyEngine.
        """
        if self._knowledge is None:
            return KnowledgeResult.no_result()
        return self._knowledge.query(KnowledgeQuery(text=text))

    # -- TelephonyListener (called by the telephony adapter) -----------------

    def _dispatch(self, call_id: str, handler: Callable[..., None], *args: object) -> None:
        """Route a telephony event; evict the session once it ends."""
        session = self._sessions.get(call_id)
        if session is None:
            return
        handler(session, *args)
        if session.state is CallState.ENDED:
            # pop: the handler may have completed synchronously through a
            # nested event (e.g. auto-confirmed hangup) and evicted already.
            self._sessions.pop(call_id, None)

    def on_answered(self, call_id: str) -> None:
        self._dispatch(call_id, CallSession.handle_answered)

    def on_caller_hangup(self, call_id: str) -> None:
        self._dispatch(call_id, CallSession.handle_caller_hangup)

    def on_hangup_completed(self, call_id: str) -> None:
        self._dispatch(call_id, CallSession.handle_hangup_completed)

    def on_transfer_result(self, call_id: str, result: TransferResult) -> None:
        self._dispatch(call_id, CallSession.handle_transfer_result, result)
