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
from receptionist.resilience import (
    CapacityLimiter,
    CircuitBreaker,
    CircuitState,
    ResilienceConfig,
)

#: Health detail marking provider-circuit degradation. Only this exact
#: degradation is lifted by a successful recovery probe.
_CIRCUIT_DEGRADED_DETAIL = "conversational provider circuit open"

#: Health detail marking lost configuration authority. Set when admission
#: detects an unreadable config authority mid-run (mirroring startup);
#: lifted when a later admission reads configuration trustworthily again.
_CONFIG_DOWN_DETAIL = "configuration authority unavailable"


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
        resilience: ResilienceConfig | None = None,
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
        self._resilience = resilience if resilience is not None else ResilienceConfig()
        self.breaker = CircuitBreaker(
            failure_threshold=self._resilience.breaker_threshold,
            probe_cooldown_seconds=self._resilience.breaker_probe_cooldown_seconds,
            clock=clock,
        )
        self.capacity = CapacityLimiter(max_sessions=self._resilience.max_ai_sessions)
        self.health = ServiceHealth()
        self._sessions: dict[str, CallSession] = {}
        self._ai_permits: set[str] = set()
        self._last_prune: float | None = None
        telephony.set_listener(self)

    def start(self) -> ServiceHealth:
        """Validate required configuration, then expose readiness.

        An unreadable config authority fails startup to NOT_READY; no
        session is admitted until configuration trust returns.
        """
        try:
            missing = self._config.missing_required()
        except Exception as error:
            self.health.mark_not_ready(f"configuration authority unavailable: {error}")
            return self.health
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
        """Synthetic inbound entry point: readiness, then policy, then answer.

        Admission order is fixed: health gates new sessions (STARTING /
        NOT_READY admit none, DEGRADED may still serve), then admission
        policy, then AI-path routing. An open provider circuit or a
        saturated AI capacity routes the call to deterministic PBX
        fallback without opening provider resources; the existing AI
        call, if any, is untouched. A call the policy refuses is
        rejected, never fallback-transferred.
        """
        call_id = self._call_ids.next_id()
        try:
            greeting = self._config.get_greeting() or ""
            transcripts_enabled = self._config.transcripts_enabled()
            config_trusted = True
        except Exception:
            greeting = ""
            transcripts_enabled = False
            config_trusted = False
            self.health.mark_not_ready(_CONFIG_DOWN_DETAIL)
        if config_trusted and (
            self.health.status is HealthStatus.NOT_READY
            and self.health.detail == _CONFIG_DOWN_DETAIL
        ):
            self.health.mark_ready("configuration loaded")
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
            transcripts_enabled=transcripts_enabled,
            resilience=self._resilience,
            breaker=self.breaker,
        )
        self._sessions[call_id] = session
        # No AI session may start while STARTING/NOT_READY. Refusing here
        # acquires no voice resources (lazy open in request_answer); exact
        # real-adapter fallback behavior belongs to a later ticket.
        admitted = self.health.status not in (HealthStatus.STARTING, HealthStatus.NOT_READY)
        if admitted and config_trusted and self._policy.should_answer(caller_id):
            self._sync_breaker_health()
            if self.breaker.is_open:
                session.begin_fallback_only("provider_unavailable")
            elif not self._acquire_ai_slot(call_id):
                session.begin_fallback_only("capacity_saturated")
            else:
                session.request_answer()
        else:
            session.reject()
            del self._sessions[call_id]
        return session

    def get_session(self, call_id: str) -> CallSession | None:
        return self._sessions.get(call_id)

    def _acquire_ai_slot(self, call_id: str) -> bool:
        """Take one AI capacity slot for an admitted call. False when the
        installation is saturated: the caller routes to PBX fallback."""
        if not self.capacity.acquire():
            return False
        self._ai_permits.add(call_id)
        return True

    def _release_ai_slot(self, call_id: str) -> None:
        """Return the AI slot exactly once per held permit, however the
        session ended (normal end, handoff, hangup, failure, fallback).
        Sessions that never held one (rejected, fallback-only) release
        nothing, so permits never leak and never go negative."""
        if call_id in self._ai_permits:
            self._ai_permits.discard(call_id)
            self.capacity.release()

    def _sync_breaker_health(self) -> None:
        """Reflect an open provider circuit in service health. DEGRADED
        still admits calls; they route to PBX fallback at admission."""
        if self.breaker.is_open and self.health.status is HealthStatus.READY:
            self.health.mark_degraded(_CIRCUIT_DEGRADED_DETAIL)

    def report_provider_probe(self, success: bool) -> bool:
        """Project-owned recovery seam for a future backend health check.

        Honors one probe outcome once the cooldown elapsed. A successful
        probe closes the circuit and lifts circuit degradation; anything
        else leaves the circuit OPEN. Returns False when no probe was
        due. No background threads: the caller drives this explicitly.
        """
        if not self.breaker.record_probe(success):
            return False
        if (
            self.health.status is HealthStatus.DEGRADED
            and self.health.detail == _CIRCUIT_DEGRADED_DETAIL
        ):
            self.health.recover("conversational provider recovered")
        return True

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
        self._sync_breaker_health()
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
        NO_RESULT (no company knowledge), never a failure. A service
        implementation that raises instead of returning FAILURE is
        contained here, so knowledge can never break the call path.
        Retrieved text is never parsed into actions: transfers still
        require a typed TransferRequest resolved by PolicyEngine.
        """
        if self._knowledge is None:
            return KnowledgeResult.no_result()
        try:
            return self._knowledge.query(KnowledgeQuery(text=text))
        except Exception as error:  # rogue service: contain, don't propagate
            return KnowledgeResult.failure(f"knowledge service failed: {error}")

    # -- TelephonyListener (called by the telephony adapter) -----------------

    def _dispatch(self, call_id: str, handler: Callable[..., None], *args: object) -> None:
        """Route a telephony event; evict the session once it ends.

        Eviction releases the AI capacity slot, records positive
        provider evidence (real provider output in the live generation)
        against the breaker streak, and surfaces observability
        degradation without touching routing. Neutral completions
        (hangup, fallback-only, no provider output) record nothing, so
        they never wipe failure evidence.
        """
        session = self._sessions.get(call_id)
        if session is None:
            return
        handler(session, *args)
        if session.observability_failed:
            self.report_degraded("runtime observability degraded")
        if session.state is CallState.ENDED:
            # pop: the handler may have completed synchronously through a
            # nested event (e.g. auto-confirmed hangup) and evicted already.
            self._sessions.pop(call_id, None)
            self._release_ai_slot(call_id)
            if session.provider_success_evidence and not session.provider_terminal_failure:
                self.breaker.record_success()

    def on_answered(self, call_id: str) -> None:
        self._dispatch(call_id, CallSession.handle_answered)

    def on_caller_hangup(self, call_id: str) -> None:
        self._dispatch(call_id, CallSession.handle_caller_hangup)

    def on_hangup_completed(self, call_id: str) -> None:
        self._dispatch(call_id, CallSession.handle_hangup_completed)

    def on_transfer_result(self, call_id: str, result: TransferResult) -> None:
        self._dispatch(call_id, CallSession.handle_transfer_result, result)
