"""ReceptionistCore: admission, per-call sessions, and service health."""

from __future__ import annotations

from collections.abc import Callable

from receptionist.boundaries import (
    AudioFrame,
    CallIdGenerator,
    Clock,
    KnowledgeQuery,
    KnowledgeResult,
    KnowledgeService,
    Policy,
    TelephonyAdapter,
    TelephonyRegistrationState,
    TransferResult,
    VoiceBackend,
)
from receptionist.alerting import (
    CODE_AUDIT_UNAVAILABLE,
    CODE_CAPACITY_SATURATED,
    CODE_CIRCUIT_OPEN,
    CODE_CONFIG_INCOMPLETE,
    CODE_CONFIG_UNAVAILABLE,
    CODE_HISTORY_UNAVAILABLE,
    CODE_TELEPHONY_LOST,
    CODE_TELEPHONY_NOT_REGISTERED,
    CODE_TRANSCRIPT_UNAVAILABLE,
    CODE_VOICE_BACKEND_NOT_READY,
    HealthComponent,
    HealthMonitor,
    HttpTransport,
    SmtpTransport,
    build_monitor,
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

#: Health detail marking a required voice backend that never became
#: ready. Sanitized and stable: never paths, versions, or stderr.
_VOICE_NOT_READY_DETAIL = "voice backend not ready"


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
        monitor: HealthMonitor | None = None,
        smtp_transport: SmtpTransport | None = None,
        http_transport: HttpTransport | None = None,
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
        # Productive alerting composition: with no monitor injected, the
        # monitor (and its sinks) is built once from config.db through
        # ConfigService, honoring the persisted notify_recovery flag.
        # Disabled channels stay local-only; invalid enabled channels
        # fail composition explicitly instead of half-applied operation.
        if monitor is None:
            monitor = build_monitor(
                config_service.alert_settings(),
                clock,
                smtp_transport=smtp_transport,
                http_transport=http_transport,
            )
        self.monitor = monitor
        self._sessions: dict[str, CallSession] = {}
        self._ai_permits: set[str] = set()
        self._last_prune: float | None = None
        telephony.set_listener(self)

    def start(self) -> ServiceHealth:
        """Validate required configuration, then expose readiness.

        An unreadable config authority fails startup to NOT_READY; no
        session is admitted until configuration trust returns. Startup
        transitions alert once per condition; a clean STARTING -> READY
        emits no recovery (no prior outage existed).
        """
        try:
            missing = self._config.missing_required()
            config_trusted = True
        except Exception:
            missing = []
            config_trusted = False
        if not config_trusted:
            self.health.mark_not_ready(_CONFIG_DOWN_DETAIL)
            self.monitor.report_unhealthy(
                HealthComponent.CONFIGURATION,
                CODE_CONFIG_UNAVAILABLE,
            )
        elif missing:
            self.health.mark_not_ready(f"missing required configuration: {', '.join(missing)}")
            self.monitor.report_unhealthy(
                HealthComponent.CONFIGURATION,
                CODE_CONFIG_INCOMPLETE,
            )
        else:
            self.health.mark_ready("configuration loaded")
            self.monitor.report_recovered(
                HealthComponent.CONFIGURATION, CODE_CONFIG_INCOMPLETE
            )
            self.monitor.report_recovered(
                HealthComponent.CONFIGURATION, CODE_CONFIG_UNAVAILABLE
            )
        self._sync_voice_readiness()
        self._reconcile_health()
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
            missing = self._config.missing_required()
            config_trusted = True
        except Exception:
            greeting = ""
            transcripts_enabled = False
            missing = ["unreadable"]
            config_trusted = False
        if not config_trusted:
            self.monitor.report_unhealthy(
                HealthComponent.CONFIGURATION,
                CODE_CONFIG_UNAVAILABLE,
            )
        elif missing:
            self.monitor.report_recovered(
                HealthComponent.CONFIGURATION, CODE_CONFIG_UNAVAILABLE
            )
            self.monitor.report_unhealthy(
                HealthComponent.CONFIGURATION,
                CODE_CONFIG_INCOMPLETE,
            )
        else:
            self.monitor.report_recovered(
                HealthComponent.CONFIGURATION, CODE_CONFIG_UNAVAILABLE
            )
            self.monitor.report_recovered(
                HealthComponent.CONFIGURATION, CODE_CONFIG_INCOMPLETE
            )
        self._reconcile_health()
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
            monitor=self.monitor,
        )
        self._sessions[call_id] = session
        # Voice readiness syncs BEFORE the admission decision, so a
        # recovered backend re-admits without a manual start(), and a
        # backend that died mid-run is observed on the next call. Ticks
        # re-sync too, so recovery never waits for new traffic.
        self._sync_voice_readiness()
        # No AI session may start while STARTING/NOT_READY. Refusing here
        # acquires no voice resources (lazy open in request_answer).
        # Deliberate AI-unadmission (distinct from service admission):
        # a voice-only outage with otherwise usable configuration routes
        # to PBX fallback exactly like an open circuit, opening nothing.
        admitted = self.health.status not in (HealthStatus.STARTING, HealthStatus.NOT_READY)
        if (
            admitted
            and config_trusted
            and not missing
            and self._policy.should_answer(caller_id)
        ):
            self._sync_breaker_health()
            if not self._voice_usable():
                session.begin_fallback_only("provider_unavailable")
            elif self.breaker.is_open:
                session.begin_fallback_only("provider_unavailable")
            elif not self._acquire_ai_slot(call_id):
                session.begin_fallback_only("capacity_saturated")
                self.monitor.report_unhealthy(
                    HealthComponent.CAPACITY,
                    CODE_CAPACITY_SATURATED,
                )
            else:
                session.request_answer()
                self.monitor.report_recovered(
                    HealthComponent.CAPACITY, CODE_CAPACITY_SATURATED
                )
        elif (
            config_trusted
            and not missing
            and self._voice_has_hook()
            and not self._voice_usable()
            and self._policy.should_answer(caller_id)
        ):
            # Voice-only outage (service config usable, AI path down):
            # deliberate AI-unadmission to PBX fallback, no provider
            # resources opened. Distinct from service-level refusal.
            session.begin_fallback_only("provider_unavailable")
        else:
            session.reject()
            del self._sessions[call_id]
        self._reconcile_health()
        return session

    def _voice_has_hook(self) -> bool:
        """Whether the voice backend reports readiness at all. Legacy
        backends without the hook never gate admission."""
        return callable(getattr(self._voice, "check_ready", None))

    def _voice_usable(self) -> bool:
        """Whether the required voice backend may accept a call right
        now. Backends without the readiness hook (legacy fakes) are
        always usable; the cascaded backend gates on warmed models."""
        check = getattr(self._voice, "check_ready", None)
        if not callable(check):
            return True
        try:
            ready, _detail = check()
        except Exception:
            return False
        return bool(ready)

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
        nothing, so permits never leak and never go negative. A freed
        slot recovers the saturation condition immediately instead of
        waiting for new traffic."""
        if call_id in self._ai_permits:
            self._ai_permits.discard(call_id)
            self.capacity.release()
            if not self.capacity.saturated:
                self.monitor.report_recovered(
                    HealthComponent.CAPACITY, CODE_CAPACITY_SATURATED
                )

    def _reconcile_health(self) -> None:
        """Derive aggregate health from active monitor conditions: the one
        authoritative mapping. Config states win over degradation; any
        other active condition degrades; READY requires no remaining cause.
        Capacity saturation alerts but never changes the aggregate (it is
        expected-load routing, not sickness). Never touches STARTING."""
        if self.health.status is HealthStatus.STARTING:
            return
        active = {(c.component, c.code) for c in self.monitor.active_conditions()}
        if (HealthComponent.CONFIGURATION, CODE_CONFIG_UNAVAILABLE) in active:
            self.health.mark_not_ready(_CONFIG_DOWN_DETAIL)
        elif (HealthComponent.CONFIGURATION, CODE_CONFIG_INCOMPLETE) in active:
            self.health.mark_not_ready("required configuration missing")
        elif (HealthComponent.PROVIDER, CODE_VOICE_BACKEND_NOT_READY) in active:
            self.health.mark_not_ready(_VOICE_NOT_READY_DETAIL)
        elif (HealthComponent.TELEPHONY, CODE_TELEPHONY_NOT_REGISTERED) in active:
            self.health.mark_not_ready("telephony endpoint not registered")
        elif (HealthComponent.TELEPHONY, CODE_TELEPHONY_LOST) in active:
            self.health.mark_not_ready("telephony registration lost")
        elif (HealthComponent.PROVIDER, CODE_CIRCUIT_OPEN) in active:
            self.health.mark_degraded(_CIRCUIT_DEGRADED_DETAIL)
        elif (HealthComponent.RUNTIME, CODE_HISTORY_UNAVAILABLE) in active or (
            HealthComponent.RUNTIME,
            CODE_AUDIT_UNAVAILABLE,
        ) in active:
            self.health.mark_degraded("runtime observability degraded")
        elif (HealthComponent.TRANSCRIPT, CODE_TRANSCRIPT_UNAVAILABLE) in active:
            self.health.mark_degraded("transcript sidecar degraded")
        else:
            self.health.mark_ready("configuration loaded")

    def close(self) -> None:
        """Shutdown: drain pending alert deliveries (bounded)."""
        self.monitor.close()

    def _sync_voice_readiness(self) -> None:
        """Reflect a required voice backend's readiness in the monitor.

        Duck-typed: backends exposing `check_ready() -> (bool, str)`
        (the cascaded backend) gate admission; legacy fakes without the
        hook are skipped untouched. The sanitized detail never reaches
        health or alerts: identity stays (PROVIDER, stable code).
        """
        check = getattr(self._voice, "check_ready", None)
        if not callable(check):
            return
        try:
            ready, _detail = check()
        except Exception:
            ready = False
        if ready:
            self.monitor.report_recovered(
                HealthComponent.PROVIDER, CODE_VOICE_BACKEND_NOT_READY
            )
        else:
            self.monitor.report_unhealthy(
                HealthComponent.PROVIDER, CODE_VOICE_BACKEND_NOT_READY
            )
        self._reconcile_health()

    def _sync_breaker_health(self) -> None:
        """Report an open provider circuit. Aggregate follows via reconcile:
        DEGRADED still admits calls; they route to PBX fallback at admission."""
        if self.breaker.is_open:
            self.monitor.report_unhealthy(
                HealthComponent.PROVIDER,
                CODE_CIRCUIT_OPEN,
            )
        self._reconcile_health()

    def report_provider_probe(self, success: bool) -> bool:
        """Project-owned recovery seam for a future backend health check.

        Honors one probe outcome once the cooldown elapsed. A successful
        probe closes the circuit and lifts circuit degradation; anything
        else leaves the circuit OPEN. Returns False when no probe was
        due. No background threads: the caller drives this explicitly.
        """
        if not self.breaker.record_probe(success):
            return False
        if success:
            self.monitor.report_recovered(
                HealthComponent.PROVIDER, CODE_CIRCUIT_OPEN
            )
        self._reconcile_health()
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
        self._sync_voice_readiness()
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
        self._reconcile_health()
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

    def on_caller_audio(self, call_id: str, frame: AudioFrame) -> None:
        """Route decoded caller PCM into the session media path.

        Unknown call ids (late frames after eviction) are ignored. The
        frame itself is project-owned PCM16 mono; vendor types never
        reach here.
        """
        session = self._sessions.get(call_id)
        if session is None:
            return
        try:
            session.push_caller_audio(frame)
        except Exception:
            pass

    def on_dtmf(self, call_id: str, digit: str) -> None:
        """Observe one DTMF digit for a live call.

        v0.1 carries no DTMF business logic (#25 transports only): the
        event is validated (known call, single valid digit) and otherwise
        ignored so digits can never become privileged actions. Unknown
        calls and malformed digits are dropped.
        """
        if self._sessions.get(call_id) is None:
            return
        if not isinstance(digit, str) or len(digit) != 1:
            return
        if digit.upper() not in "0123456789ABCD*#":
            return

    def on_remote_hold(self, call_id: str, held: bool) -> None:
        """Observe a remote hold/resume re-INVITE. Transport-level only:
        media routing already pauses/resumes in the adapter; the session
        needs no state change in v0.1."""
        if self._sessions.get(call_id) is None:
            return

    # -- telephony registration health -----------------------------------

    def report_telephony_state(
        self, state: TelephonyRegistrationState, detail: str = ""
    ) -> None:
        """Project-owned hook wiring telephony registration into READY.

        REGISTERED lifts telephony conditions; anything else raises the
        stable sanitized condition (no REGISTER retries alert per call:
        edge-triggering in the monitor dedups). Re-registration recovers
        through the same path. Never touches STARTING.
        """
        if state is TelephonyRegistrationState.REGISTERED:
            self.monitor.report_recovered(
                HealthComponent.TELEPHONY, CODE_TELEPHONY_NOT_REGISTERED
            )
            self.monitor.report_recovered(
                HealthComponent.TELEPHONY, CODE_TELEPHONY_LOST
            )
        elif state in (
            TelephonyRegistrationState.STARTING,
            TelephonyRegistrationState.STOPPING,
            TelephonyRegistrationState.STOPPED,
        ):
            self.monitor.report_unhealthy(
                HealthComponent.TELEPHONY,
                CODE_TELEPHONY_NOT_REGISTERED,
            )
        elif state is TelephonyRegistrationState.REGISTRATION_FAILED:
            self.monitor.report_unhealthy(
                HealthComponent.TELEPHONY,
                CODE_TELEPHONY_NOT_REGISTERED,
            )
        else:  # REGISTRATION_LOST
            self.monitor.report_unhealthy(
                HealthComponent.TELEPHONY,
                CODE_TELEPHONY_LOST,
            )
        self._reconcile_health()
