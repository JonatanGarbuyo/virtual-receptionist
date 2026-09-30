"""ReceptionistCore: admission, per-call sessions, and service health."""

from __future__ import annotations

from collections.abc import Callable

from receptionist.boundaries import (
    CallRepository,
    Clock,
    Policy,
    TelephonyAdapter,
    VoiceBackend,
)
from receptionist.call_session import CallSession, CallState
from receptionist.config import ConfigService
from receptionist.health import HealthStatus, ServiceHealth


class ReceptionistCore:
    """Owns the call lifecycle across sessions; one instance per service."""

    def __init__(
        self,
        *,
        telephony: TelephonyAdapter,
        voice: VoiceBackend,
        config_service: ConfigService,
        policy: Policy,
        calls: CallRepository,
        clock: Clock,
    ) -> None:
        self._telephony = telephony
        self._voice = voice
        self._config = config_service
        self._policy = policy
        self._calls = calls
        self._clock = clock
        self.health = ServiceHealth()
        self._sessions: dict[str, CallSession] = {}
        self._next_call = 1
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

    def incoming_call(self, caller_id: str) -> CallSession:
        """Synthetic inbound entry point: readiness, then policy, then answer."""
        call_id = f"call-{self._next_call}"
        self._next_call += 1
        greeting = self._config.get_greeting() or ""
        session = CallSession(
            call_id=call_id,
            caller_id=caller_id,
            telephony=self._telephony,
            voice=self._voice,
            greeting=greeting,
            calls=self._calls,
            clock=self._clock,
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

    # -- TelephonyListener (called by the telephony adapter) -----------------

    def _dispatch(self, call_id: str, handler: Callable[[CallSession], None]) -> None:
        """Route a telephony event; evict the session once it ends."""
        session = self._sessions.get(call_id)
        if session is None:
            return
        handler(session)
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
