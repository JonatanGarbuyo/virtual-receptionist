"""Deterministic failure/recovery components (#22).

Small separable pieces, each testable with FakeClock:

- RetryPolicy: bounded per-turn retry budget with a single retryability
  table. Only CANCELLED is never retryable.
- CircuitBreaker: consecutive terminal provider failures open the
  conversational-provider circuit across calls. Time alone never closes
  it; an explicit successful health probe does, once the cooldown
  elapsed. App cancellations, knowledge outcomes, and observability
  failures never touch it (callers simply never record them here).
- CapacityLimiter: bounded concurrent AI sessions. Permits are held by
  the core, never leak, and never go negative.

No sleeps, no wall clock, no threads, no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from receptionist.boundaries import Clock, ProviderFailureCategory


#: v0.1 hard ceiling: no configuration may result in more retries.
HARD_MAX_RETRIES = 3


@dataclass(frozen=True)
class ResilienceConfig:
    """Typed resilience configuration. Invalid values are rejected at
    construction (fail-closed); anything above the hard max never yields
    extra attempts. Defaults are conservative and documented here."""

    provider_retries: int = 1
    breaker_threshold: int = 3
    breaker_probe_cooldown_seconds: float = 60.0
    max_ai_sessions: int = 1
    turn_deadline_seconds: float = 60.0
    no_input_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not 0 <= self.provider_retries <= HARD_MAX_RETRIES:
            raise ValueError(
                f"provider_retries must be 0..{HARD_MAX_RETRIES}, "
                f"got {self.provider_retries}"
            )
        if self.breaker_threshold < 1:
            raise ValueError(
                f"breaker_threshold must be >= 1, got {self.breaker_threshold}"
            )
        if self.breaker_probe_cooldown_seconds < 0:
            raise ValueError("breaker_probe_cooldown_seconds must be >= 0")
        if self.max_ai_sessions < 1:
            raise ValueError(
                f"max_ai_sessions must be >= 1, got {self.max_ai_sessions}"
            )
        if self.turn_deadline_seconds <= 0:
            raise ValueError("turn_deadline_seconds must be > 0")
        if self.no_input_seconds <= 0:
            raise ValueError("no_input_seconds must be > 0")


class RetryPolicy:
    """One explicit retryability table. Retries always stay within the
    configured budget, the hard max, and the turn deadline (enforced by
    the session, which owns the clock); they never reset the deadline
    and never loop."""

    def __init__(self, max_retries: int = 1) -> None:
        if not 0 <= max_retries <= HARD_MAX_RETRIES:
            raise ValueError(
                f"max_retries must be 0..{HARD_MAX_RETRIES}, got {max_retries}"
            )
        self._max_retries = max_retries

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def retryable(self, category: ProviderFailureCategory) -> bool:
        """Whether the category may ever be retried. CANCELLED (barge-in,
        hangup, handoff, call limit, shutdown) never is: a cancellation
        is app-owned, not provider sickness, and must not feed the
        breaker either."""
        return category is not ProviderFailureCategory.CANCELLED

    def allows(self, category: ProviderFailureCategory, attempts_used: int) -> bool:
        """Whether another attempt fits the remaining budget."""
        return self.retryable(category) and attempts_used < self._max_retries


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"


class CircuitBreaker:
    """Consecutive-failure circuit for the conversational provider.

    CLOSED admits the AI path. ``failure_threshold`` consecutive terminal
    provider failures open it; any call ending without a terminal
    provider failure resets the streak. While OPEN, bypassed calls do
    not feed it (only an explicit probe can close it), so time alone or
    fallback traffic never recovers the provider: recovery requires a
    successful health signal once the probe cooldown elapsed.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        probe_cooldown_seconds: float = 60.0,
        clock: Clock | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")
        if probe_cooldown_seconds < 0:
            raise ValueError("probe_cooldown_seconds must be >= 0")
        self._threshold = failure_threshold
        self._cooldown = probe_cooldown_seconds
        self._clock = clock
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state is CircuitState.OPEN

    def _now(self) -> float:
        return self._clock.now() if self._clock is not None else 0.0

    def record_failure(self) -> bool:
        """Record one terminal provider failure. Returns True when this
        record just opened the circuit. Ignored while OPEN."""
        if self._state is CircuitState.OPEN:
            return False
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._threshold:
            self._state = CircuitState.OPEN
            self._opened_at = self._now()
            return True
        return False

    def record_success(self) -> None:
        """A call ended with no terminal provider failure: reset the
        streak. Never closes an OPEN circuit (only a probe does)."""
        if self._state is CircuitState.CLOSED:
            self._consecutive_failures = 0

    def probe_allowed(self) -> bool:
        """Whether a recovery probe may be honored right now: OPEN plus
        cooldown elapsed since opening."""
        if self._state is not CircuitState.OPEN or self._opened_at is None:
            return False
        return self._now() - self._opened_at >= self._cooldown

    def record_probe(self, success: bool) -> bool:
        """Honor one health-probe outcome. Returns False when no probe
        was due (closed circuit or cooldown). A successful due probe
        closes the circuit; a failed one keeps it OPEN."""
        if not self.probe_allowed():
            return False
        if success:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = None
        return True


class CapacityLimiter:
    """Bounded concurrent AI sessions. The core acquires before opening
    provider resources and releases exactly once per held permit when
    the session ends, however it ends. Releases without a matching
    acquire are ignored so the count never goes negative."""

    def __init__(self, max_sessions: int = 1) -> None:
        if max_sessions < 1:
            raise ValueError(f"max_sessions must be >= 1, got {max_sessions}")
        self._max_sessions = max_sessions
        self._active = 0

    @property
    def limit(self) -> int:
        return self._max_sessions

    @property
    def active(self) -> int:
        return self._active

    def acquire(self) -> bool:
        """Take one AI slot. False when saturated: no resources opened."""
        if self._active >= self._max_sessions:
            return False
        self._active += 1
        return True

    def release(self) -> None:
        """Return one AI slot. Safe to call for sessions that never held
        one (fallback-only admissions): the count never goes negative."""
        if self._active > 0:
            self._active -= 1
