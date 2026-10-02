"""Unit tests for deterministic failure/recovery components (#22).

RetryPolicy, CircuitBreaker, and CapacityLimiter are tested here in
isolation with FakeClock. Session/core integration lives in
test_failure_recovery.py. No sleeps, no wall clock, no network.
"""

import unittest

from receptionist.boundaries import ProviderFailureCategory
from receptionist.resilience import (
    CapacityLimiter,
    CircuitBreaker,
    CircuitState,
    ResilienceConfig,
    RetryPolicy,
)

from fakes import FakeClock


class RetryPolicyTest(unittest.TestCase):
    def test_default_budget_is_one(self) -> None:
        self.assertEqual(RetryPolicy().max_retries, 1)

    def test_zero_disables_retry(self) -> None:
        policy = RetryPolicy(max_retries=0)
        self.assertFalse(policy.allows(ProviderFailureCategory.TIMEOUT, 0))

    def test_budget_counts_attempts_used(self) -> None:
        policy = RetryPolicy(max_retries=2)
        self.assertTrue(policy.allows(ProviderFailureCategory.TIMEOUT, 0))
        self.assertTrue(policy.allows(ProviderFailureCategory.TIMEOUT, 1))
        self.assertFalse(policy.allows(ProviderFailureCategory.TIMEOUT, 2))

    def test_hard_max_three_rejects_more(self) -> None:
        with self.assertRaises(ValueError):
            RetryPolicy(max_retries=4)
        with self.assertRaises(ValueError):
            RetryPolicy(max_retries=-1)
        self.assertTrue(RetryPolicy(max_retries=3).allows(ProviderFailureCategory.INTERNAL, 2))

    def test_cancelled_is_never_retryable(self) -> None:
        policy = RetryPolicy(max_retries=3)
        for used in (0, 1, 2):
            self.assertFalse(policy.allows(ProviderFailureCategory.CANCELLED, used))

    def test_all_other_categories_retryable_within_budget(self) -> None:
        policy = RetryPolicy(max_retries=1)
        for category in (
            ProviderFailureCategory.TIMEOUT,
            ProviderFailureCategory.UNAVAILABLE,
            ProviderFailureCategory.INVALID_OUTPUT,
            ProviderFailureCategory.RESOURCE_EXHAUSTED,
            ProviderFailureCategory.INTERNAL,
        ):
            self.assertTrue(policy.retryable(category))
            self.assertTrue(policy.allows(category, 0))
            self.assertFalse(policy.allows(category, 1))
        self.assertFalse(policy.retryable(ProviderFailureCategory.CANCELLED))


class CircuitBreakerTest(unittest.TestCase):
    def test_closed_initially(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, clock=FakeClock())
        self.assertEqual(breaker.state, CircuitState.CLOSED)
        self.assertFalse(breaker.is_open)

    def test_below_threshold_stays_closed(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, clock=FakeClock())
        breaker.record_failure()
        self.assertEqual(breaker.state, CircuitState.CLOSED)

    def test_threshold_opens(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, clock=FakeClock())
        self.assertFalse(breaker.record_failure())
        self.assertTrue(breaker.record_failure())
        self.assertTrue(breaker.is_open)

    def test_success_resets_streak(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, clock=FakeClock())
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        self.assertFalse(breaker.is_open)

    def test_failures_while_open_do_not_change_state(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, clock=FakeClock())
        breaker.record_failure()
        self.assertTrue(breaker.is_open)
        breaker.record_failure()
        self.assertTrue(breaker.is_open)

    def test_success_while_open_does_not_close(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, clock=FakeClock())
        breaker.record_failure()
        breaker.record_success()
        self.assertTrue(breaker.is_open)

    def test_probe_needs_cooldown_then_success_closes(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(
            failure_threshold=1, probe_cooldown_seconds=60.0, clock=clock
        )
        breaker.record_failure()
        self.assertFalse(breaker.record_probe(True))
        self.assertTrue(breaker.is_open)
        clock.advance(60.0)
        self.assertTrue(breaker.record_probe(True))
        self.assertFalse(breaker.is_open)

    def test_failed_probe_rearms_cooldown(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(
            failure_threshold=1, probe_cooldown_seconds=60.0, clock=clock
        )
        breaker.record_failure()
        clock.advance(60.0)
        self.assertTrue(breaker.record_probe(False))
        self.assertTrue(breaker.is_open)
        # Cooldown rearmed: an immediate second probe is not due.
        self.assertFalse(breaker.record_probe(True))
        clock.advance(60.0)
        self.assertTrue(breaker.record_probe(True))
        self.assertFalse(breaker.is_open)

    def test_probe_when_closed_is_rejected(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, clock=FakeClock())
        self.assertFalse(breaker.record_probe(True))

    def test_invalid_threshold_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CircuitBreaker(failure_threshold=0, clock=FakeClock())


class CapacityLimiterTest(unittest.TestCase):
    def test_default_limit_is_one(self) -> None:
        limiter = CapacityLimiter()
        self.assertTrue(limiter.acquire())
        self.assertFalse(limiter.acquire())
        self.assertEqual(limiter.active, 1)

    def test_release_frees_slot(self) -> None:
        limiter = CapacityLimiter(max_sessions=1)
        self.assertTrue(limiter.acquire())
        limiter.release()
        self.assertEqual(limiter.active, 0)
        self.assertTrue(limiter.acquire())

    def test_release_without_acquire_never_goes_negative(self) -> None:
        limiter = CapacityLimiter(max_sessions=1)
        limiter.release()
        self.assertEqual(limiter.active, 0)

    def test_configured_limit(self) -> None:
        limiter = CapacityLimiter(max_sessions=2)
        self.assertTrue(limiter.acquire())
        self.assertTrue(limiter.acquire())
        self.assertFalse(limiter.acquire())

    def test_invalid_limit_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CapacityLimiter(max_sessions=0)


class ResilienceConfigTest(unittest.TestCase):
    def test_defaults(self) -> None:
        config = ResilienceConfig()
        self.assertEqual(config.provider_retries, 1)
        self.assertEqual(config.breaker_threshold, 3)
        self.assertEqual(config.max_ai_sessions, 1)
        self.assertEqual(config.turn_deadline_seconds, 60.0)
        self.assertEqual(config.no_input_seconds, 30.0)

    def test_invalid_values_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ResilienceConfig(provider_retries=4)
        with self.assertRaises(ValueError):
            ResilienceConfig(provider_retries=-1)
        with self.assertRaises(ValueError):
            ResilienceConfig(breaker_threshold=0)
        with self.assertRaises(ValueError):
            ResilienceConfig(max_ai_sessions=0)
        with self.assertRaises(ValueError):
            ResilienceConfig(turn_deadline_seconds=0)
        with self.assertRaises(ValueError):
            ResilienceConfig(no_input_seconds=-5)


if __name__ == "__main__":
    unittest.main()
