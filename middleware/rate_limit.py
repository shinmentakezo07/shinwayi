"""
Shin Proxy — request admission controls.

Contains:
- Legacy per-key rate limit hook (disabled/no-op)
- Adaptive global concurrency gate for upstream protection
- Upstream circuit breaker for fail-fast protection
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import structlog

from shin.analytics import prom_metrics
from shin.config import settings

log = structlog.get_logger()


def enforce_rate_limit(api_key: str) -> None:
    """Legacy per-key rate limiting hook (currently disabled)."""
    return


class AdaptiveConcurrencyController:
    """AIMD-style global concurrency controller.

    - Increases limit slowly after sustained successes.
    - Decreases limit quickly when upstream rate-limit signals appear.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        initial: int,
        minimum: int,
        maximum: int,
        success_window: int,
        rate_limit_cooldown_seconds: float,
    ) -> None:
        minimum = max(1, int(minimum))
        maximum = max(minimum, int(maximum))
        initial = max(minimum, min(int(initial), maximum))

        self.enabled = bool(enabled)
        self.limit = initial
        self.minimum = minimum
        self.maximum = maximum
        self.success_window = max(1, int(success_window))
        self.rate_limit_cooldown_seconds = max(0.0, float(rate_limit_cooldown_seconds))

        self._in_flight = 0
        self._success_since_adjust = 0
        self._last_rate_limited_at = 0.0
        self._cond = asyncio.Condition()

    async def acquire(self) -> None:
        if not self.enabled:
            return
        async with self._cond:
            while self._in_flight >= self.limit:
                await self._cond.wait()
            self._in_flight += 1
            prom_metrics.set_adaptive_concurrency(
                limit=self.limit,
                in_flight=self._in_flight,
                cooldown_remaining_seconds=self._cooldown_remaining(),
            )

    async def release(self) -> None:
        if not self.enabled:
            return
        async with self._cond:
            if self._in_flight > 0:
                self._in_flight -= 1
            prom_metrics.set_adaptive_concurrency(
                limit=self.limit,
                in_flight=self._in_flight,
                cooldown_remaining_seconds=self._cooldown_remaining(),
            )
            self._cond.notify_all()

    async def report_success(self) -> None:
        if not self.enabled:
            return
        async with self._cond:
            self._success_since_adjust += 1
            now = time.monotonic()
            in_cooldown = (
                self.rate_limit_cooldown_seconds > 0
                and self._last_rate_limited_at > 0
                and (now - self._last_rate_limited_at) < self.rate_limit_cooldown_seconds
            )

            should_increase = (
                not in_cooldown
                and self.limit < self.maximum
                and self._success_since_adjust >= self.success_window
                and self._in_flight >= self.limit
            )
            if should_increase:
                self.limit += 1
                self._success_since_adjust = 0
                log.info(
                    "adaptive_concurrency_increase",
                    limit=self.limit,
                    in_flight=self._in_flight,
                )
                self._cond.notify_all()

            prom_metrics.set_adaptive_concurrency(
                limit=self.limit,
                in_flight=self._in_flight,
                cooldown_remaining_seconds=self._cooldown_remaining(),
            )

    async def report_rate_limited(self) -> None:
        if not self.enabled:
            return
        async with self._cond:
            self._last_rate_limited_at = time.monotonic()
            # Multiplicative decrease, but do not drop below configured floor.
            new_limit = max(self.minimum, max(1, int(self.limit * 0.8)))
            changed = new_limit != self.limit
            self.limit = new_limit
            self._success_since_adjust = 0
            if changed:
                log.warning(
                    "adaptive_concurrency_decrease",
                    limit=self.limit,
                    in_flight=self._in_flight,
                )
            prom_metrics.set_adaptive_concurrency(
                limit=self.limit,
                in_flight=self._in_flight,
                cooldown_remaining_seconds=self._cooldown_remaining(),
            )
            self._cond.notify_all()

    async def report_failure(self) -> None:
        if not self.enabled:
            return
        async with self._cond:
            self._success_since_adjust = 0
            prom_metrics.set_adaptive_concurrency(
                limit=self.limit,
                in_flight=self._in_flight,
                cooldown_remaining_seconds=self._cooldown_remaining(),
            )

    async def snapshot(self) -> dict:
        async with self._cond:
            cooldown_remaining = self._cooldown_remaining()
            return {
                "enabled": self.enabled,
                "limit": self.limit,
                "minimum": self.minimum,
                "maximum": self.maximum,
                "in_flight": self._in_flight,
                "success_window": self.success_window,
                "success_since_adjust": self._success_since_adjust,
                "rate_limit_cooldown_seconds": self.rate_limit_cooldown_seconds,
                "cooldown_remaining_seconds": round(cooldown_remaining, 3),
            }

    def _cooldown_remaining(self) -> float:
        if self._last_rate_limited_at <= 0 or self.rate_limit_cooldown_seconds <= 0:
            return 0.0
        now = time.monotonic()
        return max(
            0.0,
            self.rate_limit_cooldown_seconds - (now - self._last_rate_limited_at),
        )


@dataclass
class AdaptiveConcurrencyTicket:
    """Outcome ticket for one admitted upstream attempt."""

    outcome: str = "failure"  # success | rate_limited | failure

    def mark_success(self) -> None:
        self.outcome = "success"

    def mark_rate_limited(self) -> None:
        self.outcome = "rate_limited"

    def mark_failure(self) -> None:
        self.outcome = "failure"


class CircuitBreakerController:
    """Shared circuit breaker guarding upstream Cursor attempts."""

    def __init__(
        self,
        *,
        enabled: bool,
        failure_threshold: int,
        rate_limit_threshold: int,
        open_seconds: float,
        half_open_max_probes: int,
    ) -> None:
        self.enabled = bool(enabled)
        self.failure_threshold = max(1, int(failure_threshold))
        self.rate_limit_threshold = max(1, int(rate_limit_threshold))
        self.open_seconds = max(0.0, float(open_seconds))
        self.half_open_max_probes = max(1, int(half_open_max_probes))

        self._state = "closed"
        self._consecutive_failures = 0
        self._consecutive_rate_limits = 0
        self._open_until = 0.0
        self._half_open_probes_in_flight = 0
        self._fast_fail_count = 0
        self._cond = asyncio.Condition()
        self._sync_metrics_locked()

    async def before_attempt(self) -> None:
        if not self.enabled:
            return
        async with self._cond:
            now = time.monotonic()
            if self._state == "open":
                if now >= self._open_until:
                    self._transition_locked("half_open")
                else:
                    self._fast_fail_count += 1
                    prom_metrics.observe_circuit_breaker_fast_fail()
                    raise RuntimeError("circuit_open")

            if self._state == "half_open":
                if self._half_open_probes_in_flight >= self.half_open_max_probes:
                    self._fast_fail_count += 1
                    prom_metrics.observe_circuit_breaker_fast_fail()
                    raise RuntimeError("circuit_open")
                self._half_open_probes_in_flight += 1
                self._sync_metrics_locked()

    async def after_success(self) -> None:
        if not self.enabled:
            return
        async with self._cond:
            if self._state == "half_open":
                self._decrement_half_open_locked()
                prom_metrics.observe_circuit_breaker_probe(outcome="success")
                self._close_locked()
            else:
                self._consecutive_failures = 0
                self._consecutive_rate_limits = 0
                self._sync_metrics_locked()

    async def after_rate_limit(self) -> None:
        if not self.enabled:
            return
        async with self._cond:
            if self._state == "half_open":
                self._decrement_half_open_locked()
                prom_metrics.observe_circuit_breaker_probe(outcome="rate_limited")
                self._open_locked()
                return
            self._consecutive_failures += 1
            self._consecutive_rate_limits += 1
            if self._consecutive_rate_limits >= self.rate_limit_threshold:
                self._open_locked()
            else:
                self._sync_metrics_locked()

    async def after_failure(self) -> None:
        if not self.enabled:
            return
        async with self._cond:
            if self._state == "half_open":
                self._decrement_half_open_locked()
                prom_metrics.observe_circuit_breaker_probe(outcome="failure")
                self._open_locked()
                return
            self._consecutive_failures += 1
            self._consecutive_rate_limits = 0
            if self._consecutive_failures >= self.failure_threshold:
                self._open_locked()
            else:
                self._sync_metrics_locked()

    async def snapshot(self) -> dict:
        async with self._cond:
            return {
                "enabled": self.enabled,
                "state": self._state,
                "failure_threshold": self.failure_threshold,
                "rate_limit_threshold": self.rate_limit_threshold,
                "open_seconds": self.open_seconds,
                "open_remaining_seconds": round(self._open_remaining_locked(), 3),
                "half_open_max_probes": self.half_open_max_probes,
                "half_open_probes_in_flight": self._half_open_probes_in_flight,
                "consecutive_failures": self._consecutive_failures,
                "consecutive_rate_limits": self._consecutive_rate_limits,
                "fast_fail_count": self._fast_fail_count,
            }

    def _close_locked(self) -> None:
        self._state = "closed"
        self._consecutive_failures = 0
        self._consecutive_rate_limits = 0
        self._open_until = 0.0
        self._half_open_probes_in_flight = 0
        prom_metrics.observe_circuit_breaker_transition(state="closed")
        self._sync_metrics_locked()
        self._cond.notify_all()

    def _open_locked(self) -> None:
        self._state = "open"
        self._open_until = time.monotonic() + self.open_seconds
        self._half_open_probes_in_flight = 0
        self._consecutive_failures = 0
        self._consecutive_rate_limits = 0
        prom_metrics.observe_circuit_breaker_transition(state="open")
        self._sync_metrics_locked()
        self._cond.notify_all()

    def _transition_locked(self, state: str) -> None:
        self._state = state
        if state == "half_open":
            self._consecutive_failures = 0
            self._consecutive_rate_limits = 0
            self._open_until = 0.0
        prom_metrics.observe_circuit_breaker_transition(state=state)
        self._sync_metrics_locked()

    def _decrement_half_open_locked(self) -> None:
        if self._half_open_probes_in_flight > 0:
            self._half_open_probes_in_flight -= 1

    def _open_remaining_locked(self) -> float:
        if self._state != "open":
            return 0.0
        return max(0.0, self._open_until - time.monotonic())

    def _sync_metrics_locked(self) -> None:
        prom_metrics.set_circuit_breaker(
            state=self._state,
            open_remaining_seconds=self._open_remaining_locked(),
            half_open_probes_in_flight=self._half_open_probes_in_flight,
        )


_controller = AdaptiveConcurrencyController(
    enabled=settings.adaptive_concurrency_enabled,
    initial=settings.adaptive_concurrency_initial,
    minimum=settings.adaptive_concurrency_min,
    maximum=settings.adaptive_concurrency_max,
    success_window=settings.adaptive_concurrency_success_window,
    rate_limit_cooldown_seconds=settings.adaptive_concurrency_rate_limit_cooldown_seconds,
)
_circuit_breaker = CircuitBreakerController(
    enabled=settings.circuit_breaker_enabled,
    failure_threshold=settings.circuit_breaker_failure_threshold,
    rate_limit_threshold=settings.circuit_breaker_rate_limit_threshold,
    open_seconds=settings.circuit_breaker_open_seconds,
    half_open_max_probes=settings.circuit_breaker_half_open_max_probes,
)


@asynccontextmanager
async def adaptive_concurrency_slot():
    """Acquire/release adaptive concurrency slot around one upstream attempt."""
    ticket = AdaptiveConcurrencyTicket()
    await _controller.acquire()
    try:
        yield ticket
    finally:
        if ticket.outcome == "success":
            await _controller.report_success()
        elif ticket.outcome == "rate_limited":
            await _controller.report_rate_limited()
        else:
            await _controller.report_failure()
        await _controller.release()


async def adaptive_concurrency_snapshot() -> dict:
    return await _controller.snapshot()


async def circuit_breaker_before_attempt() -> None:
    await _circuit_breaker.before_attempt()


async def circuit_breaker_report_success() -> None:
    await _circuit_breaker.after_success()


async def circuit_breaker_report_rate_limited() -> None:
    await _circuit_breaker.after_rate_limit()


async def circuit_breaker_report_failure() -> None:
    await _circuit_breaker.after_failure()


async def circuit_breaker_report_ignored() -> None:
    async with _circuit_breaker._cond:
        if _circuit_breaker._state == "half_open" and _circuit_breaker._half_open_probes_in_flight > 0:
            _circuit_breaker._decrement_half_open_locked()
            _circuit_breaker._sync_metrics_locked()


async def circuit_breaker_snapshot() -> dict:
    return await _circuit_breaker.snapshot()
