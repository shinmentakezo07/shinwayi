from __future__ import annotations

import asyncio

from shin.handlers import CircuitOpenError
from shin.middleware import rate_limit as rate_limit_module
from shin.pipeline import _call_with_retry, PipelineParams


async def _reset_breaker() -> None:
    rate_limit_module._circuit_breaker = rate_limit_module.CircuitBreakerController(
        enabled=True,
        failure_threshold=2,
        rate_limit_threshold=2,
        open_seconds=0.01,
        half_open_max_probes=1,
    )


def test_circuit_breaker_opens_after_failures() -> None:
    async def run() -> None:
        await _reset_breaker()
        await rate_limit_module.circuit_breaker_report_failure()
        snapshot = await rate_limit_module.circuit_breaker_snapshot()
        assert snapshot["state"] == "closed"

        await rate_limit_module.circuit_breaker_report_failure()
        snapshot = await rate_limit_module.circuit_breaker_snapshot()
        assert snapshot["state"] == "open"

    asyncio.run(run())


def test_circuit_breaker_rejects_when_open() -> None:
    async def run() -> None:
        await _reset_breaker()
        await rate_limit_module.circuit_breaker_report_failure()
        await rate_limit_module.circuit_breaker_report_failure()
        try:
            await rate_limit_module.circuit_breaker_before_attempt()
        except RuntimeError as exc:
            assert str(exc) == "circuit_open"
        else:
            raise AssertionError("expected circuit to reject attempt")

    asyncio.run(run())


def test_circuit_breaker_half_open_success_closes() -> None:
    async def run() -> None:
        await _reset_breaker()
        await rate_limit_module.circuit_breaker_report_failure()
        await rate_limit_module.circuit_breaker_report_failure()
        await asyncio.sleep(0.02)
        await rate_limit_module.circuit_breaker_before_attempt()
        snapshot = await rate_limit_module.circuit_breaker_snapshot()
        assert snapshot["state"] == "half_open"
        assert snapshot["half_open_probes_in_flight"] == 1

        await rate_limit_module.circuit_breaker_report_success()
        snapshot = await rate_limit_module.circuit_breaker_snapshot()
        assert snapshot["state"] == "closed"
        assert snapshot["half_open_probes_in_flight"] == 0

    asyncio.run(run())


def test_call_with_retry_does_not_retry_circuit_open() -> None:
    class StubClient:
        def __init__(self) -> None:
            self.calls = 0

        async def call(self, cursor_messages, model, anthropic_tools, cred=None):
            self.calls += 1
            raise CircuitOpenError("open")

    async def run() -> None:
        params = PipelineParams(
            api_style="openai",
            model="anthropic/claude-sonnet-4.6",
            messages=[{"role": "user", "content": "hi"}],
            cursor_messages=[{"role": "user", "parts": [{"type": "text", "text": "hi"}], "id": "1"}],
        )
        client = StubClient()
        try:
            await _call_with_retry(client, params, None)
        except CircuitOpenError:
            pass
        else:
            raise AssertionError("expected CircuitOpenError")
        assert client.calls == 1

    asyncio.run(run())


def test_circuit_breaker_ignored_error_releases_half_open_probe() -> None:
    async def run() -> None:
        await _reset_breaker()
        await rate_limit_module.circuit_breaker_report_failure()
        await rate_limit_module.circuit_breaker_report_failure()
        await asyncio.sleep(0.02)
        await rate_limit_module.circuit_breaker_before_attempt()
        await rate_limit_module.circuit_breaker_report_ignored()
        snapshot = await rate_limit_module.circuit_breaker_snapshot()
        assert snapshot["state"] == "half_open"
        assert snapshot["half_open_probes_in_flight"] == 0

    asyncio.run(run())
