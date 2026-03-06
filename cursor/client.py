"""
Shin Proxy — Async Cursor API client.

Wraps httpx.AsyncClient to communicate with Cursor's /api/chat endpoint.
Handles request building, SSE streaming, and error classification.
"""

from __future__ import annotations

import asyncio
import json
import platform
import time
import uuid
from typing import AsyncIterator, Protocol

import httpx
import structlog

from shin.analytics import prom_metrics
from shin.config import settings
from shin.cursor.credentials import CredentialPool, CredentialInfo, credential_pool
from shin.cursor.sse import parse_line, extract_delta
from shin.handlers import (
    BackendError,
    CircuitOpenError,
    CredentialError,
    EmptyResponseError,
    RateLimitError,
    TimeoutError,
)
from shin.middleware.rate_limit import (
    adaptive_concurrency_slot,
    circuit_breaker_before_attempt,
    circuit_breaker_report_failure,
    circuit_breaker_report_ignored,
    circuit_breaker_report_rate_limited,
    circuit_breaker_report_success,
)

log = structlog.get_logger()


class StreamObserver(Protocol):
    def on_stream_opened(self) -> None: ...

    def on_first_sse_line(self) -> None: ...

    def on_first_delta(self) -> None: ...

    def on_retry_sleep(self, seconds: float) -> None: ...

    def on_pre_first_delta_retry(self) -> None: ...


def _build_headers(
    cred: CredentialInfo | None = None,
    pool: CredentialPool | None = None,
) -> dict[str, str]:
    """Build request headers for Cursor's /api/chat endpoint."""
    _pool = pool or credential_pool
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": "RooCode/3.50.5",
        "x-stainless-retry-count": "0",
        "x-stainless-lang": "python",
        "x-stainless-package-version": "1.0.0",
        "x-stainless-os": platform.system() or "Windows",
        "x-stainless-arch": platform.machine().replace("AMD64", "x64") or "x64",
        "x-stainless-runtime": "python",
        "x-stainless-runtime-version": platform.python_version(),
        "Origin": settings.cursor_base_url,
        "Referer": f"{settings.cursor_base_url}/",
        "http-referer": "https://github.com/RooVetGit/Roo-Cline",
        "x-title": "Roo Code",
    }
    headers.update(_pool.get_auth_headers(cred))
    return headers


def _build_payload(
    cursor_messages: list[dict],
    model: str,
    anthropic_tools: list[dict] | None = None,
) -> dict:
    """Build the Cursor API request body."""
    payload: dict = {
        "context": [
            {
                "type": "file",
                "content": "",
                "filePath": settings.cursor_context_file_path,
            }
        ],
        "model": model or settings.cursor_model,
        "id": uuid.uuid4().hex[:16],
        "trigger": "submit-message",
        "messages": cursor_messages,
    }
    if anthropic_tools:
        payload["tools"] = anthropic_tools
    return payload


def classify_cursor_error(status: int, body: str) -> BackendError | CredentialError | RateLimitError:
    """Map a Cursor HTTP error to a typed ProxyError."""
    match status:
        case 401:
            return CredentialError("Cursor credential rejected — cookie may be expired")
        case 403:
            return CredentialError("Cursor access forbidden — account may be banned")
        case 429:
            return RateLimitError("Cursor rate limit hit", retry_after=60)
        case 500 | 502 | 503:
            return BackendError(
                f"Cursor backend error {status}", upstream_status=status
            )
        case _:
            return BackendError(
                f"Unexpected Cursor response {status}: {body[:200]}"
            )


class CursorClient:
    """Async wrapper around Cursor's /api/chat SSE endpoint."""

    _MAX_INITIAL_RETRIES = 4

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        pool: CredentialPool | None = None,
    ) -> None:
        self._http = http_client
        self._pool = pool or credential_pool

    @property
    def chat_url(self) -> str:
        return f"{settings.cursor_base_url}/api/chat"

    async def stream(
        self,
        cursor_messages: list[dict],
        model: str,
        anthropic_tools: list[dict] | None = None,
        cred: CredentialInfo | None = None,
        observer: StreamObserver | None = None,
    ) -> AsyncIterator[str]:
        """Stream text deltas from Cursor /api/chat.

        Yields plain strings — never tool call JSON.
        The caller (pipeline) is responsible for tool-call detection.

        Resilience behavior:
        - Retry on upstream 429 / timeout before first token is emitted.
        - Fail over to another credential between retries.
        - Once first token is emitted, do not retry (to avoid duplicated output).
        """
        payload = _build_payload(cursor_messages, model, anthropic_tools)
        attempts = self._MAX_INITIAL_RETRIES if cred is None else 1
        last_exc: Exception | None = None

        for attempt in range(attempts):
            _cred = cred or self._pool.next()
            if cred is None and _cred is None:
                raise CredentialError("No Cursor credentials available")

            try:
                await circuit_breaker_before_attempt()
            except RuntimeError as exc:
                raise CircuitOpenError(
                    "Cursor upstream circuit breaker is open"
                ) from exc

            headers = _build_headers(_cred, self._pool)
            got_any = False
            attempt_started = time.perf_counter()

            try:
                async with adaptive_concurrency_slot() as slot:
                    async with self._http.stream(
                        "POST",
                        self.chat_url,
                        headers=headers,
                        content=json.dumps(payload),
                    ) as response:
                        if observer is not None and response.status_code == 200:
                            observer.on_stream_opened()

                        if response.status_code != 200:
                            body = await response.aread()
                            body_text = body.decode("utf-8", errors="ignore")
                            classified = classify_cursor_error(response.status_code, body_text)

                            if _cred:
                                # Smart failover: cool down bad credential, move to next.
                                if isinstance(classified, RateLimitError):
                                    self._pool.mark_rate_limited(_cred)
                                else:
                                    self._pool.mark_error(_cred)

                            if isinstance(classified, RateLimitError):
                                slot.mark_rate_limited()
                                await circuit_breaker_report_rate_limited()
                                prom_metrics.observe_upstream_attempt(
                                    provider="cursor",
                                    outcome="rate_limited",
                                    status_code=response.status_code,
                                    duration_seconds=(time.perf_counter() - attempt_started),
                                )
                            elif isinstance(classified, CredentialError):
                                slot.mark_failure()
                                await circuit_breaker_report_ignored()
                                prom_metrics.observe_upstream_attempt(
                                    provider="cursor",
                                    outcome="failure",
                                    status_code=response.status_code,
                                    duration_seconds=(time.perf_counter() - attempt_started),
                                )
                            else:
                                slot.mark_failure()
                                await circuit_breaker_report_failure()
                                prom_metrics.observe_upstream_attempt(
                                    provider="cursor",
                                    outcome="failure",
                                    status_code=response.status_code,
                                    duration_seconds=(time.perf_counter() - attempt_started),
                                )

                            if isinstance(classified, RateLimitError) and cred is None and attempt + 1 < attempts:
                                retry_sleep = min(1.0 * (attempt + 1), 3.0)
                                if observer is not None and not got_any:
                                    observer.on_pre_first_delta_retry()
                                    observer.on_retry_sleep(retry_sleep)
                                await asyncio.sleep(retry_sleep)
                                continue

                            raise classified

                        _LINE_TIMEOUT = settings.stream_line_timeout_seconds
                        _aiter = response.aiter_lines().__aiter__()
                        saw_sse_line = False
                        while True:
                            try:
                                raw_line = await asyncio.wait_for(
                                    _aiter.__anext__(),
                                    timeout=_LINE_TIMEOUT,
                                )
                            except StopAsyncIteration:
                                break
                            except asyncio.TimeoutError as exc:
                                raise TimeoutError(
                                    f"No SSE line for {_LINE_TIMEOUT}s — upstream stalled"
                                ) from exc

                            line = raw_line.strip()
                            if not line:
                                continue
                            if observer is not None and not saw_sse_line:
                                observer.on_first_sse_line()
                                saw_sse_line = True
                            event = parse_line(line)
                            if not event:
                                continue
                            if isinstance(event, dict) and event.get("done"):
                                break
                            delta = extract_delta(event)
                            if delta:
                                if not got_any:
                                    got_any = True
                                    if observer is not None:
                                        observer.on_first_delta()
                                yield delta

                        if not got_any:
                            slot.mark_failure()
                            await circuit_breaker_report_failure()
                            prom_metrics.observe_upstream_attempt(
                                provider="cursor",
                                outcome="empty",
                                status_code=200,
                                duration_seconds=(time.perf_counter() - attempt_started),
                            )
                            raise EmptyResponseError("Cursor returned 200 but empty body")

                        if _cred:
                            self._pool.mark_success(_cred)
                        slot.mark_success()
                        await circuit_breaker_report_success()
                        prom_metrics.observe_upstream_attempt(
                            provider="cursor",
                            outcome="success",
                            status_code=200,
                            duration_seconds=(time.perf_counter() - attempt_started),
                        )
                        return

            except httpx.ReadTimeout:
                if _cred:
                    self._pool.mark_timeout(_cred)
                last_exc = TimeoutError("Cursor read timeout")
                await circuit_breaker_report_failure()
                prom_metrics.observe_upstream_attempt(
                    provider="cursor",
                    outcome="timeout",
                    status_code=None,
                    duration_seconds=(time.perf_counter() - attempt_started),
                )
                if not got_any and cred is None and attempt + 1 < attempts:
                    retry_sleep = min(0.8 * (attempt + 1), 2.5)
                    if observer is not None:
                        observer.on_pre_first_delta_retry()
                        observer.on_retry_sleep(retry_sleep)
                    await asyncio.sleep(retry_sleep)
                    continue
                raise last_exc
            except httpx.ConnectTimeout:
                if _cred:
                    self._pool.mark_timeout(_cred)
                last_exc = TimeoutError("Cursor connect timeout")
                await circuit_breaker_report_failure()
                prom_metrics.observe_upstream_attempt(
                    provider="cursor",
                    outcome="timeout",
                    status_code=None,
                    duration_seconds=(time.perf_counter() - attempt_started),
                )
                if not got_any and cred is None and attempt + 1 < attempts:
                    retry_sleep = min(0.8 * (attempt + 1), 2.5)
                    if observer is not None:
                        observer.on_pre_first_delta_retry()
                        observer.on_retry_sleep(retry_sleep)
                    await asyncio.sleep(retry_sleep)
                    continue
                raise last_exc
            except httpx.ConnectError as exc:
                if _cred:
                    self._pool.mark_timeout(_cred)
                last_exc = BackendError(f"Cursor connection error: {exc}")
                await circuit_breaker_report_failure()
                prom_metrics.observe_upstream_attempt(
                    provider="cursor",
                    outcome="connect_error",
                    status_code=None,
                    duration_seconds=(time.perf_counter() - attempt_started),
                )
                if not got_any and cred is None and attempt + 1 < attempts:
                    retry_sleep = min(0.8 * (attempt + 1), 2.5)
                    if observer is not None:
                        observer.on_pre_first_delta_retry()
                        observer.on_retry_sleep(retry_sleep)
                    await asyncio.sleep(retry_sleep)
                    continue
                raise last_exc
            except httpx.RemoteProtocolError as exc:
                last_exc = BackendError(f"Cursor mid-stream protocol error: {exc}")
                await circuit_breaker_report_failure()
                prom_metrics.observe_upstream_attempt(
                    provider="cursor",
                    outcome="failure",
                    status_code=None,
                    duration_seconds=(time.perf_counter() - attempt_started),
                )
                raise last_exc
            except asyncio.IncompleteReadError as exc:
                last_exc = BackendError(f"Cursor mid-stream incomplete read: {exc}")
                await circuit_breaker_report_failure()
                prom_metrics.observe_upstream_attempt(
                    provider="cursor",
                    outcome="failure",
                    status_code=None,
                    duration_seconds=(time.perf_counter() - attempt_started),
                )
                raise last_exc
            except EmptyResponseError as exc:
                if _cred:
                    self._pool.mark_timeout(_cred)
                await circuit_breaker_report_failure()
                last_exc = exc
                if cred is None and attempt + 1 < attempts:
                    retry_sleep = 0.4
                    if observer is not None and not got_any:
                        observer.on_pre_first_delta_retry()
                        observer.on_retry_sleep(retry_sleep)
                    await asyncio.sleep(retry_sleep)
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        raise BackendError("All upstream attempts failed")

    async def call(
        self,
        cursor_messages: list[dict],
        model: str,
        anthropic_tools: list[dict] | None = None,
        cred: CredentialInfo | None = None,
    ) -> str:
        """Non-streaming call — collects full response text."""
        chunks: list[str] = []
        async for delta in self.stream(
            cursor_messages, model, anthropic_tools, cred
        ):
            chunks.append(delta)
        return "".join(chunks)
