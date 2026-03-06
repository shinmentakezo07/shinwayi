"""
Shin Proxy — OpenAI-compatible router.

Endpoints:
    GET  /health
    GET  /v1/models
    GET  /v1/internal/stats
    GET  /metrics
    POST /v1/chat/completions
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from shin.analytics import analytics, prom_metrics
from shin.app import get_http_client
from shin.cache import response_cache
from shin.config import settings
from shin.converters.from_cursor import now_ts
from shin.converters.to_cursor import openai_to_cursor
from shin.cursor.client import CursorClient
from shin.middleware.auth import verify_bearer
from shin.middleware.rate_limit import adaptive_concurrency_snapshot, circuit_breaker_snapshot
from shin.pipeline import (
    PipelineParams,
    _openai_stream,
    handle_openai_non_streaming,
    with_sse_heartbeat,
)
from shin.tools.normalize import normalize_openai_tools, to_anthropic_tool_format
from shin.validators import validate_openai_request, validate_payload_object

router = APIRouter()


@router.get("/health")
async def health(authorization: str | None = Header(default=None)):
    verify_bearer(authorization)
    return {"ok": True}


@router.get("/v1/models")
async def models(authorization: str | None = Header(default=None)):
    verify_bearer(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": "anthropic/claude-sonnet-4.6",
                "object": "model",
                "created": now_ts(),
                "owned_by": "anthropic",
                "context_length": 200_000,
            }
        ],
    }


@router.get("/v1/internal/stats")
async def internal_stats(authorization: str | None = Header(default=None)):
    verify_bearer(authorization)
    snapshot = analytics.snapshot()
    adaptive = await adaptive_concurrency_snapshot()
    breaker = await circuit_breaker_snapshot()
    snapshot["adaptive_concurrency"] = adaptive
    snapshot["circuit_breaker"] = breaker
    snapshot["cache"] = {
        "enabled": settings.cache_enabled,
        "backend": response_cache.backend_name,
        "redis_available": response_cache.redis_available,
    }
    return snapshot


@router.get("/metrics")
async def metrics():
    if not settings.metrics_enabled:
        return PlainTextResponse("# metrics disabled\n", media_type="text/plain; version=0.0.4")

    adaptive = await adaptive_concurrency_snapshot()
    breaker = await circuit_breaker_snapshot()
    prom_metrics.set_adaptive_concurrency(
        limit=adaptive.get("limit", 0),
        in_flight=adaptive.get("in_flight", 0),
        cooldown_remaining_seconds=adaptive.get("cooldown_remaining_seconds", 0.0),
    )
    prom_metrics.set_circuit_breaker(
        state=breaker.get("state", "closed"),
        open_remaining_seconds=breaker.get("open_remaining_seconds", 0.0),
        half_open_probes_in_flight=breaker.get("half_open_probes_in_flight", 0),
    )

    return PlainTextResponse(
        prom_metrics.render(),
        media_type="text/plain; version=0.0.4",
    )


@router.get("/v1/internal/credentials")
async def credential_status(authorization: str | None = Header(default=None)):
    verify_bearer(authorization)
    from shin.cursor.credentials import credential_pool
    return {
        "pool_size": credential_pool.size,
        "credentials": credential_pool.snapshot(),
    }


@router.post("/v1/internal/credentials/reset")
async def credential_reset(authorization: str | None = Header(default=None)):
    verify_bearer(authorization)
    from shin.cursor.credentials import credential_pool
    credential_pool.reset_all()
    return {"ok": True, "message": f"Reset {credential_pool.size} credentials"}


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(default=None),
):
    api_key = verify_bearer(authorization)
    request_started_at = time.perf_counter()

    payload = validate_payload_object(await request.json())

    stream = bool(payload.get("stream", False))
    model = payload.get("model") or settings.cursor_model
    messages = payload.get("messages", [])
    tools = normalize_openai_tools(payload.get("tools", []))
    tool_choice = payload.get("tool_choice", "auto")

    validate_openai_request(
        payload,
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
    )

    reasoning_effort = payload.get("reasoning_effort")
    if not reasoning_effort and isinstance(payload.get("reasoning"), dict):
        reasoning_effort = payload["reasoning"].get("effort")

    show_reasoning = bool(payload.get("show_reasoning", False))
    if isinstance(payload.get("reasoning"), dict):
        show_reasoning = show_reasoning or bool(
            payload["reasoning"].get("show", False)
        )

    parallel_tool_calls = bool(payload.get("parallel_tool_calls", True))

    # KEY INVARIANT: cursor_messages built BEFORE any stream/cache branch
    cursor_messages = openai_to_cursor(
        messages,
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
        show_reasoning=show_reasoning,
        model=model,
    )

    anthropic_tools = to_anthropic_tool_format(tools) if tools else None

    params = PipelineParams(
        api_style="openai",
        model=model,
        messages=messages,
        cursor_messages=cursor_messages,
        tools=tools,
        tool_choice=tool_choice,
        stream=stream,
        show_reasoning=show_reasoning,
        reasoning_effort=reasoning_effort,
        parallel_tool_calls=parallel_tool_calls,
        api_key=api_key,
        request_started_at=request_started_at,
    )

    client = CursorClient(get_http_client())

    if stream:
        return StreamingResponse(
            with_sse_heartbeat(
                _openai_stream(client, params, anthropic_tools),
                api_style="openai",
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    resp = await handle_openai_non_streaming(client, params, anthropic_tools)
    return JSONResponse(resp)
