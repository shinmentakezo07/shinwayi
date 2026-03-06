"""
Shin Proxy — Anthropic-compatible router.

Endpoint:
    POST /v1/messages
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from shin.app import get_http_client
from shin.config import settings
from shin.converters.to_cursor import anthropic_to_cursor
from shin.cursor.client import CursorClient
from shin.middleware.auth import verify_bearer
from shin.pipeline import (
    PipelineParams,
    _anthropic_stream,
    handle_anthropic_non_streaming,
    with_sse_heartbeat,
)
from shin.tools.normalize import normalize_anthropic_tools
from shin.validators import validate_anthropic_request, validate_payload_object

router = APIRouter()


@router.post("/v1/messages")
async def anthropic_messages(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
):
    # Support both Bearer token and x-api-key header
    if authorization:
        api_key = verify_bearer(authorization)
    else:
        from shin.handlers import AuthError

        if x_api_key != settings.master_key:
            raise AuthError("invalid x-api-key")
        api_key = x_api_key

    request_started_at = time.perf_counter()
    payload = validate_payload_object(await request.json())

    stream = bool(payload.get("stream", False))
    model = payload.get("model") or settings.cursor_model
    messages = payload.get("messages", [])

    # System text — can be string or list of text blocks
    system = payload.get("system", "")
    if isinstance(system, list):
        system_text = "\n".join(
            b.get("text", "")
            for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        system_text = system if isinstance(system, str) else ""

    tools = normalize_anthropic_tools(payload.get("tools", []))
    tool_choice = payload.get("tool_choice", "auto")

    validate_anthropic_request(
        payload,
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        system_text=system_text,
    )

    thinking = (
        payload.get("thinking")
        if isinstance(payload.get("thinking"), dict)
        else None
    )

    # KEY INVARIANT: cursor_messages built BEFORE any stream/cache branch
    cursor_messages, show_reasoning = anthropic_to_cursor(
        messages,
        system_text=system_text,
        tools=tools,
        tool_choice=tool_choice,
        thinking=thinking,
        model=model,
    )

    # Anthropic tools are already in the right format after normalisation
    from shin.tools.normalize import to_anthropic_tool_format

    anthropic_tools = to_anthropic_tool_format(tools) if tools else None

    params = PipelineParams(
        api_style="anthropic",
        model=model,
        messages=messages,
        cursor_messages=cursor_messages,
        tools=tools,
        tool_choice=tool_choice,
        stream=stream,
        show_reasoning=show_reasoning,
        reasoning_effort="medium" if show_reasoning else None,
        api_key=api_key,
        system_text=system_text,
        request_started_at=request_started_at,
    )

    client = CursorClient(get_http_client())

    if stream:
        return StreamingResponse(
            with_sse_heartbeat(
                _anthropic_stream(client, params, anthropic_tools),
                api_style="anthropic",
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    resp = await handle_anthropic_non_streaming(client, params, anthropic_tools)
    return JSONResponse(resp)
