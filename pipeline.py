"""
Shin Proxy — Request pipeline.

Orchestrates the full lifecycle: validate → convert → call Cursor → parse → format.

KEY INVARIANT: `cursor_messages` is ALWAYS built before any branch on stream/cache.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import structlog

from shin.analytics import RequestLog, analytics, estimate_cost, prom_metrics
from shin.cache import response_cache
from shin.config import settings
from shin.tokens import estimate_from_messages, estimate_from_text, count_message_tokens
from shin.converters.from_cursor import (
    anthropic_content_block_delta,
    anthropic_content_block_start,
    anthropic_content_block_stop,
    anthropic_message_delta,
    anthropic_message_start,
    anthropic_message_stop,
    anthropic_non_streaming_response,
    convert_tool_calls_to_anthropic,
    enforce_output_policy,
    openai_chunk,
    openai_content_sse,
    openai_done,
    openai_non_streaming_response,
    openai_role_sse,
    openai_sse,
    openai_tool_call_argument_sse,
    openai_tool_call_start_sse,
    openai_usage_chunk,
    sanitize_visible_text,
    split_visible_reasoning,
)
from shin.cursor.client import CursorClient
from shin.handlers import (
    BackendError,
    CircuitOpenError,
    CredentialError,
    ProxyError,
    RateLimitError,
    StreamAbortError,
    TimeoutError,
)
from shin.tools.parse import (
    find_tool_hint_offset,
    parse_tool_calls_details,
    parse_tool_calls_from_text,
)


log = structlog.get_logger()


# ── Pipeline params ─────────────────────────────────────────────────────────

@dataclass
class PipelineParams:
    """All parameters needed for a single request through the pipeline."""

    api_style: str  # "openai" or "anthropic"
    model: str
    messages: list[dict]
    cursor_messages: list[dict]
    tools: list[dict] = field(default_factory=list)
    tool_choice: Any = "auto"
    stream: bool = False
    show_reasoning: bool = False
    reasoning_effort: str | None = None
    parallel_tool_calls: bool = True
    api_key: str = ""
    system_text: str = ""  # Anthropic only
    request_started_at: float | None = None


# ── Provider detection ──────────────────────────────────────────────────────

def _provider_from_model(model: str) -> str:
    ml = model.lower()
    if "gpt" in ml or "o1" in ml or "openai" in ml:
        return "openai"
    return "anthropic"


_HEARTBEAT_DONE_SENTINEL = object()


def _heartbeat_interval_seconds() -> float | None:
    """Return the validated downstream heartbeat interval or None if disabled."""
    if not settings.stream_heartbeat_enabled:
        return None
    interval = max(0.0, float(settings.stream_heartbeat_interval_seconds))
    if interval <= 0:
        return None
    line_timeout = float(settings.stream_line_timeout_seconds)
    if line_timeout > 0:
        interval = min(interval, max(0.001, line_timeout * 0.5))
    return interval


async def with_sse_heartbeat(
    stream: AsyncIterator[str],
    *,
    api_style: str,
) -> AsyncIterator[str]:
    """Wrap an SSE stream and emit keep-alive comments while idle."""
    interval = _heartbeat_interval_seconds()
    if interval is None:
        async for chunk in stream:
            yield chunk
        return

    queue: asyncio.Queue[object] = asyncio.Queue()

    async def _pump() -> None:
        try:
            async for chunk in stream:
                await queue.put(chunk)
        except Exception as exc:
            await queue.put(exc)
        finally:
            await queue.put(_HEARTBEAT_DONE_SENTINEL)

    pump_task = asyncio.create_task(_pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval)
            except asyncio.TimeoutError:
                prom_metrics.observe_stream_event(api_style=api_style, event="heartbeat")
                yield ": keep-alive\n\n"
                continue

            if item is _HEARTBEAT_DONE_SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        pump_task.cancel()
        with suppress(asyncio.CancelledError):
            await pump_task


@dataclass
class StreamStageTracker:
    """Tracks stream lifecycle timestamps for latency instrumentation."""

    api_style: str
    request_started_at: float
    stream_opened_at: float | None = None
    first_upstream_sse_line_at: float | None = None
    first_upstream_delta_at: float | None = None
    first_visible_content_at: float | None = None
    cumulative_retry_sleep_before_first_delta: float = 0.0
    pre_first_delta_retry_count: int = 0
    _observed: bool = False

    def mark_stream_opened(self) -> None:
        if self.stream_opened_at is None:
            self.stream_opened_at = time.perf_counter()

    def mark_first_upstream_sse_line(self) -> None:
        if self.first_upstream_sse_line_at is None:
            self.first_upstream_sse_line_at = time.perf_counter()

    def mark_first_upstream_delta(self) -> None:
        if self.first_upstream_delta_at is None:
            self.first_upstream_delta_at = time.perf_counter()

    def mark_first_visible_content(self) -> float | None:
        if self.first_visible_content_at is None:
            self.first_visible_content_at = time.perf_counter()
            self.observe_stage_metrics()
        return self.first_visible_content_at

    def add_retry_sleep(self, seconds: float) -> None:
        if self.first_upstream_delta_at is None and seconds > 0:
            self.cumulative_retry_sleep_before_first_delta += seconds

    def mark_retry_before_first_delta(self) -> None:
        if self.first_upstream_delta_at is None:
            self.pre_first_delta_retry_count += 1
            prom_metrics.observe_stream_event(
                api_style=self.api_style,
                event="retry_before_first_delta",
            )

    def observe_stage_metrics(self) -> None:
        if self._observed:
            return
        self._observed = True

        pairs = (
            (
                "request_to_stream_open",
                self.request_started_at,
                self.stream_opened_at,
            ),
            (
                "stream_open_to_first_upstream_sse_line",
                self.stream_opened_at,
                self.first_upstream_sse_line_at,
            ),
            (
                "first_upstream_sse_line_to_first_upstream_delta",
                self.first_upstream_sse_line_at,
                self.first_upstream_delta_at,
            ),
            (
                "stream_open_to_first_upstream_delta",
                self.stream_opened_at,
                self.first_upstream_delta_at,
            ),
            (
                "first_upstream_delta_to_first_visible",
                self.first_upstream_delta_at,
                self.first_visible_content_at,
            ),
            (
                "request_to_first_visible",
                self.request_started_at,
                self.first_visible_content_at,
            ),
        )
        for stage, start, end in pairs:
            if start is None or end is None:
                continue
            prom_metrics.observe_stream_stage_latency(
                api_style=self.api_style,
                stage=stage,
                seconds=max(0.0, end - start),
            )

        if self.cumulative_retry_sleep_before_first_delta > 0:
            prom_metrics.observe_stream_stage_latency(
                api_style=self.api_style,
                stage="retry_sleep_before_first_delta",
                seconds=max(0.0, self.cumulative_retry_sleep_before_first_delta),
            )


class _CursorStreamHooks:
    """Bridge Cursor client events into pipeline stage tracking."""

    def __init__(self, tracker: StreamStageTracker) -> None:
        self._tracker = tracker

    def on_stream_opened(self) -> None:
        self._tracker.mark_stream_opened()

    def on_first_sse_line(self) -> None:
        self._tracker.mark_first_upstream_sse_line()

    def on_first_delta(self) -> None:
        self._tracker.mark_first_upstream_delta()

    def on_retry_sleep(self, seconds: float) -> None:
        self._tracker.add_retry_sleep(seconds)

    def on_pre_first_delta_retry(self) -> None:
        self._tracker.mark_retry_before_first_delta()


class _ToolsStreamState:
    """Incremental parser state for tool-enabled streams."""

    HINT_LOOKBACK = 512

    def __init__(self) -> None:
        self.structured_hint_seen = False
        self.hint_offset: int | None = None
        self.parsed_calls: list[dict] = []
        self.streamed_visible_len = 0

    def mark_hint(self, hint_offset: int) -> None:
        self.structured_hint_seen = True
        if self.hint_offset is None:
            self.hint_offset = hint_offset


@dataclass
class _StreamSemanticState:
    """Shared semantic stream state reused by both streaming APIs."""

    acc: str = ""
    tool_state: _ToolsStreamState = field(default_factory=_ToolsStreamState)
    tool_mode: bool = False
    suppressed_tool_payload_pending: bool = False
    suppressed_tool_payload_logged: bool = False


@dataclass
class _StreamSemanticProjection:
    """Protocol-agnostic semantic view of the current stream buffer."""

    current_calls: list[dict]
    thinking_text: str | None
    visible_text: str
    suppressed: bool
    thinking_in_progress: bool


_REASONING_HINT_TOKENS = ("<thinking>", "</thinking>", "<final>", "</final>")
_TOOL_PREFIX_HINT_TOKENS = (
    "[assistant_tool_calls]",
    '"tool_calls"',
    '{"tool_calls"',
)
_STRUCTURED_PREFIX_TOKENS = _REASONING_HINT_TOKENS + _TOOL_PREFIX_HINT_TOKENS
_IDENTITY_QUERY_RE = re.compile(
    r"(?i)\b(who are you|what are you|your identity|identify yourself|what is your identity)\b"
)


def _find_structured_stream_hint_offset(text: str, start: int = 0) -> int | None:
    """Return the earliest reasoning/tool hint offset at or after start."""
    begin = max(0, start)
    hits: list[int] = []

    tool_hint = find_tool_hint_offset(text, begin)
    if tool_hint is not None:
        hits.append(tool_hint)

    for token in _REASONING_HINT_TOKENS:
        idx = (text or "").find(token, begin)
        if idx != -1:
            hits.append(idx)

    if not hits:
        return None
    return min(hits)


def _structured_prefix_holdback(text: str) -> int:
    """Return trailing chars to hold back while a structured prefix is ambiguous."""
    t = text or ""
    max_holdback = 0
    for token in _STRUCTURED_PREFIX_TOKENS:
        upper = min(len(t), len(token) - 1)
        for prefix_len in range(1, upper + 1):
            if t.endswith(token[:prefix_len]):
                max_holdback = max(max_holdback, prefix_len)
    return max_holdback


def _mark_first_stream_token(
    api_style: str,
    started_at: float,
    first_token_at: float | None,
) -> float:
    """Observe TTFT exactly once for the first emitted non-role stream payload."""
    if first_token_at is None:
        first_token_at = time.perf_counter()
        prom_metrics.observe_stream_ttft(
            api_style=api_style,
            seconds=max(0.0, first_token_at - started_at),
        )
    return first_token_at


def _maybe_mark_structured_hint(
    tool_state: _ToolsStreamState,
    acc: str,
    delta_text: str,
) -> None:
    """Switch to structured parsing only on strong tool/reasoning hints."""
    if tool_state.structured_hint_seen:
        return
    search_start = max(0, len(acc) - len(delta_text) - _ToolsStreamState.HINT_LOOKBACK)
    hint_offset = _find_structured_stream_hint_offset(acc, search_start)
    if hint_offset is None:
        suffix = acc[search_start:]
        if any(token in suffix for token in _STRUCTURED_PREFIX_TOKENS):
            return
        return
    tool_state.mark_hint(hint_offset)


def _sanitize_stream_visible_text(
    acc: str,
    tool_state: _ToolsStreamState,
) -> tuple[str, bool, str | None]:
    """Return visible text for tool-enabled streams without false suppression."""
    if tool_state.structured_hint_seen:
        thinking_text, final_text = split_visible_reasoning(acc)
        base_visible = final_text if thinking_text is not None else acc
        safe_text, suppressed = sanitize_visible_text(
            base_visible,
            tool_state.parsed_calls,
        )
        return safe_text, suppressed, thinking_text

    holdback = _structured_prefix_holdback(acc)
    safe_text = acc[:-holdback] if holdback else acc
    return safe_text, False, None


def _parse_stream_tool_calls(
    acc: str,
    params: PipelineParams,
    tool_state: _ToolsStreamState,
) -> list[dict]:
    """Parse current tool calls from the stream once structured hints exist."""
    if not params.tools or not tool_state.structured_hint_seen:
        return []

    parse_details = parse_tool_calls_details(
        acc,
        params.tools,
        start=tool_state.hint_offset or 0,
    )
    current_calls = parse_details.calls or []
    if current_calls and not params.parallel_tool_calls:
        current_calls = current_calls[:1]
    if current_calls:
        tool_state.parsed_calls = current_calls
    return current_calls


def _project_stream_visible_text(
    acc: str,
    params: PipelineParams,
    tool_state: _ToolsStreamState,
    force_identity: bool,
    *,
    preserve_reasoning_tags: bool = False,
    prefer_reasoning_projection: bool = False,
) -> tuple[str, bool, str | None]:
    """Project the current stream buffer into safe visible text."""
    thinking_text: str | None = None
    if params.tools or prefer_reasoning_projection:
        visible_text, suppressed, thinking_text = _sanitize_stream_visible_text(
            acc,
            tool_state,
        )
    else:
        if tool_state.structured_hint_seen:
            thinking_text, _ = split_visible_reasoning(acc)
        holdback = _structured_prefix_holdback(acc)
        visible_text = acc[:-holdback] if holdback else acc
        visible_text, suppressed = sanitize_visible_text(visible_text)

    visible_text = enforce_output_policy(
        visible_text,
        force_identity=force_identity,
        preserve_reasoning_tags=preserve_reasoning_tags,
    )
    return visible_text, suppressed, thinking_text


def _thinking_in_progress(acc: str) -> bool:
    """Return True while a thinking block has opened but not closed yet."""
    return "<thinking>" in acc and "</thinking>" not in acc


def _advance_stream_semantics(
    state: _StreamSemanticState,
    params: PipelineParams,
    *,
    delta_text: str,
    force_identity: bool,
    preserve_reasoning_tags: bool = False,
    prefer_reasoning_projection: bool = False,
) -> _StreamSemanticProjection:
    """Advance the shared semantic stream state by one delta."""
    state.acc += delta_text
    _maybe_mark_structured_hint(state.tool_state, state.acc, delta_text)
    current_calls = _parse_stream_tool_calls(state.acc, params, state.tool_state)
    visible_text, suppressed, thinking_text = _project_stream_visible_text(
        state.acc,
        params,
        state.tool_state,
        force_identity,
        preserve_reasoning_tags=preserve_reasoning_tags,
        prefer_reasoning_projection=prefer_reasoning_projection,
    )
    if suppressed:
        state.suppressed_tool_payload_pending = True
    elif visible_text:
        state.suppressed_tool_payload_pending = False
    return _StreamSemanticProjection(
        current_calls=current_calls,
        thinking_text=thinking_text,
        visible_text=visible_text,
        suppressed=suppressed,
        thinking_in_progress=bool(
            params.show_reasoning and _thinking_in_progress(state.acc)
        ),
    )


def _finalize_stream_visible_text(
    state: _StreamSemanticState,
    *,
    force_identity: bool,
) -> tuple[str, bool, str | None]:
    """Build the final visible-text projection from the accumulated stream."""
    thinking_text, final_text = split_visible_reasoning(state.acc)
    final_candidate = final_text if thinking_text is not None else state.acc
    visible_text, suppressed = sanitize_visible_text(
        final_candidate,
        state.tool_state.parsed_calls,
    )
    visible_text = enforce_output_policy(
        visible_text,
        force_identity=force_identity,
    )
    return visible_text, suppressed, thinking_text


def _take_visible_text_delta(
    tool_state: _ToolsStreamState,
    visible_text: str,
) -> str:
    """Return newly visible text and advance the emitted-length cursor.

    Clamps the cursor to visible_text length in case sanitization
    shrinks the text (e.g., a tool payload that was streamed then suppressed).
    """
    safe_cursor = min(tool_state.streamed_visible_len, len(visible_text))
    if len(visible_text) <= safe_cursor:
        tool_state.streamed_visible_len = safe_cursor
        return ""
    delta = visible_text[safe_cursor:]
    tool_state.streamed_visible_len = len(visible_text)
    return delta


def _take_reasoning_delta(
    thinking_text: str | None,
    thinking_sent: int,
) -> tuple[str, int]:
    """Return the newly available reasoning suffix and updated cursor."""
    if not thinking_text or len(thinking_text) <= thinking_sent:
        return "", thinking_sent
    return thinking_text[thinking_sent:], len(thinking_text)


def _maybe_log_suppressed_tool_payload(
    api_style: str,
    state: _StreamSemanticState,
    suppressed: bool,
    *,
    final: bool = False,
) -> None:
    """Log raw tool-payload suppression only when it persists to final output."""
    if suppressed:
        state.suppressed_tool_payload_pending = True
    if not final:
        return
    if not suppressed:
        state.suppressed_tool_payload_pending = False
        return
    if not state.suppressed_tool_payload_logged:
        log.warning(f"suppressed_raw_tool_payload_{api_style}_stream")
        state.suppressed_tool_payload_logged = True


def _emit_openai_content_delta(
    chunk_id: str,
    model: str,
    content: str,
) -> str:
    """Build an OpenAI content SSE chunk for an already-decided delta."""
    return openai_content_sse(chunk_id, model, content)


def _emit_openai_tool_call_start(
    chunk_id: str,
    model: str,
    *,
    index: int,
    call_id: str | None,
    name: str | None,
) -> str:
    """Build an OpenAI tool-call header SSE chunk."""
    return openai_tool_call_start_sse(
        chunk_id,
        model,
        index=index,
        call_id=call_id,
        name=name,
    )


def _emit_openai_tool_argument_delta(
    chunk_id: str,
    model: str,
    *,
    index: int,
    arguments: str,
) -> str:
    """Build an OpenAI tool-call argument SSE chunk."""
    return openai_tool_call_argument_sse(
        chunk_id,
        model,
        index=index,
        arguments=arguments,
    )


def _emit_anthropic_text_start(index: int) -> str:
    """Build an Anthropic text block start event."""
    return anthropic_content_block_start(index, {"type": "text", "text": ""})


def _emit_anthropic_text_delta(index: int, text: str) -> str:
    """Build an Anthropic text delta event."""
    return anthropic_content_block_delta(index, {"type": "text_delta", "text": text})


def _emit_anthropic_thinking_start(index: int) -> str:
    """Build an Anthropic thinking block start event."""
    return anthropic_content_block_start(index, {"type": "thinking", "thinking": ""})


def _emit_anthropic_thinking_delta(index: int, thinking: str) -> str:
    """Build an Anthropic thinking delta event."""
    return anthropic_content_block_delta(
        index,
        {"type": "thinking_delta", "thinking": thinking},
    )


def _emit_anthropic_tool_use_start(
    index: int,
    *,
    tool_id: str | None,
    name: str | None,
    input_payload: dict,
) -> str:
    """Build an Anthropic tool_use block start event."""
    return anthropic_content_block_start(
        index,
        {
            "type": "tool_use",
            "id": tool_id,
            "name": name,
            "input": input_payload,
        },
    )


def _extract_user_text(messages: list[dict]) -> str:
    """Extract concatenated text content from user messages only."""
    out: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                ):
                    out.append(block.get("text", ""))
    return "\n".join(out)


def _identity_forced(messages: list[dict]) -> bool:
    """Return True when the user asks a direct identity question."""
    return bool(_IDENTITY_QUERY_RE.search(_extract_user_text(messages) or ""))


# ── Retry wrapper ───────────────────────────────────────────────────────────

async def _call_with_retry(
    client: CursorClient,
    params: PipelineParams,
    anthropic_tools: list[dict] | None,
) -> str:
    """Non-streaming call with retry logic per error class."""
    last_exc: Exception | None = None

    for attempt in range(settings.retry_attempts):
        try:
            return await client.call(
                params.cursor_messages,
                params.model,
                anthropic_tools,
            )
        except CircuitOpenError:
            raise
        except (CredentialError, TimeoutError) as exc:
            last_exc = exc
            if attempt + 1 < settings.retry_attempts:
                await asyncio.sleep(settings.retry_backoff_seconds * (attempt + 1))
            continue
        except RateLimitError as exc:
            last_exc = exc
            await asyncio.sleep(settings.retry_backoff_seconds * (attempt + 1))
            continue
        except BackendError as exc:
            last_exc = exc
            if attempt + 1 < settings.retry_attempts:
                await asyncio.sleep(settings.retry_backoff_seconds * (attempt + 1))
            continue

    if isinstance(last_exc, ProxyError):
        raise last_exc
    raise BackendError(f"All upstream attempts failed: {last_exc}")


# ── OpenAI streaming generator ──────────────────────────────────────────────

async def _openai_stream(
    client: CursorClient,
    params: PipelineParams,
    anthropic_tools: list[dict] | None,
) -> AsyncIterator[str]:
    """Generate OpenAI SSE chunks from Cursor stream."""
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    model = params.model
    started = time.perf_counter()
    request_started_at = params.request_started_at or started
    stage_tracker = StreamStageTracker("openai", request_started_at)
    stream_observer = _CursorStreamHooks(stage_tracker)

    input_tokens = count_message_tokens(params.messages, model)

    yield openai_role_sse(cid, model)

    state = _StreamSemanticState()
    emitted_signatures: dict[str, dict] = {}
    emitted_count = 0
    finish_reason = "stop"
    first_token_at: float | None = None
    thinking_opened = False
    thinking_sent = 0
    force_identity = _identity_forced(params.messages)

    prom_metrics.observe_stream_event(api_style="openai", event="start")

    try:
        async for delta_text in client.stream(
            params.cursor_messages,
            model,
            anthropic_tools,
            observer=stream_observer,
        ):
            projection = _advance_stream_semantics(
                state,
                params,
                delta_text=delta_text,
                force_identity=force_identity,
                preserve_reasoning_tags=bool(params.show_reasoning and not params.tools),
            )
            _maybe_log_suppressed_tool_payload(
                "openai",
                state,
                projection.suppressed,
            )

            if projection.current_calls:
                if params.show_reasoning and projection.thinking_text:
                    if not thinking_opened:
                        first_token_at = _mark_first_stream_token(
                            "openai", started, first_token_at
                        )
                        stage_tracker.mark_first_visible_content()
                        yield _emit_openai_content_delta(cid, model, "<thinking>")
                        thinking_opened = True
                    reasoning_delta, thinking_sent = _take_reasoning_delta(
                        projection.thinking_text,
                        thinking_sent,
                    )
                    if reasoning_delta:
                        first_token_at = _mark_first_stream_token(
                            "openai", started, first_token_at
                        )
                        stage_tracker.mark_first_visible_content()
                        yield _emit_openai_content_delta(cid, model, reasoning_delta)

                if not projection.thinking_in_progress:
                    visible_delta = _take_visible_text_delta(
                        state.tool_state,
                        projection.visible_text,
                    )
                    if visible_delta:
                        first_token_at = _mark_first_stream_token(
                            "openai", started, first_token_at
                        )
                        stage_tracker.mark_first_visible_content()
                        yield _emit_openai_content_delta(cid, model, visible_delta)

                state.tool_mode = True
                for tc in projection.current_calls:
                    sig = json.dumps(
                        tc.get("function", {}), sort_keys=True, ensure_ascii=False
                    )
                    fn = tc.get("function", {})
                    call_id = tc.get("id")
                    fn_name = fn.get("name")
                    args_text = fn.get("arguments", "{}")

                    rec = emitted_signatures.get(sig)
                    if rec is None:
                        rec = {"index": emitted_count, "sent": 0}
                        emitted_signatures[sig] = rec
                        first_token_at = _mark_first_stream_token(
                            "openai", started, first_token_at
                        )
                        yield _emit_openai_tool_call_start(
                            cid,
                            model,
                            index=emitted_count,
                            call_id=call_id,
                            name=fn_name,
                        )
                        emitted_count += 1

                    sent = rec["sent"]
                    if len(args_text) > sent:
                        remaining = args_text[sent:]
                        rec["sent"] = len(args_text)
                        first_token_at = _mark_first_stream_token(
                            "openai", started, first_token_at
                        )
                        yield _emit_openai_tool_argument_delta(
                            cid,
                            model,
                            index=rec["index"],
                            arguments=remaining,
                        )
                continue

            if state.tool_mode:
                continue

            if params.tools and projection.thinking_in_progress:
                continue

            visible_delta = _take_visible_text_delta(
                state.tool_state,
                projection.visible_text,
            )
            if visible_delta:
                first_token_at = _mark_first_stream_token(
                    "openai", started, first_token_at
                )
                stage_tracker.mark_first_visible_content()
                yield _emit_openai_content_delta(cid, model, visible_delta)

        if state.tool_mode:
            finish_reason = "tool_calls"
            if thinking_opened:
                first_token_at = _mark_first_stream_token(
                    "openai", started, first_token_at
                )
                stage_tracker.mark_first_visible_content()
                yield _emit_openai_content_delta(cid, model, "</thinking>\n\n")
        elif params.tools:
            final_visible, suppressed, thinking_text = _finalize_stream_visible_text(
                state,
                force_identity=force_identity,
            )
            _maybe_log_suppressed_tool_payload(
                "openai",
                state,
                suppressed,
                final=True,
            )

            if (
                params.show_reasoning
                and thinking_text
                and state.tool_state.streamed_visible_len == 0
                and final_visible
                and not force_identity
            ):
                final_visible = (
                    f"<thinking>{thinking_text}</thinking>\n\n<final>{final_visible}</final>"
                )
            final_visible = enforce_output_policy(
                final_visible,
                force_identity=force_identity,
                preserve_reasoning_tags=bool(params.show_reasoning and thinking_text),
            )

            final_delta = _take_visible_text_delta(
                state.tool_state,
                final_visible,
            )
            if final_delta:
                first_token_at = _mark_first_stream_token(
                    "openai", started, first_token_at
                )
                stage_tracker.mark_first_visible_content()
                yield _emit_openai_content_delta(cid, model, final_delta)

    except StreamAbortError:
        prom_metrics.observe_stream_event(api_style="openai", event="aborted")
    except Exception as exc:
        log.exception("stream_error_openai")
        prom_metrics.observe_stream_event(api_style="openai", event="error")
        yield openai_sse(
            {"error": {"message": str(exc)[:200], "type": "stream_error"}}
        )

    stage_tracker.observe_stage_metrics()

    yield openai_sse(openai_chunk(cid, model, finish_reason=finish_reason))
    output_tokens = estimate_from_text(state.acc, model)
    yield openai_usage_chunk(cid, model, input_tokens, output_tokens)
    yield openai_done()

    prom_metrics.observe_stream_event(api_style="openai", event="done")

    latency_ms = (time.perf_counter() - started) * 1000.0
    provider = _provider_from_model(model)
    _record(params, provider, state.acc, latency_ms)


# ── Anthropic streaming generator ──────────────────────────────────────────

async def _anthropic_stream(
    client: CursorClient,
    params: PipelineParams,
    anthropic_tools: list[dict] | None,
) -> AsyncIterator[str]:
    """Generate Anthropic SSE events from Cursor stream."""
    mid = f"msg_{uuid.uuid4().hex[:24]}"
    model = params.model
    started = time.perf_counter()
    request_started_at = params.request_started_at or started
    stage_tracker = StreamStageTracker("anthropic", request_started_at)
    stream_observer = _CursorStreamHooks(stage_tracker)

    input_tokens = count_message_tokens(params.messages, model)

    yield anthropic_message_start(mid, model, input_tokens)

    state = _StreamSemanticState()
    idx = 0
    thinking_opened = False
    thinking_completed = False
    thinking_sent = 0
    text_opened = False
    emitted_signatures: set[str] = set()
    first_token_at: float | None = None
    force_identity = _identity_forced(params.messages)

    prom_metrics.observe_stream_event(api_style="anthropic", event="start")

    try:
        async for delta_text in client.stream(
            params.cursor_messages,
            model,
            anthropic_tools,
            observer=stream_observer,
        ):
            projection = _advance_stream_semantics(
                state,
                params,
                delta_text=delta_text,
                force_identity=force_identity,
                prefer_reasoning_projection=True,
            )
            _maybe_log_suppressed_tool_payload(
                "anthropic",
                state,
                projection.suppressed,
            )

            if params.show_reasoning and projection.thinking_text and not state.tool_mode:
                if not thinking_opened and not thinking_completed:
                    first_token_at = _mark_first_stream_token(
                        "anthropic", started, first_token_at
                    )
                    stage_tracker.mark_first_visible_content()
                    yield _emit_anthropic_thinking_start(idx)
                    thinking_opened = True
                reasoning_delta, thinking_sent = _take_reasoning_delta(
                    projection.thinking_text,
                    thinking_sent,
                )
                if reasoning_delta:
                    first_token_at = _mark_first_stream_token(
                        "anthropic", started, first_token_at
                    )
                    stage_tracker.mark_first_visible_content()
                    yield _emit_anthropic_thinking_delta(idx, reasoning_delta)
                if thinking_opened and not projection.thinking_in_progress:
                    yield anthropic_content_block_stop(idx)
                    thinking_opened = False
                    thinking_completed = True
                    idx += 1

            if projection.current_calls:
                if thinking_opened:
                    yield anthropic_content_block_stop(idx)
                    thinking_opened = False
                    thinking_completed = True
                    idx += 1
                state.tool_mode = True
                for tc in projection.current_calls:
                    sig = json.dumps(
                        tc.get("function", {}), sort_keys=True, ensure_ascii=False
                    )
                    if sig in emitted_signatures:
                        continue
                    emitted_signatures.add(sig)
                    fn = tc.get("function", {})
                    try:
                        inp = (
                            json.loads(fn.get("arguments", "{}"))
                            if isinstance(fn.get("arguments"), str)
                            else fn.get("arguments", {})
                        )
                    except Exception:
                        inp = {}

                    first_token_at = _mark_first_stream_token(
                        "anthropic", started, first_token_at
                    )
                    yield _emit_anthropic_tool_use_start(
                        idx,
                        tool_id=tc.get("id"),
                        name=fn.get("name"),
                        input_payload=inp,
                    )
                    yield anthropic_content_block_stop(idx)
                    idx += 1
                continue

            if state.tool_mode:
                continue

            if projection.thinking_in_progress:
                continue

            text_delta = _take_visible_text_delta(
                state.tool_state,
                projection.visible_text,
            )
            if text_delta:
                if thinking_opened:
                    yield anthropic_content_block_stop(idx)
                    thinking_opened = False
                    thinking_completed = True
                    idx += 1
                if not text_opened:
                    first_token_at = _mark_first_stream_token(
                        "anthropic", started, first_token_at
                    )
                    stage_tracker.mark_first_visible_content()
                    yield _emit_anthropic_text_start(idx)
                    text_opened = True
                first_token_at = _mark_first_stream_token(
                    "anthropic", started, first_token_at
                )
                stage_tracker.mark_first_visible_content()
                yield _emit_anthropic_text_delta(idx, text_delta)

        if not state.tool_mode:
            if thinking_opened:
                yield anthropic_content_block_stop(idx)
                thinking_opened = False
                thinking_completed = True
                idx += 1
            final_visible, suppressed, _ = _finalize_stream_visible_text(
                state,
                force_identity=force_identity,
            )
            _maybe_log_suppressed_tool_payload(
                "anthropic",
                state,
                suppressed,
                final=True,
            )
            final_delta = _take_visible_text_delta(
                state.tool_state,
                final_visible,
            )
            if final_delta:
                if not text_opened:
                    first_token_at = _mark_first_stream_token(
                        "anthropic", started, first_token_at
                    )
                    stage_tracker.mark_first_visible_content()
                    yield _emit_anthropic_text_start(idx)
                    text_opened = True
                first_token_at = _mark_first_stream_token(
                    "anthropic", started, first_token_at
                )
                stage_tracker.mark_first_visible_content()
                yield _emit_anthropic_text_delta(idx, final_delta)

    except StreamAbortError:
        prom_metrics.observe_stream_event(api_style="anthropic", event="aborted")
    except Exception as exc:
        log.exception("stream_error_anthropic")
        prom_metrics.observe_stream_event(api_style="anthropic", event="error")
        from shin.converters.from_cursor import anthropic_sse_event

        yield anthropic_sse_event(
            "error",
            {"type": "error", "error": {"type": "api_error", "message": str(exc)[:200]}},
        )

    stage_tracker.observe_stage_metrics()

    output_tokens = estimate_from_text(state.acc, model)
    if state.tool_mode:
        yield anthropic_message_delta("tool_use", output_tokens)
    else:
        if thinking_opened:
            yield anthropic_content_block_stop(idx)
            idx += 1
        if text_opened:
            yield anthropic_content_block_stop(idx)
        yield anthropic_message_delta("end_turn", output_tokens)

    yield anthropic_message_stop()

    prom_metrics.observe_stream_event(api_style="anthropic", event="done")

    latency_ms = (time.perf_counter() - started) * 1000.0
    provider = _provider_from_model(model)
    _record(params, provider, state.acc, latency_ms)


# ── Non-streaming handlers ─────────────────────────────────────────────────

async def handle_openai_non_streaming(
    client: CursorClient,
    params: PipelineParams,
    anthropic_tools: list[dict] | None,
) -> dict:
    """Handle a non-streaming OpenAI request."""
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    # Cache check
    cache_key = response_cache.build_key(
        api_style="openai",
        model=params.model,
        messages=params.messages,
        tools=params.tools,
        tool_choice=params.tool_choice,
        reasoning_effort=params.reasoning_effort,
        show_reasoning=params.show_reasoning,
    )
    cached = response_cache.get(cache_key)
    if cached is not None:
        _record(params, _provider_from_model(params.model), "", 0.0, cache_hit=True)
        prom_metrics.observe_cache(
            api_style="openai", backend=response_cache.backend_name, result="hit"
        )
        return cached

    prom_metrics.observe_cache(
        api_style="openai", backend=response_cache.backend_name, result="miss"
    )

    started = time.time()
    text = await _call_with_retry(client, params, anthropic_tools)
    latency_ms = (time.time() - started) * 1000.0
    force_identity = _identity_forced(params.messages)

    parsed_calls = parse_tool_calls_from_text(text, params.tools)
    if parsed_calls and not params.parallel_tool_calls:
        parsed_calls = parsed_calls[:1]

    thinking_text, final_text = split_visible_reasoning(text)
    base_visible = final_text if thinking_text is not None else text
    visible_text, suppressed = sanitize_visible_text(base_visible, parsed_calls)
    visible_text = enforce_output_policy(
        visible_text,
        force_identity=force_identity,
    )
    if suppressed:
        log.warning("suppressed_raw_tool_payload_openai_nonstream")

    if parsed_calls:
        message = {"role": "assistant", "content": None, "tool_calls": parsed_calls}
        finish_reason = "tool_calls"
    else:
        if (
            params.show_reasoning
            and thinking_text
            and visible_text
            and not force_identity
        ):
            visible_text = f"<thinking>{thinking_text}</thinking>\n\n<final>{visible_text}</final>"
        message = {"role": "assistant", "content": visible_text}
        finish_reason = "stop"

    input_tokens = count_message_tokens(params.messages, params.model)
    output_tokens = estimate_from_text(visible_text or text, params.model)

    resp = openai_non_streaming_response(
        cid,
        params.model,
        message,
        finish_reason,
        params.reasoning_effort,
        params.show_reasoning,
        thinking_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    provider = _provider_from_model(params.model)
    _record(params, provider, visible_text or text, latency_ms)
    response_cache.set(cache_key, resp)
    prom_metrics.observe_cache(
        api_style="openai", backend=response_cache.backend_name, result="set"
    )
    return resp


async def handle_anthropic_non_streaming(
    client: CursorClient,
    params: PipelineParams,
    anthropic_tools: list[dict] | None,
) -> dict:
    """Handle a non-streaming Anthropic request."""
    mid = f"msg_{uuid.uuid4().hex[:24]}"

    # Cache check
    cache_key = response_cache.build_key(
        api_style="anthropic",
        model=params.model,
        messages=params.messages,
        tools=params.tools,
        tool_choice=params.tool_choice,
        reasoning_effort=(params.reasoning_effort),
        show_reasoning=params.show_reasoning,
        system_text=params.system_text,
    )
    cached = response_cache.get(cache_key)
    if cached is not None:
        _record(params, _provider_from_model(params.model), "", 0.0, cache_hit=True)
        prom_metrics.observe_cache(
            api_style="anthropic", backend=response_cache.backend_name, result="hit"
        )
        return cached

    prom_metrics.observe_cache(
        api_style="anthropic", backend=response_cache.backend_name, result="miss"
    )

    started = time.time()
    text = await _call_with_retry(client, params, anthropic_tools)
    latency_ms = (time.time() - started) * 1000.0
    force_identity = _identity_forced(params.messages)

    parsed_calls = parse_tool_calls_from_text(text, params.tools)
    thinking_text, final_text = split_visible_reasoning(text)

    content_blocks: list[dict] = []
    if params.show_reasoning and thinking_text:
        content_blocks.append({"type": "thinking", "thinking": thinking_text})

    if parsed_calls:
        tool_blocks = convert_tool_calls_to_anthropic(parsed_calls)
        # If we had thinking blocks and tool calls, keep thinking; otherwise start fresh
        if not content_blocks:
            content_blocks = []
        content_blocks.extend(tool_blocks)
        stop_reason = "tool_use"
    else:
        base_text = final_text if thinking_text is not None else text
        safe_text, suppressed = sanitize_visible_text(base_text)
        safe_text = enforce_output_policy(
            safe_text,
            force_identity=force_identity,
        )
        if suppressed:
            log.warning("suppressed_raw_tool_payload_anthropic_nonstream")
        content_blocks.append({"type": "text", "text": safe_text})
        stop_reason = "end_turn"

    input_tokens = count_message_tokens(params.messages, params.model)
    output_tokens = estimate_from_text(text, params.model)

    resp = anthropic_non_streaming_response(
        mid, params.model, content_blocks, stop_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    provider = _provider_from_model(params.model)
    _record(params, provider, text, latency_ms)
    response_cache.set(cache_key, resp)
    prom_metrics.observe_cache(
        api_style="anthropic", backend=response_cache.backend_name, result="set"
    )
    return resp


# ── Analytics helper ────────────────────────────────────────────────────────

def _record(
    params: PipelineParams,
    provider: str,
    text: str,
    latency_ms: float,
    cache_hit: bool = False,
) -> None:
    input_tokens = estimate_from_messages(params.messages, params.model)
    output_tokens = estimate_from_text(text, params.model)
    cost = estimate_cost(provider, input_tokens, output_tokens)
    analytics.record(
        RequestLog(
            api_key=params.api_key,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
        )
    )
