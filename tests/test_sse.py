import asyncio

import shin.pipeline as pipeline_module
from shin.cursor.sse import extract_delta
from shin.pipeline import _StreamSemanticState, _maybe_log_suppressed_tool_payload, with_sse_heartbeat


def test_extract_delta_openai_nested_choices_content() -> None:
    assert extract_delta({"choices": [{"delta": {"content": "hello"}}]}) == "hello"


def test_extract_delta_nested_delta_text() -> None:
    assert extract_delta({"delta": {"text": "world"}}) == "world"


def test_extract_delta_plain_string_delta() -> None:
    assert extract_delta({"delta": "plain"}) == "plain"


def test_extract_delta_empty_event() -> None:
    assert extract_delta({}) == ""


def test_sse_heartbeat_emits_keepalive_while_idle() -> None:
    async def source():
        yield "data: start\n\n"
        await asyncio.sleep(0.02)
        yield "data: end\n\n"

    async def collect() -> list[str]:
        out: list[str] = []
        async for chunk in with_sse_heartbeat(source(), api_style="openai"):
            out.append(chunk)
        return out

    from shin.config import settings

    old_enabled = settings.stream_heartbeat_enabled
    old_interval = settings.stream_heartbeat_interval_seconds
    old_timeout = settings.stream_line_timeout_seconds
    settings.stream_heartbeat_enabled = True
    settings.stream_heartbeat_interval_seconds = 0.01
    settings.stream_line_timeout_seconds = 1
    try:
        chunks = asyncio.run(collect())
    finally:
        settings.stream_heartbeat_enabled = old_enabled
        settings.stream_heartbeat_interval_seconds = old_interval
        settings.stream_line_timeout_seconds = old_timeout

    assert chunks[0] == "data: start\n\n"
    assert ": keep-alive\n\n" in chunks
    assert chunks[-1] == "data: end\n\n"


def test_sse_heartbeat_preserves_fast_stream_without_keepalive() -> None:
    async def source():
        yield "data: one\n\n"
        yield "data: two\n\n"

    async def collect() -> list[str]:
        out: list[str] = []
        async for chunk in with_sse_heartbeat(source(), api_style="openai"):
            out.append(chunk)
        return out

    from shin.config import settings

    old_enabled = settings.stream_heartbeat_enabled
    old_interval = settings.stream_heartbeat_interval_seconds
    old_timeout = settings.stream_line_timeout_seconds
    settings.stream_heartbeat_enabled = True
    settings.stream_heartbeat_interval_seconds = 0.05
    settings.stream_line_timeout_seconds = 1
    try:
        chunks = asyncio.run(collect())
    finally:
        settings.stream_heartbeat_enabled = old_enabled
        settings.stream_heartbeat_interval_seconds = old_interval
        settings.stream_line_timeout_seconds = old_timeout

    assert chunks == ["data: one\n\n", "data: two\n\n"]


def test_suppressed_tool_payload_warning_is_not_logged_if_final_projection_is_clean(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(pipeline_module.log, "warning", lambda message: warnings.append(message))

    state = _StreamSemanticState()
    _maybe_log_suppressed_tool_payload("openai", state, True)
    _maybe_log_suppressed_tool_payload("openai", state, False, final=True)

    assert warnings == []
    assert state.suppressed_tool_payload_pending is False
    assert state.suppressed_tool_payload_logged is False


def test_suppressed_tool_payload_warning_is_logged_if_final_projection_still_suppressed(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(pipeline_module.log, "warning", lambda message: warnings.append(message))

    state = _StreamSemanticState()
    _maybe_log_suppressed_tool_payload("openai", state, True)
    _maybe_log_suppressed_tool_payload("openai", state, True, final=True)

    assert warnings == ["suppressed_raw_tool_payload_openai_stream"]
    assert state.suppressed_tool_payload_logged is True
