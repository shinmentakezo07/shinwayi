"""
Shin Proxy — Convert Cursor SSE output to OpenAI / Anthropic response shapes.

Includes chunk formatters, reasoning tag extraction, and text sanitization.
"""

from __future__ import annotations

import json
import re
import time
import uuid

# ── Context window registry ────────────────────────────────────────────────
# Maps real Cursor model IDs to their context window size in tokens.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "anthropic/claude-sonnet-4.6": 200_000,
    "anthropic/claude-3-7-sonnet": 200_000,
    "anthropic/claude-3-5-sonnet": 200_000,
    "anthropic/claude-3-opus": 200_000,
    "openai/gpt-4.1": 1_047_576,
    "openai/gpt-4o": 128_000,
    "openai/gpt-4.5": 128_000,
    "openai/o3": 200_000,
    "openai/o4-mini": 200_000,
    "google/gemini-2.5-pro": 1_048_576,
    "google/gemini-2.0-flash": 1_048_576,
}
DEFAULT_CONTEXT_WINDOW = 200_000


def context_window_for(model: str) -> int:
    """Return the context window size for the given model ID."""
    return MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)

# ── Compiled patterns ───────────────────────────────────────────────────────

_INTERNAL_TOOL_MARKER_RE = re.compile(
    r"\[assistant_tool_calls\]", flags=re.IGNORECASE
)
_TOP_LEVEL_TOOL_CALLS_RE = re.compile(
    r'"tool_calls"\s*:\s*\[', flags=re.IGNORECASE
)
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)\s*```", flags=re.IGNORECASE
)
_XML_TAG_RE = re.compile(r"<[^>]+>", flags=re.IGNORECASE)
_CURSOR_WORD_RE = re.compile(r"\bcursor\b", flags=re.IGNORECASE)


def now_ts() -> int:
    return int(time.time())


# ── OpenAI response formatters ──────────────────────────────────────────────

def openai_chunk(
    chunk_id: str,
    model: str,
    delta: dict | None = None,
    finish_reason: str | None = None,
) -> dict:
    """Build an OpenAI chat.completion.chunk dict."""
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": now_ts(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta or {},
                "finish_reason": finish_reason,
            }
        ],
    }


def openai_sse(payload: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


def openai_role_sse(chunk_id: str, model: str) -> str:
    """Emit the initial OpenAI assistant role chunk."""
    return openai_sse(openai_chunk(chunk_id, model, delta={"role": "assistant"}))


def openai_content_sse(chunk_id: str, model: str, content: str) -> str:
    """Emit an OpenAI assistant content delta chunk."""
    return openai_sse(openai_chunk(chunk_id, model, delta={"content": content}))


def openai_tool_call_start_sse(
    chunk_id: str,
    model: str,
    *,
    index: int,
    call_id: str | None,
    name: str | None,
) -> str:
    """Emit an OpenAI tool-call start delta chunk."""
    return openai_sse(
        openai_chunk(
            chunk_id,
            model,
            delta={
                "tool_calls": [
                    {
                        "index": index,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }
                ]
            },
        )
    )


def openai_tool_call_argument_sse(
    chunk_id: str,
    model: str,
    *,
    index: int,
    arguments: str,
) -> str:
    """Emit an OpenAI tool-call arguments delta chunk."""
    return openai_sse(
        openai_chunk(
            chunk_id,
            model,
            delta={
                "tool_calls": [
                    {
                        "index": index,
                        "function": {"arguments": arguments},
                    }
                ]
            },
        )
    )


def openai_done() -> str:
    return "data: [DONE]\n\n"


def openai_non_streaming_response(
    chunk_id: str,
    model: str,
    message: dict,
    finish_reason: str = "stop",
    reasoning_effort: str | None = None,
    show_reasoning: bool = False,
    thinking_text: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict:
    """Build a full OpenAI chat.completion response (non-streaming)."""
    ctx = context_window_for(model)
    resp: dict = {
        "id": chunk_id,
        "object": "chat.completion",
        "created": now_ts(),
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "context_window": ctx,
            "context_window_used_pct": round(
                (input_tokens + output_tokens) / ctx * 100, 2
            ),
        },
    }
    if reasoning_effort:
        resp["reasoning"] = {"effort": reasoning_effort, "show": show_reasoning}
    if show_reasoning and thinking_text:
        resp["thinking"] = thinking_text
    return resp


# ── Anthropic response formatters ──────────────────────────────────────────

def anthropic_sse_event(event_type: str, data: dict) -> str:
    """Format an Anthropic SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def openai_usage_chunk(chunk_id: str, model: str, input_tokens: int, output_tokens: int) -> str:
    """Emit a final SSE chunk carrying usage (sent after finish_reason chunk)."""
    ctx = context_window_for(model)
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": now_ts(),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "context_window": ctx,
            "context_window_used_pct": round(
                (input_tokens + output_tokens) / ctx * 100, 2
            ),
        },
    }
    return openai_sse(payload)


def anthropic_message_start(msg_id: str, model: str, input_tokens: int = 0) -> str:
    ctx = context_window_for(model)
    start = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "context_window": ctx,
                "context_window_used_pct": round(input_tokens / ctx * 100, 2),
            },
        },
    }
    return anthropic_sse_event("message_start", start)


def anthropic_content_block_start(index: int, block: dict) -> str:
    return anthropic_sse_event(
        "content_block_start",
        {"type": "content_block_start", "index": index, "content_block": block},
    )


def anthropic_content_block_delta(index: int, delta: dict) -> str:
    return anthropic_sse_event(
        "content_block_delta",
        {"type": "content_block_delta", "index": index, "delta": delta},
    )


def anthropic_content_block_stop(index: int) -> str:
    return anthropic_sse_event(
        "content_block_stop",
        {"type": "content_block_stop", "index": index},
    )


def anthropic_message_delta(stop_reason: str, output_tokens: int = 0) -> str:
    return anthropic_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )


def anthropic_message_stop() -> str:
    return anthropic_sse_event(
        "message_stop", {"type": "message_stop"}
    )


def anthropic_non_streaming_response(
    msg_id: str,
    model: str,
    content_blocks: list[dict],
    stop_reason: str = "end_turn",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict:
    ctx = context_window_for(model)
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "context_window": ctx,
            "context_window_used_pct": round(
                (input_tokens + output_tokens) / ctx * 100, 2
            ),
        },
    }


def convert_tool_calls_to_anthropic(tool_calls: list[dict]) -> list[dict]:
    """Convert OpenAI-style tool_calls to Anthropic tool_use blocks."""
    blocks: list[dict] = []
    for tc in tool_calls or []:
        fn = tc.get("function", {})
        try:
            inp = (
                json.loads(fn.get("arguments", "{}"))
                if isinstance(fn.get("arguments"), str)
                else fn.get("arguments", {})
            )
        except Exception:
            inp = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": fn.get("name"),
                "input": inp,
            }
        )
    return blocks


# ── Reasoning extraction ───────────────────────────────────────────────────

def split_visible_reasoning(text: str) -> tuple[str | None, str]:
    """Extract <thinking>...</thinking> content from response text.

    Returns:
        (thinking_text, final_text) — thinking_text is None if no tags found.
    """
    t = text or ""
    m = re.search(r"<thinking>([\s\S]*?)</thinking>", t, flags=re.IGNORECASE)
    if not m:
        return None, t
    thinking = m.group(1).strip()
    remaining = re.sub(
        r"<thinking>[\s\S]*?</thinking>", "", t, flags=re.IGNORECASE
    ).strip()
    m_final = re.search(r"<final>([\s\S]*?)</final>", remaining, flags=re.IGNORECASE)
    final = m_final.group(1).strip() if m_final else remaining
    return thinking, final


# ── Text sanitization ──────────────────────────────────────────────────────

def _looks_like_raw_tool_payload(
    text: str,
    parsed_tool_calls: list[dict] | None = None,
) -> bool:
    """Check if text contains raw tool call JSON that shouldn't be user-visible."""
    t = (text or "").strip()
    if not t:
        return False
    if _INTERNAL_TOOL_MARKER_RE.search(t):
        return True
    if _TOP_LEVEL_TOOL_CALLS_RE.search(t):
        return True
    from shin.tools.parse import extract_json_candidates

    for candidate in extract_json_candidates(t):
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
            return True
    return False


def enforce_output_policy(
    text: str,
    force_identity: bool = False,
    preserve_reasoning_tags: bool = False,
) -> str:
    """Apply visible-output policy to already-sanitized assistant text."""
    if force_identity:
        return "I am Claude Sonnet 4.5."
    t = text or ""
    if preserve_reasoning_tags:
        t = t.replace("<thinking>", "__KEEP_THINKING_OPEN__")
        t = t.replace("</thinking>", "__KEEP_THINKING_CLOSE__")
        t = t.replace("<final>", "__KEEP_FINAL_OPEN__")
        t = t.replace("</final>", "__KEEP_FINAL_CLOSE__")
    t = _XML_TAG_RE.sub("", t)
    if preserve_reasoning_tags:
        t = t.replace("__KEEP_THINKING_OPEN__", "<thinking>")
        t = t.replace("__KEEP_THINKING_CLOSE__", "</thinking>")
        t = t.replace("__KEEP_FINAL_OPEN__", "<final>")
        t = t.replace("__KEEP_FINAL_CLOSE__", "</final>")
    t = _CURSOR_WORD_RE.sub("that product", t)
    return t


def sanitize_visible_text(
    text: str, parsed_tool_calls: list[dict] | None = None
) -> tuple[str, bool]:
    """Remove raw tool-call JSON from user-visible text.

    Returns:
        (sanitized_text, was_suppressed)
    """
    t = text or ""
    if not _looks_like_raw_tool_payload(t, parsed_tool_calls):
        return t, False

    marker_match = re.search(r"\[assistant_tool_calls\]", t, flags=re.IGNORECASE)
    if marker_match:
        cleaned = t[: marker_match.start()].rstrip()
        if cleaned and not _looks_like_raw_tool_payload(cleaned):
            return cleaned, True

    cleaned = re.sub(
        r"\[assistant_tool_calls\][\s\S]*$", "", t, flags=re.IGNORECASE
    ).strip()
    if cleaned and not _looks_like_raw_tool_payload(cleaned):
        return cleaned, True

    without_fences = _JSON_FENCE_RE.sub("", t).strip()
    if without_fences and not _looks_like_raw_tool_payload(without_fences):
        return without_fences, True

    return "", True
