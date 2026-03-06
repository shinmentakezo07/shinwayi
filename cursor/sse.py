"""
Shin Proxy — SSE line parser.

Stateless functions that parse raw SSE lines from Cursor's /api/chat endpoint.
Tool-call detection is NOT done here — it belongs in the streaming loop.
"""

from __future__ import annotations

import json


def parse_line(line: str) -> dict | None:
    """Parse a single SSE line into a dict, or None if not a data line.

    Returns:
        {"done": True}          — stream finished
        {"raw": str}            — unparseable data line
        dict                    — parsed JSON payload
        None                    — blank / non-data line
    """
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data:
        return None
    if data == "[DONE]":
        return {"done": True}
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {"raw": data}


def extract_delta(event: dict) -> str:
    """Pull the text delta from a parsed SSE event dict.

    Checks Cursor-native top-level shapes first, then falls back to
    OpenAI-style nested choices[0].delta.content.
    Returns empty string if no text content found.
    """
    if not isinstance(event, dict):
        return ""

    # Cursor-native: top-level string value
    for key in ("delta", "text", "content", "token"):
        val = event.get(key)
        if isinstance(val, str):
            return val

    # OpenAI-style nested: {"choices": [{"delta": {"content": "..."}}]}
    choices = event.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            delta = first.get("delta", {})
            if isinstance(delta, dict):
                for key in ("content", "text"):
                    val = delta.get(key)
                    if isinstance(val, str):
                        return val

    # Anthropic-style nested: {"type": "content_block_delta", "delta": {"text": "..."}}
    top_delta = event.get("delta")
    if isinstance(top_delta, dict):
        for key in ("text", "content"):
            val = top_delta.get(key)
            if isinstance(val, str):
                return val

    return ""
