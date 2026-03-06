"""
Shin Proxy — Tool normalization.

Converts between OpenAI and Anthropic tool schemas, ensuring a consistent
internal representation (OpenAI format) for the Cursor API.
"""

from __future__ import annotations


def normalize_openai_tools(tools: list[dict] | None) -> list[dict]:
    """Normalise OpenAI-format tools, discarding malformed entries."""
    out: list[dict] = []
    for t in tools or []:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                },
            }
        )
    return out


def normalize_anthropic_tools(tools: list[dict] | None) -> list[dict]:
    """Convert Anthropic-format tools to internal (OpenAI) format."""
    out: list[dict] = []
    for t in tools or []:
        if not isinstance(t, dict) or not isinstance(t.get("name"), str):
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get(
                        "input_schema", {"type": "object", "properties": {}}
                    ),
                },
            }
        )
    return out


def to_anthropic_tool_format(openai_tools: list[dict] | None) -> list[dict]:
    """Convert normalised OpenAI tools to Anthropic tool_use format for Cursor."""
    result: list[dict] = []
    for t in openai_tools or []:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function", {})
        if not isinstance(fn.get("name"), str):
            continue
        result.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        )
    return result
