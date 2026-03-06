"""
Shin Proxy — Tool call parser.

Extracts structured tool call objects from raw Cursor response text.
All `arguments` values are guaranteed to be JSON strings on output.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)\s*```", flags=re.IGNORECASE
)


def extract_json_candidates(text: str) -> list[str]:
    """Extract JSON-like substrings from text.

    Handles:
    - Fenced ```json ... ``` code blocks
    - Bare JSON objects/arrays using bracket-matching
    """
    t = (text or "").strip()
    if not t:
        return []

    candidates: list[str] = []

    # Code-fenced blocks
    for m in _JSON_FENCE_RE.finditer(t):
        block = m.group(1).strip()
        if block:
            candidates.append(block)

    # Bracket-matched JSON segments
    stack: list[str] = []
    start: int | None = None
    in_str = False
    esc = False

    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "[{":
            if start is None:
                start = i
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                continue
            top = stack[-1]
            if (top == "[" and ch == "]") or (top == "{" and ch == "}"):
                stack.pop()
                if not stack and start is not None:
                    seg = t[start : i + 1].strip()
                    if seg:
                        candidates.append(seg)
                    start = None
            else:
                stack.clear()
                start = None

    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


@dataclass
class ToolParseResult:
    """Structured result for tool-call parsing plus bounded reparse hints."""

    calls: list[dict] | None
    hint_offset: int | None = None


_TOOL_HINT_TOKENS = (
    "[assistant_tool_calls]",
    '"tool_calls"',
)


def find_tool_hint_offset(text: str, start: int = 0) -> int | None:
    """Return the earliest tool-payload hint offset at or after start."""
    t = text or ""
    if not t:
        return None
    begin = max(0, start)
    hits = [idx for token in _TOOL_HINT_TOKENS if (idx := t.find(token, begin)) != -1]
    if not hits:
        return None
    return min(hits)


def parse_tool_calls_details(
    text: str,
    tools: list[dict] | None,
    *,
    start: int = 0,
) -> ToolParseResult:
    """Parse tool calls and report the first hint offset used for reparsing."""
    if not tools:
        return ToolParseResult(calls=None)

    allowed = {
        t.get("function", {}).get("name")
        for t in tools
        if isinstance(t, dict)
    }

    hint_offset = find_tool_hint_offset(text, start)
    if hint_offset is None:
        return ToolParseResult(calls=None)

    segment = text[hint_offset:]
    objs: list = []
    for raw in extract_json_candidates(segment):
        try:
            objs.append(json.loads(raw))
        except Exception:
            pass
    if not objs:
        return ToolParseResult(calls=None, hint_offset=hint_offset)

    merged: list[dict] = []
    for obj in objs:
        if isinstance(obj, list):
            merged.extend(obj)
        elif isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
            merged.extend(obj["tool_calls"])
    if not merged:
        return ToolParseResult(calls=None, hint_offset=hint_offset)

    out: list[dict] = []
    for c in merged:
        if not isinstance(c, dict):
            continue
        # Handle both {"function": {"name": ...}} and {"name": ...}
        if isinstance(c.get("function"), dict):
            name = c["function"].get("name")
            args = c["function"].get("arguments", {})
        else:
            name = c.get("name")
            args = c.get("arguments", {})

        if not isinstance(name, str) or name not in allowed:
            continue

        # INVARIANT: arguments is always a JSON string
        arg_str = (
            args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        )
        out.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": arg_str},
            }
        )

    return ToolParseResult(calls=(out or None), hint_offset=hint_offset)


def parse_tool_calls_from_text(
    text: str, tools: list[dict] | None
) -> list[dict] | None:
    """Parse tool calls from assistant response text.

    Returns None if no valid tool calls are found.
    Each returned item has the shape:
        {"id": "call_...", "type": "function",
         "function": {"name": "...", "arguments": "<json string>"}}
    """
    return parse_tool_calls_details(text, tools).calls
