"""
Shin Proxy — Convert OpenAI / Anthropic messages to Cursor format.

Cursor's /api/chat expects messages in a "parts" format:
    {"parts": [{"type": "text", "text": "..."}], "id": "...", "role": "..."}

This module converts from both OpenAI and Anthropic message schemas.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from shin.config import settings
from shin.handlers import ConfigError


# ── Helpers ─────────────────────────────────────────────────────────────────

def _msg(role: str, text: str) -> dict:
    """Create a single Cursor-format message."""
    return {
        "parts": [{"type": "text", "text": text}],
        "id": uuid.uuid4().hex[:16],
        "role": role,
    }


def _extract_text(content) -> str:
    """Extract plain text from various content shapes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return ""


def _load_prompt_pack_blocks(path: str) -> list[str]:
    """Load prompt-pack blocks from a raw reminder file or JSON payload."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except Exception as exc:
        raise ConfigError(f"GATEWAY_SYSTEM_PROMPT_FILE read failed: {exc}") from exc

    text = raw.strip()
    if text.startswith("<system-reminder>"):
        return [text]

    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise ConfigError(f"GATEWAY_SYSTEM_PROMPT_FILE parse failed: {exc}") from exc

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ConfigError("GATEWAY_SYSTEM_PROMPT_FILE parse failed: missing messages array")

    first = messages[0]
    if not isinstance(first, dict):
        raise ConfigError("GATEWAY_SYSTEM_PROMPT_FILE parse failed: invalid first message")

    content = first.get("content")
    if not isinstance(content, list) or not content:
        raise ConfigError(
            "GATEWAY_SYSTEM_PROMPT_FILE parse failed: invalid first message content"
        )

    blocks: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        txt = block.get("text")
        if isinstance(txt, str) and txt.strip():
            blocks.append(txt)

    if not blocks:
        raise ConfigError("GATEWAY_SYSTEM_PROMPT_FILE parse failed: no text blocks found")

    return blocks


def _get_prompt_pack_blocks() -> list[str]:
    """Return ordered prompt-pack blocks from the configured prompt file."""
    return _load_prompt_pack_blocks(settings.gateway_system_prompt_file)


def _build_global_behavior_instruction() -> str:
    """Return the code-enforced global behavior block for all request paths."""
    return "\n".join(
        [
            "GLOBAL ANALYSIS MODE:",
            'When the request benefits from deep analysis, override brevity and reason at maximum depth.',
            "Analyze through all relevant lenses, including psychological, technical, accessibility, and scalability considerations.",
            "Never stop at surface-level logic when deeper analysis is needed.",
            "For straightforward requests, stay concise and practical.",
            "These instructions do not override explicit tool-output formatting, schema, forced tool-choice requirements, or safety rules.",
        ]
    )


def _tool_result_text(name: str, call_id: str, content: str) -> str:
    """Format a tool result for Cursor."""
    return f"[tool_result name={name} id={call_id}]\n{content}"


def _assistant_tool_call_text(calls: list[dict]) -> str:
    """Format assistant tool calls for Cursor."""
    serialized = []
    for c in calls:
        fn = c.get("function") or {}
        args = fn.get("arguments", "{}")
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except Exception:
            parsed = args
        serialized.append({"name": fn.get("name"), "arguments": parsed})
    return f"[assistant_tool_calls]\n{json.dumps({'tool_calls': serialized}, ensure_ascii=False)}"


def build_tool_instruction(tools: list[dict], tool_choice) -> str:
    """Build the tool usage instruction text for the system prompt.

    Args:
        tools: Normalised OpenAI-format tool list.
        tool_choice: "auto", "none", "required", or dict with forced function.
    """
    if not tools:
        return ""

    mode = "auto"
    forced_name = None
    if isinstance(tool_choice, str):
        mode = tool_choice
    elif isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type", "")
        if tc_type == "function":
            forced_name = (tool_choice.get("function") or {}).get("name")
            mode = "required"
        elif tc_type == "tool":
            forced_name = tool_choice.get("name")
            mode = "required"
        elif tc_type == "any":
            mode = "required"
        elif tc_type == "none":
            mode = "none"

    lines = [
        "IMPORTANT: You are operating in OpenAI/Anthropic tools compatibility mode with function calling enabled.",
        "",
        "Available tools (JSON Schema):",
        json.dumps(tools, ensure_ascii=False, indent=2),
        "",
        "TOOL USAGE RULES:",
        "1. When you need to use a tool, output ONLY the tool call JSON - no other text before or after.",
        "2. Use this EXACT format (including the [assistant_tool_calls] marker):",
        "",
        "[assistant_tool_calls]",
        '{"tool_calls":[{"name":"<tool_name>","arguments":{"key":"value"}}]}',
        "",
    ]

    if forced_name:
        lines.append(f"CRITICAL: You MUST call tool '{forced_name}' in your response.")
    elif mode == "required":
        lines.append("CRITICAL: You MUST call at least one tool in your response.")
    elif mode == "none":
        lines.append("Do not call any tool. Respond with normal text only.")
    else:
        lines.append("You may choose to call a tool OR respond with normal text.")

    lines.extend([
        "",
        "RESPONSE FORMAT:",
        "- If using tools: Start with [assistant_tool_calls] on its own line, followed by the JSON.",
        "- If NOT using tools: Respond with natural language only, no JSON markers.",
        "- NEVER combine tool calls with explanatory text in the same message.",
    ])
    return "\n".join(lines)


# ── Public converters ───────────────────────────────────────────────────────

def openai_to_cursor(
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_choice=None,
    reasoning_effort: str | None = None,
    show_reasoning: bool = False,
    model: str = "",
) -> list[dict]:
    """Convert OpenAI-format messages to Cursor parts-format.

    KEY INVARIANT: this function ALWAYS returns cursor_messages,
    regardless of stream/cache path. Call it BEFORE branching.
    """
    cursor_messages: list[dict] = []

    for block in _get_prompt_pack_blocks():
        cursor_messages.append(_msg("user", block))

    cursor_messages.append(_msg("system", _build_global_behavior_instruction()))

    # Tool instruction
    tool_inst = build_tool_instruction(tools or [], tool_choice)
    if tool_inst:
        cursor_messages.append(_msg("system", tool_inst))

    # Reasoning effort
    if reasoning_effort:
        txt = f"Reasoning effort preference: {reasoning_effort}."
        if show_reasoning:
            txt += (
                " Include a visible short 'thinking' section before the final answer"
                " using: <thinking>...</thinking> then <final>...</final>."
            )
        else:
            txt += " Keep reasoning internal; return concise final answers."
        cursor_messages.append(_msg("system", txt))

    # User / assistant / tool messages
    for msg in messages or []:
        role = msg.get("role", "user")

        if role == "tool":
            text = _tool_result_text(
                msg.get("name") or "tool",
                msg.get("tool_call_id") or "",
                _extract_text(msg.get("content", "")),
            )
            cursor_messages.append(_msg("user", text))
        elif (
            role == "assistant"
            and isinstance(msg.get("tool_calls"), list)
            and msg.get("tool_calls")
        ):
            text = _assistant_tool_call_text(msg["tool_calls"])
            cursor_messages.append(_msg("assistant", text))
        else:
            text = _extract_text(msg.get("content", ""))
            cursor_messages.append(_msg(role, text))

    return cursor_messages


def anthropic_to_cursor(
    messages: list[dict],
    system_text: str = "",
    tools: list[dict] | None = None,
    tool_choice=None,
    thinking: dict | None = None,
    model: str = "",
) -> tuple[list[dict], bool]:
    """Convert Anthropic-format messages to Cursor parts-format.

    Returns:
        (cursor_messages, show_reasoning)
    """
    show_reasoning = bool(
        isinstance(thinking, dict) and thinking.get("type") == "enabled"
    )
    reasoning_effort = "medium" if show_reasoning else None
    cursor_messages: list[dict] = []

    for block in _get_prompt_pack_blocks():
        cursor_messages.append(_msg("user", block))

    cursor_messages.append(_msg("system", _build_global_behavior_instruction()))

    # Additional system text from Anthropic request
    if system_text:
        cursor_messages.append(_msg("system", system_text))

    # Tool instruction
    tool_inst = build_tool_instruction(tools or [], tool_choice)
    if tool_inst:
        cursor_messages.append(_msg("system", tool_inst))

    # Reasoning
    if reasoning_effort:
        cursor_messages.append(
            _msg(
                "system",
                "Reasoning effort preference: medium. Include a visible short "
                "'thinking' section before the final answer using: "
                "<thinking>...</thinking> then <final>...</final>.",
            )
        )

    # Messages
    for msg in messages or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        mapped = "assistant" if role == "assistant" else "user"

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    parts.append(
                        _tool_result_text(
                            block.get("name") or "tool",
                            block.get("tool_use_id") or "",
                            _extract_text(block.get("content", "")),
                        )
                    )
                elif btype == "tool_use":
                    one = {
                        "tool_calls": [
                            {
                                "name": block.get("name"),
                                "arguments": block.get("input", {}),
                            }
                        ]
                    }
                    parts.append(
                        f"[assistant_tool_calls]\n{json.dumps(one, ensure_ascii=False)}"
                    )
            text = "\n".join(p for p in parts if p)
        else:
            text = ""

        cursor_messages.append(_msg(mapped, text))

    return cursor_messages, show_reasoning
