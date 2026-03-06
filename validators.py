"""
Shin Proxy — Request validation.

Validates OpenAI-compatible and Anthropic-compatible request payloads at the
router boundary so malformed input fails with typed ProxyErrors instead of
reaching downstream converters and stream handlers.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from shin.converters.from_cursor import MODEL_CONTEXT_WINDOWS, context_window_for
from shin.handlers import ContextWindowError, RequestValidationError
from shin.tokens import count_message_tokens, count_tokens, count_tool_tokens

log = structlog.get_logger()

_MAX_MESSAGES = 2048
_MAX_MESSAGE_CHARS = 2_000_000
_CONTEXT_WINDOW_WARN_PCT = 0.9
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_OPENAI_ROLES = {"system", "user", "assistant", "tool"}
_ANTHROPIC_ROLES = {"user", "assistant"}


def _require(condition: bool, message: str, **detail: object) -> None:
    if not condition:
        raise RequestValidationError(message, **detail)


def validate_payload_object(payload: Any) -> dict[str, Any]:
    _require(isinstance(payload, dict), "request body must be a JSON object", field="body")
    return payload


def _text_like_length(content: Any, field: str) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return len(text)
        nested = content.get("content")
        if isinstance(nested, str):
            return len(nested)
        raise RequestValidationError(
            f"{field} must be a string or text-like content",
            field=field,
        )
    if isinstance(content, list):
        total = 0
        for index, block in enumerate(content):
            total += _text_like_length(block, f"{field}[{index}]")
        return total
    raise RequestValidationError(
        f"{field} must be a string or text-like content",
        field=field,
    )


def _block_length(block: Any, api_style: str, field: str) -> int:
    if isinstance(block, str):
        return len(block)

    _require(isinstance(block, dict), f"{field} must be an object or string", field=field)

    block_type = block.get("type")
    if block_type == "text":
        text = block.get("text")
        _require(isinstance(text, str), f"{field}.text must be a string", field=field)
        return len(text)

    if api_style == "anthropic" and block_type == "tool_use":
        name = block.get("name")
        _require(
            isinstance(name, str) and bool(name.strip()),
            f"{field}.name must be a non-empty string",
            field=field,
        )
        input_payload = block.get("input", {})
        _require(
            input_payload is None or isinstance(input_payload, dict),
            f"{field}.input must be an object",
            field=field,
        )
        return 0

    if api_style == "anthropic" and block_type == "tool_result":
        return _text_like_length(block.get("content"), f"{field}.content")

    fallback_text = block.get("text")
    if isinstance(fallback_text, str):
        return len(fallback_text)

    fallback_content = block.get("content")
    if isinstance(fallback_content, str):
        return len(fallback_content)

    raise RequestValidationError(
        f"{field} contains an unsupported content block",
        field=field,
    )


def validate_messages_not_empty(messages: Any) -> None:
    _require(
        isinstance(messages, list) and bool(messages),
        "messages must be a non-empty array",
        field="messages",
    )


def validate_message_count(messages: list[dict]) -> None:
    _require(
        len(messages) <= _MAX_MESSAGES,
        f"messages must contain at most {_MAX_MESSAGES} items",
        field="messages",
    )


def validate_message_roles(messages: list[dict], api_style: str) -> None:
    allowed_roles = _OPENAI_ROLES if api_style == "openai" else _ANTHROPIC_ROLES
    for index, message in enumerate(messages):
        _require(
            isinstance(message, dict),
            f"messages[{index}] must be an object",
            field=f"messages[{index}]",
        )
        role = message.get("role")
        _require(
            isinstance(role, str) and role in allowed_roles,
            f"messages[{index}].role must be one of {sorted(allowed_roles)}",
            field=f"messages[{index}].role",
        )


def validate_message_content(messages: list[dict], api_style: str) -> None:
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        field = f"messages[{index}].content"

        if content is None:
            _require(
                role == "assistant",
                f"{field} cannot be null for role '{role}'",
                field=field,
            )
            continue

        if isinstance(content, str):
            _require(
                len(content) <= _MAX_MESSAGE_CHARS,
                f"messages[{index}] exceeds {_MAX_MESSAGE_CHARS} characters",
                field=field,
            )
            continue

        _require(isinstance(content, list), f"{field} must be a string or block list", field=field)
        total_chars = 0
        for block_index, block in enumerate(content):
            total_chars += _block_length(
                block,
                api_style,
                f"messages[{index}].content[{block_index}]",
            )
        _require(
            total_chars <= _MAX_MESSAGE_CHARS,
            f"messages[{index}] exceeds {_MAX_MESSAGE_CHARS} characters",
            field=field,
        )


def validate_role_alternation(messages: list[dict], api_style: str) -> None:
    previous_role: str | None = None
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == previous_role:
            log.warning(
                "consecutive_same_role_messages",
                api_style=api_style,
                index=index,
                role=role,
            )
        previous_role = role if isinstance(role, str) else previous_role


def validate_model(model: Any) -> None:
    _require(
        isinstance(model, str) and bool(model.strip()),
        "model must be a non-empty string",
        field="model",
    )
    if model not in MODEL_CONTEXT_WINDOWS:
        log.warning(
            "unknown_model_context_window",
            model=model,
            default_context_window=context_window_for(model),
        )


def validate_tools(tools: list[dict]) -> None:
    seen_names: set[str] = set()
    for index, tool in enumerate(tools):
        _require(
            isinstance(tool, dict) and tool.get("type") == "function",
            f"tools[{index}] must be a function tool",
            field=f"tools[{index}]",
        )
        fn = tool.get("function")
        _require(
            isinstance(fn, dict),
            f"tools[{index}].function must be an object",
            field=f"tools[{index}].function",
        )
        name = fn.get("name")
        _require(
            isinstance(name, str) and bool(_TOOL_NAME_RE.fullmatch(name)),
            (
                f"tools[{index}].function.name must match "
                f"{_TOOL_NAME_RE.pattern}"
            ),
            field=f"tools[{index}].function.name",
        )
        _require(
            name not in seen_names,
            f"duplicate tool name: {name}",
            field=f"tools[{index}].function.name",
        )
        seen_names.add(name)

        parameters = fn.get("parameters")
        _require(
            isinstance(parameters, dict) and parameters.get("type") == "object",
            f"tools[{index}].function.parameters.type must be 'object'",
            field=f"tools[{index}].function.parameters",
        )


def validate_tool_choice(tool_choice: Any, tools: list[dict]) -> None:
    tool_names = {
        tool["function"]["name"]
        for tool in tools
        if isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and isinstance(tool["function"].get("name"), str)
    }

    if isinstance(tool_choice, str):
        _require(
            tool_choice in {"auto", "none", "required"},
            "tool_choice must be 'auto', 'none', or 'required'",
            field="tool_choice",
        )
        if tool_choice == "required":
            _require(
                bool(tool_names),
                "tool_choice 'required' requires at least one tool",
                field="tool_choice",
            )
        return

    _require(isinstance(tool_choice, dict), "tool_choice must be a string or object", field="tool_choice")

    tc_type = tool_choice.get("type")
    if tc_type == "none":
        return
    if tc_type == "any":
        _require(
            bool(tool_names),
            "tool_choice 'any' requires at least one tool",
            field="tool_choice",
        )
        return
    if tc_type == "function":
        function = tool_choice.get("function")
        _require(
            isinstance(function, dict) and isinstance(function.get("name"), str),
            "tool_choice.function.name must be a string",
            field="tool_choice.function.name",
        )
        name = function["name"]
        _require(
            name in tool_names,
            f"tool_choice references unknown tool '{name}'",
            field="tool_choice.function.name",
        )
        return
    if tc_type == "tool":
        name = tool_choice.get("name")
        _require(
            isinstance(name, str),
            "tool_choice.name must be a string",
            field="tool_choice.name",
        )
        _require(
            name in tool_names,
            f"tool_choice references unknown tool '{name}'",
            field="tool_choice.name",
        )
        return

    raise RequestValidationError(
        "tool_choice must be 'auto', 'none', 'required', or a named tool reference",
        field="tool_choice",
    )


def validate_context_window(
    messages: list[dict],
    model: str,
    tools: list[dict],
    *,
    system_text: str = "",
) -> None:
    context_window = context_window_for(model)
    input_tokens = count_message_tokens(messages, model=model)
    input_tokens += count_tool_tokens(tools, model=model)
    if system_text:
        input_tokens += count_tokens(system_text, model=model)

    used_pct = input_tokens / context_window if context_window else 0.0
    if used_pct >= _CONTEXT_WINDOW_WARN_PCT:
        log.warning(
            "context_window_near_limit",
            model=model,
            input_tokens=input_tokens,
            context_window=context_window,
            context_window_used_pct=round(used_pct * 100, 2),
        )

    if input_tokens > context_window:
        raise ContextWindowError(
            f"Estimated input tokens exceed the context window for model '{model}'",
            model=model,
            input_tokens=input_tokens,
            context_window=context_window,
            context_window_used_pct=round(used_pct * 100, 2),
        )


def validate_openai_request(
    payload: Any,
    *,
    model: Any,
    messages: Any,
    tools: list[dict],
    tool_choice: Any,
) -> None:
    validate_payload_object(payload)
    validate_model(model)
    validate_messages_not_empty(messages)
    validate_message_count(messages)
    validate_message_roles(messages, "openai")
    validate_message_content(messages, "openai")
    validate_role_alternation(messages, "openai")
    validate_tools(tools)
    validate_tool_choice(tool_choice, tools)
    validate_context_window(messages, model, tools)


def validate_anthropic_request(
    payload: Any,
    *,
    model: Any,
    messages: Any,
    tools: list[dict],
    tool_choice: Any,
    system_text: str = "",
) -> None:
    validate_payload_object(payload)
    validate_model(model)
    validate_messages_not_empty(messages)
    validate_message_count(messages)
    validate_message_roles(messages, "anthropic")
    validate_message_content(messages, "anthropic")
    validate_role_alternation(messages, "anthropic")
    validate_tools(tools)
    validate_tool_choice(tool_choice, tools)
    validate_context_window(messages, model, tools, system_text=system_text)
