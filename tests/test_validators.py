from __future__ import annotations

import pytest

from shin.handlers import ContextWindowError, RequestValidationError
from shin.validators import (
    validate_anthropic_request,
    validate_openai_request,
    validate_payload_object,
)


VALID_OPENAI_PAYLOAD = {
    "model": "anthropic/claude-sonnet-4.6",
    "messages": [{"role": "user", "content": "hi"}],
    "tools": [],
    "tool_choice": "auto",
}


def test_validate_payload_object_rejects_non_object() -> None:
    with pytest.raises(RequestValidationError, match="request body must be a JSON object"):
        validate_payload_object([])


def test_validate_openai_request_rejects_empty_messages() -> None:
    payload = {**VALID_OPENAI_PAYLOAD, "messages": []}

    with pytest.raises(RequestValidationError, match="messages must be a non-empty array"):
        validate_openai_request(
            payload,
            model=payload["model"],
            messages=payload["messages"],
            tools=[],
            tool_choice="auto",
        )


def test_validate_openai_request_rejects_bad_role() -> None:
    payload = {
        **VALID_OPENAI_PAYLOAD,
        "messages": [{"role": "badrole", "content": "hi"}],
    }

    with pytest.raises(RequestValidationError, match=r"messages\[0\]\.role"):
        validate_openai_request(
            payload,
            model=payload["model"],
            messages=payload["messages"],
            tools=[],
            tool_choice="auto",
        )


def test_validate_openai_request_rejects_tool_name_with_spaces() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bad tool",
                "description": "desc",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    with pytest.raises(RequestValidationError, match="function.name"):
        validate_openai_request(
            VALID_OPENAI_PAYLOAD,
            model=VALID_OPENAI_PAYLOAD["model"],
            messages=VALID_OPENAI_PAYLOAD["messages"],
            tools=tools,
            tool_choice="auto",
        )


def test_validate_openai_request_rejects_duplicate_tool_names() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "dup_tool",
                "description": "desc",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "dup_tool",
                "description": "desc",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    with pytest.raises(RequestValidationError, match="duplicate tool name"):
        validate_openai_request(
            VALID_OPENAI_PAYLOAD,
            model=VALID_OPENAI_PAYLOAD["model"],
            messages=VALID_OPENAI_PAYLOAD["messages"],
            tools=tools,
            tool_choice="auto",
        )


def test_validate_openai_request_rejects_context_window_overflow() -> None:
    payload = {
        **VALID_OPENAI_PAYLOAD,
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "x " * 130_000}],
    }

    with pytest.raises(ContextWindowError, match="context window"):
        validate_openai_request(
            payload,
            model=payload["model"],
            messages=payload["messages"],
            tools=[],
            tool_choice="auto",
        )


def test_validate_anthropic_request_rejects_system_role_in_messages() -> None:
    payload = {
        "model": "anthropic/claude-sonnet-4.6",
        "messages": [{"role": "system", "content": "hi"}],
        "tools": [],
        "tool_choice": "auto",
    }

    with pytest.raises(RequestValidationError, match=r"messages\[0\]\.role"):
        validate_anthropic_request(
            payload,
            model=payload["model"],
            messages=payload["messages"],
            tools=[],
            tool_choice="auto",
            system_text="",
        )


def test_validate_openai_request_accepts_valid_payload() -> None:
    validate_openai_request(
        VALID_OPENAI_PAYLOAD,
        model=VALID_OPENAI_PAYLOAD["model"],
        messages=VALID_OPENAI_PAYLOAD["messages"],
        tools=[],
        tool_choice="auto",
    )
