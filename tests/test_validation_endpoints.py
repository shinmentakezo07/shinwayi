from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from shin.app import create_app

AUTH_HEADERS = {"Authorization": "Bearer sk-local-dev"}


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


def test_openai_empty_messages_returns_invalid_request(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"model": "anthropic/claude-sonnet-4.6", "messages": []},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_openai_bad_role_returns_invalid_request(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={
            "model": "anthropic/claude-sonnet-4.6",
            "messages": [{"role": "badrole", "content": "hi"}],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_openai_bad_tool_name_returns_invalid_request(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={
            "model": "anthropic/claude-sonnet-4.6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bad tool",
                        "description": "desc",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_openai_duplicate_tool_names_return_invalid_request(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={
            "model": "anthropic/claude-sonnet-4.6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
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
            ],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_openai_context_window_overflow_returns_context_error(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "x " * 130_000}],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "context_length_exceeded"


def test_anthropic_system_role_in_messages_returns_invalid_request(client: TestClient) -> None:
    response = client.post(
        "/v1/messages",
        headers=AUTH_HEADERS,
        json={
            "model": "anthropic/claude-sonnet-4.6",
            "messages": [{"role": "system", "content": "hi"}],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
