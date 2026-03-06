"""
Shin Proxy — Custom exception hierarchy.

Every error that enters the proxy is classified before a response is sent.
Each subclass carries an HTTP status code and can render itself in both
OpenAI and Anthropic error response formats.
"""

from __future__ import annotations


class ProxyError(Exception):
    """Base class — all proxy errors carry an HTTP status and structured detail."""

    status_code: int = 500
    error_type: str = "proxy_error"

    def __init__(self, message: str, **detail: object) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message)

    def to_openai(self) -> dict:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": str(self.status_code),
                **self.detail,
            }
        }

    def to_anthropic(self) -> dict:
        return {
            "type": "error",
            "error": {
                "type": self.error_type,
                "message": self.message,
                **self.detail,
            },
        }


class AuthError(ProxyError):
    status_code = 401
    error_type = "authentication_error"


class RequestValidationError(ProxyError):
    status_code = 400
    error_type = "invalid_request_error"


class ContextWindowError(ProxyError):
    status_code = 400
    error_type = "context_length_exceeded"


class CredentialError(ProxyError):
    status_code = 401
    error_type = "credential_error"


class RateLimitError(ProxyError):
    status_code = 429
    error_type = "rate_limit_error"


class CircuitOpenError(ProxyError):
    status_code = 503
    error_type = "service_unavailable_error"


class BackendError(ProxyError):
    status_code = 502
    error_type = "backend_error"


class ConfigError(ProxyError):
    status_code = 500
    error_type = "config_error"


class TimeoutError(ProxyError):
    status_code = 504
    error_type = "timeout_error"


class EmptyResponseError(ProxyError):
    status_code = 502
    error_type = "empty_response_error"


class ToolParseError(ProxyError):
    """Not surfaced as HTTP error — handled inline in pipeline."""

    status_code = 200
    error_type = "tool_parse_error"


class StreamAbortError(ProxyError):
    """Client disconnected — close stream cleanly, do not log as error."""

    status_code = 499
    error_type = "stream_aborted"
