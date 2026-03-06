"""
Shin Proxy — typed configuration via pydantic-settings.

All environment variables are declared here and nowhere else.
Usage: `from shin.config import settings`
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All proxy configuration — loaded from env vars and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Server ──────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=4000, alias="PORT")

    # ── Auth ────────────────────────────────────────────────────────────────
    master_key: str = Field(default="sk-local-dev", alias="LITELLM_MASTER_KEY")

    # ── Cursor upstream ─────────────────────────────────────────────────────
    cursor_base_url: str = Field(default="https://cursor.com", alias="CURSOR_BASE_URL")
    cursor_model: str = Field(
        default="anthropic/claude-sonnet-4.6", alias="CURSOR_MODEL"
    )
    # Primary cookie (backwards-compatible)
    cursor_cookie: str = Field(default="", alias="CURSOR_COOKIE")
    # Additional cookies for round-robin pool (comma OR newline separated full cookie strings)
    # Example: CURSOR_COOKIES=WorkosCursorSessionToken=token1...,WorkosCursorSessionToken=token2...
    cursor_cookies: str = Field(default="", alias="CURSOR_COOKIES")
    cursor_auth_header: str = Field(default="", alias="CURSOR_AUTH_HEADER")
    cursor_context_file_path: str = Field(
        default="/docs/", alias="CURSOR_CONTEXT_FILE_PATH"
    )

    # ── User-Agent ──────────────────────────────────────────────────────────
    user_agent: str = Field(
        default="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
        alias="USER_AGENT",
    )

    # ── Upstream timeout ────────────────────────────────────────────────────
    # Read timeout for Cursor streaming/non-streaming responses.
    # Set to 0 (or negative) to disable read timeout.
    upstream_read_timeout_seconds: float = Field(
        default=300.0,
        alias="GATEWAY_UPSTREAM_READ_TIMEOUT_SECONDS",
    )
    stream_line_timeout_seconds: int = Field(
        default=30,
        alias="GATEWAY_STREAM_LINE_TIMEOUT_SECONDS",
    )
    stream_heartbeat_enabled: bool = Field(
        default=True,
        alias="GATEWAY_STREAM_HEARTBEAT_ENABLED",
    )
    stream_heartbeat_interval_seconds: float = Field(
        default=15.0,
        alias="GATEWAY_STREAM_HEARTBEAT_INTERVAL_SECONDS",
    )

    # ── Retry ───────────────────────────────────────────────────────────────
    retry_attempts: int = Field(default=2, alias="GATEWAY_RETRY_ATTEMPTS")
    retry_backoff_seconds: float = Field(
        default=0.6, alias="GATEWAY_RETRY_BACKOFF_SECONDS"
    )

    # ── Adaptive concurrency (upstream protection) ─────────────────────────
    adaptive_concurrency_enabled: bool = Field(
        default=True,
        alias="GATEWAY_ADAPTIVE_CONCURRENCY_ENABLED",
    )
    adaptive_concurrency_initial: int = Field(
        default=21,
        alias="GATEWAY_ADAPTIVE_CONCURRENCY_INITIAL",
    )
    adaptive_concurrency_min: int = Field(
        default=21,
        alias="GATEWAY_ADAPTIVE_CONCURRENCY_MIN",
    )
    adaptive_concurrency_max: int = Field(
        default=21,
        alias="GATEWAY_ADAPTIVE_CONCURRENCY_MAX",
    )
    adaptive_concurrency_success_window: int = Field(
        default=8,
        alias="GATEWAY_ADAPTIVE_CONCURRENCY_SUCCESS_WINDOW",
    )
    adaptive_concurrency_rate_limit_cooldown_seconds: float = Field(
        default=6.0,
        alias="GATEWAY_ADAPTIVE_CONCURRENCY_RATE_LIMIT_COOLDOWN_SECONDS",
    )

    # ── Circuit breaker (upstream protection) ──────────────────────────────
    circuit_breaker_enabled: bool = Field(
        default=True,
        alias="GATEWAY_CIRCUIT_BREAKER_ENABLED",
    )
    circuit_breaker_failure_threshold: int = Field(
        default=4,
        alias="GATEWAY_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
    )
    circuit_breaker_rate_limit_threshold: int = Field(
        default=2,
        alias="GATEWAY_CIRCUIT_BREAKER_RATE_LIMIT_THRESHOLD",
    )
    circuit_breaker_open_seconds: float = Field(
        default=12.0,
        alias="GATEWAY_CIRCUIT_BREAKER_OPEN_SECONDS",
    )
    circuit_breaker_half_open_max_probes: int = Field(
        default=1,
        alias="GATEWAY_CIRCUIT_BREAKER_HALF_OPEN_MAX_PROBES",
    )

    # ── Cache ───────────────────────────────────────────────────────────────
    cache_enabled: bool = Field(default=True, alias="GATEWAY_CACHE_ENABLED")
    cache_ttl_seconds: int = Field(default=45, alias="GATEWAY_CACHE_TTL_SECONDS")
    cache_max_entries: int = Field(default=500, alias="GATEWAY_CACHE_MAX_ENTRIES")
    cache_backend: str = Field(default="memory", alias="GATEWAY_CACHE_BACKEND")

    # ── Redis cache backend ─────────────────────────────────────────────────
    redis_url: str = Field(default="redis://redis:6379/0", alias="REDIS_URL")
    redis_cache_prefix: str = Field(default="shin:cache:", alias="REDIS_CACHE_PREFIX")

    # ── Metrics / observability ─────────────────────────────────────────────
    metrics_enabled: bool = Field(default=True, alias="GATEWAY_METRICS_ENABLED")
    otel_enabled: bool = Field(default=False, alias="GATEWAY_OTEL_ENABLED")
    otel_service_name: str = Field(default="shin-proxy", alias="OTEL_SERVICE_NAME")
    otel_exporter_otlp_endpoint: str = Field(
        default="", alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )

    # ── Pricing ─────────────────────────────────────────────────────────────
    price_anthropic_per_1k: float = Field(
        default=0.015, alias="GATEWAY_PRICE_ANTHROPIC_PER_1K"
    )
    price_openai_per_1k: float = Field(
        default=0.01, alias="GATEWAY_PRICE_OPENAI_PER_1K"
    )

    # ── Prompt pack / legacy prompt ────────────────────────────────────────
    gateway_system_prompt_file: str = Field(
        default="/app/sysprompt.txt",
        alias="GATEWAY_SYSTEM_PROMPT_FILE",
    )
    system_prompt: str = Field(
        default="You are a helpful, friendly, and knowledgeable AI assistant. "
        "Be concise, natural, and engaging in conversation. For coding tasks, "
        "think step by step and return clean, practical solutions.",
        alias="GATEWAY_SYSTEM_PROMPT",
    )


settings = Settings()
