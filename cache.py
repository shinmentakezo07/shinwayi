"""
Shin Proxy — response cache abstraction.

Supports:
- In-memory TTL cache (default)
- Optional Redis backend for shared cache across instances
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from cachetools import TTLCache as _TTLCache

from shin.config import settings

try:
    import redis
except Exception:  # pragma: no cover - optional dependency
    redis = None


class _CacheBackend:
    """Protocol-like base class for cache backends."""

    backend_name = "base"

    def get(self, key: str) -> Any | None:
        raise NotImplementedError

    def set(self, key: str, value: Any) -> None:
        raise NotImplementedError


class _MemoryCacheBackend(_CacheBackend):
    """In-process TTL cache backend."""

    backend_name = "memory"

    def __init__(self) -> None:
        self._cache: _TTLCache = _TTLCache(
            maxsize=max(1, settings.cache_max_entries),
            ttl=max(1, settings.cache_ttl_seconds),
        )

    def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value


class _RedisCacheBackend(_CacheBackend):
    """Redis-backed cache backend."""

    backend_name = "redis"

    def __init__(self) -> None:
        if redis is None:
            raise RuntimeError(
                "redis package not installed but GATEWAY_CACHE_BACKEND=redis configured"
            )
        self._prefix = (settings.redis_cache_prefix or "shin:cache:").strip() or "shin:cache:"
        self._ttl = max(1, settings.cache_ttl_seconds)
        self._client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=1.5,
            socket_timeout=1.5,
            health_check_interval=30,
        )
        self._client.ping()

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Any | None:
        raw = self._client.get(self._full_key(key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        self._client.setex(self._full_key(key), self._ttl, json.dumps(value, ensure_ascii=False))


class ResponseCache:
    """Cache façade with backend fallback behavior."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requested_backend = (settings.cache_backend or "memory").strip().lower()
        self._backend: _CacheBackend = _MemoryCacheBackend()
        self._backend_name = "memory"
        self._redis_available = False
        self._init_backend()

    def _init_backend(self) -> None:
        if self._requested_backend != "redis":
            self._backend = _MemoryCacheBackend()
            self._backend_name = "memory"
            self._redis_available = False
            return

        try:
            self._backend = _RedisCacheBackend()
            self._backend_name = "redis"
            self._redis_available = True
        except Exception:
            # Fallback to in-memory cache if Redis is unavailable or misconfigured.
            self._backend = _MemoryCacheBackend()
            self._backend_name = "memory"
            self._redis_available = False

    def _ensure_backend(self) -> None:
        if self._requested_backend == "redis" and not self._redis_available:
            self._init_backend()

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def redis_available(self) -> bool:
        return self._redis_available

    def get(self, key: str | None) -> Any | None:
        if not settings.cache_enabled or not key:
            return None
        with self._lock:
            self._ensure_backend()
            try:
                return self._backend.get(key)
            except Exception:
                # Keep serving traffic even if cache backend is transiently failing.
                if self._requested_backend == "redis":
                    self._backend = _MemoryCacheBackend()
                    self._backend_name = "memory"
                    self._redis_available = False
                return None

    def set(self, key: str | None, value: Any) -> None:
        if not settings.cache_enabled or not key or value is None:
            return
        with self._lock:
            self._ensure_backend()
            try:
                self._backend.set(key, value)
            except Exception:
                if self._requested_backend == "redis":
                    self._backend = _MemoryCacheBackend()
                    self._backend_name = "memory"
                    self._redis_available = False
                return

    @staticmethod
    def build_key(
        *,
        api_style: str,
        model: str,
        messages: list,
        tools: list,
        tool_choice: Any,
        reasoning_effort: Any,
        show_reasoning: bool,
        system_text: str = "",
    ) -> str:
        """Deterministic cache key from request parameters."""
        payload = {
            "api_style": api_style,
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "reasoning_effort": reasoning_effort,
            "show_reasoning": show_reasoning,
            "system": system_text,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Module-level singleton
response_cache = ResponseCache()
