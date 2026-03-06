"""
Shin Proxy — Credential pool for Cursor cookies.

Supports two env var formats:

    CURSOR_COOKIE=WorkosCursorSessionToken=token1...
        → single cookie, backwards compatible

    CURSOR_COOKIES=WorkosCursorSessionToken=token1...,WorkosCursorSessionToken=token2...
        → comma-separated list of full cookie strings (up to 15)
        → can also use newlines instead of commas

Both are combined and deduplicated into one round-robin pool.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from dataclasses import dataclass, field

from shin.config import settings

log = logging.getLogger(__name__)


@dataclass
class CredentialInfo:
    """Per-credential tracking."""

    cookie: str
    index: int = 0
    request_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    last_used: float = 0.0
    last_error: float = 0.0
    cooldown_until: float = 0.0
    healthy: bool = True


def _parse_cookies(raw: str) -> list[str]:
    """
    Parse a cookie string that may contain multiple cookies separated by
    commas or newlines.

    Handles the awkward edge case where a cookie value itself contains commas
    by requiring each entry to contain 'WorkosCursorSessionToken' to be valid.
    Falls back to treating the entire string as one cookie if splitting fails.
    """
    if not raw or not raw.strip():
        return []

    candidates: list[str] = []

    # Try splitting on newlines first (cleanest format)
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if len(lines) > 1:
        candidates = lines
    else:
        # Try splitting on comma — but guard against cookie values with commas
        # by checking that each part looks like a cookie header
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) > 1:
            # Validate: all parts should look like key=value pairs
            cookie_parts = [p for p in parts if "=" in p]
            if len(cookie_parts) == len(parts):
                # Try to reconstruct as full cookies —
                # if every part has a known token prefix, they're separate cookies
                if all("WorkosCursorSessionToken" in p or "=" in p for p in parts):
                    # Check if first part already is a complete token (long enough)
                    if any(len(p) > 100 for p in parts):
                        candidates = parts
                    else:
                        candidates = [raw]  # single cookie with commas in value
                else:
                    candidates = [raw]
            else:
                candidates = [raw]
        else:
            candidates = [raw]

    return candidates


class CredentialPool:
    """Thread-safe round-robin credential rotator."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._creds: list[CredentialInfo] = []
        self._current_index = 0
        self._calls_on_current = 0
        self.calls_per_rotation = 21  # Rotate after 21 requests on current credential
        self.cooldown_on_rate_limit_seconds = 30.0
        self.cooldown_on_timeout_seconds = 8.0
        self._load()

    # ── Init ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        raw_cookies: list[str] = []

        # 1. Primary: CURSOR_COOKIE (single, backwards-compat)
        if settings.cursor_cookie:
            raw_cookies.extend(_parse_cookies(settings.cursor_cookie))

        # 2. Multi-cookie: CURSOR_COOKIES (comma or newline separated)
        if settings.cursor_cookies:
            raw_cookies.extend(_parse_cookies(settings.cursor_cookies))

        # Deduplicate preserving order, max 15
        seen: set[str] = set()
        for raw in raw_cookies[:15]:
            cookie = raw.strip()
            if cookie and cookie not in seen:
                seen.add(cookie)
                idx = len(self._creds)
                self._creds.append(CredentialInfo(cookie=cookie, index=idx))

        if self._creds:
            log.info(
                "credential_pool_loaded",
                extra={"count": len(self._creds)},
            )
        else:
            log.warning("credential_pool_empty — set CURSOR_COOKIE in .env")

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._creds)

    def next(self) -> CredentialInfo | None:
        """Get the next available credential.

        Rotation behavior:
        - Uses `calls_per_rotation` while the current credential is healthy.
        - Immediately skips credentials in cooldown/unhealthy state.
        - If all credentials are unavailable, clears cooldowns as emergency fallback.
        """
        if not self._creds:
            return None

        with self._lock:
            now = time.time()
            for _ in range(len(self._creds)):
                cred = self._creds[self._current_index]

                # Auto-recover from cooldown when timer passes
                if cred.cooldown_until and cred.cooldown_until <= now:
                    cred.cooldown_until = 0.0
                    cred.healthy = True
                    cred.consecutive_errors = 0

                if cred.healthy and cred.cooldown_until <= now:
                    self._calls_on_current += 1

                    # Prepare next index when current credential reached quota
                    if self._calls_on_current >= self.calls_per_rotation:
                        self._calls_on_current = 0
                        self._current_index = (self._current_index + 1) % len(self._creds)

                    cred.request_count += 1
                    cred.last_used = now
                    return cred

                # Current is unavailable, move on
                self._calls_on_current = 0
                self._current_index = (self._current_index + 1) % len(self._creds)

            # All unavailable — emergency reset of cooldown/health
            log.warning("all_credentials_unavailable — emergency recovery")
            self._reset_all_unsafe()
            if self._creds:
                cred = self._creds[self._current_index]
                self._calls_on_current = 1
                if self._calls_on_current >= self.calls_per_rotation:
                    self._calls_on_current = 0
                    self._current_index = (self._current_index + 1) % len(self._creds)

                cred.request_count += 1
                cred.last_used = now
                return cred

        return None

    def mark_error(self, cred: CredentialInfo, *, mark_unhealthy: bool = True) -> None:
        """Record a failed request against a credential.

        Args:
            cred: Credential to update.
            mark_unhealthy: If False, only increments counters and keeps credential healthy.
        """
        with self._lock:
            cred.error_count += 1
            cred.consecutive_errors += 1
            cred.last_error = time.time()
            if not mark_unhealthy:
                return

            # Mark unhealthy after repeated errors.
            if cred.consecutive_errors >= 3:
                cred.healthy = False
                log.warning(
                    "credential_marked_unhealthy",
                    extra={"index": cred.index, "total_errors": cred.error_count},
                )

    def mark_rate_limited(self, cred: CredentialInfo) -> None:
        """Immediately cool down a credential that hit upstream 429."""
        with self._lock:
            now = time.time()
            cred.error_count += 1
            cred.consecutive_errors += 1
            cred.last_error = now
            cred.healthy = False
            cred.cooldown_until = now + self.cooldown_on_rate_limit_seconds

            # Force next() to move away from this credential quickly
            self._calls_on_current = 0
            if self._creds:
                self._current_index = (cred.index + 1) % len(self._creds)

            log.warning(
                "credential_rate_limited_cooldown",
                extra={
                    "index": cred.index,
                    "cooldown_seconds": self.cooldown_on_rate_limit_seconds,
                    "cooldown_until": round(cred.cooldown_until, 3),
                },
            )

    def mark_timeout(self, cred: CredentialInfo) -> None:
        """Temporarily cool down a credential on timeout-related failures."""
        with self._lock:
            now = time.time()
            cred.error_count += 1
            cred.consecutive_errors += 1
            cred.last_error = now
            cred.healthy = False
            cred.cooldown_until = max(
                cred.cooldown_until,
                now + self.cooldown_on_timeout_seconds,
            )

            self._calls_on_current = 0
            if self._creds:
                self._current_index = (cred.index + 1) % len(self._creds)

            log.warning(
                "credential_timeout_cooldown",
                extra={
                    "index": cred.index,
                    "cooldown_seconds": self.cooldown_on_timeout_seconds,
                    "cooldown_until": round(cred.cooldown_until, 3),
                },
            )

    def mark_success(self, cred: CredentialInfo) -> None:
        """Reset consecutive error counter after a successful request."""
        with self._lock:
            cred.consecutive_errors = 0
            cred.healthy = True
            cred.cooldown_until = 0.0

    def _reset_all_unsafe(self) -> None:
        """Reset all credentials — must hold self._lock."""
        for cred in self._creds:
            cred.healthy = True
            cred.consecutive_errors = 0
            cred.cooldown_until = 0.0

    def reset_all(self) -> None:
        """Manually reset all credentials to healthy."""
        with self._lock:
            self._reset_all_unsafe()

    def snapshot(self) -> list[dict]:
        """Return pool status (cookie values are never exposed)."""
        with self._lock:
            return [
                {
                    "index": c.index,
                    "healthy": c.healthy,
                    "requests": c.request_count,
                    "total_errors": c.error_count,
                    "consecutive_errors": c.consecutive_errors,
                    "last_used": round(c.last_used, 3) if c.last_used else None,
                    "last_error": round(c.last_error, 3) if c.last_error else None,
                    "cooldown_until": round(c.cooldown_until, 3) if c.cooldown_until else None,
                    # show only first 12 chars of cookie for identification
                    "cookie_prefix": c.cookie[:12] + "..." if c.cookie else "",
                }
                for c in self._creds
            ]

    def get_auth_headers(self, cred: CredentialInfo | None = None) -> dict[str, str]:
        """Build auth headers for a Cursor API request."""
        headers: dict[str, str] = {}
        cookie_val = cred.cookie if cred else settings.cursor_cookie
        if cookie_val:
            headers["Cookie"] = cookie_val
        if settings.cursor_auth_header:
            headers["Authorization"] = settings.cursor_auth_header
        return headers


# ── Module-level singleton ──────────────────────────────────────────────────
credential_pool = CredentialPool()
