"""
Shin Proxy — Bearer token auth middleware.
"""

from __future__ import annotations

from shin.config import settings
from shin.handlers import AuthError


def verify_bearer(authorization: str | None) -> str:
    """Validate Bearer token and return the API key.

    Raises AuthError if token is missing or invalid.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthError("Authentication Error, No api key passed in.")
    token = authorization.split(" ", 1)[1]
    if token != settings.master_key:
        raise AuthError("Authentication Error, Invalid api key.")
    return token
