"""
Shin Proxy — Token counting.

Uses tiktoken when available (accurate, model-aware encoding).
Falls back to the char/4 heuristic when tiktoken is not installed
or when the model has no known encoding.

Per-model encoder instances are cached at module level after first use
so subsequent calls pay only the encoding cost, not the lookup cost.
"""

from __future__ import annotations

import functools
import json
import logging

from shin.converters.to_cursor import _extract_text

log = logging.getLogger(__name__)

# ── tiktoken bootstrap ──────────────────────────────────────────────────────

try:
    import tiktoken as _tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _tiktoken = None  # type: ignore[assignment]
    _TIKTOKEN_AVAILABLE = False

# ── Per-message role overhead (matches OpenAI's counting method) ────────────
# Each message costs: 4 tokens (role + separators + reply priming)
_PER_MESSAGE_OVERHEAD = 4
_REPLY_PRIMING = 3  # every reply is primed with <im_start>assistant


# ── Encoder cache ───────────────────────────────────────────────────────────

def _sanitize_model(model: str) -> str:
    """Strip vendor prefix from Cursor model IDs before tiktoken lookup.

    tiktoken knows bare names like 'gpt-4o', not prefixed ones like
    'openai/gpt-4o' or 'anthropic/claude-sonnet-4.6'.
    """
    # e.g. 'anthropic/claude-sonnet-4.6' -> 'claude-sonnet-4.6'
    if "/" in model:
        _, bare = model.split("/", 1)
        return bare
    return model


@functools.lru_cache(maxsize=16)
def _get_encoder(model: str):
    """Return a cached tiktoken encoder for the given model name.

    Falls back to cl100k_base (GPT-4o / Claude compatible) on unknown models.
    Returns None if tiktoken is unavailable.
    """
    if not _TIKTOKEN_AVAILABLE:
        return None
    bare = _sanitize_model(model)
    try:
        return _tiktoken.encoding_for_model(bare)
    except KeyError:
        try:
            return _tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def _count_text_tokens(text: str, encoder) -> int:
    """Count tokens in a string using the given encoder."""
    if encoder is None:
        return max(1, len(text) // 4)
    try:
        return len(encoder.encode(text, disallowed_special=()))
    except Exception:
        return max(1, len(text) // 4)


def _heuristic(text: str) -> int:
    """Char/4 heuristic — fast, no dependencies."""
    return max(1, len(text) // 4)


# ── Public API ───────────────────────────────────────────────────────────────

def count_tokens(text: str, model: str = "") -> int:
    """Count tokens in a plain text string.

    Uses tiktoken when available; otherwise falls back to char/4.
    """
    enc = _get_encoder(model) if model else _get_encoder("gpt-4o")
    return _count_text_tokens(text, enc)


def count_message_tokens(messages: list[dict], model: str = "") -> int:
    """Count total tokens for a list of messages.

    Matches OpenAI's method: sum of content tokens + 4 per message +
    3 reply-priming tokens for the assistant turn.

    Handles:
    - str content
    - list[dict] content blocks  (Anthropic / vision format)
    - tool_calls  (counted as their serialised JSON string)
    """
    enc = _get_encoder(model) if model else _get_encoder("gpt-4o")
    total = _REPLY_PRIMING

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        total += _PER_MESSAGE_OVERHEAD

        # role (always a short string like "user" / "assistant")
        total += _count_text_tokens(msg.get("role", ""), enc)

        # content — str, list, or None
        content = msg.get("content")
        if isinstance(content, str):
            total += _count_text_tokens(content, enc)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    t = block.get("text") or block.get("content", "")
                    if isinstance(t, str):
                        total += _count_text_tokens(t, enc)
        elif content is not None:
            total += _count_text_tokens(str(content), enc)

        # tool_calls (serialised JSON for counting)
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                total += _count_text_tokens(fn.get("name", ""), enc)
                total += _count_text_tokens(fn.get("arguments", ""), enc)

        # name field adds 1 token overhead
        if msg.get("name"):
            total += 1

    return max(1, total)


def count_tool_tokens(tools: list[dict], model: str = "") -> int:
    """Estimate tokens consumed by the tools block (JSON schema injection)."""
    if not tools:
        return 0
    enc = _get_encoder(model) if model else _get_encoder("gpt-4o")
    try:
        serialised = json.dumps(tools, ensure_ascii=False)
    except Exception:
        serialised = str(tools)
    # Tool definitions are injected as a system message — add overhead
    return _count_text_tokens(serialised, enc) + _PER_MESSAGE_OVERHEAD


# ── Legacy compat (used by analytics.py) ────────────────────────────────────

def estimate_from_messages(messages: list[dict] | None, model: str = "") -> int:
    """Legacy alias — uses accurate counting when tiktoken is available."""
    return count_message_tokens(messages or [], model)


def estimate_from_text(text: str | None, model: str = "") -> int:
    """Legacy alias — uses accurate counting when tiktoken is available."""
    return count_tokens(text or "", model)
