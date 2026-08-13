"""Shared LLM helpers — request-scoped API keys (never logged or written to disk)."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Per-request / per-thread override from the Gradio password field.
# Never persist this value to disk, logs, or status text.
_api_key_override: ContextVar[str | None] = ContextVar("openai_api_key_override", default=None)

_PLACEHOLDER_KEYS = {"", "sk-your-key-here", "sk-your-key", "your-key-here"}

DEFAULT_MODEL = "gpt-4o-mini"


def on_vercel() -> bool:
    return bool(os.getenv("VERCEL"))


def openai_model() -> str:
    """Model name, falling back when OPENAI_MODEL is missing *or* blank.

    A hosting dashboard can define the variable with an empty value, which
    os.getenv's default does not cover; an empty model reaches the API as
    "you must provide a model parameter".
    """
    return (os.getenv("OPENAI_MODEL") or "").strip() or DEFAULT_MODEL


def _normalize_key(key: str | None) -> str:
    return (key or "").strip()


def looks_like_openai_key(key: str | None) -> bool:
    """Lightweight shape check — does not call OpenAI."""
    k = _normalize_key(key)
    if not k or k.lower() in _PLACEHOLDER_KEYS:
        return False
    # Project keys (sk-proj-…) or classic sk-…
    return bool(re.match(r"^sk-[A-Za-z0-9_\-]{20,}$", k))


def get_api_key() -> str:
    """Resolve API key: UI override first, then environment / .env."""
    override = _api_key_override.get()
    if override:
        return override
    return _normalize_key(os.getenv("OPENAI_API_KEY"))


def require_api_key() -> str:
    key = get_api_key()
    if not key or key.lower() in _PLACEHOLDER_KEYS:
        raise ValueError(
            "OpenAI API key missing. Paste it in the secure key field at the top of the app, "
            "or set OPENAI_API_KEY in .env / Vercel project environment variables. "
            "Get a key at https://platform.openai.com/api-keys"
        )
    if not looks_like_openai_key(key):
        raise ValueError(
            "That does not look like a valid OpenAI API key (expected sk-…). "
            "Check for extra spaces or a truncated paste."
        )
    return key


@contextmanager
def openai_api_key(key: str | None) -> Iterator[None]:
    """Temporarily use a user-supplied key for this request/thread only."""
    normalized = _normalize_key(key)
    if not normalized:
        yield
        return
    token = _api_key_override.set(normalized)
    try:
        yield
    finally:
        _api_key_override.reset(token)


def redact_secrets(text: str, key: str | None = None) -> str:
    """Strip API keys from any text that might be shown in the UI."""
    out = text or ""
    candidates = [get_api_key(), _normalize_key(key), _normalize_key(os.getenv("OPENAI_API_KEY"))]
    for secret in candidates:
        if secret and len(secret) >= 12:
            out = out.replace(secret, "sk-***REDACTED***")
    out = re.sub(r"sk-[A-Za-z0-9_\-]{10,}", "sk-***REDACTED***", out)
    return out


def llm(temperature: float = 0.2) -> ChatOpenAI:
    key = require_api_key()
    return ChatOpenAI(
        model=openai_model(),
        api_key=key,
        temperature=temperature,
    )
