"""Normalize environment variables before any module reads them.

Hosting dashboards (and `.env` files) happily define a variable with an empty
value. ``os.getenv(name, default)`` only returns ``default`` when the name is
absent, so an empty value flows straight into ``int()`` and raises
``ValueError``. Gradio hits this during import via ``GRADIO_SERVER_PORT``, which
takes down the whole app before any of our code runs.

Deleting the blank entries lets every existing ``os.getenv(name, default)`` call
fall back to its intended default, including the ones inside third-party code.
"""

from __future__ import annotations

import os

NUMERIC_VARS: tuple[str, ...] = (
    "GRADIO_SERVER_PORT",
    "MAX_AGENT_STEPS",
    "MAX_LOOPS",
    "MAX_TEAM_ROUNDS",
    "MIN_PAPER_YEAR",
    "CHARS_PER_PAGE",
    "RUN_TIMEOUT_SEC",
    "STREAMLIT_SMOKE_TIMEOUT_OK",
)


def _is_intlike(raw: str) -> bool:
    return raw.strip().lstrip("+-").isdigit()


def drop_blank_numeric_vars(names: tuple[str, ...] = NUMERIC_VARS) -> list[str]:
    """Unset numeric env vars whose value is blank or not an integer.

    Returns the names that were removed, for logging.
    """
    dropped: list[str] = []
    for name in names:
        raw = os.environ.get(name)
        if raw is None or _is_intlike(raw):
            continue
        del os.environ[name]
        dropped.append(name)
    return dropped
