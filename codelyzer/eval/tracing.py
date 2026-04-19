"""Optional LangSmith ``@traceable`` wrappers (no-op if ``langsmith`` is not installed)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def traceable_step(name: str | None = None) -> Callable[[F], F]:
    """
    Decorate pipeline stages for LangSmith when ``LANGCHAIN_TRACING_V2=true`` and API key set.

    If ``langsmith`` is missing, returns the function unchanged.
    """
    try:
        from langsmith import traceable as _traceable  # type: ignore[import-untyped]

        return _traceable(name=name) if name else _traceable()
    except ImportError:

        def _noop(decorated: F) -> F:
            return decorated

        return _noop
