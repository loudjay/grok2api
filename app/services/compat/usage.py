"""Lightweight token usage estimation helpers.

The project historically returned placeholder usage fields. These helpers keep
the numbers deterministic without introducing extra runtime dependencies.
"""

from __future__ import annotations

import math
from typing import Any

import orjson

PROMPT_OVERHEAD = 4


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return orjson.dumps(value).decode()
    except Exception:
        return str(value)


def estimate_tokens(value: Any) -> int:
    text = _coerce_text(value).strip()
    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def estimate_prompt_tokens(value: Any, *, overhead: int = PROMPT_OVERHEAD) -> int:
    base = estimate_tokens(value)
    if base <= 0:
        return 0
    return base + max(0, overhead)


def estimate_tool_call_tokens(tool_calls: list[Any]) -> int:
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        if isinstance(call, dict):
            normalized.append(call)
            continue
        normalized.append(
            {
                "id": getattr(call, "call_id", ""),
                "name": getattr(call, "name", ""),
                "arguments": getattr(call, "arguments", ""),
            }
        )
    return estimate_tokens(normalized)


__all__ = [
    "estimate_prompt_tokens",
    "estimate_tokens",
    "estimate_tool_call_tokens",
]

