from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def json_safe(value: Any, *, depth: int = 0) -> Any:
    """Coerce an arbitrary value into a JSON-serializable shape.

    Shared by the adapters (provider response envelopes) and `trace` (event
    fields). Lives in its own leaf module so both layers can import it without
    pulling in unrelated concerns or creating an import cycle.
    """
    if depth >= 12:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): json_safe(item, depth=depth + 1) for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item, depth=depth + 1) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value), depth=depth + 1)
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json")
        except TypeError:
            dumped = value.model_dump()
        return json_safe(dumped, depth=depth + 1)
    if hasattr(value, "dict"):
        try:
            dumped = value.dict()
        except TypeError:
            dumped = value.dict
        return json_safe(dumped, depth=depth + 1)
    if hasattr(value, "__dict__"):
        return json_safe(vars(value), depth=depth + 1)
    return repr(value)
