"""Exact token accounting for the framework's context-window strategy.

The server enforces its context limit in *model tokens*; any character-based
heuristic is a nonlinear approximation of that unit and fails unboundedly on
adversarial content (a NUL-heavy binary dump tokenizes at ~1 token/char -- 4x
the chars/4 guess -- which crashed real rollouts). This module measures requests
in the server's own unit via the model's actual tokenizer from the HuggingFace
Hub.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

TRUNCATION_MARKER = "\n...[truncated]"


class HfTokenCounter:
    """Counts and clips text in model tokens (`tokenizers.Tokenizer` adapter)."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def truncate(self, text: str, max_tokens: int) -> str:
        """Clip to ~max_tokens, cutting the original text at a token boundary
        (via encoding offsets), marked as truncated."""
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        if len(encoding.ids) <= max_tokens:
            return text
        keep = max_tokens - self.count(TRUNCATION_MARKER)
        return text[: encoding.offsets[keep - 1][1]] + TRUNCATION_MARKER


@lru_cache(maxsize=4)
def resolve_token_counter(
    tokenizer_name: str | None, model_name: str
) -> HfTokenCounter | None:
    """The counter for a provider config: explicit ``tokenizer_name`` must load
    (fail fast); when unset, ``model_name`` is tried as the tokenizer id if it
    is Hub-id-shaped (org/name). A verified missing Hub repo falls back to the
    chars/4 heuristic (None); operational failures propagate. Non-Hub names
    also use the heuristic. Cached so a missing lookup is not repeated."""
    from huggingface_hub.errors import RepositoryNotFoundError
    from tokenizers import Tokenizer

    if tokenizer_name is None and "/" not in model_name:
        return None
    try:
        return HfTokenCounter(Tokenizer.from_pretrained(tokenizer_name or model_name))
    except RepositoryNotFoundError:
        if tokenizer_name is not None:
            raise
        logger.warning(
            "no Hub tokenizer found for model %r; "
            "context-window accounting falls back to the chars/4 heuristic",
            model_name,
        )
        return None
