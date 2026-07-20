"""HfTokenCounter contract: exact counts, offset-based truncation, and
config-level counter resolution."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from huggingface_hub.errors import RepositoryNotFoundError
from requests import Response

from src.llm.token_counter import HfTokenCounter, resolve_token_counter


def _repository_not_found(name: str) -> RepositoryNotFoundError:
    response = Response()
    response.status_code = 404
    response.url = f"https://huggingface.co/{name}"
    return RepositoryNotFoundError(f"no such repo: {name}", response=response)


@dataclass
class _Encoding:
    ids: list[int]
    offsets: list[tuple[int, int]]


class _CharTokenizer:
    """One token per char, so budgets map 1:1 to text lengths."""

    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        del add_special_tokens
        return _Encoding(
            ids=[ord(c) for c in text], offsets=[(i, i + 1) for i in range(len(text))]
        )


def test_count_matches_tokenizer():
    assert HfTokenCounter(_CharTokenizer()).count("xyz") == 3


def test_truncate_returns_short_text_unchanged():
    assert HfTokenCounter(_CharTokenizer()).truncate("short", 100) == "short"


def test_truncate_clips_to_budget_at_token_boundary():
    counter = HfTokenCounter(_CharTokenizer())
    clipped = counter.truncate("\x00" * 500, 50)
    assert clipped.endswith("...[truncated]")
    assert counter.count(clipped) == 50
    assert clipped.startswith("\x00")


@pytest.fixture(autouse=True)
def _isolate_resolve_cache():
    resolve_token_counter.cache_clear()
    yield
    resolve_token_counter.cache_clear()


class _StubHub:
    """Stands in for tokenizers.Tokenizer at the module seam."""

    def __init__(self, known: dict[str, object]) -> None:
        self.known = known
        self.requested: list[str] = []

    def from_pretrained(self, name: str) -> object:
        self.requested.append(name)
        if name not in self.known:
            raise RuntimeError(f"no such repo: {name}")
        return self.known[name]


def _patch_hub(monkeypatch, known: dict[str, object]) -> _StubHub:
    import tokenizers

    hub = _StubHub(known)
    monkeypatch.setattr(tokenizers, "Tokenizer", hub)
    return hub


class _MissingHub:
    """Tokenizer hub whose every lookup 404s (repo absent), unlike _StubHub's
    generic RuntimeError for an operational/network failure."""

    def from_pretrained(self, name: str) -> object:
        raise _repository_not_found(name)


@pytest.mark.parametrize(
    ("explicit_default", "model_name"),
    [
        pytest.param("Org/Model", "Org/Model", id="explicit-load-failure"),
        pytest.param(
            None, "openai/gpt-oss-network-failure", id="derived-default-operational"
        ),
    ],
)
def test_resolve_propagates_operational_load_failure(
    monkeypatch, explicit_default, model_name
):
    _patch_hub(monkeypatch, known={})
    with pytest.raises(RuntimeError, match="no such repo"):
        resolve_token_counter(explicit_default, model_name)


def test_resolve_explicit_tokenizer_not_found_fails_fast(monkeypatch):
    import tokenizers

    monkeypatch.setattr(tokenizers, "Tokenizer", _MissingHub())

    with pytest.raises(RepositoryNotFoundError, match="no such repo"):
        resolve_token_counter("Org/Missing", "Org/Model")


def test_resolve_defaults_to_hub_shaped_model_name(monkeypatch):
    hub = _patch_hub(monkeypatch, known={"Qwen/Some-Model": _CharTokenizer()})
    counter = resolve_token_counter(None, "Qwen/Some-Model")
    assert counter is not None and counter.count("ab") == 2
    assert hub.requested == ["Qwen/Some-Model"]


def test_resolve_skips_non_hub_model_names_without_lookup(monkeypatch):
    hub = _patch_hub(monkeypatch, known={})
    assert resolve_token_counter(None, "gpt-test") is None
    assert hub.requested == []


def test_resolve_derived_default_falls_back_to_heuristic_on_load_failure(monkeypatch):
    import tokenizers

    monkeypatch.setattr(tokenizers, "Tokenizer", _MissingHub())
    assert resolve_token_counter(None, "openai/gpt-oss-nonexistent") is None
