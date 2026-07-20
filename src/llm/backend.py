from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import LlmProviderConfig


Emit = Callable[[str], None]


class CompletionRequestError(Exception):
    """The completion request is invalid; the caller owns the failure."""


class CompletionInfraError(Exception):
    """External completion infrastructure failed to serve the request."""


class ProviderRejectedToolCallError(Exception):
    """The provider rejected malformed model-emitted tool arguments."""


class MalformedProviderStreamError(CompletionInfraError):
    """A transient malformed provider stream prevented completion decoding."""


class FrameworkError(Exception):
    """Frozen framework code failed between the policy and the provider."""


class ContextWindowExceededError(CompletionRequestError):
    """The request exceeded the model's context window (a 400 the server cannot serve).

    Unlike a transient 5xx, retrying the same request never succeeds -- it must be made
    smaller. Typed so the action loop catches it and re-trims replayed history instead of
    crashing the rollout. Carries the server-reported token counts when parseable so the
    caller can shrink decisively rather than guessing. Any may be None if the server's
    message did not surface them."""

    def __init__(
        self,
        message: str,
        *,
        limit: int | None = None,
        requested: int | None = None,
        input_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.limit = limit
        self.requested = requested
        self.input_tokens = input_tokens


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """One rendered model request.

    ``enable_thinking`` overrides the provider config's thinking mode for this
    call only; None keeps the configured behavior. Callers may pass non-None
    only for deployments that declare a togglable thinking channel (config
    enable_thinking set). Providers whose reasoning is provider-managed must
    treat it as a no-op."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    enable_thinking: bool | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: str  # raw JSON string emitted by the model


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Completion:
    """Provider-normalized raw completion."""

    tool_calls: tuple[ToolCall, ...] = ()
    content: str | None = None
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    # JSON-safe reasoning text/blocks, or None if the provider did not
    # surface any. Written verbatim to the trace event.
    reasoning_content: Any | None = None
    # JSON-safe slim envelope built by the adapter so callers do not need to
    # know provider-specific message shapes.
    response: dict[str, Any] = field(default_factory=dict)
    # Transport provenance for measurement; policy code must not branch on it.
    served_from_cache: bool = False


class CompletionBackend:
    """One completion source. The caller owns any tool loop.

    Providers override ``_complete`` (one raw model call; the inherited
    ``complete`` ladder classifies their foreign exceptions). Wrappers override
    ``complete`` itself and forward to an inner backend -- the ladder already
    ran at the provider, so a wrapper must never re-classify."""

    @classmethod
    def complete_duration_bound_sec(cls, max_tokens: int) -> float:
        """Worst-case wall time for one healthy ``complete()`` attempt.

        The provider's half of the stall watchdog (episode._stall_timeout_sec):
        the frozen loop bounds act() by wall clock without trusting this
        editable plane to police itself, and only the provider knows how long
        a healthy attempt may take. Every registered provider class overrides
        it with the same transport timeout it enforces, so bound and behavior
        cannot drift. A capability fact: resolved via ``backend_class``, never
        through wrapper stacks, which inherit this raising default."""
        raise NotImplementedError

    async def complete(self, request: CompletionRequest) -> Completion:
        """One completion. Raises only the typed failures defined in this module."""
        from src.llm.transport import classified_completion_failure

        try:
            return await self._complete(request)
        except (
            CompletionRequestError,
            ProviderRejectedToolCallError,
            CompletionInfraError,
            FrameworkError,
        ):
            raise
        except Exception as exc:
            raise classified_completion_failure(exc) from exc

    async def _complete(self, request: CompletionRequest) -> Completion:
        """One raw model call. Every provider class must override it."""
        raise NotImplementedError

    def cache_key(self, request: CompletionRequest) -> str:
        """A stable hash uniquely identifying this request's output. Only called on
        backends ``make_backend`` wraps with the completion cache, which it does
        solely for deterministic configs; such a backend must derive the key from
        the exact request it sends, so the key can never stand for a different
        completion than the one produced."""
        raise NotImplementedError

    async def close(self) -> None:
        pass


def backend_class(provider: str) -> type[CompletionBackend]:
    # Late imports: the concrete backends import this module. One entry
    # per provider value in LlmProviderConfig.
    from src.llm.openai_completion_backend import OpenAICompletionBackend

    backend_classes = {
        "openai_compatible": OpenAICompletionBackend,
    }
    return backend_classes[provider]


def make_backend(
    config: "LlmProviderConfig", *, cache: bool = False
) -> CompletionBackend:
    """Build one backend instance.

    The sampler makes one backend per rollout attempt (each owns its own kept-alive
    client); the smoke check makes a single live throwaway. ``cache=True`` wraps the
    backend with the cross-run completion cache.
    """
    cls = backend_class(config.provider)
    backend = cls(config=config)
    if not cache or not config.is_deterministic:
        return backend
    # Cache only deterministic transports; memoizing randomness pins one draw.
    # Late import: the cache module imports this module.
    from src.plugins.caching.llm_cache import maybe_wrap

    return maybe_wrap(backend, revision=config.provider_revision)


@dataclass(frozen=True, slots=True)
class TurnResult:
    """The outcome of one completed workspace agent turn."""

    thread_id: str
    progress_summary: str


class AgentBackend(ABC):
    """One full repo-mutating agent turn. The backend owns tool execution."""

    @abstractmethod
    def run_turn(
        self,
        *,
        prompt: str,
        repo_root: Path,
        emit: Emit,
        thread_id: str | None = None,
    ) -> TurnResult:
        pass

    def _assert_ready(self) -> None:
        """Framework-internal preflight; raise if this backend cannot serve a turn."""
