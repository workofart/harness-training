"""Provider-neutral transport policy (retry + timeout constants + stall math)
shared by the completion backends."""

from __future__ import annotations

import asyncio
import logging

import httpx
from openai import APIConnectionError, APIError, APIStatusError, BadRequestError
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.llm.backend import (
    CompletionInfraError,
    CompletionRequestError,
    MalformedProviderStreamError,
    ProviderRejectedToolCallError,
)

# Bounds connect/write/pool phases, not total stream duration; reads use the
# stall window below.
REQUEST_TIMEOUT_SECONDS = 300.0

# Tenacity owns transient retries; SDK retries stay disabled because nested
# budgets can multiply into (budget+1)^2 attempts.
OPENAI_CLIENT_RETRY_BUDGET = 5

# Retry inter-chunk stalls. Scale for buffered tool calls at a conservative 20
# tok/s plus 60s, with a 240s floor; per-chunk timing leaves healthy long streams
# unbounded.
STREAM_STALL_TIMEOUT_FLOOR_SECONDS = 240.0
_BUFFERED_CALL_DECODE_FLOOR_TOKENS_PER_SEC = 20.0
_FIRST_TOKEN_MARGIN_SECONDS = 60.0
_MALFORMED_TOOL_ARGS = "Failed to parse tool call arguments as JSON"


def _is_provider_rejected_tool_call(exc: BaseException) -> bool:
    return isinstance(exc, APIError) and _MALFORMED_TOOL_ARGS in str(exc)


def stream_stall_timeout_seconds(max_tokens: int) -> float:
    """Longest stream silence tolerated before the attempt is retried."""
    return max(
        STREAM_STALL_TIMEOUT_FLOOR_SECONDS,
        max_tokens / _BUFFERED_CALL_DECODE_FLOOR_TOKENS_PER_SEC
        + _FIRST_TOKEN_MARGIN_SECONDS,
    )


def transient_retrying(log: logging.Logger) -> AsyncRetrying:
    """One completion call's retry policy, used by the OpenAI-compatible
    backend. Backoff is sized
    to the serving endpoint, not the SDK default (0.5/8: 15.5s of sleep across
    the budget): the Vast serverless autoscaler stops idle workers and takes
    60-110s to bring one back, so the total sleep (4+8+16+32+45 = 105s) must
    span a worker-restart window even when each attempt fails fast."""
    return AsyncRetrying(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(OPENAI_CLIENT_RETRY_BUDGET + 1),
        wait=wait_exponential(multiplier=4.0, max=45.0),
        before_sleep=before_sleep_log(log, logging.WARNING),
        sleep=asyncio.sleep,
        reraise=True,
    )


def _is_transient(exc: BaseException) -> bool:
    """Retry transient transport/stream failures; surface client errors (4xx),
    which a retry cannot fix. A statusless stream error, a stalled or
    dropped/timed-out connection, malformed provider output, a 5xx, and the
    retryable statuses (408 timeout, 409 conflict, 429 rate limit) are transient;
    any other classified 4xx is not."""
    if _is_provider_rejected_tool_call(exc):
        return False
    if isinstance(exc, MalformedProviderStreamError):
        return True
    # Stream-consumption failures surface as raw httpx errors, unlike create()
    # failures.
    if isinstance(exc, (httpx.RemoteProtocolError, httpx.TimeoutException)):
        return True
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    return isinstance(exc, APIError)  # bare APIError: mid-stream `error` event


def classified_completion_failure(exc: Exception) -> Exception:
    """Map an unclassified completion failure onto the backend exception contract.

    Unrecognized types default to infra so an unlisted SDK exception crashes
    the task, not the whole experiment."""
    if _is_provider_rejected_tool_call(exc):
        return ProviderRejectedToolCallError(str(exc))
    if isinstance(exc, BadRequestError):
        return CompletionRequestError(exc.message)
    return CompletionInfraError(str(exc))
