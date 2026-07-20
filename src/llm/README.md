# llm

*Read this to add a model provider, or to see how model calls are made,
classified, and retried.*

Two unrelated jobs share this directory: talking to the task model (the LLM
the harness drives to solve tasks) and running the estimator's agent turns
(the coding agent that edits the harness between measurements). They
deliberately don't mix — task-model calls are measured, classified, and
cached; estimator turns are none of those.

## Completion backends (the task model)

`backend.py` defines `CompletionBackend`. A provider implements `_complete()`
(one raw call) and `complete_duration_bound_sec()` (how long the measurement
watchdog lets one healthy attempt run before treating it as hung); the
inherited `complete()` classifies every failure into the four-exception
contract (`CompletionRequestError`, `CompletionInfraError`,
`ProviderRejectedToolCallError`, `FrameworkError`). Wrappers like the LLM
cache override `complete()` instead and forward. Adding a provider is one
subclass, plus its entry in `backend_class()` and a new value on the
`LlmProviderConfig.provider` literal — a subclass alone is unreachable. Use
`OpenAICompletionBackend` as the concrete implementation of the contract.

| file | role |
| --- | --- |
| `openai_completion_backend.py` | any OpenAI-compatible `/v1` endpoint (SGLang, vLLM, llama-server, Ollama, OpenRouter); the deterministic path — `seed` set ⇒ `is_deterministic` |
| `transport.py` | shared retry/timeout policy: which failures count as transient, stall windows sized to `max_tokens` |
| `token_counter.py` | HF-tokenizer counting for exact context accounting; an explicitly configured tokenizer id fails hard, only an unset one falls back to chars/4 |

Determinism is a configured fact, not a class capability: one predicate
resolved from provider config (`LlmProviderConfig.is_deterministic` — seed
present on an openai-compatible provider) feeds cache admission, replay
gating, and retry policy. Backends do not self-declare determinism.
`max_tokens` is required — a finite decode bound is an upstream contract —
and stream read timeouts derive from it.

## Agent backends (the estimator's turn)

`agent_backend.py` runs one repo-mutating turn of a coding agent inside a
worktree. `ClaudeAgentBackend` shells out to `claude -p`, `CodexAgentBackend`
to `codex exec`; both stream JSON progress events and return a resumable
thread id. Pass `model` to either backend to select its CLI model; omit it to
use the CLI's configured default. `AgenticEstimator` (`src/trainer/estimator.py`)
drives these; the CLI must be installed and authenticated on the host.
