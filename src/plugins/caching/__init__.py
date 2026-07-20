"""Cache substrate shared by plugins.

| Plugin | Enabling flag | Exact injection site |
|---|---|---|
| LLM completion cache | `plugins.llm_cache` | `src/llm/backend.py:make_backend` |

The replay-regime caches (env step cache, SWE verify cache) live in
`src/plugins/replay/` and share this package's `store`.
"""
