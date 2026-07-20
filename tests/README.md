# Test design contract

Keep tests small by testing behavior classes, not by hiding test logic. A change is
an improvement only when the resulting test still makes the input, expected
observable, and failure boundary obvious.

## Choose the smallest framing

| What changes between cases? | Framing |
|---|---|
| Only input and expected output | One `pytest.mark.parametrize` table |
| Repeated setup around one production boundary | One module-local scenario helper |
| Order, concurrency, cancellation, cleanup, or lifecycle | An explicit focused test |
| Real Docker, network, Git, or subprocess behavior | A test marked `@pytest.mark.integration` (excluded from the default run) |

Start by listing the behavior classes. Add a row to an existing table when the
arrange/act path is the same. Use a separate test when the failure phase, ordering,
cleanup, or artifact contract differs.

Prefer ordinary values and descriptive IDs:

```python
@pytest.mark.parametrize(
    ("method", "status", "request_headers", "response_headers"),
    [
        pytest.param("GET", 404, None, None, id="non-success"),
        pytest.param("HEAD", 200, None, None, id="unsupported-method"),
        pytest.param("GET", 200, [("Range", "bytes=10-")], None, id="range"),
    ],
)
def test_non_cacheable_responses_leave_no_entry(...):
    ...
```

Use factories for mutable values, streams, iterators, and exceptions. Parameter
rows must not share state.

## Boundary helpers

A helper owns one real boundary and returns collaborators that tests inspect. For
example, `_TrainerCase` owns trainer/repository/measurement wiring, while
`_MeasurementCase` owns process/pipe/event wiring. They remain separate because a
single helper spanning both would reproduce production control flow in tests.

A helper is justified only when it removes repeated orchestration from multiple
cases. Do not add flags for unrelated phases, a generic test DSL, or a wrapper that
only renames one call. Keep cross-module isolation in `conftest.py`; prefer
module-local helpers otherwise.

## Generalize only to a stronger invariant

After creating a parameter table, challenge every child row and expected pattern.
Two cases may consolidate when all of these are true:

1. They traverse the same behavior boundary and production branch obligation.
2. One input/invariant is a genuine superset, not merely a looser assertion.
3. The surviving test retains all relevant expected fields, calls, order, cleanup,
   metrics, persistence, and artifacts.
4. The old regression remains identifiable from the surviving test name or row ID.

For static text contracts, one required/forbidden fragment set is stronger and
smaller than many tests that each read the same file and assert one substring. Keep
ordering assertions separate. For example, the Terminal-Bench setup test checks the
complete fragment superset, while CA installation order remains its own test.

Do not merge foreign and owned timeouts, retry exhaustion and ordinary transient
failures, stateful recovery steps, or capability-presence cases merely because their
fixtures look similar.

## Assertions and regressions

Assert the full relevant projection, not only the headline result:

- outcome plus reason, origin, and error;
- calls and event order;
- cleanup and closed-resource facts;
- metrics and persisted rows;
- generated artifacts and stable payloads.

Keep full protocol and persistence goldens. Do not move them to an uncounted data
file, minify them, suppress formatting, or replace them with a weaker snapshot for a
line-count reduction.

For a bug, first add a failing regression at the closest public boundary. Once the
fix passes, fold it into the nearest existing table or scenario with a stable,
descriptive ID when the framing is the same.

## Required checks

Activate the project environment, then run the changed module before the suite:

```bash
source .venv/bin/activate
ruff format --check tests/path/to/test_module.py
ruff check tests/path/to/test_module.py
pytest -q tests/path/to/test_module.py
```

Before handoff, run the whole set:

```bash
ruff format --check . && ruff check .
pytest -q                      # default: unit suite, parallel
pytest -q -m integration       # needs real Docker and live network
pytest -q --cov --cov-branch   # branch coverage
```

The default run is parallel and skips integration tests
(`-m 'not integration' -n 4 --dist loadgroup`), so no test may depend on
execution order or on another test's process state; cross-module isolation lives
in the autouse `conftest.py` fixtures. Reviewing coverage is not a substitute for
reviewing the expected projections above.
