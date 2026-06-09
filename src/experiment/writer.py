"""Write-only atomic persistence for ``experiment.json``.

The orchestrator's sole writer (§4): it takes an ``ExperimentResult`` and writes
it atomically (temp file in the same dir + ``os.replace``) so a crash never
leaves a half-written record. Holds no business logic, reads nothing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from src.experiment.record import ExperimentResult


def write_experiment_result(result: ExperimentResult, *, root: Path) -> None:
    write_json_atomic(
        ExperimentResult.path(result.experiment_id, root=root),
        result.model_dump(mode="json"),
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)
