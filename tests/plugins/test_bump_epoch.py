from __future__ import annotations

import gc
import sqlite3
import subprocess
import sys
import warnings
from contextlib import closing
from pathlib import Path

from src.plugins.replay import bump_epoch


def test_bump_replay_epoch_increments_namespace(tmp_path: Path) -> None:
    db = tmp_path / "cache" / "cache.db"
    command = [
        # sys.executable, not cwd-relative .venv: suite must pass from any checkout.
        sys.executable,
        "-m",
        "src.plugins.replay.bump_epoch",
        "swe:task-a",
        "--db",
        str(db),
    ]

    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)

    assert first.stdout == "0→1\n"
    assert second.stdout == "1→2\n"
    with closing(sqlite3.connect(db)) as conn:
        row = conn.execute(
            "SELECT value FROM counters WHERE key=?",
            ("env:epoch:swe:task-a",),
        ).fetchone()
    assert row == (2,)


def test_bump_replay_epoch_closes_database(tmp_path: Path, monkeypatch, capsys) -> None:
    db = tmp_path / "cache.db"
    monkeypatch.setattr(
        sys,
        "argv",
        ["bump_epoch", "swe:task-a", "--db", str(db)],
    )
    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        bump_epoch.main()
        gc.collect()

    assert capsys.readouterr().out == "0→1\n"
    assert not [warning for warning in caught if warning.category is ResourceWarning]
