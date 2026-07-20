"""Operator lever: bump one replay-cache epoch (see src/plugins/README.md)."""

from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

from src.plugins.caching.store import COUNTERS_DDL, DB_PATH
from src.plugins.replay.contract import epoch_counter_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Bump one replay cache epoch.")
    parser.add_argument("namespace")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()
    key = epoch_counter_key(args.namespace)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(args.db)) as conn, conn:
        conn.execute(COUNTERS_DDL)
        row = conn.execute(
            "INSERT INTO counters(key, value) VALUES(?, 1) "
            "ON CONFLICT(key) DO UPDATE SET value=value + 1 RETURNING value",
            (key,),
        ).fetchone()
        assert row is not None
        new = int(row[0])
        old = new - 1
    print(f"{old}→{new}")


if __name__ == "__main__":
    main()
