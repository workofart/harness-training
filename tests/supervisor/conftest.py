"""Shared fixtures for the supervisor loop + end-to-end tests.

``repo_root`` is a real temp git repo (the primary) carrying the editable harness
surface + program.md; ``experiments_dir`` is outside it, so seeding/writing
``loop.json``/``experiment.json`` never dirties the repo (in production
``experiments/`` is gitignored -- here we sidestep gitignore entirely).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "primary"
    root.mkdir()
    _git("init", "-q", "-b", "main", cwd=root)
    _git("config", "user.email", "t@example.com", cwd=root)
    _git("config", "user.name", "tester", cwd=root)
    _git("config", "commit.gpgsign", "false", cwd=root)
    (root / "src" / "harness").mkdir(parents=True)
    (root / "tests" / "harness").mkdir(parents=True)
    (root / "src" / "harness" / "core.py").write_text("VALUE = 1\n")
    (root / "tests" / "harness" / "test_core.py").write_text(
        "def test_ok():\n    assert True\n"
    )
    (root / "program.md").write_text("# program\n")
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "init", cwd=root)
    return root


@pytest.fixture
def experiments_dir(tmp_path: Path) -> Path:
    # Outside the git repo, so seeding files never makes repo_root dirty.
    return tmp_path / "experiments"
