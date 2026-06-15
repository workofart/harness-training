"""Reaping the whole ``uv run exp`` subtree on teardown.

The supervisor spawns ``exp`` as a GRANDCHILD: ``auto`` -> ``uv`` (direct child)
-> ``exp``. Killing only the direct child (``process.kill()``) orphans ``exp``
(reparented to PID 1) because ``uv`` cannot forward the signal -- the leak these
tests pin shut. The fix puts the child in its own session/process group
(``start_new_session=True``) so ``terminate_process_group`` can ``killpg`` the
whole subtree, grandchild included.
"""

from __future__ import annotations

import inspect
import os
import signal
import subprocess
import sys
import tempfile
import time

import pytest

from src.supervisor import subproc
from src.supervisor.subproc import (
    run_streamed,
    terminate_live_children,
    terminate_process_group,
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _spawn_grandchild_subtree() -> tuple[subprocess.Popen, int]:
    """Re-create the ``auto -> uv -> exp`` shape: a parent shell that itself spawns
    a child ``sleep`` (the "grandchild") and ``wait``s on it, in its OWN session.
    The parent prints the grandchild pid so the test can watch it independently.
    Returns ``(parent_popen, grandchild_pid)``; the parent's pid is its pgid."""
    parent = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 60 & echo $! ; wait"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    assert parent.stdout is not None
    grandchild_pid = int(parent.stdout.readline().strip())
    # Sanity: the grandchild really is in the parent's process group (so killpg
    # reaches it) and is a distinct process (so kill(parent) alone would miss it).
    assert grandchild_pid != parent.pid
    assert os.getpgid(grandchild_pid) == parent.pid
    return parent, grandchild_pid


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.02)
    return False


def test_plain_kill_of_direct_child_orphans_the_grandchild() -> None:
    """Baseline that PROVES the bug exists: signalling only the direct child (the
    old ``process.kill()`` behavior) leaves the grandchild running."""
    parent, grandchild_pid = _spawn_grandchild_subtree()
    try:
        os.kill(parent.pid, signal.SIGKILL)  # direct child only -- what kill() did
        parent.wait(timeout=5)
        # The grandchild is reparented to PID 1 and keeps running: the leak.
        assert _pid_alive(grandchild_pid), "grandchild should survive a direct kill"
    finally:
        with _suppress_lookup():
            os.kill(grandchild_pid, signal.SIGKILL)


def test_terminate_process_group_reaps_the_whole_subtree() -> None:
    """The fix: ``killpg`` the session takes the grandchild down too."""
    parent, grandchild_pid = _spawn_grandchild_subtree()
    try:
        terminate_process_group(parent.pid, grace_sec=0.1)
        parent.wait(timeout=5)
        assert _wait_gone(grandchild_pid), "grandchild must be reaped with the group"
        assert _wait_gone(parent.pid), "parent must be reaped too"
    finally:
        with _suppress_lookup():
            os.kill(grandchild_pid, signal.SIGKILL)


def test_terminate_process_group_is_a_noop_for_a_dead_group() -> None:
    parent, grandchild_pid = _spawn_grandchild_subtree()
    terminate_process_group(parent.pid, grace_sec=0.1)
    parent.wait(timeout=5)
    assert _wait_gone(grandchild_pid)
    # Group already gone -- must not raise.
    terminate_process_group(parent.pid, grace_sec=0.1)


def test_run_streamed_spawns_child_in_its_own_session() -> None:
    """The child must be a session leader so its pid == pgid and ``killpg`` reaches
    the whole subtree. Assert the child's pgid equals its own pid."""
    completed = run_streamed(
        [
            sys.executable,
            "-c",
            "import os; print(os.getpid(), os.getpgrp())",
        ],
        on_chunk=lambda *_: None,
    )
    pid_str, pgrp_str = completed.stdout.split()
    assert pid_str == pgrp_str, "child should lead its own process group"


def test_run_streamed_kill_path_reaps_grandchild_on_timeout() -> None:
    """End-to-end through ``run_streamed``: a python parent that spawns a detached
    ``sleep`` grandchild then blocks forever. The timeout kill must reap the
    grandchild, not just the parent -- exactly the ``uv``/``exp`` shape."""
    marker_fd, marker_path = tempfile.mkstemp(prefix="grandchild-pid-")
    os.close(marker_fd)
    code = (
        "import subprocess, sys, time\n"
        f"gc = subprocess.Popen(['sleep', '60'])\n"
        f"open({marker_path!r}, 'w').write(str(gc.pid))\n"
        "sys.stdout.write('ready\\n'); sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )
    with pytest.raises(subproc.StreamTimeout):
        run_streamed(
            [sys.executable, "-u", "-c", code],
            timeout_sec=1.5,
            on_chunk=lambda *_: None,
        )
    grandchild_pid = int(open(marker_path).read().strip())
    os.unlink(marker_path)
    try:
        assert _wait_gone(grandchild_pid), "timeout kill must reap the grandchild too"
    finally:
        with _suppress_lookup():
            os.kill(grandchild_pid, signal.SIGKILL)


def test_run_streamed_registers_and_deregisters_live_child() -> None:
    """The registry the signal handler reaps from is populated only while a child
    runs and cleared afterwards -- no stale pids."""
    assert not subproc._LIVE_CHILD_PIDS  # clean baseline for this test
    run_streamed(
        [sys.executable, "-c", "pass"],
        on_chunk=lambda *_: None,
    )
    assert not subproc._LIVE_CHILD_PIDS, "child pid must be deregistered after exit"


def test_terminate_live_children_reaps_a_registered_group() -> None:
    """``terminate_live_children`` (what the signal handler calls) reaps a group
    registered in ``_LIVE_CHILD_PIDS`` -- the path used when the active child is
    blocked deep in the select loop."""
    parent, grandchild_pid = _spawn_grandchild_subtree()
    subproc._LIVE_CHILD_PIDS.add(parent.pid)
    try:
        terminate_live_children()
        parent.wait(timeout=5)
        assert _wait_gone(grandchild_pid)
        assert _wait_gone(parent.pid)
    finally:
        subproc._LIVE_CHILD_PIDS.discard(parent.pid)
        with _suppress_lookup():
            os.kill(grandchild_pid, signal.SIGKILL)


def test_signal_handler_masks_sigint_during_cleanup() -> None:
    """The double-Ctrl-C trap: the installed handler must set SIGINT to ``SIG_IGN``
    before running cleanup, so a second Ctrl-C cannot raise ``KeyboardInterrupt``
    through it and abort the reap. Asserted by source inspection plus a live check
    that the handler ignores re-entrant SIGINT."""
    from src import cli

    src = inspect.getsource(cli._install_runner_teardown_signal_handler)
    assert "SIG_IGN" in src, "handler must mask SIGINT during cleanup"

    # Install the real handler in a child process. The reap is stubbed to send a
    # SECOND SIGINT mid-cleanup, then sleep -- the window where a non-masked SIGINT
    # would raise KeyboardInterrupt and abort cleanup. The handler must instead
    # finish (print REAPED) and exit 130.
    child_code = (
        "import os, signal, time\n"
        "from src.supervisor import subproc\n"
        "def fake():\n"
        "    os.kill(os.getpid(), signal.SIGINT)\n"  # second Ctrl-C mid-cleanup
        "    time.sleep(0.3)\n"  # window a non-masked SIGINT would fire in
        "    print('REAPED', flush=True)\n"
        "subproc.terminate_live_children = fake\n"
        "from src import cli\n"
        "cli._install_runner_teardown_signal_handler()\n"
        "os.kill(os.getpid(), signal.SIGINT)\n"  # first Ctrl-C -> enters handler
        "time.sleep(5)\n"
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    proc = subprocess.run(
        [sys.executable, "-c", child_code],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=repo_root,
    )
    assert proc.returncode == 130, proc.stderr
    assert "REAPED" in proc.stdout, (
        f"cleanup must finish despite the second SIGINT; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )


class _suppress_lookup:
    def __enter__(self) -> "_suppress_lookup":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is not None and issubclass(exc_type, ProcessLookupError)
