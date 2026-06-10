"""The shared streamed-subprocess engine, tested against real child processes.

Both consumers (the `uv run exp` PTY seam and the agent backends' line-JSON
streaming) ride on `run_streamed`, so these pin the behaviors they rely on:
live chunk delivery, full capture for the post-mortem, the deadline kill, and
PTY mode keeping the child's `isatty()` true (what makes the exp progress bar
render through the seam).
"""

from __future__ import annotations

import sys
import time

import pytest

from src.supervisor.subproc import LineSplitter, StreamTimeout, run_streamed


def test_streams_chunks_live_and_captures_everything() -> None:
    chunks: list[tuple[str, str]] = []
    completed = run_streamed(
        [
            sys.executable,
            "-c",
            "import sys; print('out line'); "
            "print('err line', file=sys.stderr); sys.exit(3)",
        ],
        on_chunk=lambda name, text: chunks.append((name, text)),
    )
    assert completed.returncode == 3
    assert "out line" in completed.stdout
    assert "err line" in completed.stderr
    # The callback saw exactly what was captured, per stream.
    assert "".join(t for n, t in chunks if n == "stdout") == completed.stdout
    assert "".join(t for n, t in chunks if n == "stderr") == completed.stderr


def test_timeout_kills_the_child_and_raises() -> None:
    start = time.monotonic()
    with pytest.raises(StreamTimeout):
        run_streamed(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_sec=0.3,
            on_chunk=lambda *_: None,
        )
    # The child was killed at the deadline, not waited out.
    assert time.monotonic() - start < 10


def test_pty_mode_gives_the_child_a_tty() -> None:
    # The exp seam depends on this: the child's progress bar only renders when
    # its stderr isatty(), and the bar must stream through the supervisor live.
    completed = run_streamed(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.stdout.isatty(), sys.stderr.isatty())",
        ],
        use_pty=True,
        on_chunk=lambda *_: None,
    )
    assert "True True" in completed.stdout


def test_pipe_mode_gives_the_child_no_tty() -> None:
    completed = run_streamed(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.stdout.isatty(), sys.stderr.isatty())",
        ],
        on_chunk=lambda *_: None,
    )
    assert "False False" in completed.stdout


def test_nonzero_exit_is_returned_not_raised() -> None:
    completed = run_streamed(
        [sys.executable, "-c", "raise SystemExit(7)"],
        on_chunk=lambda *_: None,
    )
    assert completed.returncode == 7


def test_line_splitter_reassembles_lines_across_chunks_and_flushes_the_tail() -> None:
    lines: list[tuple[str, str]] = []
    splitter = LineSplitter(lambda name, line: lines.append((name, line)))
    splitter.on_chunk("stdout", "a\nb")
    splitter.on_chunk("stdout", "c\nno newline")
    splitter.on_chunk("stderr", "whole\n")
    splitter.flush()
    assert ("stdout", "a\n") in lines
    assert ("stdout", "bc\n") in lines
    assert ("stderr", "whole\n") in lines
    assert ("stdout", "no newline") in lines  # flush delivers the tail
