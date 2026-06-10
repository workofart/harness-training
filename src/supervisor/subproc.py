"""One streamed-subprocess engine for the supervisor's two consumers.

``run_streamed`` runs a child process, forwards its decoded output to a
callback as it arrives (so progress is live), captures everything for the
post-mortem on failure, and kills the child when ``timeout_sec`` expires.
Single-threaded: both streams are multiplexed with ``selectors``.

Two consumers with different needs share it:

- the ``uv run exp`` seam (``loop._run_with_live_tty_output``) runs the child
  under PTYs (``use_pty=True``) so its tty-gated progress bar streams live, and
  forwards raw chunks straight to our stdout/stderr;
- the agent backends (``agent_backend._run_streamed_process``) run codex/claude
  over pipes and parse stdout line-by-line (JSON events) via ``LineSplitter``.
"""

from __future__ import annotations

import codecs
import errno
import locale
import os
import pty
import selectors
import subprocess
import time
import tty
from collections.abc import Callable
from pathlib import Path

# (stream_name, decoded_chunk) where stream_name is "stdout" or "stderr".
OnChunk = Callable[[str, str], None]

_READ_CHUNK_BYTES = 8192


class StreamTimeout(RuntimeError):
    """The child exceeded ``timeout_sec`` and was killed."""

    def __init__(self, timeout_sec: float) -> None:
        self.timeout_sec = timeout_sec
        super().__init__(f"subprocess timed out after {timeout_sec} seconds")


class LineSplitter:
    """Reassemble ``on_chunk`` text into newline-terminated lines per stream.

    A trailing unterminated line is delivered by ``flush()`` -- call it after
    ``run_streamed`` returns so a final line without ``\\n`` is not dropped.
    """

    def __init__(self, on_line: Callable[[str, str], None]) -> None:
        self._on_line = on_line
        self._buffers: dict[str, str] = {}

    def on_chunk(self, stream_name: str, chunk: str) -> None:
        buffer = self._buffers.get(stream_name, "") + chunk
        while True:
            line, sep, rest = buffer.partition("\n")
            if not sep:
                break
            self._on_line(stream_name, line + "\n")
            buffer = rest
        self._buffers[stream_name] = buffer

    def flush(self) -> None:
        for stream_name, buffer in self._buffers.items():
            if buffer:
                self._on_line(stream_name, buffer)
        self._buffers.clear()


def run_streamed(
    command: list[str],
    *,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    use_pty: bool = False,
    timeout_sec: float | None = None,
    on_chunk: OnChunk,
) -> subprocess.CompletedProcess[str]:
    """Run ``command``, streaming decoded output to ``on_chunk`` as it arrives.

    Returns a ``CompletedProcess`` carrying the full captured stdout/stderr
    text. ``use_pty=True`` gives the child PTYs on both streams so tty-gated
    output (progress bars) stays live. Raises ``StreamTimeout`` (child killed)
    when ``timeout_sec`` elapses before the child finishes.
    """
    encoding = locale.getencoding()
    popen_kwargs: dict = dict(
        cwd=cwd, env=env, stdin=subprocess.DEVNULL, close_fds=True
    )
    if use_pty:
        stdout_master, stdout_slave = pty.openpty()
        stderr_master, stderr_slave = pty.openpty()
        tty.setraw(stdout_slave)
        tty.setraw(stderr_slave)
        process = subprocess.Popen(
            command, stdout=stdout_slave, stderr=stderr_slave, **popen_kwargs
        )
        os.close(stdout_slave)
        os.close(stderr_slave)
        fd_streams = {stdout_master: "stdout", stderr_master: "stderr"}
    else:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **popen_kwargs
        )
        assert process.stdout is not None and process.stderr is not None
        fd_streams = {
            process.stdout.fileno(): "stdout",
            process.stderr.fileno(): "stderr",
        }
    decoders = {
        fd: codecs.getincrementaldecoder(encoding)(errors="replace")
        for fd in fd_streams
    }
    captured: dict[str, list[str]] = {"stdout": [], "stderr": []}
    deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
    selector = selectors.DefaultSelector()
    for fd in fd_streams:
        selector.register(fd, selectors.EVENT_READ)

    def deliver(fd: int, text: str) -> None:
        if not text:
            return
        stream_name = fd_streams[fd]
        captured[stream_name].append(text)
        on_chunk(stream_name, text)

    try:
        while selector.get_map():
            select_timeout = None
            if deadline is not None:
                select_timeout = deadline - time.monotonic()
                if select_timeout <= 0:
                    raise StreamTimeout(timeout_sec)  # type: ignore[arg-type]
            for key, _ in selector.select(select_timeout):
                try:
                    chunk = os.read(key.fd, _READ_CHUNK_BYTES)
                except OSError as exc:
                    # A PTY master raises EIO when the child closes its end:
                    # that is this stream's EOF, not an error.
                    if exc.errno != errno.EIO:
                        raise
                    chunk = b""
                if not chunk:
                    selector.unregister(key.fd)
                    deliver(key.fd, decoders[key.fd].decode(b"", final=True))
                    if use_pty:
                        os.close(key.fd)
                    continue
                deliver(key.fd, decoders[key.fd].decode(chunk))
        return subprocess.CompletedProcess(
            args=command,
            returncode=process.wait(),
            stdout="".join(captured["stdout"]),
            stderr="".join(captured["stderr"]),
        )
    finally:
        if use_pty:
            for fd in list(selector.get_map()):
                try:
                    os.close(int(fd))  # registered fds are always ints here
                except OSError:
                    pass
        selector.close()
        if not use_pty:
            assert process.stdout is not None and process.stderr is not None
            process.stdout.close()
            process.stderr.close()
        if process.poll() is None:
            process.kill()
            process.wait()
