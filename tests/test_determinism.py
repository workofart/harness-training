"""Unit tests for the centralized determinism controls.

Each scrub case is a regression for a specific run-to-run fork observed in experiment
artifacts; comments or stable parameter IDs retain provenance. False-positive cases
guard the signal the agent needs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import pytest

from src.determinism import (
    CONTAINER_HOSTNAME,
    GIT_HOOKS_INIT_COMMAND,
    GDB_INIT_COMMAND,
    MTIME_RESET_COMMAND,
    PINS_FINGERPRINT,
    SOLVE_EXEC_ENV,
    drift_audit_canonical,
    scrub_nondeterminism,
)


def test_pins_fingerprint_covers_all_source_pins():
    payload = json.dumps(
        {
            "container_hostname": CONTAINER_HOSTNAME,
            "solve_exec_env": SOLVE_EXEC_ENV,
            "mtime_reset_command": MTIME_RESET_COMMAND,
            "git_hooks_init_command": GIT_HOOKS_INIT_COMMAND,
            "gdb_init_command": GDB_INIT_COMMAND,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    assert re.fullmatch(r"[0-9a-f]{12}", PINS_FINGERPRINT)
    assert PINS_FINGERPRINT == hashlib.sha256(payload.encode()).hexdigest()[:12]


def test_drift_audit_canonical_scrubs_and_strips_trailing_whitespace_per_line():
    text = "built at 15:11:50   \nserver: ok\t\n"

    assert drift_audit_canonical(text) == "\nbuilt at <TIME>\nserver: ok"


def test_drift_audit_canonical_ignores_line_order_but_preserves_content():
    first = "worker one warning\nworker two warning\nworker one warning"
    reordered = "worker one warning\nworker one warning\nworker two warning"
    changed = "worker one warning\nworker one warning\nworker three warning"

    assert drift_audit_canonical(first) == drift_audit_canonical(reordered)
    assert drift_audit_canonical(first) != drift_audit_canonical(changed)


def test_scrubs_hex_object_address():
    # sympy-17318: `<... object at 0x...>` in a pip/urllib3 repr.
    a = scrub_nondeterminism("broken by '<pool object at 0x7ffffd51dd30>'")
    b = scrub_nondeterminism("broken by '<pool object at 0x7ffffd544d30>'")
    assert a == b
    assert "0x" not in a


def test_scrubs_wall_clock_duration_keeps_passfail():
    out = scrub_nondeterminism("===== 1 passed, 2 warnings in 0.34s =====")
    assert "1 passed" in out
    assert "0.34s" not in out


def test_scrubs_spelled_out_duration_seconds():
    # sympy-13091: `... in 0.53 seconds =` -- the spelled-out unit the `0.34s`-only
    # form missed, which forked the trajectory and flipped the task at the gate.
    a = scrub_nondeterminism("= tests finished: 73 passed, in 0.53 seconds =")
    b = scrub_nondeterminism("= tests finished: 73 passed, in 0.59 seconds =")
    assert a == b
    assert "seconds" not in a


def test_scrubs_minutes_seconds_duration():
    # pytest prints `in 1m 23s` once a run crosses 60s; same intrinsic-timing class.
    a = scrub_nondeterminism("===== 5 passed in 1m 23s =====")
    b = scrub_nondeterminism("===== 5 passed in 1m 47s =====")
    assert a == b
    assert "1m 23s" not in a


def test_scrubs_pytest_elapsed_parenthetical_and_banner_width():
    # pylint-4970: pytest adds (H:MM:SS) after 60s and derives banner width from the variable line; both forked 8 candidate runs.
    slow = scrub_nondeterminism(
        "====================== 1 passed, 3 warnings in 61.20s (0:01:01) ======================"
    )
    fast = scrub_nondeterminism(
        "=========================== 1 passed, 3 warnings in 2.34s ==========================="
    )
    assert slow == fast
    assert "0:01:01" not in slow and "61.20s" not in slow


def test_scrubs_duration_generic_across_phrasings_and_units():
    # Use one generic duration rule so new tool and unit spellings cannot re-leak.
    variants = [
        "Build took 1.2s",
        "Build took 3.74s",
        "Build took 1m4s",
        "Build took 500ms",
        "Build took 2 minutes",
    ]
    assert {scrub_nondeterminism(v) for v in variants} == {"Build took <DUR>"}


def test_scrubs_clock_timestamp():
    assert "15:11:50" not in scrub_nondeterminism("built at 15:11:50 done")
    # single-digit hour (the form pytest's elapsed parenthetical uses) is also caught.
    assert "0:01:01" not in scrub_nondeterminism("elapsed (0:01:01) total")


def test_scrubs_certificate_fingerprint():
    # terminal-bench/openssl-selfsigned-cert: `openssl x509 -fingerprint` prints
    # the digest of a keypair generated inside the trial, so it is new every run.
    # Two eval runs of the same commit forked here at step 9, turning a 22-step
    # solve into a 30-step cap.
    a = scrub_nondeterminism(
        "sha256 Fingerprint=2A:D7:A2:72:AF:F7:BE:50:D3:2D:01:F2:64:17:1E:2E:"
        "DF:30:0B:97:11:0B:EF:B7:28:6A:78:56:FA:C3:FF:27"
    )
    b = scrub_nondeterminism(
        "sha256 Fingerprint=9D:4F:71:BF:D6:4C:04:84:AC:C9:E4:D8:4B:03:3B:03:"
        "C1:D2:99:C0:26:1A:D5:A1:9A:26:11:99:F6:76:48:F3"
    )
    assert a == b == "sha256 Fingerprint=<FINGERPRINT>"


def test_clock_rule_does_not_split_colon_separated_hex():
    # The unguarded HH:MM:SS rule rewrote an all-numeric octet run inside a
    # digest (`...:9A:26:11:99:F6:...` -> `...:9A:<TIME>:F6:...`), which left the
    # surrounding random hex in the observation instead of masking it. 72% of
    # random SHA-256 fingerprints contain such a run.
    masked = scrub_nondeterminism("ether 00:11:22:33:44:55 txqueuelen 1000")
    assert "<TIME>" not in masked
    assert "00:11:22:33:44:55" in masked


def test_scrubs_random_tempfile_name():
    # pytest-10051 / sphinx-10323: tempfile's 8-char random component.
    a = scrub_nondeterminism("Source dir: /tmp/tmpkce_suy0/source")
    b = scrub_nondeterminism("Source dir: /tmp/tmp6e5qoqik/source")
    assert a == b
    a = scrub_nondeterminism("../tmp/tmpxiubt8w8_test.py::test FAILED")
    b = scrub_nondeterminism("../tmp/tmpef94lah0_test.py::test FAILED")
    assert a == b


def test_scrubs_sphinx_random_error_log_name():
    # sphinx-10673: Sphinx's tempfile suffix forked every trajectory after step 85.
    a = scrub_nondeterminism("saved in /tmp/sphinx-err-zukakzni.log")
    b = scrub_nondeterminism("saved in /tmp/sphinx-err-m0nj2_k1.log")

    assert a == b == "saved in /tmp/sphinx-err-<RAND>.log"
    assert (
        scrub_nondeterminism("saved in /tmp/sphinx-err-reference.log")
        == "saved in /tmp/sphinx-err-reference.log"
    )


def test_scrubs_sympy_random_seed():
    # sympy-23950: test runner's per-process random seed.
    a = scrub_nondeterminism("random seed:        73158167")
    b = scrub_nondeterminism("random seed:        73018636")
    assert a == b


def test_scrubs_ls_l_mtime():
    # django-14155: `.git` line of `ls -la /testbed` forked on the container mtime.
    a = scrub_nondeterminism("drwxrwxrwx 1 root root    10 Jun 29 18:19 .git")
    b = scrub_nondeterminism("drwxrwxrwx 1 root root    10 Jun 29 19:05 .git")
    assert a == b == "drwxrwxrwx 1 root root    10 <MTIME> .git"
    # Sticky/setuid/setgid mode bits are still ordinary `ls -l` permission output.
    a = scrub_nondeterminism("drwxrwxrwt   1 root root  26 Jan  1  1970 tmp")
    b = scrub_nondeterminism("drwxrwxrwt   1 root root  26 Jul  5 20:40 tmp")
    assert a == b == "drwxrwxrwt   1 root root  26 <MTIME> tmp"
    assert "2025" not in scrub_nondeterminism(
        "-rw-r--r-- 1 root root 1407 Sep 10  2025 .eslintrc"
    )


def test_keeps_ps_header_and_uppercase_pid_without_number():
    # `ps` prints a bare `PID` column header (no colon+number); it is not a
    # launch label and must survive.
    text = "  PID TTY          TIME CMD\n 1234 pts/0    00:00:00 bash"
    assert "PID TTY" in scrub_nondeterminism(text)


def test_scrubs_netstat_pid_program_column():
    # mailman: normalize the run-varying PID in PID/program; preserve the stable program and unattributed "-".
    row = (
        "tcp        0      0 127.0.0.1:8024          0.0.0.0:*"
        "               LISTEN      {pid}/python3"
    )
    a = scrub_nondeterminism(row.format(pid="2138"))
    b = scrub_nondeterminism(row.format(pid="94"))
    assert a == b
    assert a.endswith("LISTEN      <PID>/python3")
    unattributed = (
        "tcp        0      0 0.0.0.0:25              0.0.0.0:*"
        "               LISTEN      -"
    )
    assert scrub_nondeterminism(unattributed).endswith("LISTEN      -")


def test_scrubs_debian_backend_header_values():
    a = scrub_nondeterminism(
        "Content-Type: application/x-gzip\r\n"
        "Backend: 4qpvL1tJyeV1P6Tmf0Lj8g--F_conova_debian_backend_mirrors_debian_org\r\n"
        "Accept-Ranges: bytes\r\n"
    )
    b = scrub_nondeterminism(
        "Content-Type: application/x-gzip\r\n"
        "Backend: 4qpvL1tJyeV1P6Tmf0Lj8g--F_accum_debian_backend_mirrors_debian_org\r\n"
        "Accept-Ranges: bytes\r\n"
    )
    assert a == b
    assert "conova" not in a and "accum" not in b
    assert "backend: <DEBIAN_BACKEND>" in a


def test_keeps_generic_server_headers_and_backend_words():
    text = "Server: nginx/1.22.1\r\nBackend: api-primary\r\n"
    scrubbed = scrub_nondeterminism(text)
    assert scrubbed == "server: nginx/1.22.1\nbackend: api-primary\n"
    assert "api-primary" in scrubbed


def test_debian_backend_near_miss_does_not_backtrack_quadratically():
    text = "Backend: " + ("a--F_" * 20_000) + "not_debian\n"

    started = time.perf_counter()
    assert scrub_nondeterminism(text) == text

    assert time.perf_counter() - started < 1.0


def test_scrubs_torchelastic_child_failure_report():
    # torch-tensor-parallelism: dead-child report timestamps and PIDs vary per run.
    a = scrub_nondeterminism(
        "  time      : 2026-07-10_21:38:08\n"
        "  host      : sandbox\n"
        "  rank      : 1 (local_rank: 1)\n"
        "  exitcode  : 1 (pid: 4631) \n"
        "  error_file: <N/A>\n"
        "worker failed (exitcode: 1) local_rank: 0 (pid: 4630) of binary: /usr/bin/python3\n"
        "worker Sending process 4631 closing signal SIGTERM"
    )
    b = scrub_nondeterminism(
        "  time      : 2026-07-11_01:02:03\n"
        "  host      : sandbox\n"
        "  rank      : 1 (local_rank: 1)\n"
        "  exitcode  : 1 (pid: 9842) \n"
        "  error_file: <N/A>\n"
        "worker failed (exitcode: 1) local_rank: 0 (pid: 9841) of binary: /usr/bin/python3\n"
        "worker Sending process 9842 closing signal SIGTERM"
    )
    assert a == b
    assert "<DATE>_<TIME>" in a
    assert "(pid: <PID>)" in a
    assert "Sending process <PID> closing signal" in a


def test_keeps_non_torchelastic_closing_signal_prose():
    # Only the glog-prefixed torchelastic line is a timing race; bare prose
    # keeps its line (with the dead-child PID still normalized).
    text = "Sending process 4631 closing signal SIGTERM"
    assert scrub_nondeterminism(text) == (
        "Sending process <PID> closing signal SIGTERM"
    )


def test_scrubs_bash_pipeline_job_status_continuation_pids():
    # write-compressor (exp-20260704-004139): bash omits the prefix on later pipeline status lines, leaking their PIDs.
    a = scrub_nondeterminism(
        "bash: line 1:   116 Done                    printf '\\x02\\x01'\n"
        "       117 Segmentation fault      | /app/decomp > /tmp/out.txt 2>&1"
    )
    b = scrub_nondeterminism(
        "bash: line 1:   118 Done                    printf '\\x02\\x01'\n"
        "       119 Segmentation fault      | /app/decomp > /tmp/out.txt 2>&1"
    )
    assert a == b
    assert "117" not in a and "119" not in b


def test_scrubs_process_termination_log_metadata():
    # Some runtimes use glog-style severity/date/time/pid prefixes and then
    # repeat another kernel PID in the message body.
    a = scrub_nondeterminism(
        "W0705 <TIME>.812000 5072 torch/multiprocessing/spawn.py:165] "
        "Terminating process 5088 via signal SIGTERM"
    )
    b = scrub_nondeterminism(
        "W0706 <TIME>.889000 5426 torch/multiprocessing/spawn.py:165] "
        "Terminating process 5442 via signal SIGTERM"
    )
    assert a == b
    assert "5072" not in a and "5088" not in a
    assert a == (
        "<LOG> <TIME>.<SUBSEC> <PID> torch/multiprocessing/spawn.py:165] "
        "Terminating process <PID> via signal SIGTERM"
    )


def test_scrubs_bare_pid_line_for_pid_producing_command():
    # pypi-server: `pypi-server run ... & ; echo $!` prints the bare child PID.
    command = "pypi-server run -p 8080 /app/dist/ > /tmp/pypi.log 2>&1 &\necho $!"
    a = scrub_nondeterminism("1410\nListening on http://0.0.0.0:8080/", command=command)
    b = scrub_nondeterminism("1404\nListening on http://0.0.0.0:8080/", command=command)
    assert a == b == "<PID>\nListening on http://0.0.0.0:8080/"
    # qemu-alpine-ssh: `pidof qemu-system-x86_64` prints bare PIDs; multiple
    # matches land space-separated on one line.
    a = scrub_nondeterminism("4189 312\nrunning", command="pidof qemu-system-x86_64")
    b = scrub_nondeterminism("4171 298\nrunning", command="pidof qemu-system-x86_64")
    assert a == b == "<PID> <PID>\nrunning"


def test_keeps_bare_numeric_line_without_pid_producing_command():
    # A digit-only line is a legitimate answer (`wc -l < file`) unless the
    # command asked for a PID by construction.
    text = "42"
    assert scrub_nondeterminism(text, command="wc -l < access.log") == text
    assert scrub_nondeterminism(text) == text


def test_keeps_prose_lines_for_pid_producing_command():
    # The bare-PID mask is line-anchored: mixed prose/digit lines survive.
    text = "started worker 1410 ok"
    assert scrub_nondeterminism(text, command="worker & echo $!") == text


_CANONICAL_SCRUB_CASES = {
    # django-11490: `qs1.query id: 140737465471272`.
    "decimal-id-address": (
        "qs1.query id: 140737465471272",
        "qs1.query id: 140737465532368",
        "qs1.query id: <ADDR>",
    ),
    # extract-elf (exp-20260705-203854): /proc's synthetic link count varies
    # with kernel process state and is not source-pinnable inside Docker.
    "proc-root-ls-link-count": (
        "dr-xr-xr-x 362 root root 0 <MTIME> proc",
        "dr-xr-xr-x 359 root root 0 <MTIME> proc",
        "dr-xr-xr-x <NLINK> root root 0 <MTIME> proc",
    ),
    # custom-memory-heap-crash (exp-20260705-203854): Valgrind prefixes every
    # diagnostic line with its process id.
    "valgrind-pid-prefix": (
        "==1321== Memcheck, a memory error detector",
        "==1320== Memcheck, a memory error detector",
        "==<PID>== Memcheck, a memory error detector",
    ),
    # hf-model-inference: uppercase PID: is a launch idiom with a kernel-assigned value; lowercase pid: prose must survive.
    "background-launch-pid-label": (
        "PID: 203",
        "PID: 197",
        "PID: <PID>",
    ),
    "background-launch-pid-label-midline": (
        "the Postfix mail system is running: PID: 1052",
        "the Postfix mail system is running: PID: 99",
        "the Postfix mail system is running: PID: <PID>",
    ),
    # mailman: `GNU Mailman is running (master pid: 1075)`; the manager PID is
    # kernel state, and the phrase anchor keeps `generic pid:` prose untouched.
    "daemon-master-pid": (
        "GNU Mailman is running (master pid: 1075)",
        "GNU Mailman is running (master pid: 22)",
        "GNU Mailman is running (master pid: <PID>)",
    ),
    # install-windows-3.11: `ss -tlnp` prints the owning process id and fd for
    # each listening socket. Both are kernel-assigned per-process handles; the
    # process name is stable signal.
    "ss-socket-pid-and-fd": (
        'users:(("qemu-system-x86",pid=2504,fd=17))',
        'users:(("qemu-system-x86",pid=61,fd=9))',
        'users:(("qemu-system-x86",pid=<PID>,fd=<FD>))',
    ),
    # build-pov-ray: wget's log timestamp carries the run's wall-clock date.
    "log-line-date-before-time": (
        "2026-07-03 20:03:16 ERROR 404: NOT FOUND.",
        "2026-07-04 09:11:52 ERROR 404: NOT FOUND.",
        "<DATE> <TIME> ERROR 404: NOT FOUND.",
    ),
    # build-pov-ray: the DNS pick and the container's bridge IP + ephemeral
    # source port all vary per run.
    "curl-connect-endpoints": (
        "connect to 172.67.69.229 port 21 from 192.168.215.14 port 45932 failed",
        "connect to 104.18.11.207 port 21 from 192.168.209.3 port 51002 failed",
        "connect to <ENDPOINT> from <ENDPOINT> failed",
    ),
    # Some network tools print the resolved endpoint in bracketed diagnostics.
    # DNS/CDN choices are not stable enough to be task signal.
    "bracketed-network-endpoints": (
        "404 Not Found [IP: 151.101.130.132 443]",
        "404 Not Found [IP: 151.101.2.132 443]",
        "404 Not Found [IP: <ENDPOINT>]",
    ),
    # nginx-request-logging: the agent forced verbose apt (-o quiet=0), so the
    # env-side quiet config can't stop the run-varying rate.
    "parenthesized-download-rates": (
        "Fetched 9004 kB in <DUR> (3564 kB/s)",
        "Fetched 9004 kB in <DUR> (11.2 MB/s)",
        "Fetched 9004 kB in <DUR> (<RATE>)",
    ),
    # Structural header names normalize to lowercase without a per-name allowlist.
    "structural-header-names-lowercase": (
        "X-New-Cache-Status: HIT\nServer: nginx\n",
        "x-new-cache-status: HIT\nserver: nginx\n",
        "x-new-cache-status: HIT\nserver: nginx\n",
    ),
    "structural-header-names-lowercase-crlf": (
        "X-New-Cache-Status: HIT\r\n",
        "x-new-cache-status: HIT\r\n",
        "x-new-cache-status: HIT\n",
    ),
    # custom-memory-heap-crash (exp-20260703-231416): bash prints the child PID
    # in job-status lines; kernel PID allocation is not pinnable.
    "bash-job-status-pid": (
        "bash: line 1:  1195 Segmentation fault      /app/release 2>&1",
        "bash: line 1:  1199 Segmentation fault      /app/release 2>&1",
        "bash: line 1: <PID> Segmentation fault      /app/release 2>&1",
    ),
    # rstan-to-pystan: cmdstanpy prints compile wall time with no leading timing
    # word, so the generic duration rule misses it.
    "cmdstanpy-build-duration": (
        "Building: 58.7s, done.Running MCMC sampling...",
        "Building: 54.4s, done.Running MCMC sampling...",
        "Building: <DUR>, done.Running MCMC sampling...",
    ),
}


@pytest.mark.parametrize(
    "input_a, input_b, canonical",
    _CANONICAL_SCRUB_CASES.values(),
    ids=_CANONICAL_SCRUB_CASES,
)
def test_scrub_collapses_to_canonical(input_a, input_b, canonical):
    a = scrub_nondeterminism(input_a)
    b = scrub_nondeterminism(input_b)
    assert a == b == canonical


_PROC_NET_TCP_ROW = (
    "   0: 00000000000000000000000000000000:14D0 "
    "00000000000000000000000000000000:0000 0A 00000000:00000000 "
    "00:00000000 00000000     0        0 {} 1 {} 100 0 0 10 0"
)

_CANONICAL_SUBSTRING_SCRUB_CASES = {
    # pypi-server (exp-20260703-231416): wheel stages the .whl in a `.tmp-` +
    # 8-random-char dir inside dist/ and prints the path while building.
    "wheel-bdist-hyphenated-tempdir": (
        "creating '/app/dist/.tmp-lwb223gt/vectorops-0.1.0-py3-none-any.whl'",
        "creating '/app/dist/.tmp-ov2ycuvx/vectorops-0.1.0-py3-none-any.whl'",
        ".tmp<RAND>/",
    ),
    # build-cython-ext (exp-20260703-231416): pip's ephemeral wheel cache is a
    # tempfile-named dir printed on every source build.
    "pip-ephem-wheel-cache-dir": (
        "Stored in directory: /tmp/pip-ephem-wheel-cache-xg6l8n1u/wheels/5e/f2",
        "Stored in directory: /tmp/pip-ephem-wheel-cache-yvw8z7ak/wheels/5e/f2",
        "pip-ephem-wheel-cache-<RAND>",
    ),
    # pypi-server: pypa/build's isolated environment uses a tempfile suffix.
    "pypa-build-isolated-env-dir": (
        "/tmp/build-env-oku54bsr/lib/python3.13/site-packages/build/__init__.py",
        "/tmp/build-env-ckxoplo8/lib/python3.13/site-packages/build/__init__.py",
        "build-env-<RAND>",
    ),
    # custom-memory-heap-crash: gcc/ld's mkstemps object name varies per run.
    "gcc-ld-temp-object-file": (
        "/usr/bin/ld: /tmp/ccC61NfV.o: in function `main':",
        "/usr/bin/ld: /tmp/ccjglZzN.o: in function `main':",
        "/tmp/cc<RAND>.o",
    ),
    # build-cython-ext (exp-20260703-231416): the C-extension wheel embeds pip's
    # random build path, so size/sha vary despite SOURCE_DATE_EPOCH.
    "built-wheel-size-and-hash": (
        "Created wheel for planarity: filename=planarity-1.0.0-cp313-cp313-"
        "linux_x86_64.whl size=1856596 sha256=" + "a" * 64,
        "Created wheel for planarity: filename=planarity-1.0.0-cp313-cp313-"
        "linux_x86_64.whl size=1856569 sha256=" + "b" * 64,
        "size=<SIZE> sha256=<HASH>",
    ),
    # kv-store-grpc (exp-20260704-004139): the agent polled /proc/net/tcp6;
    # socket inode and kernel sk cookie vary per boot/socket.
    "proc-net-tcp-inode-and-socket-cookie": (
        _PROC_NET_TCP_ROW.format("1291505", "000000006b40b35b"),
        _PROC_NET_TCP_ROW.format("60683225", "00000000fc62b435"),
        "<INODE> 1 <SK>",
    ),
}


@pytest.mark.parametrize(
    "input_a, input_b, expected_substr",
    _CANONICAL_SUBSTRING_SCRUB_CASES.values(),
    ids=_CANONICAL_SUBSTRING_SCRUB_CASES,
)
def test_scrub_collapses_with_substring(input_a, input_b, expected_substr):
    a = scrub_nondeterminism(input_a)
    b = scrub_nondeterminism(input_b)
    assert a == b
    assert expected_substr in a


# Run-conditional output must be deleted to converge emitting and quiet runs.


_KEEPALIVE_LINE = "  Building wheel for planarity (pyproject.toml): still running..."
_CLOSING_SIGNAL_LINE = (
    "W0710 21:38:08.123456 4629 torch/distributed/elastic/multiprocessing"
    "/api.py:1028] Sending process 4631 closing signal SIGTERM"
)
_HF_REQUEST_ID = (
    " (Request ID: Root=1-6a51652a-4fcedbc27853d5e809c526b2;"
    "e81e5b3a-dc2d-4c72-ac3b-53df1dcb5e83)"
)

_DROP_LINE_SCRUB_CASES = {
    # build-cython-ext: pip emits keepalive chatter only when a build runs long.
    "pip-non-tty-keepalive-line": (
        "before\n",
        _KEEPALIVE_LINE + "\n",
        "after",
        _KEEPALIVE_LINE,
    ),
    # torch-tensor-parallelism: SIGTERM is sent only when a sibling child is
    # still alive when the failure is noticed, a child-exit timing race.
    "torchelastic-closing-signal-line": (
        "before\n",
        _CLOSING_SIGNAL_LINE + "\n",
        "after",
        "closing signal",
    ),
    # count-dataset-tokens: the proxy adds a request ID only on a cache miss.
    "hf-hub-request-id-parenthetical": (
        "huggingface_hub.errors.RemoteEntryNotFoundError: 404 Client Error.",
        _HF_REQUEST_ID,
        "",
        "Request ID:",
    ),
}


@pytest.mark.parametrize(
    "before, droppable, after, absent_marker",
    _DROP_LINE_SCRUB_CASES.values(),
    ids=_DROP_LINE_SCRUB_CASES,
)
def test_scrub_drops_run_conditional_line(before, droppable, after, absent_marker):
    with_line = scrub_nondeterminism(before + droppable + after)
    without_line = scrub_nondeterminism(before + after)
    assert with_line == without_line
    assert absent_marker not in with_line


_PRESERVED_SIGNAL_CASES = {
    "netstat-numeric-path-segments": (
        "GET http://host:8080/api",
        "released on 10/11/2026",
    ),
    "make-mips-interpreter-real-tmp-symbols": (
        "tmp_s3_floorheight = patch_map[tmp_s3_floorheight]",
    ),
    "structural-header-single-prose-label": ("Note: Preserve this label\n",),
    "pypa-nonrandom-build-env": (
        "/tmp/build-env-development/lib/python3.13/site-packages",
    ),
    "gcc-nearby-nonrandom-text": ("/tmp/ccache.o\ngcc file.c",),
    "pip-finished-status-line": (
        "  Building wheel for planarity (pyproject.toml): finished with status 'done'",
    ),
    "request-id-prose-without-payload": (
        "Retry only when (Request ID required) appears.",
    ),
    "generic-process-and-pid-prose": ("kill the process 1234 now\ngeneric pid: 999",),
    "proc-net-near-miss-number-sequence": ("commit 12 7 0123456789abcdef0 applied",),
    "agent-chosen-paths-and-small-numbers": (
        "wrote /tmp/test_bytes.py; 42 lines; record id: 42",
    ),
    "source-hex-literal": ("MASK = 0xdeadbeef",),
    "prose-date-without-ls-prefix": ("Released Jun 29 18:19 per the changelog",),
    "sparql-university-bare-iso-date": ("the university was founded on 1959-06-11",),
    "configured-server-port": ("server listening on port 8080",),
    "unparenthesized-prose-rate": ("the link sustains 10 MB/s under load",),
}


@pytest.mark.parametrize(
    "observations",
    _PRESERVED_SIGNAL_CASES.values(),
    ids=_PRESERVED_SIGNAL_CASES,
)
def test_preserves_task_signal(observations):
    assert tuple(map(scrub_nondeterminism, observations)) == observations


def test_solve_env_pins_git_date_filter_timezone(tmp_path: Path):
    # pytest-5840: this boundary commit appeared under UTC and vanished under
    # America/Los_Angeles for `git log --after=2019-08-26`.
    def git(*args: str, env: dict[str, str] | None = None) -> str:
        return subprocess.run(
            ("git", *args),
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    git("init", "-q")
    (tmp_path / "tracked").write_text("content\n")
    git("add", "tracked")
    commit_env = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2019-08-26T17:18:46+02:00",
        "GIT_COMMITTER_DATE": "2019-08-26T17:18:46+02:00",
    }
    git(
        "-c",
        "user.name=Harness",
        "-c",
        "user.email=harness@example.com",
        "commit",
        "-q",
        "-m",
        "boundary",
        env=commit_env,
    )

    def filtered_subjects(after: str, ambient_timezone: str) -> str:
        return git(
            "log",
            "--format=%s",
            f"--after={after}",
            env={**os.environ, "TZ": ambient_timezone, **SOLVE_EXEC_ENV},
        )

    # Git's date-only --after injects current time; assert TZ invariance plus a clock-independent one-day-earlier control.
    assert filtered_subjects("2019-08-26", "UTC") == filtered_subjects(
        "2019-08-26", "America/Los_Angeles"
    )
    assert filtered_subjects("2019-08-25", "UTC") == "boundary\n"
    assert filtered_subjects("2019-08-25", "America/Los_Angeles") == "boundary\n"


def test_container_env_pins_each_observed_source():
    env = SOLVE_EXEC_ENV
    assert env["PYTHONHASHSEED"] == "0"
    assert env["PERL_HASH_SEED"] == "0"
    assert env["PERL_PERTURB_KEYS"] == "0"
    # raw `@0` is rejected by git; the zone-qualified form is required.
    assert env["GIT_AUTHOR_DATE"] == "@0 +0000"
    assert env["GIT_COMMITTER_DATE"] == "@0 +0000"
    assert env["SOURCE_DATE_EPOCH"] == "0"
    # suppresses __pycache__ writes, whose dir-mtime bump leaked into `ls -l`
    # (django-11490, exp-20260627-025036).
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_mtime_reset_command_stamps_container_root_not_testbed():
    cmd = MTIME_RESET_COMMAND
    assert cmd.startswith("touch -c -d @0 ")
    assert " / " in f" {cmd} "
    assert "/testbed" not in cmd


def test_git_template_hooks_pin_future_checkout_mtimes():
    cmd = GIT_HOOKS_INIT_COMMAND
    assert "init.templateDir" in cmd
    assert "core.hooksPath" not in cmd
    assert ".cache/framework-git-template" in cmd
    assert "post-checkout" in cmd
    assert "post-merge" in cmd
    assert "198001010000.00" in cmd


def test_gdb_init_command_suppresses_inferior_pid_events():
    cmd = GDB_INIT_COMMAND
    assert "/root/.gdbinit" in cmd
    assert "set print inferior-events off" in cmd


def test_container_hostname_is_fixed():
    assert CONTAINER_HOSTNAME and " " not in CONTAINER_HOSTNAME
