"""Determinism controls for solve-container observations.

Prefer source pins when the environment can be made stable. Scrub only residual
tokens that Docker/tools still vary; ``base.scrub_raw_env_output`` applies this once
before the agent and trace see the bytes.
"""

from __future__ import annotations

import hashlib
import json
import re

# Source-level pins

# Docker's default hostname is the per-run container id.
CONTAINER_HOSTNAME = "sandbox"

_FIXED_EPOCH = "@0"

# These pins trade Harbor verifier parity for deterministic Terminal-Bench grading.
SOLVE_EXEC_ENV = {
    "PYTHONHASHSEED": "0",
    "PERL_HASH_SEED": "0",
    "PERL_PERTURB_KEYS": "0",
    "GIT_AUTHOR_DATE": f"{_FIXED_EPOCH} +0000",
    "GIT_COMMITTER_DATE": f"{_FIXED_EPOCH} +0000",
    "SOURCE_DATE_EPOCH": "0",
    "TZ": "UTC",
    "PYTHONDONTWRITEBYTECODE": "1",
}

# Reset only container-created dirs whose mtimes leak through ls; preserve /testbed mtimes and git's stat cache.
MTIME_RESET_COMMAND = f"touch -c -d {_FIXED_EPOCH} / /tmp /root /run"

# Checkout mtimes leak into Python source archives; hooks pin metadata for future clones.
GIT_HOOKS_INIT_COMMAND = r"""if command -v git >/dev/null 2>&1; then
  home_dir=${HOME:-/tmp}
  template_dir="$home_dir/.cache/framework-git-template"
  hooks_dir="$template_dir/hooks"
  mkdir -p "$hooks_dir"
  cat > "$hooks_dir/post-checkout" <<'EOF'
#!/usr/bin/env sh
worktree=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
find "$worktree" -exec touch -h -t 198001010000.00 {} + 2>/dev/null || \
  find "$worktree" -exec touch -t 198001010000.00 {} + 2>/dev/null || true
EOF
  cp "$hooks_dir/post-checkout" "$hooks_dir/post-merge"
  chmod +x "$hooks_dir/post-checkout" "$hooks_dir/post-merge"
  HOME="$home_dir" git config --global init.templateDir "$template_dir"
fi"""

# GDB otherwise prints run-varying inferior PIDs on fork/exec events.
GDB_INIT_COMMAND = "printf '%s\\n' 'set print inferior-events off' >> /root/.gdbinit"

# Any pin change shifts every env content fingerprint; envs never hand-enumerate pins.
PINS_FINGERPRINT: str = hashlib.sha256(
    json.dumps(
        {
            "container_hostname": CONTAINER_HOSTNAME,
            "solve_exec_env": SOLVE_EXEC_ENV,
            "mtime_reset_command": MTIME_RESET_COMMAND,
            "git_hooks_init_command": GIT_HOOKS_INIT_COMMAND,
            "gdb_init_command": GDB_INIT_COMMAND,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()[:12]


# Residual observation scrubbing

_RegexSub = tuple[re.Pattern[str], str]

_DRIFT_AUDIT_HEADER_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9-]*):[ \t]*(.*)$")
_BASH_JOB_STATUS_LINE = re.compile(r"(bash: line \d+:)\s+\d+\s+")
_BASH_PIPELINE_JOB_STATUS_LINE = re.compile(
    r"^(\s+)\d+ (Done(?:\(\d+\))?|Exit \d+|Killed|Terminated|Aborted"
    r"|Segmentation fault|Broken pipe|Bus error|Hangup|Interrupt"
    r"|Floating point exception|Illegal instruction|Stopped|Running)\b"
)
_GLOG_PREFIX = re.compile(r"\b[IWEF]\d{4}\s+<TIME>\.\d+\s+\d+(\s+\S+:\d+\])")
_TERMINATING_PROCESS = re.compile(r"\b(Terminating process) \d+ (?=via signal\b)")
_DROPPED_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # pip emits elapsed-time keepalives only for slow subprocesses (build-cython-ext).
    re.compile(
        r"\s*(?:Building (?:wheel|editable) for \S+|Installing build dependencies"
        r"|Getting requirements to build wheel|Preparing (?:editable )?metadata)"
        r" \((?:pyproject\.toml|setup\.py)\): still running\.\.\.\s*"
    ),
    # Drop torchelastic survivor-signal logs caused by child-exit races; match after glog scrubbing.
    re.compile(
        r"<LOG> <TIME>\.<SUBSEC> <PID> torch/distributed/elastic/multiprocessing"
        r"/api\.py:\d+\] Sending process <PID> closing signal \w+\s*"
    ),
)

# Rules stay narrow to protect task signal; substitution order is significant.
_ADDRESS_IDENTITY_SUBS: tuple[_RegexSub, ...] = (
    # CPython repr/id addresses remain a portability and post-divergence risk.
    (re.compile(r"at 0x[0-9a-fA-F]+"), "at <ADDR>"),
    (re.compile(r"\b(id[ =:]+)\d{12,}\b"), r"\1<ADDR>"),
    (re.compile(r"\b(?:[0-9A-Fa-f]{2}:){15,}[0-9A-Fa-f]{2}\b"), "<FINGERPRINT>"),
)

_WALL_CLOCK_SUBS: tuple[_RegexSub, ...] = (
    # Timing words plus durations, including pytest's optional H:MM:SS parenthetical.
    (
        re.compile(
            r"""
            \b
            (in|took|after|elapsed)
            \s+
            (?: \d+ \s* m \s* )?
            \d+ (?: \.\d+ )?
            \s*
            (?: ns | µs | us | ms | seconds? | secs? | s
              | minutes? | mins? | m | hours? | hrs? | h )
            \b
            (?: \s* \( \d{1,2} : \d{2} : \d{2} \) )?
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
        r"\1 <DUR>",
    ),
    # cmdstanpy prints the model compile wall time with no leading timing word
    # (rstan-to-pystan `Building: 54.4s, done.`).
    (re.compile(r"\b(Building:) \d+(?:\.\d+)?s\b"), r"\1 <DUR>"),
    # hex (MAC addresses, short digests)
    (
        re.compile(r"(?<![0-9A-Fa-f]{2}:)\b\d{1,2}:\d{2}:\d{2}\b(?!:[0-9A-Fa-f]{2})"),
        "<TIME>",
    ),
    # Must run after the clock-stamp rule: adjacency to `<TIME>` is what proves
    # a date is a log stamp; bare prose/data dates carry task signal.
    (re.compile(r"\b\d{4}-\d{2}-\d{2}(?= <TIME>)"), "<DATE>"),
)

_NETWORK_DRIFT_SUBS: tuple[_RegexSub, ...] = (
    # Delete cache-miss-only HF/AWS request IDs rather than tokenizing them.
    (re.compile(r" \(Request ID: [^)]{1,200}\)"), ""),
    # DNS pick and container bridge IP + ephemeral source port vary per run.
    (
        re.compile(r"\b(connect to|from) \d{1,3}(?:\.\d{1,3}){3} port \d{1,5}\b"),
        r"\1 <ENDPOINT>",
    ),
    # Network tools also report resolved endpoints in bracketed diagnostics
    # (`[IP: 151.101.2.132 443]`); DNS/CDN choices are not task signal.
    (
        re.compile(r"\[IP: \d{1,3}(?:\.\d{1,3}){3} \d{1,5}\]"),
        "[IP: <ENDPOINT>]",
    ),
    (
        re.compile(
            r"(?m)^(Backend:\s*)[A-Za-z0-9]+--F_[A-Za-z0-9-]+"
            r"_debian_backend_mirrors_debian_org(\r?)$"
        ),
        r"\1<DEBIAN_BACKEND>\2",
    ),
    # Parentheses anchor volatile apt/wget rates without masking bare prose rates such as "10 MB/s link".
    (
        re.compile(r"\(\d[\d.]* ?[kKMG]?B/s\)"),
        "(<RATE>)",
    ),
)

_FILESYSTEM_METADATA_SUBS: tuple[_RegexSub, ...] = (
    # `ls -l` dates after mutable dirs/files are touched; prose dates must survive.
    (
        re.compile(
            r"""
            ^
            (
                [-dlbcps] [rwxstST-]{9} [.+@]?
                \s+ \d+
                \s+ \S+
                \s+ \S+
                \s+ \S+
                \s+
            )
            (?: Jan | Feb | Mar | Apr | May | Jun
              | Jul | Aug | Sep | Oct | Nov | Dec )
            \s+ \d{1,2}
            \s+
            (?: \d{1,2} : \d{2} | \d{4} )
            """,
            re.MULTILINE | re.VERBOSE,
        ),
        r"\1<MTIME>",
    ),
    # /proc is synthetic; its root link count varies with kernel process state.
    (
        re.compile(
            r"^(dr-xr-xr-x\s+)\d+(\s+root\s+root\s+\S+\s+<MTIME>\s+proc)$", re.MULTILINE
        ),
        r"\1<NLINK>\2",
    ),
)

_TOOL_RANDOMNESS_SUBS: tuple[_RegexSub, ...] = (
    # pypa/build's isolated-env tempfile suffix varies (pypi-server).
    (re.compile(r"\bbuild-env-[a-z0-9_]{8}\b"), "build-env-<RAND>"),
    # gcc/ld's mkstemps object name varies (custom-memory-heap-crash).
    (
        re.compile(r"(?<=/tmp/)cc[A-Za-z0-9]{6}(?=\.[a-z]{1,4}\b)"),
        "cc<RAND>",
    ),
    # Sphinx reports its unpinnable tempfile error-log path.
    (
        re.compile(r"(?<=/tmp/)sphinx-err-[a-z0-9_]{8}(?=\.log\b)"),
        "sphinx-err-<RAND>",
    ),
    # tempfile uses an unseedable private RNG; optional hyphens cover wheel staging dirs.
    (
        re.compile(
            r"(?:(?:(?<=/)|(?<=\.))tmp-?[a-z0-9_]{8}(?=[/._-])"
            r"|\btmp-?[a-z0-9_]{8}(?![a-z0-9_]))"
        ),
        "tmp<RAND>",
    ),
    # pip working-directory suffixes use the same unpinnable RNG.
    (
        re.compile(
            r"\bpip-(ephem-wheel-cache|install|req-build|build-env|unpack|target)"
            r"-[a-z0-9_]{8}\b"
        ),
        r"pip-\1-<RAND>",
    ),
    # C-extension wheels embed pip's random build path, varying reported size/hash despite SOURCE_DATE_EPOCH.
    (re.compile(r"\bsize=\d+ sha256=[0-9a-f]{64}\b"), "size=<SIZE> sha256=<HASH>"),
)

_PROCESS_IDENTITY_SUBS: tuple[_RegexSub, ...] = (
    # Valgrind prefixes diagnostics with its process id (`==1321==`); PID
    # allocation is kernel state, not task signal.
    (re.compile(r"==\d+=="), "==<PID>=="),
    # /proc/net/tcp socket inode/cookie vary; zero retransmit/UID/timeout fields anchor rows without matching prose.
    (
        re.compile(r"(00000000\s+0\s+0 )\d+ (\d+) [0-9a-f]{16}\b"),
        r"\1<INODE> \2 <SK>",
    ),
    # Match uppercase PID: only; lowercase prose and the uncolonized ps header must survive.
    (re.compile(r"\b(PID: )\d+\b"), r"\1<PID>"),
    # GNU Mailman/systemd report the manager PID (mailman `master pid: N`); the
    # phrase anchor keeps unrelated lowercase `pid:` prose.
    (re.compile(r"\b(master pid: )\d+"), r"\1<PID>"),
    # ss -p exposes variable pid=/fd= handles; those labels are sufficient anchors.
    (re.compile(r"\bpid=\d+"), "pid=<PID>"),
    (re.compile(r"\bfd=\d+"), "fd=<FD>"),
    # netstat -p ends rows in PID/Program; require a whitespace-led terminal field to avoid URLs, /proc paths, and dates.
    (re.compile(r"(?m)(?<=\s)\d+/([A-Za-z][\w.+-]*)\s*$"), r"<PID>/\1"),
)

_PRESENTATION_SUBS: tuple[_RegexSub, ...] = (
    # Pytest banner padding changes with pre-scrub duration width.
    (re.compile(r"={8,}"), "========"),
)

_RUNNER_RANDOMNESS_SUBS: tuple[_RegexSub, ...] = (
    # sympy's test runner prints a per-process random seed.
    (re.compile(r"(random seed:\s+)\d+"), r"\1<SEED>"),
)

_TORCHELASTIC_REPORT_SUBS: tuple[_RegexSub, ...] = (
    # Torchelastic repeats volatile child timestamps and PIDs across its failure report.
    (
        re.compile(r"(?m)^(\s*time\s+: )\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}\b"),
        r"\1<DATE>_<TIME>",
    ),
    (
        re.compile(r"(?m)^(\s*exitcode\s+: \d+ \(pid: )\d+(\))"),
        r"\1<PID>\2",
    ),
    (
        re.compile(r"(\(exitcode: \d+\) local_rank: \d+ \(pid: )\d+(\) of binary:)"),
        r"\1<PID>\2",
    ),
    (
        re.compile(r"\b(Sending process )\d+( closing signal )"),
        r"\1<PID>\2",
    ),
)

_RESIDUAL_OBSERVATION_SUBS: tuple[_RegexSub, ...] = (
    *_ADDRESS_IDENTITY_SUBS,
    *_TORCHELASTIC_REPORT_SUBS,
    *_WALL_CLOCK_SUBS,
    *_NETWORK_DRIFT_SUBS,
    *_FILESYSTEM_METADATA_SUBS,
    *_TOOL_RANDOMNESS_SUBS,
    *_PROCESS_IDENTITY_SUBS,
    *_PRESENTATION_SUBS,
    *_RUNNER_RANDOMNESS_SUBS,
)


# Bare PIDs lack an output anchor; scrub digit-only lines only for $! or pidof commands to preserve numeric task output.
_PID_PRODUCING_COMMAND = re.compile(r"\$!|\bpidof\b")
_BARE_PID_LINE = re.compile(r"(?m)^\d+(?: \d+)*$")


def scrub_nondeterminism(text: str, *, command: str | None = None) -> str:
    """Normalize residual volatile tokens in one observation text stream.

    ``command`` is the shell command that produced the text, when known; it
    gates scrubs that are only sound for output the command shaped by
    construction (bare-PID lines).
    """
    if command is not None and _PID_PRODUCING_COMMAND.search(command):
        text = _BARE_PID_LINE.sub(
            lambda match: re.sub(r"\d+", "<PID>", match.group()), text
        )
    for pattern, repl in _RESIDUAL_OBSERVATION_SUBS:
        text = pattern.sub(repl, text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = _canonicalize_structural_headers(text.split("\n"))
    normalized = []
    for line in lines:
        line = _BASH_JOB_STATUS_LINE.sub(r"\1 <PID> ", line)
        line = _BASH_PIPELINE_JOB_STATUS_LINE.sub(r"\1<PID> \2", line)
        line = _GLOG_PREFIX.sub(r"<LOG> <TIME>.<SUBSEC> <PID>\1", line)
        line = _TERMINATING_PROCESS.sub(r"\1 <PID> ", line)
        # Dropped-line patterns expect the line-local normalized form.
        if any(pattern.fullmatch(line) for pattern in _DROPPED_LINE_PATTERNS):
            continue
        normalized.append(line)
    return "\n".join(normalized)


def drift_audit_canonical(text: str, *, command: str | None = None) -> str:
    """Line-order-insensitive replay identity after model-visible scrubbing.

    ``scrub_nondeterminism`` is the model-visible canonical form; this is the
    slightly more lenient replay drift-audit identity. Concurrent writers, such
    as parallel workers' warning streams, can interleave lines nondeterministically;
    reordering those lines is not world contamination. Step cache stores raw bytes
    and scrubs at serve/audit time, so scrub-rule changes self-heal via cache misses
    and are deliberately not part of any env fingerprint.
    """
    return "\n".join(
        sorted(
            line.rstrip()
            for line in scrub_nondeterminism(text, command=command).split("\n")
        )
    )


def _canonicalize_structural_headers(lines: list[str]) -> list[str]:
    # This is mixed stdout, not a header document; preserve lone prose labels.
    header_matches = [
        _DRIFT_AUDIT_HEADER_LINE.fullmatch(line.rstrip()) for line in lines
    ]
    canonicalized = []
    for index, line in enumerate(lines):
        match = header_matches[index]
        if match is not None:
            name = match.group(1)
            adjacent_header = (index > 0 and header_matches[index - 1] is not None) or (
                index + 1 < len(header_matches)
                and header_matches[index + 1] is not None
            )
            if "-" in name or adjacent_header:
                line = f"{name.lower()}: {match.group(2)}"
        canonicalized.append(line)
    return canonicalized
