# Fakerandom shim (fetched, not vendored)

`libfaketimeMT.so.1` is the amd64 shared object from Debian's
`libfaketime` package, version **0.9.10-2.1**. This build interposes
`getrandom(2)`/`getentropy(3)` and, when `FAKERANDOM_SEED` is set, returns
deterministic entropy — which makes `os.urandom`, `secrets`, `random`, and
numpy-default seeding reproducible across runs. (It does *not* fake time
unless `FAKETIME` is set, and it does not intercept direct `/dev/urandom`
`read()`s.)

libfaketime is GPL-2.0, so the binary is **not** checked into this MIT
repo. `_fakerandom_lib_path()` in `src/env/terminal_bench.py` downloads
the pinned `.deb` from snapshot.debian.org's immutable by-hash endpoint at
first use, extracts the shared object into this directory (gitignored),
and asserts sha256 of both the `.deb` and the extracted `.so` — so the
bytes, and the env-step-cache fingerprint they feed, can never drift.
Corresponding source for the exact binary:
<https://snapshot.debian.org/package/faketime/0.9.10-2.1/>

The framework bind-mounts it read-only at `/opt/framework/libfaketimeMT.so.1`
for `pin_urandom` tasks (see `src/env/terminal_bench.py`) instead of
`apt-get install`-ing it per trial, which used to overrun the 30s setup
budget on a cold apt index.

Regenerate manually (pinned version, amd64; the rolling `apt-get download`
route 404s once Debian supersedes the version, so fetch from snapshot):

    curl -sfL https://snapshot.debian.org/file/781fba01c508a6a61e6f22a44ef43a9bd4f3d419 \
      -o libfaketime_0.9.10-2.1_amd64.deb
    docker run --rm --platform linux/amd64 -v "$PWD:/out" debian:stable-slim bash -c '
      dpkg-deb -x /out/libfaketime_0.9.10-2.1_amd64.deb ex &&
      cp ex/usr/lib/x86_64-linux-gnu/faketime/libfaketimeMT.so.1 /out/'
