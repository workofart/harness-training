"""Pins the tunable/frozen dependency rule: frozen code and tests bind
src.policy.core only through its contract (build_policy / build_env_action,
see src/policy/base.py); pins beyond that belong in tests/policy/."""

from __future__ import annotations

import ast
from pathlib import Path

CONTRACT_EXPORTS = {"build_policy", "build_env_action"}
CORE_MODULE = "src.policy.core"
# Zones that may bind core freely: the module itself + the proposer-visible tests.
FREE_ZONES = ("src/policy/", "tests/policy/")
# Frozen files allowed a static core binding, contract exports only.
CONTRACT_ONLY = {"tests/rollout/test_episode.py"}


def _core_violations(source: str) -> list[str]:
    """Return the non-contract names a module binds from src.policy.core."""
    tree = ast.parse(source)
    violations: list[str] = []
    aliases: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == CORE_MODULE:
                violations.extend(
                    alias.name
                    for alias in node.names
                    if alias.name not in CONTRACT_EXPORTS
                )
            elif node.module == "src.policy":
                aliases.extend(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "core"
                )
        elif isinstance(node, ast.Import):
            aliases.extend(
                alias.asname or alias.name.split(".")[0]
                for alias in node.names
                if alias.name == CORE_MODULE
            )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
            and node.attr not in CONTRACT_EXPORTS
        ):
            violations.append(node.attr)
    return violations


def _binds_core(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == CORE_MODULE or (
                node.module == "src.policy"
                and any(alias.name == "core" for alias in node.names)
            ):
                return True
        elif isinstance(node, ast.Import):
            if any(alias.name == CORE_MODULE for alias in node.names):
                return True
    return False


def test_frozen_zone_binds_core_only_through_the_contract() -> None:
    repo = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    for top in ("src", "tests"):
        for path in sorted((repo / top).rglob("*.py")):
            rel = path.relative_to(repo).as_posix()
            if rel.startswith(FREE_ZONES):
                continue
            source = path.read_text()
            if rel in CONTRACT_ONLY:
                offenders.extend(f"{rel}: {name}" for name in _core_violations(source))
            elif _binds_core(source):
                offenders.append(f"{rel}: static import of {CORE_MODULE}")
    assert not offenders, (
        "frozen files bind the editable policy beyond its contract:\n"
        + "\n".join(offenders)
    )


def test_tripwire_detects_internal_binding() -> None:
    # Reachability check: the sweep must actually fire on an offending file.
    assert _binds_core("from src.policy.core import LlmAgent\n")
    assert _core_violations(
        "import src.policy.core as core_module\ncore_module.LlmAgent\n"
    ) == ["LlmAgent"]
    assert not _core_violations(
        "import src.policy.core as core_module\ncore_module.build_policy\n"
    )
