from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
_EXPECTED_IMPORTERS = {
    "src/llm/backend.py",
    "src/measurement.py",
    "src/rollout/execution.py",
    "src/worker.py",
}


def _imports_plugins(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == "src.plugins" or alias.name.startswith("src.plugins.")
            for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "src.plugins" or module.startswith("src.plugins."):
                return True
            if module == "src" and any(alias.name == "plugins" for alias in node.names):
                return True
    return False


def test_plugin_import_boundary() -> None:
    importers = {
        path.relative_to(_REPO_ROOT).as_posix()
        for path in _SRC_ROOT.rglob("*.py")
        if not path.is_relative_to(_SRC_ROOT / "plugins")
        and _imports_plugins(ast.parse(path.read_text(), filename=str(path)))
    }

    assert importers == _EXPECTED_IMPORTERS
    # Control flow depends only on the core StepExecutor contract.
    assert "src/rollout/episode.py" not in importers
    assert "src/rollout/sampler.py" not in importers
    # Env grades purely; the verify cache is an injected plugin wrapper, so env
    # imports no plugin.
    assert {path for path in importers if path.startswith("src/env/")} == set()


def test_core_is_importable_and_config_validates_without_plugins() -> None:
    code = r"""
import importlib
import importlib.abc
from pathlib import Path
import sys

class _ForbidPlugins(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "src.plugins" or fullname.startswith("src.plugins."):
            raise ImportError(f"forbidden plugin import: {fullname}")
        return None

sys.meta_path.insert(0, _ForbidPlugins())
root = Path.cwd() / "src"
for path in sorted(root.rglob("*.py")):
    if path.is_relative_to(root / "plugins"):
        continue
    relative = path.relative_to(Path.cwd()).with_suffix("")
    parts = relative.parts
    module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
    importlib.import_module(module)

from src.config import RunConfig
RunConfig.model_validate({
    "schema_version": 13,
    "training_target": {"module": "src.policy.core"},
    "environment": {"kind": "swe", "task_names": ["task-a"]},
    "plugins": {
        "llm_cache": False,
        "execution": "eager",
    },
    "llm_provider_config": {
        "model_name": "model",
        "base_url": "http://127.0.0.1:18000/v1",
        "api_key_env": "OPENAI_API_KEY",
        "max_context_length": 4096,
        "max_tokens": 1024,
    },
})
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
