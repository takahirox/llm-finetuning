"""Check the bootstrap environment without downloading models or evaluating tensors."""

import importlib
import importlib.metadata
import platform
from pathlib import Path
import sys


def main() -> int:
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.system()} {platform.release()} / {platform.machine()}")
    print(f"macOS: {platform.mac_ver()[0] or 'not detected'}")
    print(f"Virtual environment: {'yes' if sys.prefix != sys.base_prefix else 'no'}")

    errors = []
    if sys.version_info[:2] != (3, 12):
        errors.append("Use Python 3.12: run `uv python install` and `uv sync --locked`.")
    if platform.system() != "Darwin":
        errors.append("Use an Apple Silicon Mac running macOS 14 or newer.")
    else:
        if platform.machine() != "arm64":
            errors.append(
                "Use native arm64 Python on Apple Silicon. Disable 'Open using Rosetta' "
                "for your terminal, then recreate .venv with `uv sync --locked`."
            )
        mac_version = platform.mac_ver()[0]
        if not mac_version or int(mac_version.split('.')[0]) < 14:
            errors.append("Upgrade to macOS 14 or newer before installing MLX.")
    if sys.prefix == sys.base_prefix:
        errors.append(
            "Use the isolated environment: run `uv sync --locked`, then "
            "`.venv/bin/python scripts/check_environment.py`."
        )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("Package imports skipped until the environment requirements are met.")
        return 1

    import tomllib

    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with project_file.open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["dependencies"]

    modules = {}
    for dependency in dependencies:
        name, expected_version = dependency.split("==")
        module_name = "mlx.core" if name == "mlx" else name.replace("-", "_")
        try:
            installed_version = importlib.metadata.version(name)
            print(f"{name}: {installed_version} (expected {expected_version})")
            if installed_version != expected_version:
                errors.append(f"{name} version differs from the project pin; run `uv sync --locked`.")
            modules[name] = importlib.import_module(module_name)
            print(f"Import {module_name}: OK")
        except Exception as exc:
            errors.append(
                f"Cannot import {module_name}: {type(exc).__name__}: {exc}. "
                "Run `uv sync --locked`; if it still fails, check native arm64 Python "
                "and macOS compatibility."
            )

    if "mlx" in modules:
        try:
            available = modules["mlx"].metal.is_available()
            print(f"Metal available: {available}")
            if not available:
                errors.append(
                    "MLX cannot access Metal. Use a native Apple Silicon macOS session "
                    "and reinstall the locked environment with `uv sync --locked`."
                )
        except Exception as exc:
            errors.append(f"Cannot query Metal: {exc}. Check your macOS/MLX installation.")

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print("Environment check passed. No models downloaded or training performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
