"""Hardware-independent checks for setup diagnostics; no MLX imports required."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import importlib.util
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_environment.py"
SPEC = importlib.util.spec_from_file_location("check_environment", SCRIPT)
check_environment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_environment)


class EnvironmentCheckTests(unittest.TestCase):
    def run_check(self, overrides=None, versions=None, import_error=None, metal=True):
        core = SimpleNamespace(metal=SimpleNamespace(is_available=lambda: metal))
        values = {
            "platform.system": "Darwin",
            "platform.machine": "arm64",
            "platform.mac_ver": ("14.0", ("", "", ""), ""),
            "sys.version_info": (3, 12, 13),
            "sys.prefix": "/project/.venv",
            "sys.base_prefix": "/python",
        }
        values.update(overrides or {})
        versions = versions or {"mlx": "0.32.3", "mlx-lm": "0.32.0"}
        stdout, stderr = StringIO(), StringIO()
        with ExitStack() as stack:
            for target, value in values.items():
                module, attribute = target.split(".")
                owner = getattr(check_environment, module)
                replacement = (
                    patch.object(owner, attribute, return_value=value)
                    if module == "platform" else patch.object(owner, attribute, value)
                )
                stack.enter_context(replacement)
            stack.enter_context(patch.object(
                check_environment.importlib.metadata, "version", side_effect=versions.__getitem__
            ))
            importer = stack.enter_context(patch.object(
                check_environment.importlib, "import_module",
                side_effect=import_error,
                return_value=core,
            ))
            stack.enter_context(redirect_stdout(stdout))
            stack.enter_context(redirect_stderr(stderr))
            status = check_environment.main()
        return status, stdout.getvalue(), stderr.getvalue(), importer.call_count

    def test_unsupported_setups_fail_before_native_imports(self):
        scenarios = [
            ({"platform.system": "Linux"}, "Apple Silicon Mac"),
            ({"platform.machine": "x86_64"}, "Rosetta"),
            ({"platform.mac_ver": ("13.6", (), "")}, "macOS 14"),
            ({"platform.mac_ver": ("", (), "")}, "macOS 14"),
            ({"sys.version_info": (3, 11, 0)}, "Python 3.12"),
            ({"sys.version_info": (3, 9, 0)}, "Python 3.12"),
            ({"sys.version_info": (3, 13, 0)}, "Python 3.12"),
            ({"sys.prefix": "/python"}, "isolated environment"),
        ]
        for overrides, diagnostic in scenarios:
            with self.subTest(overrides=overrides):
                status, output, errors, imports = self.run_check(overrides)
                self.assertEqual(status, 1)
                self.assertIn(diagnostic, errors)
                self.assertIn("imports skipped", output)
                self.assertEqual(imports, 0)

    def test_supported_setup_imports_both_packages(self):
        status, output, errors, imports = self.run_check()
        self.assertEqual(status, 0)
        self.assertIn("Metal available: True", output)
        self.assertIn("Environment check passed", output)
        self.assertEqual(errors, "")
        self.assertEqual(imports, 2)

    def test_missing_distribution_has_install_instructions(self):
        class MissingPackages(dict):
            def __getitem__(self, key):
                raise check_environment.importlib.metadata.PackageNotFoundError(key)

        status, _, errors, imports = self.run_check(versions=MissingPackages(missing=True))
        self.assertEqual(status, 1)
        self.assertIn("PackageNotFoundError", errors)
        self.assertIn("uv sync --locked", errors)
        self.assertEqual(imports, 0)

    def test_transitive_and_native_import_failures_are_actionable(self):
        for error in (ModuleNotFoundError("tokenizers"), OSError("native library unavailable")):
            with self.subTest(error=error):
                status, _, errors, imports = self.run_check(import_error=error)
                self.assertEqual(status, 1)
                self.assertIn(str(error), errors)
                self.assertIn("uv sync --locked", errors)
                self.assertEqual(imports, 2)

    def test_wrong_package_version_fails(self):
        status, _, errors, _ = self.run_check(versions={"mlx": "0.1.0", "mlx-lm": "0.32.0"})
        self.assertEqual(status, 1)
        self.assertIn("version differs", errors)

    def test_unavailable_metal_fails(self):
        status, _, errors, _ = self.run_check(metal=False)
        self.assertEqual(status, 1)
        self.assertIn("cannot access Metal", errors)


if __name__ == "__main__":
    unittest.main()
