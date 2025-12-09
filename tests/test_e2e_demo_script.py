"""
Tests for the e2e_demo.sh and e2e_demo.py scripts.

Tests run the demo scripts and verify their output contains
expected artifacts and response data.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


class TestE2EDemoShellScript:
    """Tests for examples/e2e_demo.sh."""

    @pytest.fixture
    def script_path(self) -> Path:
        """Path to the e2e demo shell script."""
        return PROJECT_ROOT / "examples" / "e2e_demo.sh"

    @pytest.fixture
    def has_bash(self) -> bool:
        """Check if bash is available."""
        # On Windows, check if bash exists (Git Bash, WSL, etc.)
        if sys.platform == "win32":
            # Try to find bash
            bash_paths = [
                shutil.which("bash"),
                r"C:\Program Files\Git\bin\bash.exe",
                r"C:\Program Files (x86)\Git\bin\bash.exe",
            ]
            for path in bash_paths:
                if path and Path(path).exists():
                    return True
            return False
        else:
            # On Unix-like systems, bash should be available
            return shutil.which("bash") is not None

    def test_script_exists(self, script_path: Path):
        """Test that the demo script exists."""
        assert script_path.exists(), f"Script not found: {script_path}"

    def test_script_is_executable_content(self, script_path: Path):
        """Test that script has proper shebang and content."""
        content = script_path.read_text()
        assert content.startswith("#!/bin/bash"), "Script should start with #!/bin/bash"
        assert "set -euo pipefail" in content, "Script should use strict mode"
        assert "mktemp" in content, "Script should create temp directory"
        assert "train.py" in content, "Script should run trainer"
        assert "builder.py" in content, "Script should run builder"
        assert "/predict" in content, "Script should test predict endpoint"

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Bash script test skipped on Windows due to Git Bash path/environment complexity. "
               "Use test_python_demo_runs_successfully instead."
    )
    def test_script_runs_successfully(self, script_path: Path, has_bash: bool):
        """Test that the demo script runs and produces expected output."""
        if not has_bash:
            pytest.skip("Bash not available")

        # Set environment variables for mock mode
        env = os.environ.copy()
        env["CONNECTOR_MOCK"] = "true"
        env["GOOGLE_OAUTH_DISABLED"] = "true"

        # On Windows with Git Bash, we need to set PYTHONPATH in Windows format
        # because Python interprets paths, not bash
        if sys.platform == "win32":
            env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

        # Find bash executable
        bash_path = shutil.which("bash")
        if sys.platform == "win32" and not bash_path:
            for path in [
                r"C:\Program Files\Git\bin\bash.exe",
                r"C:\Program Files (x86)\Git\bin\bash.exe",
            ]:
                if Path(path).exists():
                    bash_path = path
                    break

        if not bash_path:
            pytest.skip("Could not find bash executable")

        # Run with --cleanup to avoid leaving temp files
        result = subprocess.run(
            [bash_path, str(script_path), "--cleanup"],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )

        # Check script succeeded
        assert result.returncode == 0, (
            f"Script failed with code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Verify expected output content
        stdout = result.stdout

        # Check for model artifact mention
        assert "model_stub.json" in stdout, "Output should mention model_stub.json"

        # Check for service directory mention
        assert "service" in stdout.lower(), "Output should mention service directory"

        # Check for JSON response with expected keys
        assert '"output"' in stdout, "Output should contain prediction response with 'output'"
        assert '"confidence"' in stdout, "Output should contain prediction response with 'confidence'"

        # Check for completion message
        assert "Demo Complete" in stdout, "Output should show completion message"


class TestE2EDemoPythonScript:
    """Tests for examples/e2e_demo.py."""

    @pytest.fixture
    def script_path(self) -> Path:
        """Path to the e2e demo Python script."""
        return PROJECT_ROOT / "examples" / "e2e_demo.py"

    def test_script_exists(self, script_path: Path):
        """Test that the Python demo script exists."""
        assert script_path.exists(), f"Script not found: {script_path}"

    def test_script_is_valid_python(self, script_path: Path):
        """Test that the script is valid Python syntax."""
        import ast
        content = script_path.read_text()
        # This will raise SyntaxError if invalid
        ast.parse(content)

    def test_script_has_required_functions(self, script_path: Path):
        """Test that script has required functions."""
        content = script_path.read_text()
        assert "def run_demo" in content, "Script should have run_demo function"
        assert "def main" in content, "Script should have main function"
        assert "--cleanup" in content, "Script should support --cleanup flag"

    def test_python_demo_runs_successfully(self, script_path: Path):
        """Test that the Python demo script runs successfully."""
        # Set environment variables for mock mode
        env = os.environ.copy()
        env["CONNECTOR_MOCK"] = "true"
        env["GOOGLE_OAUTH_DISABLED"] = "true"

        # Run with --cleanup to avoid leaving temp files
        result = subprocess.run(
            [sys.executable, str(script_path), "--cleanup"],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )

        # Check script succeeded
        assert result.returncode == 0, (
            f"Script failed with code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Verify expected output content
        stdout = result.stdout

        # Check for model artifact mention
        assert "model_stub.json" in stdout, "Output should mention model_stub.json"

        # Check for service directory mention
        assert "service" in stdout.lower(), "Output should mention service directory"

        # Check for JSON response with expected keys
        assert '"output"' in stdout, "Output should contain prediction response with 'output'"
        assert '"confidence"' in stdout, "Output should contain prediction response with 'confidence'"

        # Check for specific prediction result (reversed "hello")
        assert "olleh" in stdout, "Output should show reversed 'hello' -> 'olleh'"

        # Check for completion message
        assert "Demo Complete" in stdout, "Output should show completion message"

    def test_run_demo_function_directly(self):
        """Test calling run_demo function directly."""
        # Import the demo module
        sys.path.insert(0, str(PROJECT_ROOT / "examples"))
        try:
            from e2e_demo import run_demo

            # Run with cleanup
            results = run_demo(cleanup=True)

            # Verify results structure
            assert "model_path" in results
            assert "service_path" in results
            assert "response" in results

            # Model and service paths should have been set (before cleanup)
            # After cleanup, workdir is None but paths were captured
            assert results["response"] is not None
            assert results["response"]["output"] == "olleh"
            assert results["response"]["confidence"] == 0.75

        finally:
            # Clean up sys.path
            if str(PROJECT_ROOT / "examples") in sys.path:
                sys.path.remove(str(PROJECT_ROOT / "examples"))

            # Remove imported module
            if "e2e_demo" in sys.modules:
                del sys.modules["e2e_demo"]
