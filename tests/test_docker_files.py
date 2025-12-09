"""Tests for Docker configuration files.

These tests validate the presence and correctness of Dockerfile and
docker-compose.yml without requiring Docker to be installed.

To run the optional Docker build test, set DOCKER_AVAILABLE=true:
    DOCKER_AVAILABLE=true pytest tests/test_docker_files.py -v
"""

import os
import subprocess
from pathlib import Path

import pytest


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DOCKER_DIR = PROJECT_ROOT / "docker"


class TestDockerfileInference:
    """Tests for docker/Dockerfile.inference."""

    @pytest.fixture
    def dockerfile_content(self):
        """Read the Dockerfile content."""
        dockerfile_path = DOCKER_DIR / "Dockerfile.inference"
        assert dockerfile_path.exists(), "Dockerfile.inference not found"
        return dockerfile_path.read_text()

    def test_dockerfile_exists(self):
        """Test that Dockerfile.inference exists."""
        dockerfile_path = DOCKER_DIR / "Dockerfile.inference"
        assert dockerfile_path.exists()

    def test_dockerfile_has_from_instruction(self, dockerfile_content):
        """Test that Dockerfile has FROM instruction with Python base image."""
        assert "FROM python:3.11-slim" in dockerfile_content or \
               "FROM python:3.10-slim" in dockerfile_content

    def test_dockerfile_has_workdir(self, dockerfile_content):
        """Test that Dockerfile sets WORKDIR to /app."""
        assert "WORKDIR /app" in dockerfile_content

    def test_dockerfile_copies_requirements(self, dockerfile_content):
        """Test that Dockerfile copies requirements.txt."""
        assert "COPY requirements.txt" in dockerfile_content or \
               "COPY ./requirements.txt" in dockerfile_content

    def test_dockerfile_copies_inference_server(self, dockerfile_content):
        """Test that Dockerfile copies inference_server directory."""
        assert "COPY inference_server" in dockerfile_content or \
               "COPY ./inference_server" in dockerfile_content

    def test_dockerfile_installs_dependencies(self, dockerfile_content):
        """Test that Dockerfile runs pip install."""
        assert "pip install" in dockerfile_content
        assert "requirements.txt" in dockerfile_content

    def test_dockerfile_exposes_port(self, dockerfile_content):
        """Test that Dockerfile exposes port 8000."""
        assert "EXPOSE 8000" in dockerfile_content

    def test_dockerfile_has_cmd_or_entrypoint(self, dockerfile_content):
        """Test that Dockerfile has CMD or ENTRYPOINT with uvicorn."""
        has_uvicorn_cmd = "uvicorn" in dockerfile_content and (
            "CMD" in dockerfile_content or "ENTRYPOINT" in dockerfile_content
        )
        assert has_uvicorn_cmd, "Dockerfile should have CMD or ENTRYPOINT with uvicorn"

    def test_dockerfile_runs_uvicorn_with_correct_app(self, dockerfile_content):
        """Test that uvicorn runs the correct app module."""
        assert "inference_server.app:app" in dockerfile_content or \
               "inference_server.app" in dockerfile_content

    def test_dockerfile_binds_to_all_interfaces(self, dockerfile_content):
        """Test that uvicorn binds to 0.0.0.0 for container access."""
        assert "0.0.0.0" in dockerfile_content


class TestDockerCompose:
    """Tests for docker/docker-compose.yml."""

    @pytest.fixture
    def compose_content(self):
        """Read the docker-compose.yml content."""
        compose_path = DOCKER_DIR / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml not found"
        return compose_path.read_text()

    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists."""
        compose_path = DOCKER_DIR / "docker-compose.yml"
        assert compose_path.exists()

    def test_compose_has_inference_service(self, compose_content):
        """Test that docker-compose defines inference service."""
        assert "inference:" in compose_content or "inference :" in compose_content

    def test_compose_has_build_section(self, compose_content):
        """Test that docker-compose has build configuration."""
        assert "build:" in compose_content

    def test_compose_references_dockerfile(self, compose_content):
        """Test that build references Dockerfile.inference."""
        assert "Dockerfile.inference" in compose_content

    def test_compose_has_port_mapping(self, compose_content):
        """Test that docker-compose maps port 8000."""
        assert "8000:8000" in compose_content

    def test_compose_has_ports_section(self, compose_content):
        """Test that docker-compose has ports section."""
        assert "ports:" in compose_content

    def test_compose_has_volumes_section(self, compose_content):
        """Test that docker-compose has volumes section."""
        assert "volumes:" in compose_content

    def test_compose_has_model_registry_volume(self, compose_content):
        """Test that docker-compose maps model_registry volume."""
        assert "model_registry" in compose_content

    def test_compose_has_environment_section(self, compose_content):
        """Test that docker-compose has environment variables."""
        assert "environment:" in compose_content

    def test_compose_sets_inference_port(self, compose_content):
        """Test that INFERENCE_PORT environment variable is set."""
        assert "INFERENCE_PORT" in compose_content


class TestDockerReadme:
    """Tests for docker/README.md."""

    def test_docker_readme_exists(self):
        """Test that docker/README.md exists."""
        readme_path = DOCKER_DIR / "README.md"
        assert readme_path.exists()

    def test_docker_readme_has_build_instructions(self):
        """Test that README has build instructions."""
        readme_path = DOCKER_DIR / "README.md"
        content = readme_path.read_text()
        assert "docker build" in content.lower() or "docker-compose" in content.lower()

    def test_docker_readme_mentions_model_registry(self):
        """Test that README explains model_registry."""
        readme_path = DOCKER_DIR / "README.md"
        content = readme_path.read_text()
        assert "model_registry" in content


class TestDockerBuildOptional:
    """Optional Docker build tests (require DOCKER_AVAILABLE=true)."""

    @pytest.fixture
    def docker_available(self):
        """Check if Docker is available for testing."""
        if os.environ.get("DOCKER_AVAILABLE", "").lower() != "true":
            pytest.skip(
                "Docker build test skipped. Set DOCKER_AVAILABLE=true to enable."
            )

        # Also check if docker command exists
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                pytest.skip("Docker command not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker command not available")

        return True

    def test_dockerfile_builds_successfully(self, docker_available):
        """Test that Dockerfile.inference builds without errors.

        This test is optional and only runs when DOCKER_AVAILABLE=true.
        """
        result = subprocess.run(
            [
                "docker", "build",
                "-f", str(DOCKER_DIR / "Dockerfile.inference"),
                "-t", "tinyforge-inference:test",
                str(PROJECT_ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for build
        )

        assert result.returncode == 0, (
            f"Docker build failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
