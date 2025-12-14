"""
Tests for configuration.md documentation.

Validates that docs/configuration.md:
- Exists and is properly formatted
- Documents all environment variables from .env.example
- Contains required sections
"""

import re
from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def config_doc_path(project_root):
    """Get path to configuration.md."""
    return project_root / "docs" / "configuration.md"


@pytest.fixture
def env_example_path(project_root):
    """Get path to .env.example."""
    return project_root / ".env.example"


@pytest.fixture
def config_doc_content(config_doc_path):
    """Load configuration.md content."""
    return config_doc_path.read_text()


@pytest.fixture
def env_example_content(env_example_path):
    """Load .env.example content."""
    return env_example_path.read_text()


class TestConfigurationDocExists:
    """Test that configuration.md exists."""

    def test_doc_file_exists(self, config_doc_path):
        """Test docs/configuration.md exists."""
        assert config_doc_path.exists(), "docs/configuration.md not found"

    def test_doc_is_markdown(self, config_doc_path):
        """Test file has .md extension."""
        assert config_doc_path.suffix == ".md"


class TestDocumentationStructure:
    """Test configuration.md has required structure."""

    def test_has_title(self, config_doc_content):
        """Test doc has a main title."""
        assert "# Configuration" in config_doc_content

    def test_has_quick_start_section(self, config_doc_content):
        """Test doc has Quick Start section."""
        assert "Quick Start" in config_doc_content

    def test_has_environment_variables_reference(self, config_doc_content):
        """Test doc has Environment Variables Reference."""
        assert "Environment Variables" in config_doc_content

    def test_has_example_configurations(self, config_doc_content):
        """Test doc has Example Configurations section."""
        assert "Example Configurations" in config_doc_content

    def test_has_tables(self, config_doc_content):
        """Test doc uses markdown tables for variable documentation."""
        assert "| Variable |" in config_doc_content
        assert "| Default |" in config_doc_content


class TestEnvironmentVariablesCoverage:
    """Test that all .env.example variables are documented."""

    def _extract_env_vars(self, content):
        """Extract environment variable names from content."""
        pattern = r'^([A-Z][A-Z0-9_]+)='
        return set(re.findall(pattern, content, re.MULTILINE))

    def test_all_env_vars_documented(self, config_doc_content, env_example_content):
        """Test all variables from .env.example are in configuration.md."""
        env_vars = self._extract_env_vars(env_example_content)

        missing_vars = []
        for var in env_vars:
            if var not in config_doc_content:
                missing_vars.append(var)

        assert not missing_vars, f"Missing documentation for: {missing_vars}"

    def test_tinyforge_vars_documented(self, config_doc_content, env_example_content):
        """Test TINYFORGE_* variables are documented."""
        env_vars = self._extract_env_vars(env_example_content)
        tinyforge_vars = [v for v in env_vars if v.startswith("TINYFORGE_")]

        for var in tinyforge_vars:
            assert var in config_doc_content, f"Missing: {var}"


class TestDocumentationQuality:
    """Test documentation quality."""

    def test_has_default_values(self, config_doc_content):
        """Test doc shows default values."""
        assert "Default" in config_doc_content

    def test_has_descriptions(self, config_doc_content):
        """Test doc has descriptions column."""
        assert "Description" in config_doc_content

    def test_has_code_examples(self, config_doc_content):
        """Test doc includes code examples."""
        assert "```" in config_doc_content

    def test_development_example(self, config_doc_content):
        """Test has development setup example."""
        assert "Development" in config_doc_content
        assert "development" in config_doc_content.lower()

    def test_production_example(self, config_doc_content):
        """Test has production setup example."""
        assert "Production" in config_doc_content
        assert "production" in config_doc_content.lower()


class TestSectionCoverage:
    """Test that required sections are present."""

    def test_application_settings_section(self, config_doc_content):
        """Test Application Settings section exists."""
        assert "Application Settings" in config_doc_content

    def test_training_settings_section(self, config_doc_content):
        """Test Training Settings section exists."""
        assert "Training" in config_doc_content

    def test_lora_settings_section(self, config_doc_content):
        """Test LoRA Settings section exists."""
        assert "LoRA" in config_doc_content

    def test_inference_settings_section(self, config_doc_content):
        """Test Inference Server Settings section exists."""
        assert "Inference" in config_doc_content

    def test_authentication_settings_section(self, config_doc_content):
        """Test Authentication Settings section exists."""
        assert "Authentication" in config_doc_content

    def test_database_settings_section(self, config_doc_content):
        """Test Database Settings section exists."""
        assert "Database" in config_doc_content

    def test_docker_settings_section(self, config_doc_content):
        """Test Docker Settings section exists."""
        assert "Docker" in config_doc_content


class TestSecurityDocumentation:
    """Test security-related documentation."""

    def test_mentions_secret_key_generation(self, config_doc_content):
        """Test doc mentions how to generate secret keys."""
        assert "openssl" in config_doc_content.lower() or "rand" in config_doc_content

    def test_warns_about_default_credentials(self, config_doc_content):
        """Test doc warns about changing default credentials."""
        assert "change" in config_doc_content.lower() or "production" in config_doc_content.lower()
