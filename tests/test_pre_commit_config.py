"""
Tests for pre-commit configuration.

Validates that .pre-commit-config.yaml:
- Exists and is valid YAML
- Contains required hooks
- Has correct structure
"""

import os
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def config_path():
    """Get path to pre-commit config file."""
    project_root = Path(__file__).parent.parent
    return project_root / ".pre-commit-config.yaml"


@pytest.fixture
def config(config_path):
    """Load and parse the pre-commit config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


class TestPreCommitConfigExists:
    """Test that pre-commit config file exists."""

    def test_config_file_exists(self, config_path):
        """Test .pre-commit-config.yaml exists in project root."""
        assert config_path.exists(), ".pre-commit-config.yaml not found"

    def test_config_is_valid_yaml(self, config_path):
        """Test config file is valid YAML."""
        with open(config_path) as f:
            try:
                yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML: {e}")


class TestPreCommitConfigStructure:
    """Test pre-commit config structure."""

    def test_has_repos_key(self, config):
        """Test config has 'repos' key."""
        assert "repos" in config, "Config must have 'repos' key"

    def test_repos_is_list(self, config):
        """Test 'repos' is a list."""
        assert isinstance(config["repos"], list), "'repos' must be a list"

    def test_each_repo_has_required_keys(self, config):
        """Test each repo entry has required keys."""
        for repo in config["repos"]:
            assert "repo" in repo, "Each repo must have 'repo' key"
            assert "rev" in repo, "Each repo must have 'rev' key"
            assert "hooks" in repo, "Each repo must have 'hooks' key"


class TestRequiredHooks:
    """Test that required hooks are configured."""

    def test_has_trailing_whitespace_hook(self, config):
        """Test trailing-whitespace hook is configured."""
        hooks = self._get_all_hook_ids(config)
        assert "trailing-whitespace" in hooks, "Missing trailing-whitespace hook"

    def test_has_end_of_file_fixer_hook(self, config):
        """Test end-of-file-fixer hook is configured."""
        hooks = self._get_all_hook_ids(config)
        assert "end-of-file-fixer" in hooks, "Missing end-of-file-fixer hook"

    def test_has_black_hook(self, config):
        """Test black formatter hook is configured."""
        hooks = self._get_all_hook_ids(config)
        assert "black" in hooks, "Missing black hook"

    def test_has_isort_hook(self, config):
        """Test isort import sorting hook is configured."""
        hooks = self._get_all_hook_ids(config)
        assert "isort" in hooks, "Missing isort hook"

    def test_has_linting_hook(self, config):
        """Test a linting hook (ruff or flake8) is configured."""
        hooks = self._get_all_hook_ids(config)
        has_ruff = "ruff" in hooks
        has_flake8 = "flake8" in hooks
        assert has_ruff or has_flake8, "Missing linting hook (ruff or flake8)"

    def _get_all_hook_ids(self, config):
        """Extract all hook IDs from config."""
        hook_ids = []
        for repo in config["repos"]:
            for hook in repo["hooks"]:
                hook_ids.append(hook["id"])
        return hook_ids


class TestHookConfiguration:
    """Test hook configurations are correct."""

    def test_black_line_length(self, config):
        """Test black is configured with correct line length."""
        for repo in config["repos"]:
            if "black" in repo["repo"]:
                for hook in repo["hooks"]:
                    if hook["id"] == "black":
                        args = hook.get("args", [])
                        has_line_length = any("--line-length" in str(arg) for arg in args)
                        assert has_line_length, "Black should have --line-length configured"

    def test_isort_profile(self, config):
        """Test isort has black profile for compatibility."""
        for repo in config["repos"]:
            if "isort" in repo["repo"]:
                for hook in repo["hooks"]:
                    if hook["id"] == "isort":
                        args = hook.get("args", [])
                        has_profile = any("--profile" in str(arg) for arg in args)
                        assert has_profile, "isort should have --profile configured"


class TestRepoVersions:
    """Test repo versions are pinned."""

    def test_all_repos_have_pinned_versions(self, config):
        """Test all repos have version pinned (not 'main' or 'master')."""
        for repo in config["repos"]:
            rev = repo.get("rev", "")
            assert rev not in ["main", "master"], f"Repo {repo['repo']} should have pinned version"
            assert rev, f"Repo {repo['repo']} must have 'rev' specified"
