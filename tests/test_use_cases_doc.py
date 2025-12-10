"""Tests for use cases documentation and example files."""

import os
from pathlib import Path

import pytest

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestUseCasesDocumentation:
    """Test suite for use cases documentation."""

    def test_use_cases_doc_exists(self) -> None:
        """Assert docs/use_cases.md exists."""
        doc_path = PROJECT_ROOT / "docs" / "use_cases.md"
        assert doc_path.exists(), f"use_cases.md not found at {doc_path}"

    def test_use_cases_doc_starts_with_heading(self) -> None:
        """Assert docs/use_cases.md starts with '# Use Cases'."""
        doc_path = PROJECT_ROOT / "docs" / "use_cases.md"
        content = doc_path.read_text(encoding="utf-8")
        assert content.startswith("# Use Cases"), (
            "use_cases.md should start with '# Use Cases'"
        )

    @pytest.mark.parametrize("heading", [
        "## 1. Internal Knowledge Assistant",
        "## 2. Customer Support Automation",
        "## 3. Domain-Specific Summarization",
        "## 4. Internal Workflow & SOP Automation",
        "## 5. Product Documentation Assistant",
        "## 6. Compliance & Audit Support",
        "## 7. Sales Enablement Assistant",
        "## 8. Manufacturing / Industrial SOP Models",
        "## 9. Healthcare (Guideline-only) Assistant",
        "## 10. Legal & Contract Assistance",
        "## 11. RAG + Tiny Model Hybrid Search",
        "## 12. Edge / Offline Deployment",
    ])
    def test_use_cases_doc_contains_required_headings(self, heading: str) -> None:
        """Assert each of the 12 required headings appears in the file."""
        doc_path = PROJECT_ROOT / "docs" / "use_cases.md"
        content = doc_path.read_text(encoding="utf-8")
        assert heading in content, f"Missing required heading: {heading}"

    def test_use_cases_doc_contains_basic_sections(self) -> None:
        """Assert each use case has Basic subsection."""
        doc_path = PROJECT_ROOT / "docs" / "use_cases.md"
        content = doc_path.read_text(encoding="utf-8")
        # Count Basic subsections (should be at least 12)
        basic_count = content.count("### Basic")
        assert basic_count >= 12, (
            f"Expected at least 12 '### Basic' subsections, found {basic_count}"
        )

    def test_use_cases_doc_contains_advanced_sections(self) -> None:
        """Assert each use case has Advanced subsection."""
        doc_path = PROJECT_ROOT / "docs" / "use_cases.md"
        content = doc_path.read_text(encoding="utf-8")
        # Count Advanced subsections (should be at least 12)
        advanced_count = content.count("### Advanced")
        assert advanced_count >= 12, (
            f"Expected at least 12 '### Advanced' subsections, found {advanced_count}"
        )


class TestUseCasesExampleFiles:
    """Test suite for use cases example files."""

    @pytest.mark.parametrize("filename", [
        "knowledge_assistant_sample.jsonl",
        "support_automation_sample.jsonl",
        "summarization_sample.txt",
        "sop_workflow_sample.jsonl",
    ])
    def test_example_file_exists(self, filename: str) -> None:
        """Assert each example file exists in examples/use_cases/."""
        file_path = PROJECT_ROOT / "examples" / "use_cases" / filename
        assert file_path.exists(), f"Example file not found: {file_path}"

    @pytest.mark.parametrize("filename", [
        "knowledge_assistant_sample.jsonl",
        "support_automation_sample.jsonl",
        "summarization_sample.txt",
        "sop_workflow_sample.jsonl",
    ])
    def test_example_file_is_non_empty(self, filename: str) -> None:
        """Assert each example file is non-empty."""
        file_path = PROJECT_ROOT / "examples" / "use_cases" / filename
        content = file_path.read_text(encoding="utf-8")
        assert len(content.strip()) > 0, f"Example file is empty: {file_path}"

    def test_jsonl_files_are_valid(self) -> None:
        """Assert JSONL example files contain valid JSON lines."""
        import json

        jsonl_files = [
            "knowledge_assistant_sample.jsonl",
            "support_automation_sample.jsonl",
            "sop_workflow_sample.jsonl",
        ]

        for filename in jsonl_files:
            file_path = PROJECT_ROOT / "examples" / "use_cases" / filename
            content = file_path.read_text(encoding="utf-8")

            for line_num, line in enumerate(content.strip().split("\n"), 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        assert "input" in data, (
                            f"{filename}:{line_num} missing 'input' field"
                        )
                        assert "output" in data, (
                            f"{filename}:{line_num} missing 'output' field"
                        )
                    except json.JSONDecodeError as e:
                        pytest.fail(
                            f"Invalid JSON in {filename}:{line_num}: {e}"
                        )


class TestReadmeLink:
    """Test suite for README.md link to use cases."""

    def test_readme_contains_use_cases_link(self) -> None:
        """Assert README.md contains a link to docs/use_cases.md."""
        readme_path = PROJECT_ROOT / "README.md"
        if readme_path.exists():
            content = readme_path.read_text(encoding="utf-8")
            assert "docs/use_cases.md" in content or "use_cases" in content.lower(), (
                "README.md should contain a link to docs/use_cases.md"
            )
        else:
            pytest.skip("README.md not found")
