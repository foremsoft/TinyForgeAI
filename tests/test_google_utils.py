"""Tests for the Google Docs utility functions."""

from connectors.google_utils import extract_text_from_html, normalize_text


def test_extract_text_from_html_removes_tags():
    """Test that HTML tags are removed."""
    html = "<p>Hello <b>world</b>!</p>"
    result = extract_text_from_html(html)
    assert result == "Hello world!"


def test_extract_text_from_html_handles_nested_tags():
    """Test that nested HTML tags are handled."""
    html = "<div><p>First <span>paragraph</span></p><p>Second</p></div>"
    result = extract_text_from_html(html)
    assert "First" in result
    assert "paragraph" in result
    assert "Second" in result


def test_extract_text_from_html_handles_empty_string():
    """Test that empty string returns empty string."""
    assert extract_text_from_html("") == ""


def test_extract_text_from_html_handles_plain_text():
    """Test that plain text without tags is returned as-is."""
    text = "No HTML here"
    assert extract_text_from_html(text) == text


def test_extract_text_from_html_handles_entities():
    """Test that HTML entities are handled."""
    html = "<p>Hello &amp; goodbye</p>"
    result = extract_text_from_html(html)
    assert "Hello" in result
    assert "goodbye" in result


def test_normalize_text_reduces_multiple_spaces():
    """Test that multiple spaces are reduced to single space."""
    text = "Too   many    spaces"
    result = normalize_text(text)
    assert result == "Too many spaces"


def test_normalize_text_reduces_multiple_newlines():
    """Test that multiple newlines are reduced to double newline."""
    text = "Line 1\n\n\n\n\nLine 2"
    result = normalize_text(text)
    assert result == "Line 1\n\nLine 2"


def test_normalize_text_strips_leading_trailing_whitespace():
    """Test that leading/trailing whitespace is stripped."""
    text = "   Hello world   "
    result = normalize_text(text)
    assert result == "Hello world"


def test_normalize_text_removes_control_characters():
    """Test that control characters are removed."""
    text = "Hello\x00\x01\x02world"
    result = normalize_text(text)
    assert "\x00" not in result
    assert "\x01" not in result
    assert "\x02" not in result
    assert "Hello" in result
    assert "world" in result


def test_normalize_text_preserves_single_newlines():
    """Test that single newlines are preserved."""
    text = "Line 1\nLine 2"
    result = normalize_text(text)
    assert result == "Line 1\nLine 2"


def test_normalize_text_handles_tabs():
    """Test that tabs are normalized to spaces."""
    text = "Column1\t\tColumn2"
    result = normalize_text(text)
    # Tabs should be replaced with single space
    assert "Column1 Column2" == result


def test_normalize_text_empty_string():
    """Test that empty string returns empty string."""
    assert normalize_text("") == ""
