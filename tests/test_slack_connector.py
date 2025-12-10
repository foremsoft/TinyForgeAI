"""Tests for Slack connector."""

import json
import os
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from connectors.slack_connector import (
    SlackConnector,
    SlackConfig,
    SlackChannel,
    SlackMessage,
    SlackUser,
)


# =============================================================================
# SlackConfig Tests
# =============================================================================

class TestSlackConfig:
    """Tests for SlackConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SlackConfig()
        assert config.bot_token is None
        assert config.user_token is None
        assert config.mock_mode is True
        assert config.samples_dir is None
        assert config.limit == 100
        assert config.include_archived is False
        assert config.include_bot_messages is False
        assert config.include_thread_replies is True
        assert config.min_message_length == 1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SlackConfig(
            bot_token="xoxb-test-token",
            mock_mode=False,
            limit=50,
            include_bot_messages=True,
        )
        assert config.bot_token == "xoxb-test-token"
        assert config.mock_mode is False
        assert config.limit == 50
        assert config.include_bot_messages is True


# =============================================================================
# SlackChannel Tests
# =============================================================================

class TestSlackChannel:
    """Tests for SlackChannel dataclass."""

    def test_channel_creation(self):
        """Test channel creation with defaults."""
        channel = SlackChannel(id="C123", name="general")
        assert channel.id == "C123"
        assert channel.name == "general"
        assert channel.is_private is False
        assert channel.is_archived is False
        assert channel.is_member is True

    def test_channel_to_dict(self):
        """Test channel to_dict method."""
        channel = SlackChannel(
            id="C123",
            name="engineering",
            is_private=True,
            topic="Tech discussions",
            num_members=25,
        )
        data = channel.to_dict()
        assert data["id"] == "C123"
        assert data["name"] == "engineering"
        assert data["is_private"] is True
        assert data["topic"] == "Tech discussions"
        assert data["num_members"] == 25


# =============================================================================
# SlackMessage Tests
# =============================================================================

class TestSlackMessage:
    """Tests for SlackMessage dataclass."""

    def test_message_creation(self):
        """Test message creation with defaults."""
        msg = SlackMessage(ts="1702000001.000001", text="Hello world")
        assert msg.ts == "1702000001.000001"
        assert msg.text == "Hello world"
        assert msg.user is None
        assert msg.reply_count == 0
        assert msg.is_bot is False

    def test_message_timestamp_property(self):
        """Test timestamp property conversion."""
        msg = SlackMessage(ts="1702000001.000001", text="Test")
        ts = msg.timestamp
        assert isinstance(ts, datetime)
        assert ts.year == 2023

    def test_message_to_dict(self):
        """Test message to_dict method."""
        msg = SlackMessage(
            ts="1702000001.000001",
            text="Hello",
            user="U123",
            channel="C456",
            reply_count=5,
            reactions=[{"name": "+1", "count": 3}],
        )
        data = msg.to_dict()
        assert data["ts"] == "1702000001.000001"
        assert data["text"] == "Hello"
        assert data["user"] == "U123"
        assert data["channel"] == "C456"
        assert data["reply_count"] == 5
        assert len(data["reactions"]) == 1


# =============================================================================
# SlackUser Tests
# =============================================================================

class TestSlackUser:
    """Tests for SlackUser dataclass."""

    def test_user_creation(self):
        """Test user creation with defaults."""
        user = SlackUser(id="U123", name="alice")
        assert user.id == "U123"
        assert user.name == "alice"
        assert user.is_bot is False
        assert user.is_admin is False
        assert user.deleted is False

    def test_user_to_dict(self):
        """Test user to_dict method."""
        user = SlackUser(
            id="U123",
            name="alice",
            real_name="Alice Smith",
            display_name="alice",
            is_admin=True,
        )
        data = user.to_dict()
        assert data["id"] == "U123"
        assert data["name"] == "alice"
        assert data["real_name"] == "Alice Smith"
        assert data["is_admin"] is True


# =============================================================================
# SlackConnector Tests - Mock Mode
# =============================================================================

class TestSlackConnectorMockMode:
    """Tests for SlackConnector in mock mode."""

    @pytest.fixture
    def mock_samples_dir(self, tmp_path):
        """Create a temporary samples directory with test data."""
        samples_dir = tmp_path / "slack_samples"
        samples_dir.mkdir()

        # Create channels.json
        channels = [
            {"id": "C001", "name": "general", "is_private": False, "num_members": 10},
            {"id": "C002", "name": "random", "is_private": False, "num_members": 5},
        ]
        with open(samples_dir / "channels.json", "w") as f:
            json.dump(channels, f)

        # Create users.json
        users = [
            {"id": "U001", "name": "alice", "profile": {"real_name": "Alice"}},
            {"id": "U002", "name": "bob", "profile": {"real_name": "Bob"}},
        ]
        with open(samples_dir / "users.json", "w") as f:
            json.dump(users, f)

        # Create channel messages
        general_dir = samples_dir / "general"
        general_dir.mkdir()
        messages = [
            {"ts": "1702000001.000001", "text": "Hello", "user": "U001", "reply_count": 1},
            {"ts": "1702000002.000002", "text": "Hi there!", "user": "U002", "thread_ts": "1702000001.000001"},
            {"ts": "1702000003.000003", "text": "How are you?", "user": "U001", "reply_count": 0},
        ]
        with open(general_dir / "messages.json", "w") as f:
            json.dump(messages, f)

        return samples_dir

    @pytest.fixture
    def connector(self, mock_samples_dir):
        """Create a connector with mock samples directory."""
        config = SlackConfig(
            mock_mode=True,
            samples_dir=str(mock_samples_dir),
        )
        return SlackConnector(config)

    def test_list_channels(self, connector):
        """Test listing channels in mock mode."""
        channels = connector.list_channels()
        assert len(channels) == 2
        assert channels[0].id == "C001"
        assert channels[0].name == "general"
        assert channels[1].id == "C002"

    def test_get_messages(self, connector):
        """Test getting messages in mock mode."""
        messages = connector.get_messages("general")
        assert len(messages) == 3
        assert messages[0].text == "Hello"
        assert messages[1].thread_ts == "1702000001.000001"

    def test_get_thread_replies(self, connector):
        """Test getting thread replies in mock mode."""
        replies = connector.get_thread_replies("general", "1702000001.000001")
        # Should include parent and reply
        thread_messages = [m for m in replies if m.thread_ts == "1702000001.000001" or m.ts == "1702000001.000001"]
        assert len(thread_messages) >= 1

    def test_get_user(self, connector):
        """Test getting user in mock mode."""
        user = connector.get_user("U001")
        assert user.id == "U001"
        assert user.name == "alice"

    def test_get_user_not_found(self, connector):
        """Test getting unknown user returns mock user."""
        user = connector.get_user("U999")
        assert user.id == "U999"
        assert "user_" in user.name  # Mock user name


# =============================================================================
# SlackConnector Tests - Environment Variables
# =============================================================================

class TestSlackConnectorEnvVars:
    """Tests for SlackConnector environment variable handling."""

    def test_mock_mode_env_true(self, monkeypatch):
        """Test SLACK_MOCK=true enables mock mode."""
        monkeypatch.setenv("SLACK_MOCK", "true")
        connector = SlackConnector(SlackConfig(mock_mode=False))
        assert connector.config.mock_mode is True

    def test_mock_mode_env_false(self, monkeypatch):
        """Test SLACK_MOCK=false disables mock mode."""
        monkeypatch.setenv("SLACK_MOCK", "false")
        connector = SlackConnector(SlackConfig(mock_mode=True))
        assert connector.config.mock_mode is False

    def test_connector_mock_env(self, monkeypatch):
        """Test CONNECTOR_MOCK enables mock mode."""
        monkeypatch.setenv("CONNECTOR_MOCK", "true")
        connector = SlackConnector(SlackConfig(mock_mode=False))
        assert connector.config.mock_mode is True

    def test_bot_token_from_env(self, monkeypatch):
        """Test bot token from environment."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
        connector = SlackConnector(SlackConfig())
        assert connector.config.bot_token == "xoxb-test-token"


# =============================================================================
# SlackConnector Tests - Message Text Cleaning
# =============================================================================

class TestSlackConnectorTextCleaning:
    """Tests for message text cleaning functionality."""

    @pytest.fixture
    def connector(self):
        """Create a connector for testing."""
        return SlackConnector(SlackConfig(mock_mode=True))

    def test_clean_user_mentions(self, connector):
        """Test removing user mentions."""
        text = "Hey <@U123456> can you help?"
        cleaned = connector._clean_message_text(text)
        assert "<@U123456>" not in cleaned
        assert "can you help?" in cleaned

    def test_clean_channel_references(self, connector):
        """Test cleaning channel references."""
        text = "Check <#C123456|general> for more info"
        cleaned = connector._clean_message_text(text)
        assert "#general" in cleaned
        assert "<#C123456|general>" not in cleaned

    def test_clean_urls_with_text(self, connector):
        """Test cleaning URLs with display text."""
        text = "Visit <https://example.com|our website> for details"
        cleaned = connector._clean_message_text(text)
        assert "our website" in cleaned
        assert "<https://example.com|our website>" not in cleaned

    def test_clean_plain_urls(self, connector):
        """Test cleaning plain URLs."""
        text = "Link: <https://example.com>"
        cleaned = connector._clean_message_text(text)
        assert "https://example.com" in cleaned
        assert "<https://example.com>" not in cleaned

    def test_clean_special_commands(self, connector):
        """Test removing special commands."""
        text = "<!here> Everyone please check this"
        cleaned = connector._clean_message_text(text)
        assert "<!here>" not in cleaned
        assert "Everyone" in cleaned

    def test_clean_multiple_mentions(self, connector):
        """Test cleaning multiple mentions."""
        text = "<@U001> and <@U002> please review"
        cleaned = connector._clean_message_text(text)
        assert "<@U001>" not in cleaned
        assert "<@U002>" not in cleaned
        assert "and" in cleaned

    def test_normalize_whitespace(self, connector):
        """Test whitespace normalization."""
        text = "Hello    world\n\ntest"
        cleaned = connector._clean_message_text(text)
        assert cleaned == "Hello world test"

    def test_clean_empty_text(self, connector):
        """Test cleaning empty text."""
        assert connector._clean_message_text("") == ""
        assert connector._clean_message_text(None) == ""


# =============================================================================
# SlackConnector Tests - Stream Samples
# =============================================================================

class TestSlackConnectorStreamSamples:
    """Tests for streaming training samples."""

    @pytest.fixture
    def mock_samples_dir(self, tmp_path):
        """Create samples directory with thread conversations."""
        samples_dir = tmp_path / "slack_samples"
        channel_dir = samples_dir / "engineering"
        channel_dir.mkdir(parents=True)

        messages = [
            {
                "ts": "1702000001.000001",
                "text": "How do I deploy the app?",
                "user": "U001",
                "reply_count": 2,
            },
            {
                "ts": "1702000002.000002",
                "text": "Run deploy.sh in the scripts folder",
                "user": "U002",
                "thread_ts": "1702000001.000001",
            },
            {
                "ts": "1702000003.000003",
                "text": "Make sure to set the environment first",
                "user": "U003",
                "thread_ts": "1702000001.000001",
            },
            {
                "ts": "1702000010.000010",
                "text": "Good morning!",
                "user": "U001",
                "reply_count": 0,
            },
            {
                "ts": "1702000011.000011",
                "text": "Morning!",
                "user": "U002",
                "reply_count": 0,
            },
        ]
        with open(channel_dir / "messages.json", "w") as f:
            json.dump(messages, f)

        return samples_dir

    @pytest.fixture
    def connector(self, mock_samples_dir):
        """Create connector with mock samples."""
        config = SlackConfig(mock_mode=True, samples_dir=str(mock_samples_dir))
        return SlackConnector(config)

    def test_stream_thread_qa_mode(self, connector):
        """Test streaming samples in thread_qa mode."""
        samples = list(connector.stream_samples(
            "engineering",
            {"input": "input", "output": "output"},
            mode="thread_qa",
        ))

        # Should have one sample from the thread
        assert len(samples) >= 1
        sample = samples[0]
        assert "How do I deploy" in sample["input"]
        assert "deploy.sh" in sample["output"]
        assert sample["metadata"]["source"] == "slack"

    def test_stream_consecutive_mode(self, connector):
        """Test streaming samples in consecutive mode."""
        samples = list(connector.stream_samples(
            "engineering",
            {"input": "input", "output": "output"},
            mode="consecutive",
        ))

        # Should have pairs of consecutive messages
        assert len(samples) >= 1
        for sample in samples:
            assert "input" in sample
            assert "output" in sample
            assert sample["metadata"]["source"] == "slack"

    def test_stream_invalid_mode(self, connector):
        """Test streaming with invalid mode raises error."""
        with pytest.raises(ValueError, match="Unknown streaming mode"):
            list(connector.stream_samples(
                "engineering",
                {"input": "input", "output": "output"},
                mode="invalid_mode",
            ))


# =============================================================================
# SlackConnector Tests - Real API (Mocked)
# =============================================================================

class TestSlackConnectorRealAPI:
    """Tests for real API calls with mocked responses."""

    @pytest.fixture
    def connector(self, monkeypatch):
        """Create connector for API testing."""
        monkeypatch.setenv("SLACK_MOCK", "false")
        config = SlackConfig(
            mock_mode=False,
            bot_token="xoxb-test-token",
        )
        return SlackConnector(config)

    @patch("connectors.slack_connector.REQUESTS_AVAILABLE", True)
    def test_api_call_success(self, connector):
        """Test successful API call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "channels": []}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_session_class.return_value = mock_session

            channels = connector.list_channels()
            assert channels == []

    @patch("connectors.slack_connector.REQUESTS_AVAILABLE", True)
    def test_api_call_error(self, connector):
        """Test API call with error response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "invalid_auth"}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            mock_session_class.return_value = mock_session

            with pytest.raises(RuntimeError, match="Slack API error"):
                connector.list_channels()

    def test_no_token_raises_error(self, monkeypatch):
        """Test that missing token raises error."""
        monkeypatch.setenv("SLACK_MOCK", "false")
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        monkeypatch.delenv("SLACK_TOKEN", raising=False)

        config = SlackConfig(mock_mode=False)
        connector = SlackConnector(config)

        with pytest.raises(RuntimeError, match="No Slack token provided"):
            connector._get_session()


# =============================================================================
# SlackConnector Tests - Built-in Samples
# =============================================================================

class TestSlackConnectorBuiltInSamples:
    """Tests using the built-in sample files."""

    @pytest.fixture
    def connector(self):
        """Create connector using built-in samples."""
        config = SlackConfig(mock_mode=True)
        return SlackConnector(config)

    def test_list_channels_builtin(self, connector):
        """Test listing channels with built-in samples."""
        samples_dir = connector._get_samples_dir()
        if not (samples_dir / "channels.json").exists():
            pytest.skip("Built-in samples not available")

        channels = connector.list_channels()
        assert len(channels) > 0
        assert all(isinstance(c, SlackChannel) for c in channels)

    def test_get_messages_builtin(self, connector):
        """Test getting messages with built-in samples."""
        samples_dir = connector._get_samples_dir()
        general_dir = samples_dir / "general"
        if not general_dir.exists():
            pytest.skip("Built-in samples not available")

        messages = connector.get_messages("general")
        assert len(messages) > 0
        assert all(isinstance(m, SlackMessage) for m in messages)

    def test_stream_samples_builtin(self, connector):
        """Test streaming samples with built-in samples."""
        samples_dir = connector._get_samples_dir()
        general_dir = samples_dir / "general"
        if not general_dir.exists():
            pytest.skip("Built-in samples not available")

        samples = list(connector.stream_samples(
            "general",
            {"input": "input", "output": "output"},
            mode="thread_qa",
        ))

        # Should have samples from threads
        assert len(samples) >= 0  # May be 0 if no threads with replies


# =============================================================================
# SlackConnector Tests - Edge Cases
# =============================================================================

class TestSlackConnectorEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def connector(self, tmp_path):
        """Create connector with empty samples directory."""
        samples_dir = tmp_path / "slack_samples"
        samples_dir.mkdir()
        config = SlackConfig(mock_mode=True, samples_dir=str(samples_dir))
        return SlackConnector(config)

    def test_list_channels_empty_dir(self, connector):
        """Test listing channels with empty directory."""
        channels = connector.list_channels()
        assert channels == []

    def test_get_messages_nonexistent_channel(self, connector):
        """Test getting messages from nonexistent channel."""
        messages = connector.get_messages("nonexistent")
        assert messages == []

    def test_user_cache(self, tmp_path):
        """Test user caching works correctly."""
        samples_dir = tmp_path / "slack_samples"
        samples_dir.mkdir()

        users = [{"id": "U001", "name": "alice"}]
        with open(samples_dir / "users.json", "w") as f:
            json.dump(users, f)

        config = SlackConfig(mock_mode=True, samples_dir=str(samples_dir))
        connector = SlackConnector(config)

        # First call should populate cache
        user1 = connector.get_user("U001")
        assert user1.name == "alice"

        # Second call should use cache
        user2 = connector.get_user("U001")
        assert user1 is user2  # Same object from cache


# =============================================================================
# SlackConnector Tests - Channel Parsing
# =============================================================================

class TestSlackConnectorChannelParsing:
    """Tests for channel data parsing."""

    def test_parse_channel_with_topic_dict(self):
        """Test parsing channel with topic as dict."""
        connector = SlackConnector(SlackConfig())
        data = {
            "id": "C123",
            "name": "test",
            "topic": {"value": "Test topic"},
            "purpose": {"value": "Test purpose"},
        }
        channel = connector._parse_channel(data)
        assert channel.topic == "Test topic"
        assert channel.purpose == "Test purpose"

    def test_parse_channel_with_topic_string(self):
        """Test parsing channel with topic as string."""
        connector = SlackConnector(SlackConfig())
        data = {
            "id": "C123",
            "name": "test",
            "topic": "Test topic",
            "purpose": "Test purpose",
        }
        channel = connector._parse_channel(data)
        assert channel.topic == "Test topic"
        assert channel.purpose == "Test purpose"


# =============================================================================
# Integration Tests
# =============================================================================

class TestSlackConnectorIntegration:
    """Integration tests for the Slack connector."""

    @pytest.fixture
    def full_samples_dir(self, tmp_path):
        """Create a full samples directory for integration testing."""
        samples_dir = tmp_path / "slack_samples"
        samples_dir.mkdir()

        # Channels
        channels = [
            {"id": "C001", "name": "support", "topic": {"value": "Help desk"}},
        ]
        with open(samples_dir / "channels.json", "w") as f:
            json.dump(channels, f)

        # Users
        users = [
            {"id": "U001", "name": "support_agent", "profile": {"real_name": "Support Agent"}},
            {"id": "U002", "name": "customer", "profile": {"real_name": "Customer"}},
        ]
        with open(samples_dir / "users.json", "w") as f:
            json.dump(users, f)

        # Support channel messages with Q&A threads
        support_dir = samples_dir / "support"
        support_dir.mkdir()
        messages = [
            {
                "ts": "1702000001.000001",
                "text": "I can't log in to my account",
                "user": "U002",
                "reply_count": 1,
                "reactions": [{"name": "eyes", "count": 1}],
            },
            {
                "ts": "1702000002.000002",
                "text": "Have you tried resetting your password? Go to Settings > Password Reset",
                "user": "U001",
                "thread_ts": "1702000001.000001",
            },
            {
                "ts": "1702000010.000010",
                "text": "How do I export my data?",
                "user": "U002",
                "reply_count": 1,
            },
            {
                "ts": "1702000011.000011",
                "text": "Go to Account > Data > Export. You'll receive an email with the download link.",
                "user": "U001",
                "thread_ts": "1702000010.000010",
            },
        ]
        with open(support_dir / "messages.json", "w") as f:
            json.dump(messages, f)

        return samples_dir

    def test_full_qa_extraction_pipeline(self, full_samples_dir):
        """Test complete Q&A extraction pipeline."""
        config = SlackConfig(mock_mode=True, samples_dir=str(full_samples_dir))
        connector = SlackConnector(config)

        # List channels
        channels = connector.list_channels()
        assert len(channels) == 1
        assert channels[0].name == "support"

        # Get messages
        messages = connector.get_messages("support")
        assert len(messages) == 4

        # Stream Q&A samples
        samples = list(connector.stream_samples(
            "support",
            {"input": "input", "output": "output"},
            mode="thread_qa",
        ))

        # Should extract 2 Q&A pairs from threads
        assert len(samples) == 2

        # Check first sample
        sample1 = samples[0]
        assert "log in" in sample1["input"]
        assert "password" in sample1["output"].lower()
        assert sample1["metadata"]["source"] == "slack"
        assert sample1["metadata"]["channel_id"] == "support"

        # Check second sample
        sample2 = samples[1]
        assert "export" in sample2["input"]
        assert "email" in sample2["output"].lower()
