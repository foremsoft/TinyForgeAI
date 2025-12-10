"""
Slack connector for TinyForgeAI.

Provides functionality to list channels, get messages, and stream training
samples from Slack conversations and threads.

Mock Mode:
    When SLACK_MOCK=true (default), the connector reads from local sample
    files in examples/slack_samples/ instead of making real API calls.

Real Mode:
    When SLACK_MOCK=false, the connector uses the Slack Web API with a
    bot token to access real channels and messages.

Usage:
    from connectors.slack_connector import SlackConnector

    # Create connector (uses mock mode by default)
    connector = SlackConnector()

    # List channels
    channels = connector.list_channels()

    # Get messages from a channel
    messages = connector.get_messages(channel_id="C123456")

    # Stream training samples (conversation pairs)
    for sample in connector.stream_samples(channel_id="C123456", mapping={...}):
        print(sample)
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# Add project root to path when run as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from connectors.mappers import row_to_sample

logger = logging.getLogger(__name__)

# Check for requests library
REQUESTS_AVAILABLE = False
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class SlackConfig:
    """Configuration for Slack connector."""

    # Authentication
    bot_token: Optional[str] = None  # Slack bot token (xoxb-...)
    user_token: Optional[str] = None  # Slack user token (xoxp-...) for extended access

    # Mock mode settings
    mock_mode: bool = True  # Use mock mode by default
    samples_dir: Optional[str] = None  # Custom samples directory

    # Query settings
    limit: int = 100  # Messages per request (max 1000)
    include_archived: bool = False  # Include archived channels

    # Message filtering
    include_bot_messages: bool = False  # Include messages from bots
    include_thread_replies: bool = True  # Include thread replies
    min_message_length: int = 1  # Minimum message length to include

    # Rate limiting
    requests_per_minute: int = 50  # Slack Tier 2 rate limit


@dataclass
class SlackChannel:
    """Represents a Slack channel."""

    id: str
    name: str
    is_private: bool = False
    is_archived: bool = False
    is_member: bool = True
    topic: Optional[str] = None
    purpose: Optional[str] = None
    num_members: Optional[int] = None
    created: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "is_private": self.is_private,
            "is_archived": self.is_archived,
            "is_member": self.is_member,
            "topic": self.topic,
            "purpose": self.purpose,
            "num_members": self.num_members,
            "created": self.created,
        }


@dataclass
class SlackMessage:
    """Represents a Slack message."""

    ts: str  # Message timestamp (unique ID)
    text: str
    user: Optional[str] = None
    channel: Optional[str] = None
    thread_ts: Optional[str] = None  # Parent thread timestamp
    reply_count: int = 0
    reactions: List[Dict[str, Any]] = field(default_factory=list)
    is_bot: bool = False
    subtype: Optional[str] = None
    edited: Optional[Dict[str, Any]] = None

    @property
    def timestamp(self) -> datetime:
        """Convert ts to datetime."""
        return datetime.fromtimestamp(float(self.ts))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ts": self.ts,
            "text": self.text,
            "user": self.user,
            "channel": self.channel,
            "thread_ts": self.thread_ts,
            "reply_count": self.reply_count,
            "reactions": self.reactions,
            "is_bot": self.is_bot,
            "subtype": self.subtype,
            "edited": self.edited,
        }


@dataclass
class SlackUser:
    """Represents a Slack user."""

    id: str
    name: str
    real_name: Optional[str] = None
    display_name: Optional[str] = None
    is_bot: bool = False
    is_admin: bool = False
    deleted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "real_name": self.real_name,
            "display_name": self.display_name,
            "is_bot": self.is_bot,
            "is_admin": self.is_admin,
            "deleted": self.deleted,
        }


class SlackConnector:
    """
    Slack connector for accessing messages and streaming training samples.

    Example:
        connector = SlackConnector()

        # List channels
        channels = connector.list_channels()

        # Get messages
        messages = connector.get_messages(channel_id="C123456")

        # Stream conversation samples (question/answer pairs from threads)
        mapping = {"input": "question", "output": "answer"}
        for sample in connector.stream_samples("C123456", mapping):
            print(sample)
    """

    BASE_URL = "https://slack.com/api"

    def __init__(self, config: Optional[SlackConfig] = None):
        """
        Initialize the Slack connector.

        Args:
            config: Configuration object. If None, uses defaults with mock mode.
        """
        self.config = config or SlackConfig()

        # Check environment variables for mock mode override
        env_mock = os.getenv("SLACK_MOCK", "").lower()
        if env_mock in ("true", "1", "yes"):
            self.config.mock_mode = True
        elif env_mock in ("false", "0", "no"):
            self.config.mock_mode = False

        # Also check global connector mock setting
        global_mock = os.getenv("CONNECTOR_MOCK", "").lower()
        if global_mock in ("true", "1", "yes"):
            self.config.mock_mode = True

        # Check for bot token in environment
        if not self.config.bot_token:
            self.config.bot_token = os.getenv("SLACK_BOT_TOKEN") or os.getenv("SLACK_TOKEN")

        self._session = None
        self._users_cache: Dict[str, SlackUser] = {}
        logger.debug(f"SlackConnector initialized (mock_mode={self.config.mock_mode})")

    def _get_samples_dir(self) -> Path:
        """Get the path to the sample files directory."""
        if self.config.samples_dir:
            return Path(self.config.samples_dir)

        # Try relative to this file first
        connector_dir = Path(__file__).parent
        samples_dir = connector_dir.parent / "examples" / "slack_samples"
        if samples_dir.exists():
            return samples_dir

        # Fallback to current working directory
        return Path("examples") / "slack_samples"

    def _get_session(self) -> "requests.Session":
        """Get or create a requests session with authentication."""
        if self._session is not None:
            return self._session

        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests library not installed. "
                "Install with: pip install requests"
            )

        token = self.config.bot_token or self.config.user_token
        if not token:
            raise RuntimeError(
                "No Slack token provided. Either:\n"
                "- Set SLACK_BOT_TOKEN environment variable\n"
                "- Pass bot_token in SlackConfig\n"
                "Or set SLACK_MOCK=true for mock mode."
            )

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        })
        return self._session

    def _api_call(self, method: str, **kwargs) -> Dict[str, Any]:
        """Make an API call to Slack."""
        session = self._get_session()

        url = f"{self.BASE_URL}/{method}"

        # Use GET for read methods, POST for write
        if method in ("conversations.list", "conversations.history",
                      "conversations.replies", "users.list", "users.info"):
            response = session.get(url, params=kwargs)
        else:
            response = session.post(url, json=kwargs)

        response.raise_for_status()
        data = response.json()

        if not data.get("ok"):
            error = data.get("error", "Unknown error")
            raise RuntimeError(f"Slack API error: {error}")

        return data

    def list_channels(
        self,
        types: Optional[str] = None,
        exclude_archived: Optional[bool] = None,
    ) -> List[SlackChannel]:
        """
        List accessible Slack channels.

        Args:
            types: Comma-separated channel types (public_channel, private_channel, mpim, im).
            exclude_archived: Whether to exclude archived channels.

        Returns:
            List of SlackChannel objects.
        """
        if self.config.mock_mode:
            return self._list_channels_mock()
        return self._list_channels_real(types, exclude_archived)

    def _list_channels_mock(self) -> List[SlackChannel]:
        """List channels from mock samples directory."""
        samples_dir = self._get_samples_dir()

        if not samples_dir.exists():
            logger.warning(f"Samples directory not found: {samples_dir}")
            return []

        # Look for channels.json file
        channels_file = samples_dir / "channels.json"
        if channels_file.exists():
            with open(channels_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                channels = data if isinstance(data, list) else data.get("channels", [])
                return [self._parse_channel(c) for c in channels]

        # Fallback: treat each subdirectory as a channel
        channels = []
        for subdir in sorted(samples_dir.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith("."):
                channels.append(SlackChannel(
                    id=subdir.name,
                    name=subdir.name,
                ))

        return channels

    def _list_channels_real(
        self,
        types: Optional[str] = None,
        exclude_archived: Optional[bool] = None,
    ) -> List[SlackChannel]:
        """List channels using Slack API."""
        if exclude_archived is None:
            exclude_archived = not self.config.include_archived

        channels = []
        cursor = None

        while True:
            params = {
                "limit": self.config.limit,
                "exclude_archived": exclude_archived,
            }
            if types:
                params["types"] = types
            if cursor:
                params["cursor"] = cursor

            data = self._api_call("conversations.list", **params)

            for channel_data in data.get("channels", []):
                channels.append(self._parse_channel(channel_data))

            # Check for pagination
            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return channels

    def _parse_channel(self, data: Dict[str, Any]) -> SlackChannel:
        """Parse channel data into SlackChannel object."""
        topic = data.get("topic", {})
        purpose = data.get("purpose", {})

        return SlackChannel(
            id=data.get("id", ""),
            name=data.get("name", ""),
            is_private=data.get("is_private", False),
            is_archived=data.get("is_archived", False),
            is_member=data.get("is_member", True),
            topic=topic.get("value") if isinstance(topic, dict) else topic,
            purpose=purpose.get("value") if isinstance(purpose, dict) else purpose,
            num_members=data.get("num_members"),
            created=data.get("created"),
        )

    def get_messages(
        self,
        channel_id: str,
        oldest: Optional[str] = None,
        latest: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SlackMessage]:
        """
        Get messages from a channel.

        Args:
            channel_id: ID of the channel.
            oldest: Only messages after this timestamp.
            latest: Only messages before this timestamp.
            limit: Maximum number of messages to retrieve.

        Returns:
            List of SlackMessage objects.
        """
        if self.config.mock_mode:
            return self._get_messages_mock(channel_id)
        return self._get_messages_real(channel_id, oldest, latest, limit)

    def _get_messages_mock(self, channel_id: str) -> List[SlackMessage]:
        """Get messages from mock samples."""
        samples_dir = self._get_samples_dir()

        # Try to find channel messages file
        for path in [
            samples_dir / channel_id / "messages.json",
            samples_dir / f"{channel_id}.json",
            samples_dir / f"{channel_id}_messages.json",
        ]:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    messages = data if isinstance(data, list) else data.get("messages", [])
                    return [self._parse_message(m, channel_id) for m in messages]

        # Search for any JSON file in channel directory
        channel_dir = samples_dir / channel_id
        if channel_dir.exists():
            for json_file in channel_dir.glob("*.json"):
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return [self._parse_message(m, channel_id) for m in data]
                    if "messages" in data:
                        return [self._parse_message(m, channel_id) for m in data["messages"]]

        logger.warning(f"No mock messages found for channel: {channel_id}")
        return []

    def _get_messages_real(
        self,
        channel_id: str,
        oldest: Optional[str] = None,
        latest: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SlackMessage]:
        """Get messages using Slack API."""
        messages = []
        cursor = None
        total_limit = limit or float("inf")

        while len(messages) < total_limit:
            params = {
                "channel": channel_id,
                "limit": min(self.config.limit, total_limit - len(messages)),
            }
            if oldest:
                params["oldest"] = oldest
            if latest:
                params["latest"] = latest
            if cursor:
                params["cursor"] = cursor

            data = self._api_call("conversations.history", **params)

            for msg_data in data.get("messages", []):
                msg = self._parse_message(msg_data, channel_id)

                # Filter based on config
                if not self.config.include_bot_messages and msg.is_bot:
                    continue
                if len(msg.text) < self.config.min_message_length:
                    continue

                messages.append(msg)

            # Check for pagination
            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor or not data.get("has_more"):
                break

        return messages

    def _parse_message(self, data: Dict[str, Any], channel_id: Optional[str] = None) -> SlackMessage:
        """Parse message data into SlackMessage object."""
        return SlackMessage(
            ts=data.get("ts", ""),
            text=data.get("text", ""),
            user=data.get("user"),
            channel=channel_id or data.get("channel"),
            thread_ts=data.get("thread_ts"),
            reply_count=data.get("reply_count", 0),
            reactions=data.get("reactions", []),
            is_bot=data.get("bot_id") is not None or data.get("subtype") == "bot_message",
            subtype=data.get("subtype"),
            edited=data.get("edited"),
        )

    def get_thread_replies(
        self,
        channel_id: str,
        thread_ts: str,
    ) -> List[SlackMessage]:
        """
        Get replies in a thread.

        Args:
            channel_id: ID of the channel.
            thread_ts: Timestamp of the parent message.

        Returns:
            List of SlackMessage objects (including parent message).
        """
        if self.config.mock_mode:
            return self._get_thread_replies_mock(channel_id, thread_ts)
        return self._get_thread_replies_real(channel_id, thread_ts)

    def _get_thread_replies_mock(
        self,
        channel_id: str,
        thread_ts: str,
    ) -> List[SlackMessage]:
        """Get thread replies from mock samples."""
        samples_dir = self._get_samples_dir()

        # Try to find thread file
        thread_file = samples_dir / channel_id / f"thread_{thread_ts}.json"
        if thread_file.exists():
            with open(thread_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                messages = data if isinstance(data, list) else data.get("messages", [])
                return [self._parse_message(m, channel_id) for m in messages]

        # Fallback: filter messages by thread_ts
        messages = self._get_messages_mock(channel_id)
        return [m for m in messages if m.thread_ts == thread_ts or m.ts == thread_ts]

    def _get_thread_replies_real(
        self,
        channel_id: str,
        thread_ts: str,
    ) -> List[SlackMessage]:
        """Get thread replies using Slack API."""
        messages = []
        cursor = None

        while True:
            params = {
                "channel": channel_id,
                "ts": thread_ts,
                "limit": self.config.limit,
            }
            if cursor:
                params["cursor"] = cursor

            data = self._api_call("conversations.replies", **params)

            for msg_data in data.get("messages", []):
                msg = self._parse_message(msg_data, channel_id)
                messages.append(msg)

            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor or not data.get("has_more"):
                break

        return messages

    def get_user(self, user_id: str) -> SlackUser:
        """
        Get user information.

        Args:
            user_id: ID of the user.

        Returns:
            SlackUser object.
        """
        # Check cache first
        if user_id in self._users_cache:
            return self._users_cache[user_id]

        if self.config.mock_mode:
            user = self._get_user_mock(user_id)
        else:
            user = self._get_user_real(user_id)

        self._users_cache[user_id] = user
        return user

    def _get_user_mock(self, user_id: str) -> SlackUser:
        """Get user from mock samples."""
        samples_dir = self._get_samples_dir()

        # Try to find users file
        users_file = samples_dir / "users.json"
        if users_file.exists():
            with open(users_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                users = data if isinstance(data, list) else data.get("members", [])
                for user_data in users:
                    if user_data.get("id") == user_id:
                        return self._parse_user(user_data)

        # Return a mock user
        return SlackUser(
            id=user_id,
            name=f"user_{user_id[-4:]}",
            real_name=f"Mock User {user_id[-4:]}",
        )

    def _get_user_real(self, user_id: str) -> SlackUser:
        """Get user using Slack API."""
        data = self._api_call("users.info", user=user_id)
        return self._parse_user(data.get("user", {}))

    def _parse_user(self, data: Dict[str, Any]) -> SlackUser:
        """Parse user data into SlackUser object."""
        profile = data.get("profile", {})
        return SlackUser(
            id=data.get("id", ""),
            name=data.get("name", ""),
            real_name=profile.get("real_name") or data.get("real_name"),
            display_name=profile.get("display_name"),
            is_bot=data.get("is_bot", False),
            is_admin=data.get("is_admin", False),
            deleted=data.get("deleted", False),
        )

    def stream_samples(
        self,
        channel_id: str,
        mapping: Dict[str, str],
        mode: str = "thread_qa",
        include_reactions: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream training samples from Slack messages.

        Supports multiple modes for creating training pairs:
        - "thread_qa": First message in thread is input, replies are outputs
        - "consecutive": Consecutive message pairs (msg1 -> msg2)
        - "reaction_filter": Messages with specific reactions

        Args:
            channel_id: ID of the channel.
            mapping: Column mapping dict with "input" and "output" keys.
            mode: Sample generation mode (thread_qa, consecutive, reaction_filter).
            include_reactions: Include reaction data in metadata.

        Yields:
            Training sample dicts with "input", "output", and "metadata" keys.
        """
        if mode == "thread_qa":
            yield from self._stream_thread_qa(channel_id, mapping, include_reactions)
        elif mode == "consecutive":
            yield from self._stream_consecutive(channel_id, mapping, include_reactions)
        elif mode == "reaction_filter":
            yield from self._stream_reaction_filter(channel_id, mapping, include_reactions)
        else:
            raise ValueError(f"Unknown streaming mode: {mode}")

    def _stream_thread_qa(
        self,
        channel_id: str,
        mapping: Dict[str, str],
        include_reactions: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """Stream samples using thread Q&A mode."""
        messages = self.get_messages(channel_id)

        # Find messages that are thread parents
        thread_parents = [m for m in messages if m.reply_count > 0]

        for parent in thread_parents:
            try:
                replies = self.get_thread_replies(channel_id, parent.ts)

                # Skip the parent message (first in replies)
                reply_texts = [r.text for r in replies if r.ts != parent.ts]

                if not reply_texts:
                    continue

                # Create sample with parent as input, combined replies as output
                row = {
                    "input": self._clean_message_text(parent.text),
                    "output": self._clean_message_text("\n".join(reply_texts)),
                }

                sample = row_to_sample(row, {"input": "input", "output": "output"})
                sample["metadata"]["source"] = "slack"
                sample["metadata"]["channel_id"] = channel_id
                sample["metadata"]["thread_ts"] = parent.ts
                sample["metadata"]["reply_count"] = len(reply_texts)

                if include_reactions:
                    sample["metadata"]["reactions"] = parent.reactions

                yield sample

            except Exception as e:
                logger.warning(f"Error processing thread {parent.ts}: {e}")
                continue

    def _stream_consecutive(
        self,
        channel_id: str,
        mapping: Dict[str, str],
        include_reactions: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """Stream samples using consecutive message pairs."""
        messages = self.get_messages(channel_id)

        # Sort by timestamp
        messages.sort(key=lambda m: float(m.ts))

        for i in range(len(messages) - 1):
            msg1 = messages[i]
            msg2 = messages[i + 1]

            # Skip if either message is too short
            if len(msg1.text) < self.config.min_message_length:
                continue
            if len(msg2.text) < self.config.min_message_length:
                continue

            row = {
                "input": self._clean_message_text(msg1.text),
                "output": self._clean_message_text(msg2.text),
            }

            sample = row_to_sample(row, {"input": "input", "output": "output"})
            sample["metadata"]["source"] = "slack"
            sample["metadata"]["channel_id"] = channel_id
            sample["metadata"]["input_ts"] = msg1.ts
            sample["metadata"]["output_ts"] = msg2.ts

            if include_reactions:
                sample["metadata"]["input_reactions"] = msg1.reactions
                sample["metadata"]["output_reactions"] = msg2.reactions

            yield sample

    def _stream_reaction_filter(
        self,
        channel_id: str,
        mapping: Dict[str, str],
        include_reactions: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """Stream samples using reaction-filtered messages."""
        messages = self.get_messages(channel_id)

        # Filter messages with reactions
        reacted_messages = [m for m in messages if m.reactions]

        for msg in reacted_messages:
            row = {
                "input": self._clean_message_text(msg.text),
                "output": "",  # Can be enriched with thread replies
            }

            # If message has thread replies, use them as output
            if msg.reply_count > 0:
                try:
                    replies = self.get_thread_replies(channel_id, msg.ts)
                    reply_texts = [r.text for r in replies if r.ts != msg.ts]
                    row["output"] = self._clean_message_text("\n".join(reply_texts))
                except Exception:
                    pass

            sample = row_to_sample(row, {"input": "input", "output": "output"})
            sample["metadata"]["source"] = "slack"
            sample["metadata"]["channel_id"] = channel_id
            sample["metadata"]["ts"] = msg.ts
            sample["metadata"]["reactions"] = msg.reactions

            yield sample

    def _clean_message_text(self, text: str) -> str:
        """Clean and normalize Slack message text."""
        if not text:
            return ""

        # Remove user mentions (<@U123456>)
        text = re.sub(r"<@[A-Z0-9]+>", "", text)

        # Remove channel references (<#C123456|channel-name>)
        text = re.sub(r"<#[A-Z0-9]+\|([^>]+)>", r"#\1", text)

        # Remove URLs but keep the text if provided (<http://url|text>)
        text = re.sub(r"<(https?://[^|>]+)\|([^>]+)>", r"\2", text)
        text = re.sub(r"<(https?://[^>]+)>", r"\1", text)

        # Remove special commands (<!here>, <!channel>, etc.)
        text = re.sub(r"<!(?:here|channel|everyone)[^>]*>", "", text)

        # Normalize whitespace
        text = " ".join(text.split())

        return text.strip()


def main() -> int:
    """CLI entry point for the Slack connector."""
    parser = argparse.ArgumentParser(
        description="Access messages and channels from Slack (or mock samples)."
    )
    parser.add_argument(
        "--channel-id",
        help="Slack channel ID to query.",
    )
    parser.add_argument(
        "--list-channels",
        action="store_true",
        help="List accessible channels.",
    )
    parser.add_argument(
        "--messages",
        action="store_true",
        help="Get messages from channel.",
    )
    parser.add_argument(
        "--thread-ts",
        help="Get replies for a specific thread.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream training samples from channel.",
    )
    parser.add_argument(
        "--mode",
        choices=["thread_qa", "consecutive", "reaction_filter"],
        default="thread_qa",
        help="Sample generation mode.",
    )
    parser.add_argument(
        "--mapping",
        default='{"input": "input", "output": "output"}',
        help='JSON mapping string for streaming samples.',
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of results.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode.",
    )

    args = parser.parse_args()

    config = SlackConfig(mock_mode=args.mock or True)
    connector = SlackConnector(config)

    try:
        if args.list_channels:
            channels = connector.list_channels()
            for ch in channels:
                print(json.dumps(ch.to_dict()))
            return 0

        if args.channel_id and args.messages:
            messages = connector.get_messages(args.channel_id)
            count = 0
            for msg in messages:
                print(json.dumps(msg.to_dict()))
                count += 1
                if args.limit and count >= args.limit:
                    break
            return 0

        if args.channel_id and args.thread_ts:
            replies = connector.get_thread_replies(args.channel_id, args.thread_ts)
            for reply in replies:
                print(json.dumps(reply.to_dict()))
            return 0

        if args.stream:
            if not args.channel_id:
                print("Error: --channel-id required for streaming", file=sys.stderr)
                return 1

            mapping = json.loads(args.mapping)
            count = 0
            for sample in connector.stream_samples(args.channel_id, mapping, mode=args.mode):
                print(json.dumps(sample))
                count += 1
                if args.limit and count >= args.limit:
                    break
            return 0

        # Default: show help
        parser.print_help()
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
