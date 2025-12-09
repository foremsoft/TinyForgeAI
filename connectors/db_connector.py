"""
Database connector for TinyForgeAI.

Provides functionality to read rows from SQL databases and convert them
into training samples using the standard TinyForgeAI format.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Iterator, Optional

# Add project root to path when run as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from connectors.db_config import db_settings
from connectors.mappers import row_to_sample


class DBConnector:
    """
    Database connector for streaming training samples from SQL databases.

    Uses Python's sqlite3 module for SQLite databases. The connector reads
    rows from the database and converts them to training samples using
    configurable column mappings.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        """
        Initialize the database connector.

        Args:
            db_url: Database connection URL. If None, uses db_settings.DB_URL.
                    For SQLite, use format: sqlite:///path/to/db.sqlite
                    or sqlite:///:memory: for in-memory database.
        """
        self.db_url = db_url or db_settings.DB_URL
        self._connection: Optional[sqlite3.Connection] = None

    def _get_sqlite_path(self) -> str:
        """Extract the file path from a SQLite URL."""
        url = self.db_url
        if url.startswith("sqlite:///"):
            return url[len("sqlite:///"):]
        elif url.startswith("sqlite://"):
            return url[len("sqlite://"):]
        return url

    def _connect(self) -> sqlite3.Connection:
        """Create a database connection."""
        path = self._get_sqlite_path()
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn

    def test_connection(self) -> bool:
        """
        Test if the database connection is working.

        Returns:
            True if connection succeeds and SELECT 1 works, False otherwise.
        """
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            conn.close()
            return result is not None and result[0] == 1
        except Exception:
            return False

    def stream_samples(
        self,
        query: str,
        mapping: dict,
        batch_size: int = 100,
    ) -> Iterator[dict]:
        """
        Stream training samples from the database.

        Executes the provided SQL query and yields training samples
        converted using the specified column mapping.

        Args:
            query: SQL query to execute.
            mapping: Column mapping dict with "input" and "output" keys.
                     Example: {"input": "question", "output": "answer"}
            batch_size: Number of rows to fetch at a time for memory efficiency.

        Yields:
            Training sample dicts with "input", "output", and "metadata" keys.
        """
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(query)

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            for row in rows:
                yield row_to_sample(row, mapping)

        conn.close()


def main() -> int:
    """CLI entry point for the database connector."""
    parser = argparse.ArgumentParser(
        description="Stream training samples from a database."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="SQL query to execute.",
    )
    parser.add_argument(
        "--mapping",
        required=True,
        help='JSON mapping string, e.g., \'{"input":"q","output":"a"}\'',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to output.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Database URL (defaults to DB_URL from environment).",
    )

    args = parser.parse_args()

    try:
        mapping = json.loads(args.mapping)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON mapping: {e}", file=sys.stderr)
        return 1

    connector = DBConnector(db_url=args.db_url)

    count = 0
    for sample in connector.stream_samples(args.query, mapping):
        print(json.dumps(sample))
        count += 1
        if args.limit and count >= args.limit:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
