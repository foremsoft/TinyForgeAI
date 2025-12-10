"""
Database connector for TinyForgeAI.

Provides functionality to read rows from SQL databases and convert them
into training samples using the standard TinyForgeAI format.

Supports:
- SQLite (built-in)
- PostgreSQL (requires psycopg2)
- MySQL (requires mysql-connector-python)
"""

import argparse
import json
import logging
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

# Add project root to path when run as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from connectors.db_config import db_settings
from connectors.mappers import row_to_sample

logger = logging.getLogger(__name__)

# Check for optional database drivers
PSYCOPG2_AVAILABLE = False
MYSQL_AVAILABLE = False

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    pass

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    pass


class DBConnector:
    """
    Database connector for streaming training samples from SQL databases.

    Supports SQLite, PostgreSQL, and MySQL databases. The connector reads
    rows from the database and converts them to training samples using
    configurable column mappings.

    Connection URL formats:
    - SQLite: sqlite:///path/to/db.sqlite or sqlite:///:memory:
    - PostgreSQL: postgresql://user:password@host:port/database
    - MySQL: mysql://user:password@host:port/database
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
    ) -> None:
        """
        Initialize the database connector.

        Args:
            db_url: Database connection URL. If None, uses db_settings.DB_URL.
            pool_size: Connection pool size (for PostgreSQL/MySQL).
            max_overflow: Maximum overflow connections beyond pool_size.
        """
        self.db_url = db_url or db_settings.DB_URL
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self._db_type = self._detect_db_type()
        self._connection_pool: list = []
        logger.debug(f"Initialized DBConnector with {self._db_type} database")

    def _detect_db_type(self) -> str:
        """Detect database type from URL."""
        url = self.db_url.lower()
        if url.startswith("postgresql://") or url.startswith("postgres://"):
            return "postgresql"
        elif url.startswith("mysql://"):
            return "mysql"
        elif url.startswith("sqlite://"):
            return "sqlite"
        else:
            # Default to SQLite for backward compatibility
            return "sqlite"

    def _parse_connection_url(self) -> Dict[str, Any]:
        """Parse connection URL into components."""
        url = self.db_url

        if self._db_type == "sqlite":
            # sqlite:///path or sqlite:///:memory:
            if url.startswith("sqlite:///"):
                path = url[len("sqlite:///"):]
            elif url.startswith("sqlite://"):
                path = url[len("sqlite://"):]
            else:
                path = url
            return {"database": path}

        # Parse PostgreSQL/MySQL URLs
        # Format: driver://user:password@host:port/database
        import re

        pattern = r"(?P<driver>\w+)://(?:(?P<user>[^:@]+)(?::(?P<password>[^@]+))?@)?(?P<host>[^:/]+)(?::(?P<port>\d+))?/(?P<database>\w+)"
        match = re.match(pattern, url)

        if not match:
            raise ValueError(f"Invalid database URL format: {url}")

        result = match.groupdict()
        if result.get("port"):
            result["port"] = int(result["port"])

        return result

    @contextmanager
    def _connect(self):
        """
        Create a database connection as a context manager.

        Yields:
            Database connection with dict-like row access.
        """
        conn = None
        try:
            if self._db_type == "sqlite":
                conn = self._connect_sqlite()
            elif self._db_type == "postgresql":
                conn = self._connect_postgresql()
            elif self._db_type == "mysql":
                conn = self._connect_mysql()
            else:
                raise ValueError(f"Unsupported database type: {self._db_type}")

            yield conn
        finally:
            if conn:
                conn.close()

    def _connect_sqlite(self) -> sqlite3.Connection:
        """Create SQLite connection."""
        params = self._parse_connection_url()
        conn = sqlite3.connect(params["database"])
        conn.row_factory = sqlite3.Row
        return conn

    def _connect_postgresql(self):
        """Create PostgreSQL connection."""
        if not PSYCOPG2_AVAILABLE:
            raise ImportError(
                "PostgreSQL support requires psycopg2. "
                "Install with: pip install psycopg2-binary"
            )

        params = self._parse_connection_url()
        conn = psycopg2.connect(
            host=params.get("host", "localhost"),
            port=params.get("port", 5432),
            database=params["database"],
            user=params.get("user"),
            password=params.get("password"),
            cursor_factory=psycopg2.extras.RealDictCursor,
        )
        return conn

    def _connect_mysql(self):
        """Create MySQL connection."""
        if not MYSQL_AVAILABLE:
            raise ImportError(
                "MySQL support requires mysql-connector-python. "
                "Install with: pip install mysql-connector-python"
            )

        params = self._parse_connection_url()
        conn = mysql.connector.connect(
            host=params.get("host", "localhost"),
            port=params.get("port", 3306),
            database=params["database"],
            user=params.get("user"),
            password=params.get("password"),
        )
        return conn

    def test_connection(self) -> bool:
        """
        Test if the database connection is working.

        Returns:
            True if connection succeeds and SELECT 1 works, False otherwise.
        """
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()

                # Handle different result formats
                if isinstance(result, dict):
                    return list(result.values())[0] == 1
                elif result:
                    return result[0] == 1
                return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def execute(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
    ) -> list:
        """
        Execute a query and return all results.

        Args:
            query: SQL query to execute.
            params: Query parameters (tuple for positional, dict for named).

        Returns:
            List of row dictionaries.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            rows = cursor.fetchall()

            # Convert to list of dicts for consistent interface
            if self._db_type == "sqlite":
                return [dict(row) for row in rows]
            elif self._db_type == "postgresql":
                return [dict(row) for row in rows]
            elif self._db_type == "mysql":
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]

            return rows

    def execute_many(
        self,
        query: str,
        params_list: list,
    ) -> int:
        """
        Execute a query multiple times with different parameters.

        Args:
            query: SQL query to execute.
            params_list: List of parameter tuples/dicts.

        Returns:
            Number of affected rows.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount

    def stream_samples(
        self,
        query: str,
        mapping: dict,
        batch_size: int = 100,
        params: Optional[Union[tuple, dict]] = None,
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
            params: Optional query parameters.

        Yields:
            Training sample dicts with "input", "output", and "metadata" keys.
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                for row in rows:
                    # Convert to dict if needed
                    if self._db_type == "mysql":
                        columns = [desc[0] for desc in cursor.description]
                        row_dict = dict(zip(columns, row))
                    elif isinstance(row, dict):
                        row_dict = row
                    else:
                        row_dict = dict(row)

                    yield row_to_sample(row_dict, mapping)

    def get_table_info(self, table_name: str) -> list:
        """
        Get column information for a table.

        Args:
            table_name: Name of the table.

        Returns:
            List of column info dictionaries.
        """
        if self._db_type == "sqlite":
            query = f"PRAGMA table_info({table_name})"
            rows = self.execute(query)
            return [
                {
                    "name": row["name"],
                    "type": row["type"],
                    "nullable": not row["notnull"],
                    "primary_key": bool(row["pk"]),
                }
                for row in rows
            ]

        elif self._db_type == "postgresql":
            query = """
                SELECT column_name as name, data_type as type,
                       is_nullable = 'YES' as nullable
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """
            rows = self.execute(query, (table_name,))
            return rows

        elif self._db_type == "mysql":
            query = f"DESCRIBE {table_name}"
            rows = self.execute(query)
            return [
                {
                    "name": row["Field"],
                    "type": row["Type"],
                    "nullable": row["Null"] == "YES",
                    "primary_key": row["Key"] == "PRI",
                }
                for row in rows
            ]

        return []

    def list_tables(self) -> list:
        """
        List all tables in the database.

        Returns:
            List of table names.
        """
        if self._db_type == "sqlite":
            query = "SELECT name FROM sqlite_master WHERE type='table'"
        elif self._db_type == "postgresql":
            query = """
                SELECT table_name as name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """
        elif self._db_type == "mysql":
            query = "SHOW TABLES"
        else:
            return []

        rows = self.execute(query)

        # Extract table names from result
        if rows and isinstance(rows[0], dict):
            key = list(rows[0].keys())[0]
            return [row[key] for row in rows]
        return [row[0] for row in rows]


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
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test database connection and exit.",
    )
    parser.add_argument(
        "--list-tables",
        action="store_true",
        help="List all tables and exit.",
    )

    args = parser.parse_args()

    try:
        mapping = json.loads(args.mapping)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON mapping: {e}", file=sys.stderr)
        return 1

    connector = DBConnector(db_url=args.db_url)

    if args.test:
        if connector.test_connection():
            print("Connection successful!")
            return 0
        else:
            print("Connection failed!", file=sys.stderr)
            return 1

    if args.list_tables:
        tables = connector.list_tables()
        for table in tables:
            print(table)
        return 0

    count = 0
    for sample in connector.stream_samples(args.query, mapping):
        print(json.dumps(sample))
        count += 1
        if args.limit and count >= args.limit:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
