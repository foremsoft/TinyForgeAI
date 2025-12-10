"""
Async Database connector for TinyForgeAI.

Provides asynchronous functionality to read rows from SQL databases and convert
them into training samples using the standard TinyForgeAI format.

Supports:
- SQLite (requires aiosqlite)
- PostgreSQL (requires asyncpg)
- MySQL (requires aiomysql)
"""

import asyncio
import json
import logging
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

# Add project root to path when run as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from connectors.db_config import db_settings
from connectors.mappers import row_to_sample

logger = logging.getLogger(__name__)

# Check for optional async database drivers
AIOSQLITE_AVAILABLE = False
ASYNCPG_AVAILABLE = False
AIOMYSQL_AVAILABLE = False

try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    pass

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    pass

try:
    import aiomysql
    AIOMYSQL_AVAILABLE = True
except ImportError:
    pass


class AsyncDBConnector:
    """
    Async database connector for streaming training samples from SQL databases.

    Supports SQLite, PostgreSQL, and MySQL databases with async operations.
    The connector reads rows from the database and converts them to training
    samples using configurable column mappings.

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
        Initialize the async database connector.

        Args:
            db_url: Database connection URL. If None, uses db_settings.DB_URL.
            pool_size: Connection pool size (for PostgreSQL/MySQL).
            max_overflow: Maximum overflow connections beyond pool_size.
        """
        self.db_url = db_url or db_settings.DB_URL
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self._db_type = self._detect_db_type()
        self._pool = None
        logger.debug(f"Initialized AsyncDBConnector with {self._db_type} database")

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
        pattern = r"(?P<driver>\w+)://(?:(?P<user>[^:@]+)(?::(?P<password>[^@]+))?@)?(?P<host>[^:/]+)(?::(?P<port>\d+))?/(?P<database>\w+)"
        match = re.match(pattern, url)

        if not match:
            raise ValueError(f"Invalid database URL format: {url}")

        result = match.groupdict()
        if result.get("port"):
            result["port"] = int(result["port"])

        return result

    async def __aenter__(self) -> "AsyncDBConnector":
        """Async context manager entry."""
        await self._init_pool()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _init_pool(self) -> None:
        """Initialize connection pool."""
        if self._pool is not None:
            return

        if self._db_type == "postgresql":
            if not ASYNCPG_AVAILABLE:
                raise ImportError(
                    "Async PostgreSQL support requires asyncpg. "
                    "Install with: pip install asyncpg"
                )
            params = self._parse_connection_url()
            self._pool = await asyncpg.create_pool(
                host=params.get("host", "localhost"),
                port=params.get("port", 5432),
                database=params["database"],
                user=params.get("user"),
                password=params.get("password"),
                min_size=1,
                max_size=self.pool_size,
            )
            logger.debug(f"Created asyncpg pool with {self.pool_size} connections")

        elif self._db_type == "mysql":
            if not AIOMYSQL_AVAILABLE:
                raise ImportError(
                    "Async MySQL support requires aiomysql. "
                    "Install with: pip install aiomysql"
                )
            params = self._parse_connection_url()
            self._pool = await aiomysql.create_pool(
                host=params.get("host", "localhost"),
                port=params.get("port", 3306),
                db=params["database"],
                user=params.get("user"),
                password=params.get("password"),
                minsize=1,
                maxsize=self.pool_size,
            )
            logger.debug(f"Created aiomysql pool with {self.pool_size} connections")

        elif self._db_type == "sqlite":
            if not AIOSQLITE_AVAILABLE:
                raise ImportError(
                    "Async SQLite support requires aiosqlite. "
                    "Install with: pip install aiosqlite"
                )
            # SQLite doesn't use pooling in the same way
            self._pool = "sqlite"
            logger.debug("Initialized aiosqlite connector")

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool is None:
            return

        if self._db_type == "postgresql" and self._pool:
            await self._pool.close()
            logger.debug("Closed asyncpg pool")
        elif self._db_type == "mysql" and self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            logger.debug("Closed aiomysql pool")

        self._pool = None

    @asynccontextmanager
    async def _connect(self):
        """
        Create a database connection as an async context manager.

        Yields:
            Database connection with dict-like row access.
        """
        await self._init_pool()

        if self._db_type == "sqlite":
            params = self._parse_connection_url()
            async with aiosqlite.connect(params["database"]) as conn:
                conn.row_factory = aiosqlite.Row
                yield conn

        elif self._db_type == "postgresql":
            async with self._pool.acquire() as conn:
                yield conn

        elif self._db_type == "mysql":
            async with self._pool.acquire() as conn:
                yield conn

    async def test_connection(self) -> bool:
        """
        Test if the database connection is working.

        Returns:
            True if connection succeeds and SELECT 1 works, False otherwise.
        """
        try:
            async with self._connect() as conn:
                if self._db_type == "sqlite":
                    async with conn.execute("SELECT 1") as cursor:
                        result = await cursor.fetchone()
                        return result[0] == 1
                elif self._db_type == "postgresql":
                    result = await conn.fetchval("SELECT 1")
                    return result == 1
                elif self._db_type == "mysql":
                    async with conn.cursor() as cursor:
                        await cursor.execute("SELECT 1")
                        result = await cursor.fetchone()
                        return result[0] == 1
            return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    async def execute(
        self,
        query: str,
        params: Optional[Union[tuple, dict, list]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a query and return all results.

        Args:
            query: SQL query to execute.
            params: Query parameters (tuple for positional, dict for named).

        Returns:
            List of row dictionaries.
        """
        async with self._connect() as conn:
            if self._db_type == "sqlite":
                if params:
                    async with conn.execute(query, params) as cursor:
                        rows = await cursor.fetchall()
                        columns = [desc[0] for desc in cursor.description]
                else:
                    async with conn.execute(query) as cursor:
                        rows = await cursor.fetchall()
                        columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]

            elif self._db_type == "postgresql":
                if params:
                    rows = await conn.fetch(query, *params if isinstance(params, (list, tuple)) else params)
                else:
                    rows = await conn.fetch(query)
                return [dict(row) for row in rows]

            elif self._db_type == "mysql":
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    if params:
                        await cursor.execute(query, params)
                    else:
                        await cursor.execute(query)
                    rows = await cursor.fetchall()
                    return list(rows)

        return []

    async def execute_many(
        self,
        query: str,
        params_list: List[Union[tuple, dict]],
    ) -> int:
        """
        Execute a query multiple times with different parameters.

        Args:
            query: SQL query to execute.
            params_list: List of parameter tuples/dicts.

        Returns:
            Number of affected rows.
        """
        async with self._connect() as conn:
            if self._db_type == "sqlite":
                await conn.executemany(query, params_list)
                await conn.commit()
                return len(params_list)

            elif self._db_type == "postgresql":
                await conn.executemany(query, params_list)
                return len(params_list)

            elif self._db_type == "mysql":
                async with conn.cursor() as cursor:
                    await cursor.executemany(query, params_list)
                    await conn.commit()
                    return cursor.rowcount

        return 0

    async def stream_samples(
        self,
        query: str,
        mapping: dict,
        batch_size: int = 100,
        params: Optional[Union[tuple, dict]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream training samples from the database asynchronously.

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
        async with self._connect() as conn:
            if self._db_type == "sqlite":
                if params:
                    async with conn.execute(query, params) as cursor:
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                        while True:
                            rows = await cursor.fetchmany(batch_size)
                            if not rows:
                                break
                            for row in rows:
                                row_dict = dict(zip(columns, row))
                                yield row_to_sample(row_dict, mapping)
                else:
                    async with conn.execute(query) as cursor:
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                        while True:
                            rows = await cursor.fetchmany(batch_size)
                            if not rows:
                                break
                            for row in rows:
                                row_dict = dict(zip(columns, row))
                                yield row_to_sample(row_dict, mapping)

            elif self._db_type == "postgresql":
                # PostgreSQL with asyncpg - use cursor for streaming
                async with conn.transaction():
                    if params:
                        cursor = await conn.cursor(query, *params if isinstance(params, (list, tuple)) else params)
                    else:
                        cursor = await conn.cursor(query)

                    while True:
                        rows = await cursor.fetch(batch_size)
                        if not rows:
                            break
                        for row in rows:
                            row_dict = dict(row)
                            yield row_to_sample(row_dict, mapping)

            elif self._db_type == "mysql":
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    if params:
                        await cursor.execute(query, params)
                    else:
                        await cursor.execute(query)

                    while True:
                        rows = await cursor.fetchmany(batch_size)
                        if not rows:
                            break
                        for row in rows:
                            yield row_to_sample(dict(row), mapping)

    async def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get column information for a table.

        Args:
            table_name: Name of the table.

        Returns:
            List of column info dictionaries.
        """
        if self._db_type == "sqlite":
            query = f"PRAGMA table_info({table_name})"
            rows = await self.execute(query)
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
                WHERE table_name = $1
                ORDER BY ordinal_position
            """
            rows = await self.execute(query, (table_name,))
            return rows

        elif self._db_type == "mysql":
            query = f"DESCRIBE {table_name}"
            rows = await self.execute(query)
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

    async def list_tables(self) -> List[str]:
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

        rows = await self.execute(query)

        # Extract table names from result
        if rows and isinstance(rows[0], dict):
            key = list(rows[0].keys())[0]
            return [row[key] for row in rows]
        return []


async def main() -> int:
    """Async CLI entry point for the database connector."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stream training samples from a database (async)."
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

    async with AsyncDBConnector(db_url=args.db_url) as connector:
        if args.test:
            if await connector.test_connection():
                print("Connection successful!")
                return 0
            else:
                print("Connection failed!", file=sys.stderr)
                return 1

        if args.list_tables:
            tables = await connector.list_tables()
            for table in tables:
                print(table)
            return 0

        count = 0
        async for sample in connector.stream_samples(args.query, mapping):
            print(json.dumps(sample))
            count += 1
            if args.limit and count >= args.limit:
                break

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
