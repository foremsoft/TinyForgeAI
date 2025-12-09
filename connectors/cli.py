"""
CLI for TinyForgeAI database connectors.

Provides command-line utilities for streaming data from databases
and converting it to JSONL training format.
"""

import json
import sys
from pathlib import Path

# Add project root to path when run as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import click

from connectors.db_connector import DBConnector


@click.group()
def cli():
    """TinyForgeAI Database Connector CLI."""
    pass


@cli.command("db-stream")
@click.option(
    "--query",
    required=True,
    help="SQL query to execute.",
)
@click.option(
    "--mapping",
    required=True,
    help='JSON mapping string, e.g., \'{"input":"q","output":"a"}\'',
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit number of samples to output.",
)
@click.option(
    "--db-url",
    default=None,
    help="Database URL (defaults to DB_URL from environment).",
)
def db_stream(query: str, mapping: str, limit: int, db_url: str):
    """
    Stream training samples from a database as JSONL.

    Executes the given SQL query, maps columns to input/output fields,
    and prints each sample as a JSON line to stdout.

    Example:
        python connectors/cli.py db-stream \\
            --query "SELECT question AS q, answer AS a FROM qa" \\
            --mapping '{"input":"q","output":"a"}'
    """
    try:
        mapping_dict = json.loads(mapping)
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON mapping: {e}", err=True)
        sys.exit(1)

    connector = DBConnector(db_url=db_url)

    count = 0
    for sample in connector.stream_samples(query, mapping_dict):
        click.echo(json.dumps(sample))
        count += 1
        if limit and count >= limit:
            break


@cli.command("test-connection")
@click.option(
    "--db-url",
    default=None,
    help="Database URL to test.",
)
def test_connection(db_url: str):
    """
    Test database connection.

    Attempts to connect to the database and execute a simple query.
    """
    connector = DBConnector(db_url=db_url)
    if connector.test_connection():
        click.echo("Connection successful!")
    else:
        click.echo("Connection failed!", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
