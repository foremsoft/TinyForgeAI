Notion Connector Sample Files
=============================

This directory contains sample files for the Notion connector's mock mode.
When NOTION_MOCK=true (default), the connector reads from this directory
instead of making real Notion API calls.

Files:
- training_database.json: Mock Notion database with Q&A pages

Usage:
    from connectors.notion_connector import NotionConnector

    # Mock mode (reads from this directory)
    connector = NotionConnector()
    pages = connector.list_pages(database_id="training_database")

    # Stream training samples
    mapping = {"input": "Question", "output": "Answer"}
    for sample in connector.stream_samples("training_database", mapping):
        print(sample)

Real API Usage:
    Set NOTION_MOCK=false and NOTION_API_TOKEN to your integration token.
