Google Drive Connector Sample Files
====================================

This directory contains sample files for the Google Drive connector's mock mode.
When GOOGLE_DRIVE_MOCK=true (default), the connector reads from this directory
instead of making real Google Drive API calls.

Files:
- training_data.jsonl: JSONL format training data with input/output pairs
- qa_pairs.json: JSON format question/answer pairs

Usage:
    from connectors.google_drive_connector import GoogleDriveConnector

    # Mock mode (reads from this directory)
    connector = GoogleDriveConnector()
    files = connector.list_files()

    # Stream training samples
    mapping = {"input": "input", "output": "output"}
    for sample in connector.stream_samples("", mapping):
        print(sample)
