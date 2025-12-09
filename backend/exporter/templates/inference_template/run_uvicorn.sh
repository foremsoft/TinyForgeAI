#!/bin/bash
# Run the inference server with uvicorn
uvicorn app:app --host 0.0.0.0 --port ${INFERENCE_PORT:-8000}
