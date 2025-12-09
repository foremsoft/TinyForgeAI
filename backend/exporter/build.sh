#!/bin/bash
# CLI wrapper for the TinyForgeAI exporter builder
# Usage: ./build.sh --model-path <path> --output-dir <dir> [--overwrite]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/builder.py" "$@"
