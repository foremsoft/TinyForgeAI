#!/usr/bin/env bash
set -euo pipefail

# TinyForgeAI Documentation Deployment Script
# Builds MkDocs documentation locally

echo "=== TinyForgeAI Docs Builder ==="
echo ""

# Check if mkdocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo "mkdocs not found. Installing..."
    pip install mkdocs mkdocs-material
fi

# Build docs
echo "Building documentation..."
mkdocs build --site-dir site_docs

echo ""
echo "Built docs at site_docs/"
echo ""
echo "Next steps:"
echo "  - Push to main branch to trigger GitHub Pages deploy"
echo "  - Or run: netlify deploy --dir=site_docs --prod"
echo "  - Or preview locally: mkdocs serve"
