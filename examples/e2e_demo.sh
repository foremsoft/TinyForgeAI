#!/bin/bash
# TinyForgeAI End-to-End Demo Script
# This script demonstrates the complete dry-run workflow:
# dataset -> train (dry-run) -> export microservice -> test inference
#
# Usage:
#   bash examples/e2e_demo.sh          # Run demo, keep artifacts
#   bash examples/e2e_demo.sh --cleanup # Run demo, then delete artifacts

set -euo pipefail

# Parse arguments
CLEANUP=false
for arg in "$@"; do
    case $arg in
        --cleanup)
            CLEANUP=true
            shift
            ;;
    esac
done

echo "=============================================="
echo "TinyForgeAI End-to-End Demo (Dry-Run Mode)"
echo "=============================================="
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Project root: $PROJECT_ROOT"
echo ""

# Set PYTHONPATH to include project root for module imports
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Create temporary workspace
WORKDIR=$(mktemp -d)
echo "Created workspace: $WORKDIR"
echo ""

# Step 1: Copy sample data
echo "[Step 1] Copying sample training data..."
cp "$PROJECT_ROOT/examples/sample_qna.jsonl" "$WORKDIR/data.jsonl"
echo "  -> Copied to: $WORKDIR/data.jsonl"
echo ""

# Step 2: Run dry-run trainer
echo "[Step 2] Running dry-run trainer..."
python "$PROJECT_ROOT/backend/training/train.py" \
    --data "$WORKDIR/data.jsonl" \
    --out "$WORKDIR/tiny_model" \
    --dry-run

if [ -f "$WORKDIR/tiny_model/model_stub.json" ]; then
    echo "  -> Model artifact created: $WORKDIR/tiny_model/model_stub.json"
else
    echo "ERROR: model_stub.json not created!"
    exit 1
fi
echo ""

# Step 3: Export to microservice
echo "[Step 3] Exporting to inference microservice..."
python "$PROJECT_ROOT/backend/exporter/builder.py" \
    --model-path "$WORKDIR/tiny_model/model_stub.json" \
    --output-dir "$WORKDIR/service" \
    --overwrite \
    --export-onnx

if [ -f "$WORKDIR/service/model_metadata.json" ] && [ -f "$WORKDIR/service/app.py" ]; then
    echo "  -> Service created at: $WORKDIR/service"
    echo "  -> Files: app.py, model_metadata.json, model_loader.py, schemas.py"
else
    echo "ERROR: Service files not created!"
    exit 1
fi
echo ""

# Step 4: Run smoke test against generated service (no uvicorn)
echo "[Step 4] Running smoke test against generated service..."

# Get the real Windows path for the service directory (handles Git Bash path translation)
SERVICE_DIR="$WORKDIR/service"

RESPONSE=$(python - "$SERVICE_DIR" <<'PY'
import sys
import os
import json
import importlib.util

# Get service directory from argument (handles path correctly on all platforms)
service_dir = sys.argv[1]

# On Windows with Git Bash, convert Unix-style path to Windows if needed
if sys.platform == "win32" and service_dir.startswith("/"):
    # Convert /c/path or /d/path to C:/path or D:/path
    if len(service_dir) >= 2 and service_dir[2] == '/':
        service_dir = service_dir[1].upper() + ":" + service_dir[2:]

# Ensure path exists
if not os.path.exists(service_dir):
    print(f"ERROR: Service directory not found: {service_dir}", file=sys.stderr)
    sys.exit(1)

# Add service directory to path for relative imports within app.py
sys.path.insert(0, service_dir)

# Load the app module dynamically
app_path = os.path.join(service_dir, "app.py")
spec = importlib.util.spec_from_file_location("service_app", app_path)
module = importlib.util.module_from_spec(spec)
sys.modules["service_app"] = module
spec.loader.exec_module(module)
app = module.app

# Use FastAPI TestClient
from fastapi.testclient import TestClient

client = TestClient(app)

# Test /health endpoint
health_response = client.get("/health")
assert health_response.status_code == 200, f"Health check failed: {health_response.status_code}"

# Test /predict endpoint
predict_response = client.post("/predict", json={"input": "hello"})
assert predict_response.status_code == 200, f"Predict failed: {predict_response.status_code}"

result = predict_response.json()
print(json.dumps(result, indent=2))
PY
)

echo "  -> Inference response:"
echo "$RESPONSE"
echo ""

# Step 5: Summary
echo "=============================================="
echo "Demo Complete!"
echo "=============================================="
echo ""
echo "Summary:"
echo "  Model path:     $WORKDIR/tiny_model/model_stub.json"
echo "  Service path:   $WORKDIR/service"
echo "  Response:       $RESPONSE"
echo ""

# Cleanup or show instructions
if [ "$CLEANUP" = true ]; then
    echo "Cleaning up workspace..."
    rm -rf "$WORKDIR"
    echo "  -> Deleted: $WORKDIR"
else
    echo "Workspace preserved at: $WORKDIR"
    echo "To delete manually:"
    echo "  rm -rf $WORKDIR"
fi

echo ""
echo "Done!"
