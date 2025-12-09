# TinyForgeAI Playground

An interactive Streamlit-based UI for testing TinyForgeAI inference services.

## Features

- Send requests to inference endpoints
- Health check monitoring
- Response visualization with confidence scores
- cURL command generation
- Example inputs for quick testing

## Quick Start

### Prerequisites

- Python 3.10+
- A running TinyForgeAI inference service

### Installation

```bash
cd playground
pip install -r requirements.txt
```

### Run the Playground

```bash
streamlit run app.py
```

The playground will open in your browser at http://localhost:8501

### Start an Inference Service

Before using the playground, start a TinyForgeAI service:

```bash
# From project root
foremforge train --data examples/data/demo_dataset.jsonl --out ./tmp/model --dry-run
foremforge export --model ./tmp/model/model_stub.json --out ./tmp/service
foremforge serve --dir ./tmp/service --port 8000
```

## Usage

1. **Configure Service URL**: Enter the URL of your inference endpoint (default: `http://127.0.0.1:8000/predict`)

2. **Check Health**: Click "Check Health" in the sidebar to verify the service is running

3. **Enter Input**: Type your text in the input area

4. **Send Request**: Click the "Send Request" button

5. **View Response**: See the model's output and confidence score

## Screenshots

```
┌──────────────────────────────────────────────────────────────┐
│  🔧 TinyForgeAI Playground                                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐           │
│  │ Input               │  │ Response            │           │
│  │                     │  │                     │           │
│  │ [What is your...]   │  │ Output: ...         │           │
│  │                     │  │ Confidence: 75%     │           │
│  │ [🚀 Send Request]   │  │ ████████░░ 75%     │           │
│  └─────────────────────┘  └─────────────────────┘           │
│                                                              │
│  📝 Example Inputs                                           │
│  [Hello...] [What is...] [Tell me...] [How do...]           │
│                                                              │
│  🔧 cURL Equivalent                                          │
│  curl -X POST http://localhost:8000/predict ...             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

- `STREAMLIT_SERVER_PORT`: Port for Streamlit (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Address to bind (default: localhost)

### Custom Configuration

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "0.0.0.0"

[theme]
primaryColor = "#4a4a6a"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```

## Troubleshooting

### Cannot connect to service

1. Verify the service is running:
   ```bash
   curl http://localhost:8000/health
   ```

2. Check the service URL in the sidebar

3. Ensure no firewall is blocking the connection

### Slow responses

- Increase the timeout in the sidebar slider
- Check service logs for performance issues

## See Also

- [API Reference](../docs/api_reference.md)
- [Deployment Guide](../deploy/README.md)
- [Dashboard](../dashboard/README.md)
