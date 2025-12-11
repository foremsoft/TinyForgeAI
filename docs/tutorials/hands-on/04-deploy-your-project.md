# Deploy Your AI Project to Production

**Time needed:** 20 minutes
**Skill level:** Beginner to Intermediate
**What you'll learn:** How to deploy your AI model so others can use it

---

## What We're Building

Turn your local AI project into a service anyone can access:

```
Local Development          Production Deployment
     ↓                           ↓
Your Computer    →    Cloud Server (always running)
http://localhost       https://your-app.com
Only you can use       Anyone can use
```

---

## Prerequisites

- Completed [03-train-your-model.md](03-train-your-model.md)
- A trained model or FAQ bot ready

---

## Option 1: Run Locally (Easiest)

### Step 1.1: Create a Production-Ready API

```python
# production_api.py - Production-ready API server

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from difflib import SequenceMatcher
from datetime import datetime

app = FastAPI(
    title="My AI Assistant API",
    description="A custom AI assistant trained on my data",
    version="1.0.0"
)

# Allow web browsers to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, list specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load knowledge base
KNOWLEDGE_BASE = []
knowledge_file = os.getenv("KNOWLEDGE_FILE", "train_data.jsonl")
if os.path.exists(knowledge_file):
    with open(knowledge_file, "r") as f:
        for line in f:
            KNOWLEDGE_BASE.append(json.loads(line))
    print(f"Loaded {len(KNOWLEDGE_BASE)} knowledge items")

# Track usage statistics
STATS = {"requests": 0, "start_time": datetime.now().isoformat()}

# Request/Response models
class ChatRequest(BaseModel):
    message: str

    class Config:
        json_schema_extra = {
            "example": {"message": "What is TinyForgeAI?"}
        }

class ChatResponse(BaseModel):
    response: str
    confidence: float
    source: str

class HealthResponse(BaseModel):
    status: str
    knowledge_items: int
    total_requests: int
    uptime_since: str

# Endpoints
@app.get("/", tags=["General"])
def home():
    """Welcome message and API information."""
    return {
        "message": "Welcome to My AI Assistant API!",
        "docs": "/docs",
        "endpoints": {
            "POST /chat": "Ask a question",
            "GET /health": "Check API health",
            "GET /stats": "View usage statistics"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """Check if the API is running properly."""
    return HealthResponse(
        status="healthy",
        knowledge_items=len(KNOWLEDGE_BASE),
        total_requests=STATS["requests"],
        uptime_since=STATS["start_time"]
    )

@app.get("/stats", tags=["General"])
def stats():
    """Get usage statistics."""
    return {
        "total_requests": STATS["requests"],
        "knowledge_items": len(KNOWLEDGE_BASE),
        "uptime_since": STATS["start_time"]
    }

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Ask a question and get an AI-powered response.

    The AI searches through its knowledge base to find
    the best matching answer for your question.
    """
    STATS["requests"] += 1

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if not KNOWLEDGE_BASE:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")

    # Find best matching response
    best_match = None
    best_score = 0

    for item in KNOWLEDGE_BASE:
        score = SequenceMatcher(
            None,
            request.message.lower(),
            item["input"].lower()
        ).ratio()
        if score > best_score:
            best_score = score
            best_match = item

    if best_score > 0.4 and best_match:
        return ChatResponse(
            response=best_match["output"],
            confidence=round(best_score, 2),
            source="knowledge_base"
        )
    else:
        return ChatResponse(
            response="I don't have specific information about that. Please try rephrasing your question or contact support for help.",
            confidence=0,
            source="default"
        )

# Run with: uvicorn production_api:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 1.2: Run Your Server

```bash
# Install dependencies
pip install fastapi uvicorn

# Run the server
python production_api.py
```

Now visit: http://localhost:8000/docs

You'll see an interactive API documentation page!

---

## Option 2: Deploy with Docker

### Step 2.1: Create Dockerfile

```dockerfile
# Dockerfile - Package your AI app

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY production_api.py .
COPY train_data.jsonl .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "production_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 2.2: Create requirements.txt

```
# requirements.txt
fastapi>=0.100.0
uvicorn>=0.20.0
pydantic>=2.0.0
```

### Step 2.3: Build and Run

```bash
# Build the Docker image
docker build -t my-ai-assistant .

# Run the container
docker run -p 8000:8000 my-ai-assistant

# Run in background
docker run -d -p 8000:8000 --name ai-assistant my-ai-assistant
```

---

## Option 3: Deploy to Cloud (Free Tiers)

### Deploy to Railway (Easiest)

1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub"
3. Connect your repository
4. Railway auto-detects and deploys!

### Deploy to Render

1. Go to [render.com](https://render.com)
2. Create new "Web Service"
3. Connect your GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn production_api:app --host 0.0.0.0 --port $PORT`

### Deploy to Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Create fly.toml
cat > fly.toml << 'EOF'
app = "my-ai-assistant"
primary_region = "ord"

[build]
  builder = "paketobuildpacks/builder:base"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true

[[services]]
  http_checks = []
  internal_port = 8000
  protocol = "tcp"
EOF

# Deploy
flyctl launch
flyctl deploy
```

---

## Option 4: Create a Web Interface

### Step 4.1: Simple HTML Chat Interface

```html
<!-- index.html - Chat interface for your AI -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .chat-container {
            width: 100%;
            max-width: 600px;
            height: 80vh;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
        }
        .chat-header {
            padding: 20px;
            background: #007bff;
            color: white;
            border-radius: 12px 12px 0 0;
            text-align: center;
        }
        .chat-header h1 {
            font-size: 1.5rem;
            margin-bottom: 5px;
        }
        .chat-header p {
            opacity: 0.9;
            font-size: 0.9rem;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
        }
        .message.user {
            justify-content: flex-end;
        }
        .message-content {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.4;
        }
        .message.user .message-content {
            background: #007bff;
            color: white;
        }
        .message.bot .message-content {
            background: #e9ecef;
            color: #333;
        }
        .confidence {
            font-size: 0.75rem;
            opacity: 0.7;
            margin-top: 5px;
        }
        .chat-input {
            padding: 20px;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
        }
        .chat-input input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s;
        }
        .chat-input input:focus {
            border-color: #007bff;
        }
        .chat-input button {
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        .chat-input button:hover {
            background: #0056b3;
        }
        .chat-input button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .typing {
            display: flex;
            gap: 4px;
            padding: 12px 16px;
        }
        .typing span {
            width: 8px;
            height: 8px;
            background: #999;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }
        .typing span:nth-child(1) { animation-delay: -0.32s; }
        .typing span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>AI Assistant</h1>
            <p>Ask me anything about our products and services</p>
        </div>

        <div class="chat-messages" id="messages">
            <div class="message bot">
                <div class="message-content">
                    Hello! I'm your AI assistant. How can I help you today?
                </div>
            </div>
        </div>

        <div class="chat-input">
            <input type="text" id="userInput" placeholder="Type your message..."
                   onkeypress="if(event.key === 'Enter') sendMessage()">
            <button onclick="sendMessage()" id="sendBtn">Send</button>
        </div>
    </div>

    <script>
        const API_URL = 'http://localhost:8000';  // Change for production
        const messagesContainer = document.getElementById('messages');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');

        function addMessage(content, isUser, confidence = null) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;

            let html = `<div class="message-content">${content}`;
            if (confidence !== null && !isUser) {
                html += `<div class="confidence">Confidence: ${(confidence * 100).toFixed(0)}%</div>`;
            }
            html += '</div>';

            messageDiv.innerHTML = html;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function addTypingIndicator() {
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message bot';
            typingDiv.id = 'typing';
            typingDiv.innerHTML = `
                <div class="message-content typing">
                    <span></span><span></span><span></span>
                </div>
            `;
            messagesContainer.appendChild(typingDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function removeTypingIndicator() {
            const typing = document.getElementById('typing');
            if (typing) typing.remove();
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            // Add user message
            addMessage(message, true);
            userInput.value = '';
            sendBtn.disabled = true;

            // Show typing indicator
            addTypingIndicator();

            try {
                const response = await fetch(`${API_URL}/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });

                const data = await response.json();
                removeTypingIndicator();
                addMessage(data.response, false, data.confidence);

            } catch (error) {
                removeTypingIndicator();
                addMessage('Sorry, I encountered an error. Please try again.', false);
                console.error('Error:', error);
            }

            sendBtn.disabled = false;
            userInput.focus();
        }
    </script>
</body>
</html>
```

### Step 4.2: Serve the Interface

Add this to your `production_api.py`:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/app")
def serve_app():
    return FileResponse("static/index.html")
```

Create the static folder and move your HTML:

```bash
mkdir static
mv index.html static/
```

Now visit: http://localhost:8000/app

---

## Option 5: Monitoring & Logging

### Add Request Logging

```python
# Add to production_api.py

import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()

    response = await call_next(request)

    duration = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"{request.method} {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Duration: {duration:.3f}s"
    )

    return response
```

---

## Deployment Checklist

Before going live, ensure:

- [ ] **Security**: Remove debug mode, use HTTPS
- [ ] **Environment Variables**: Don't hardcode secrets
- [ ] **Error Handling**: Graceful error messages
- [ ] **Rate Limiting**: Prevent abuse
- [ ] **Monitoring**: Log requests and errors
- [ ] **Backup**: Keep copies of your knowledge base
- [ ] **Testing**: Test all endpoints before launch

### Security Best Practices

```python
# Add rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/chat")
@limiter.limit("10/minute")  # Max 10 requests per minute per IP
def chat(request: ChatRequest):
    ...
```

---

## Summary

| Deployment Option | Best For | Difficulty |
|------------------|----------|------------|
| Local (uvicorn) | Development & Testing | Easy |
| Docker | Consistent deployments | Medium |
| Railway/Render | Quick cloud deploy | Easy |
| Fly.io | Global distribution | Medium |
| AWS/GCP/Azure | Enterprise scale | Advanced |

---

## What You've Learned

1. Creating a production-ready API
2. Adding interactive documentation
3. Containerizing with Docker
4. Deploying to cloud platforms
5. Building a web chat interface
6. Monitoring and logging

---

## Next Steps

- Add user authentication
- Connect to a real AI model (not just similarity matching)
- Set up continuous deployment (CI/CD)
- Add analytics to understand user questions
- Create a mobile app

---

**Congratulations!** Your AI assistant is now live and ready to help users!
