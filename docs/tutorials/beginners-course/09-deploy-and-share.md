# Module 9: Deploy & Share Your AI

**Time needed:** 25 minutes
**Prerequisites:** Module 8 (tested model)
**Goal:** Put your AI online for others to use

---

## Deployment Options

```
┌─────────────────────────────────────────────────────────────┐
│                  Deployment Options                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Level 1: Local API (localhost)                            │
│   └── Test on your computer                                 │
│                                                             │
│   Level 2: Docker Container                                 │
│   └── Works anywhere Docker runs                            │
│                                                             │
│   Level 3: Cloud Deployment                                 │
│   └── Railway, Render, Fly.io (free tiers)                  │
│                                                             │
│   Level 4: Production                                       │
│   └── AWS, GCP, Azure (enterprise)                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Level 1: Create a REST API

### Production-Ready API

```python
# api_server.py - Production-ready API for your trained model

"""
REST API for your TinyForgeAI trained model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime
import os


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.getenv("MODEL_PATH", "./my_faq_model")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))


# ============================================================
# Model Loading
# ============================================================

class ModelService:
    """Handles model loading and inference."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, model_path: str):
        """Load the model."""
        if self._initialized:
            return

        print(f"Loading model from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Load label mapping
        with open(f"{model_path}/label_mapping.json", 'r') as f:
            mapping = json.load(f)
            self.id2label = {int(k): v for k, v in mapping['id2label'].items()}

        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

        self._initialized = True
        print(f"✓ Model loaded on {self.device}")

    def predict(self, text: str) -> dict:
        """Get prediction for text."""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        top_prob, top_idx = torch.max(probs, dim=-1)

        # Get top 3 predictions
        top_k = torch.topk(probs[0], k=min(3, len(self.id2label)))
        alternatives = [
            {"answer": self.id2label[idx.item()], "confidence": prob.item()}
            for prob, idx in zip(top_k.values, top_k.indices)
        ]

        return {
            "answer": self.id2label[top_idx.item()],
            "confidence": top_prob.item(),
            "alternatives": alternatives
        }


# Initialize model service
model_service = ModelService()


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="TinyForgeAI FAQ Bot",
    description="AI-powered FAQ chatbot API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str

    class Config:
        json_schema_extra = {"example": {"message": "What are your hours?"}}


class PredictionResult(BaseModel):
    answer: str
    confidence: float


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    alternatives: List[PredictionResult]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str


# Statistics
stats = {
    "requests": 0,
    "start_time": datetime.now().isoformat()
}


# ============================================================
# Endpoints
# ============================================================

@app.on_event("startup")
async def startup():
    """Initialize model on startup."""
    model_service.initialize(MODEL_PATH)


@app.get("/", tags=["General"])
def home():
    """API information."""
    return {
        "name": "TinyForgeAI FAQ Bot API",
        "docs": "/docs",
        "endpoints": {
            "POST /chat": "Ask a question",
            "GET /health": "Health check",
            "GET /stats": "Usage statistics"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_service._initialized,
        device=model_service.device if model_service._initialized else "not loaded",
        version="1.0.0"
    )


@app.get("/stats", tags=["General"])
def get_stats():
    """Usage statistics."""
    return {
        **stats,
        "uptime": stats["start_time"]
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Ask a question and get an AI-powered answer.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    stats["requests"] += 1

    result = model_service.predict(request.message)

    return ChatResponse(
        answer=result["answer"],
        confidence=round(result["confidence"], 4),
        alternatives=[
            PredictionResult(
                answer=alt["answer"],
                confidence=round(alt["confidence"], 4)
            )
            for alt in result["alternatives"]
        ],
        timestamp=datetime.now().isoformat()
    )


@app.post("/batch", tags=["Chat"])
def batch_chat(messages: List[str]):
    """Process multiple messages at once."""
    if len(messages) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 messages per batch")

    results = []
    for msg in messages:
        if msg.strip():
            result = model_service.predict(msg)
            results.append({
                "message": msg,
                "answer": result["answer"],
                "confidence": round(result["confidence"], 4)
            })

    stats["requests"] += len(results)
    return {"results": results, "count": len(results)}


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
```

### Run Locally

```bash
# Install dependencies
pip install fastapi uvicorn

# Run the server
python api_server.py
```

Visit http://localhost:8000/docs to see the interactive API documentation!

---

## Level 2: Docker Container

### Create Dockerfile

```dockerfile
# Dockerfile - Container for your AI

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY api_server.py .
COPY my_faq_model/ ./my_faq_model/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Create requirements.txt

```
fastapi>=0.100.0
uvicorn>=0.20.0
torch>=2.0.0
transformers>=4.30.0
pydantic>=2.0.0
```

### Build and Run

```bash
# Build the image
docker build -t my-faq-bot .

# Run the container
docker run -p 8000:8000 my-faq-bot

# Run in background
docker run -d -p 8000:8000 --name faq-bot my-faq-bot
```

---

## Level 3: Cloud Deployment (Free)

### Option A: Railway.app (Easiest)

1. **Create a GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/faq-bot.git
   git push -u origin main
   ```

2. **Deploy to Railway**
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway auto-detects Python and deploys!

3. **Add Environment Variables** (if needed)
   - `MODEL_PATH`: Path to model directory

### Option B: Render.com

1. **Create `render.yaml`**
   ```yaml
   services:
     - type: web
       name: faq-bot
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: uvicorn api_server:app --host 0.0.0.0 --port $PORT
       envVars:
         - key: MODEL_PATH
           value: ./my_faq_model
   ```

2. **Deploy**
   - Go to [render.com](https://render.com)
   - Connect your GitHub repo
   - Render will auto-deploy!

### Option C: Fly.io

1. **Install flyctl**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Create `fly.toml`**
   ```toml
   app = "my-faq-bot"
   primary_region = "ord"

   [build]
     builder = "paketobuildpacks/builder:base"

   [http_service]
     internal_port = 8000
     force_https = true
     auto_stop_machines = true
     auto_start_machines = true

   [[services]]
     internal_port = 8000
     protocol = "tcp"
   ```

3. **Deploy**
   ```bash
   fly auth login
   fly launch
   fly deploy
   ```

---

## Level 4: Create a Web Interface

### Simple Chat Interface

```html
<!-- index.html - Chat interface -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FAQ Bot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .chat-container {
            width: 100%;
            max-width: 500px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }
        .chat-header h1 { font-size: 1.5rem; margin-bottom: 5px; }
        .chat-header p { opacity: 0.9; font-size: 0.9rem; }
        .chat-messages {
            height: 400px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-end;
        }
        .message.user { justify-content: flex-end; }
        .message-content {
            max-width: 80%;
            padding: 12px 18px;
            border-radius: 20px;
            line-height: 1.4;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 5px;
        }
        .message.bot .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .confidence {
            font-size: 0.75rem;
            opacity: 0.7;
            margin-top: 5px;
        }
        .chat-input {
            padding: 20px;
            background: white;
            display: flex;
            gap: 10px;
            border-top: 1px solid #eee;
        }
        .chat-input input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e9ecef;
            border-radius: 30px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s;
        }
        .chat-input input:focus { border-color: #667eea; }
        .chat-input button {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 30px;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .chat-input button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .chat-input button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .typing {
            display: flex;
            gap: 5px;
            padding: 15px;
        }
        .typing span {
            width: 10px;
            height: 10px;
            background: #667eea;
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
            <h1>🤖 FAQ Bot</h1>
            <p>Powered by TinyForgeAI</p>
        </div>
        <div class="chat-messages" id="messages">
            <div class="message bot">
                <div class="message-content">
                    Hello! I'm your AI assistant. Ask me anything about our products and services!
                </div>
            </div>
        </div>
        <div class="chat-input">
            <input type="text" id="userInput" placeholder="Type your question..."
                   onkeypress="if(event.key === 'Enter') sendMessage()">
            <button onclick="sendMessage()" id="sendBtn">Send</button>
        </div>
    </div>

    <script>
        // Change this to your deployed API URL
        const API_URL = 'http://localhost:8000';

        const messagesContainer = document.getElementById('messages');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');

        function addMessage(content, isUser, confidence = null) {
            const div = document.createElement('div');
            div.className = `message ${isUser ? 'user' : 'bot'}`;

            let html = `<div class="message-content">${escapeHtml(content)}`;
            if (confidence !== null && !isUser) {
                html += `<div class="confidence">Confidence: ${(confidence * 100).toFixed(0)}%</div>`;
            }
            html += '</div>';

            div.innerHTML = html;
            messagesContainer.appendChild(div);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function addTyping() {
            const div = document.createElement('div');
            div.className = 'message bot';
            div.id = 'typing';
            div.innerHTML = `<div class="message-content typing">
                <span></span><span></span><span></span>
            </div>`;
            messagesContainer.appendChild(div);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function removeTyping() {
            const typing = document.getElementById('typing');
            if (typing) typing.remove();
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            // Add user message
            addMessage(message, true);
            userInput.value = '';
            sendBtn.disabled = true;
            addTyping();

            try {
                const response = await fetch(`${API_URL}/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });

                const data = await response.json();
                removeTyping();
                addMessage(data.answer, false, data.confidence);

            } catch (error) {
                removeTyping();
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

### Serve the Interface

Add to your `api_server.py`:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/app")
def serve_app():
    return FileResponse("static/index.html")
```

Create `static/` folder and put `index.html` there.

---

## Production Checklist

```
┌─────────────────────────────────────────────────────────────┐
│              Production Deployment Checklist                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Security                                                   │
│  □ Use HTTPS                                                │
│  □ Add API authentication                                   │
│  □ Rate limit requests                                      │
│  □ Validate all inputs                                      │
│  □ Don't expose sensitive data                              │
│                                                             │
│  Performance                                                │
│  □ Use GPU if available                                     │
│  □ Implement caching                                        │
│  □ Set reasonable timeouts                                  │
│  □ Monitor response times                                   │
│                                                             │
│  Reliability                                                │
│  □ Health check endpoint                                    │
│  □ Automatic restarts                                       │
│  □ Error logging                                            │
│  □ Backup model files                                       │
│                                                             │
│  Monitoring                                                 │
│  □ Request logging                                          │
│  □ Error tracking                                           │
│  □ Usage statistics                                         │
│  □ Alerting setup                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Adding API Authentication

```python
# Add to api_server.py

from fastapi.security import APIKeyHeader
from fastapi import Security

API_KEY = os.getenv("API_KEY", "your-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

@app.post("/chat")
def chat(request: ChatRequest, api_key: str = Security(verify_api_key)):
    # ... rest of function
```

---

## Checkpoint Quiz

**1. What is the benefit of using Docker for deployment?**
<details>
<summary>Click for answer</summary>

Docker containers include everything your app needs to run, ensuring it works the same way everywhere. No "works on my machine" problems. Easy to deploy to any cloud provider.

</details>

**2. Why add a health check endpoint?**
<details>
<summary>Click for answer</summary>

Health checks let monitoring systems verify your service is running properly. Cloud platforms use them to restart crashed containers. Load balancers use them to route traffic away from unhealthy instances.

</details>

**3. Why is HTTPS important for production?**
<details>
<summary>Click for answer</summary>

HTTPS encrypts data between user and server. Without it, API keys and user messages could be intercepted. Most cloud platforms provide HTTPS automatically.

</details>

---

## What's Next?

In **Module 10: Next Steps**, you'll:
- Learn advanced TinyForgeAI features
- Explore larger models
- Discover additional resources
- Plan your next AI project

**Your AI is live! Let's see what else you can build.**

---

[← Back to Module 8](08-test-and-improve.md) | [Continue to Module 10 →](10-next-steps.md)
