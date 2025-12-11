"""
Playground Server - Self-hosted model playground.

Creates a local web server with a beautiful UI for testing models.
Supports optional sharing via ngrok/cloudflare tunnels.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

# HTML template for the playground UI
PLAYGROUND_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - TinyForge Playground</title>
    <style>
        :root {{
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --bg: #0f172a;
            --bg-card: #1e293b;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
            --success: #22c55e;
            --error: #ef4444;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}

        header {{
            background: var(--bg-card);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .logo {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .logo-icon {{
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--primary), #a855f7);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.25rem;
        }}

        .logo-text {{
            font-size: 1.25rem;
            font-weight: 600;
        }}

        .badge {{
            background: var(--primary);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
        }}

        main {{
            flex: 1;
            padding: 2rem;
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
        }}

        .model-info {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}

        .model-info h1 {{
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }}

        .model-info p {{
            color: var(--text-muted);
            font-size: 0.875rem;
        }}

        .model-meta {{
            display: flex;
            gap: 1.5rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }}

        .meta-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
            color: var(--text-muted);
        }}

        .meta-item span {{
            color: var(--text);
            font-weight: 500;
        }}

        .playground-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }}

        .input-section {{
            padding: 1.5rem;
            border-bottom: 1px solid var(--border);
        }}

        .input-section label {{
            display: block;
            font-size: 0.875rem;
            font-weight: 500;
            margin-bottom: 0.75rem;
            color: var(--text-muted);
        }}

        textarea {{
            width: 100%;
            min-height: 120px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            color: var(--text);
            font-size: 1rem;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.2s;
        }}

        textarea:focus {{
            outline: none;
            border-color: var(--primary);
        }}

        .actions {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            background: rgba(0,0,0,0.2);
        }}

        .btn {{
            background: var(--primary);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .btn:hover {{
            background: var(--primary-dark);
        }}

        .btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}

        .btn-secondary {{
            background: transparent;
            border: 1px solid var(--border);
        }}

        .btn-secondary:hover {{
            background: var(--bg);
        }}

        .output-section {{
            padding: 1.5rem;
        }}

        .output-section label {{
            display: block;
            font-size: 0.875rem;
            font-weight: 500;
            margin-bottom: 0.75rem;
            color: var(--text-muted);
        }}

        .output-box {{
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            min-height: 120px;
            white-space: pre-wrap;
            font-size: 1rem;
            line-height: 1.6;
        }}

        .output-box.empty {{
            color: var(--text-muted);
            font-style: italic;
        }}

        .output-box.error {{
            border-color: var(--error);
            color: var(--error);
        }}

        .metrics {{
            display: flex;
            gap: 1.5rem;
            margin-top: 1rem;
            font-size: 0.875rem;
            color: var(--text-muted);
        }}

        .metric {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .metric-value {{
            color: var(--success);
            font-weight: 500;
        }}

        footer {{
            background: var(--bg-card);
            border-top: 1px solid var(--border);
            padding: 1rem 2rem;
            text-align: center;
            font-size: 0.875rem;
            color: var(--text-muted);
        }}

        footer a {{
            color: var(--primary);
            text-decoration: none;
        }}

        footer a:hover {{
            text-decoration: underline;
        }}

        .loading {{
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid transparent;
            border-top-color: currentColor;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        .examples {{
            margin-top: 1rem;
        }}

        .examples-title {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        .example-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .example-chip {{
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 9999px;
            padding: 0.375rem 0.75rem;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .example-chip:hover {{
            border-color: var(--primary);
            color: var(--primary);
        }}

        @media (max-width: 640px) {{
            header {{
                padding: 1rem;
            }}

            main {{
                padding: 1rem;
            }}

            .model-meta {{
                flex-direction: column;
                gap: 0.5rem;
            }}

            .actions {{
                flex-direction: column;
                gap: 1rem;
            }}

            .btn {{
                width: 100%;
                justify-content: center;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <div class="logo-icon">TF</div>
            <div class="logo-text">TinyForge Playground</div>
        </div>
        <div class="badge">{model_type}</div>
    </header>

    <main>
        <div class="model-info">
            <h1>{title}</h1>
            <p>{description}</p>
            <div class="model-meta">
                <div class="meta-item">Model: <span>{model_name}</span></div>
                <div class="meta-item">Type: <span>{task_type}</span></div>
                {extra_meta}
            </div>
        </div>

        <div class="playground-card">
            <div class="input-section">
                <label for="input">Your Input</label>
                <textarea id="input" placeholder="{placeholder}">{default_input}</textarea>

                {examples_html}
            </div>

            <div class="actions">
                <button class="btn btn-secondary" onclick="clearAll()">Clear</button>
                <button class="btn" id="submitBtn" onclick="runInference()">
                    <span id="btnText">Run Model</span>
                    <span id="btnLoading" class="loading" style="display: none;"></span>
                </button>
            </div>

            <div class="output-section">
                <label>Model Output</label>
                <div class="output-box empty" id="output">Output will appear here...</div>
                <div class="metrics" id="metrics" style="display: none;">
                    <div class="metric">Latency: <span class="metric-value" id="latency">-</span></div>
                    <div class="metric">Tokens: <span class="metric-value" id="tokens">-</span></div>
                </div>
            </div>
        </div>
    </main>

    <footer>
        Powered by <a href="https://github.com/foremsoft/TinyForgeAI" target="_blank">TinyForgeAI</a>
        {expiry_notice}
    </footer>

    <script>
        const API_URL = '{api_url}';

        async function runInference() {{
            const input = document.getElementById('input').value.trim();
            if (!input) {{
                alert('Please enter some text');
                return;
            }}

            const btn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const btnLoading = document.getElementById('btnLoading');
            const output = document.getElementById('output');
            const metrics = document.getElementById('metrics');

            btn.disabled = true;
            btnText.textContent = 'Running...';
            btnLoading.style.display = 'inline-block';
            output.className = 'output-box empty';
            output.textContent = 'Processing...';

            const startTime = performance.now();

            try {{
                const response = await fetch(API_URL + '/infer', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ input: input }}),
                }});

                const endTime = performance.now();
                const latency = Math.round(endTime - startTime);

                if (!response.ok) {{
                    throw new Error('Inference failed: ' + response.status);
                }}

                const data = await response.json();

                output.className = 'output-box';
                output.textContent = data.output || data.result || JSON.stringify(data);

                document.getElementById('latency').textContent = latency + 'ms';
                document.getElementById('tokens').textContent = data.tokens || '-';
                metrics.style.display = 'flex';

            }} catch (error) {{
                output.className = 'output-box error';
                output.textContent = 'Error: ' + error.message;
                metrics.style.display = 'none';
            }} finally {{
                btn.disabled = false;
                btnText.textContent = 'Run Model';
                btnLoading.style.display = 'none';
            }}
        }}

        function clearAll() {{
            document.getElementById('input').value = '';
            document.getElementById('output').className = 'output-box empty';
            document.getElementById('output').textContent = 'Output will appear here...';
            document.getElementById('metrics').style.display = 'none';
        }}

        function useExample(text) {{
            document.getElementById('input').value = text;
        }}

        // Allow Ctrl+Enter to submit
        document.getElementById('input').addEventListener('keydown', function(e) {{
            if (e.ctrlKey && e.key === 'Enter') {{
                runInference();
            }}
        }});
    </script>
</body>
</html>
'''


@dataclass
class PlaygroundConfig:
    """Configuration for the playground server."""

    # Model settings
    model_path: str = ""
    model_name: str = "TinyForge Model"
    model_type: str = "Q&A"
    task_type: str = "question-answering"
    description: str = "Test this model by entering your input below."

    # UI settings
    title: str = "Model Playground"
    placeholder: str = "Enter your question or text here..."
    default_input: str = ""
    examples: List[str] = field(default_factory=list)

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080

    # Sharing settings
    share: bool = False
    share_duration: int = 72  # hours
    share_provider: str = "ngrok"  # ngrok, cloudflare, or localtunnel


class PlaygroundServer:
    """
    Shareable playground server for TinyForge models.

    Usage:
        server = PlaygroundServer(config)
        server.start()  # Starts server at http://localhost:8080

        # Or with sharing
        server.start(share=True)  # Returns public URL
    """

    def __init__(self, config: Optional[PlaygroundConfig] = None):
        """
        Initialize the playground server.

        Args:
            config: Playground configuration.
        """
        self.config = config or PlaygroundConfig()
        self._model = None
        self._tokenizer = None
        self._inference_fn: Optional[Callable] = None
        self._app = None
        self._public_url: Optional[str] = None

    def set_inference_function(self, fn: Callable[[str], str]) -> None:
        """
        Set a custom inference function.

        Args:
            fn: Function that takes input string and returns output string.
        """
        self._inference_fn = fn

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load a trained model.

        Args:
            model_path: Path to the model directory.
        """
        model_path = Path(model_path or self.config.model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Check for stub model
        stub_path = model_path / "model_stub.json"
        if stub_path.exists():
            logger.info("Loading stub model (dry-run mode)")
            self._setup_stub_inference(stub_path)
            return

        # Try to load real model
        try:
            from backend.training.real_trainer import RealTrainer
            self._model = RealTrainer.load_trained_model(model_path)

            def inference_fn(text: str) -> str:
                return self._model.predict(text)

            self._inference_fn = inference_fn
            logger.info(f"Loaded real model from {model_path}")

        except ImportError:
            logger.warning("Real trainer not available, using stub inference")
            self._setup_stub_inference(stub_path if stub_path.exists() else None)

    def _setup_stub_inference(self, stub_path: Optional[Path] = None) -> None:
        """Setup stub inference for dry-run models."""
        stub_data = {}
        if stub_path and stub_path.exists():
            with open(stub_path) as f:
                stub_data = json.load(f)

        def stub_inference(text: str) -> str:
            # Simple echo response for stub
            return f"[Stub Response] Received: {text[:100]}..."

        self._inference_fn = stub_inference
        logger.info("Using stub inference (model not trained)")

    def _create_app(self):
        """Create the FastAPI application."""
        try:
            from fastapi import FastAPI, Request
            from fastapi.responses import HTMLResponse, JSONResponse
            from fastapi.middleware.cors import CORSMiddleware
        except ImportError:
            raise ImportError("FastAPI required: pip install fastapi uvicorn")

        app = FastAPI(title="TinyForge Playground")

        # CORS for cross-origin requests
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        config = self.config

        @app.get("/", response_class=HTMLResponse)
        async def home():
            # Generate examples HTML
            examples_html = ""
            if config.examples:
                chips = "".join(
                    f'<span class="example-chip" onclick="useExample(\'{ex}\')">{ex[:40]}...</span>'
                    if len(ex) > 40 else
                    f'<span class="example-chip" onclick="useExample(\'{ex}\')">{ex}</span>'
                    for ex in config.examples[:5]
                )
                examples_html = f'''
                <div class="examples">
                    <div class="examples-title">Try these examples</div>
                    <div class="example-chips">{chips}</div>
                </div>
                '''

            # Expiry notice for shared playgrounds
            expiry_notice = ""
            if config.share:
                expiry_notice = f" | Link expires in {config.share_duration} hours"

            html = PLAYGROUND_HTML.format(
                title=config.title,
                description=config.description,
                model_name=config.model_name,
                model_type=config.model_type,
                task_type=config.task_type,
                placeholder=config.placeholder,
                default_input=config.default_input,
                examples_html=examples_html,
                extra_meta="",
                api_url="",
                expiry_notice=expiry_notice,
            )
            return HTMLResponse(content=html)

        @app.post("/infer")
        async def infer(request: Request):
            data = await request.json()
            input_text = data.get("input", "")

            if not input_text:
                return JSONResponse(
                    {"error": "No input provided"},
                    status_code=400
                )

            if not self._inference_fn:
                return JSONResponse(
                    {"error": "Model not loaded"},
                    status_code=500
                )

            try:
                start_time = time.time()
                output = self._inference_fn(input_text)
                latency = int((time.time() - start_time) * 1000)

                return JSONResponse({
                    "output": output,
                    "latency_ms": latency,
                    "tokens": len(output.split()),
                })

            except Exception as e:
                logger.error(f"Inference error: {e}")
                return JSONResponse(
                    {"error": str(e)},
                    status_code=500
                )

        @app.get("/health")
        async def health():
            return {"status": "ok", "model_loaded": self._inference_fn is not None}

        self._app = app
        return app

    def start(
        self,
        share: Optional[bool] = None,
        blocking: bool = True,
    ) -> Optional[str]:
        """
        Start the playground server.

        Args:
            share: Whether to create a public share link.
            blocking: Whether to block (True) or run in background (False).

        Returns:
            Public URL if sharing, None otherwise.
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError("uvicorn required: pip install uvicorn")

        # Load model if not already loaded
        if self._inference_fn is None and self.config.model_path:
            self.load_model()

        # Create app
        app = self._create_app()

        share = share if share is not None else self.config.share

        # Setup sharing if requested
        if share:
            self._public_url = self._setup_sharing()
            if self._public_url:
                print(f"\n{'='*50}")
                print(f"Playground is live!")
                print(f"Local:  http://{self.config.host}:{self.config.port}")
                print(f"Public: {self._public_url}")
                print(f"{'='*50}\n")

        else:
            print(f"\nPlayground running at: http://localhost:{self.config.port}\n")

        if blocking:
            uvicorn.run(
                app,
                host=self.config.host,
                port=self.config.port,
                log_level="info",
            )
        else:
            # Run in background thread
            import threading
            thread = threading.Thread(
                target=uvicorn.run,
                args=(app,),
                kwargs={
                    "host": self.config.host,
                    "port": self.config.port,
                    "log_level": "warning",
                },
                daemon=True,
            )
            thread.start()

        return self._public_url

    def _setup_sharing(self) -> Optional[str]:
        """Setup public sharing via tunnel."""
        provider = self.config.share_provider

        if provider == "ngrok":
            return self._setup_ngrok()
        elif provider == "cloudflare":
            return self._setup_cloudflare()
        elif provider == "localtunnel":
            return self._setup_localtunnel()
        else:
            logger.warning(f"Unknown share provider: {provider}")
            return None

    def _setup_ngrok(self) -> Optional[str]:
        """Setup ngrok tunnel."""
        try:
            import ngrok

            # Start tunnel
            listener = ngrok.forward(
                self.config.port,
                authtoken_from_env=True,
            )
            url = listener.url()
            logger.info(f"ngrok tunnel created: {url}")
            return url

        except ImportError:
            logger.warning("ngrok not installed: pip install ngrok")
            return self._fallback_share_instructions()
        except Exception as e:
            logger.error(f"ngrok setup failed: {e}")
            return self._fallback_share_instructions()

    def _setup_cloudflare(self) -> Optional[str]:
        """Setup Cloudflare tunnel."""
        # Cloudflare requires cloudflared to be installed separately
        logger.info("Cloudflare tunnel requires 'cloudflared' CLI")
        print("\nTo share via Cloudflare, run in another terminal:")
        print(f"  cloudflared tunnel --url http://localhost:{self.config.port}")
        return None

    def _setup_localtunnel(self) -> Optional[str]:
        """Setup localtunnel."""
        logger.info("localtunnel requires 'lt' CLI (npm install -g localtunnel)")
        print("\nTo share via localtunnel, run in another terminal:")
        print(f"  lt --port {self.config.port}")
        return None

    def _fallback_share_instructions(self) -> Optional[str]:
        """Show fallback sharing instructions."""
        print("\n" + "="*50)
        print("Quick sharing options:")
        print("="*50)
        print(f"\n1. ngrok (recommended):")
        print(f"   pip install ngrok")
        print(f"   ngrok http {self.config.port}")
        print(f"\n2. Cloudflare Tunnel:")
        print(f"   cloudflared tunnel --url http://localhost:{self.config.port}")
        print(f"\n3. localtunnel:")
        print(f"   npx localtunnel --port {self.config.port}")
        print("="*50 + "\n")
        return None

    def get_public_url(self) -> Optional[str]:
        """Get the public URL if sharing is enabled."""
        return self._public_url


def create_playground(
    model_path: str,
    title: str = "Model Playground",
    description: str = "Test this model",
    examples: Optional[List[str]] = None,
    port: int = 8080,
    share: bool = False,
) -> PlaygroundServer:
    """
    Convenience function to create and start a playground.

    Args:
        model_path: Path to the trained model.
        title: Playground title.
        description: Model description.
        examples: Example inputs.
        port: Port to run on.
        share: Whether to create a public link.

    Returns:
        PlaygroundServer instance.
    """
    config = PlaygroundConfig(
        model_path=model_path,
        title=title,
        description=description,
        examples=examples or [],
        port=port,
        share=share,
    )

    server = PlaygroundServer(config)
    return server


def main():
    """CLI entry point for playground server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Start a shareable playground for your model"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--title", "-t",
        default="TinyForge Playground",
        help="Playground title"
    )
    parser.add_argument(
        "--description", "-d",
        default="Test this model by entering your input below.",
        help="Model description"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Port to run on (default: 8080)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--share-provider",
        choices=["ngrok", "cloudflare", "localtunnel"],
        default="ngrok",
        help="Sharing provider (default: ngrok)"
    )
    parser.add_argument(
        "--examples",
        nargs="+",
        help="Example inputs to show"
    )

    args = parser.parse_args()

    config = PlaygroundConfig(
        model_path=args.model,
        title=args.title,
        description=args.description,
        port=args.port,
        share=args.share,
        share_provider=args.share_provider,
        examples=args.examples or [],
    )

    server = PlaygroundServer(config)
    server.start()


if __name__ == "__main__":
    main()
