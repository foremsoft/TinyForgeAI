#!/usr/bin/env python3
"""
Deployment script for Customer Support Bot.

This script launches the inference server for the trained support bot model.

Usage:
    python deploy.py --model ./output/support_bot --port 8000
    python deploy.py --model ./output/support_bot --port 8000 --host 0.0.0.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ModelLoader:
    """Load and manage trained models for inference."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.config = None
        self._load_model()

    def _load_model(self):
        """Load the trained model."""
        # Check for model stub (dry-run mode)
        stub_path = self.model_path / "model_stub.json"
        if stub_path.exists():
            with open(stub_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            print(f"Loaded model stub: {self.config.get('model_name', 'unknown')}")
            print("Note: Running in stub mode - responses will be simulated")
            return

        # Try to load real model
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            print(f"Loading model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path))
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load real model ({e})")
            print("Running in stub mode")
            self.config = {"model_name": "stub", "stub_mode": True}

    def predict(
        self,
        text: str,
        max_length: int = 256,
        temperature: float = 0.7,
        num_beams: int = 4,
    ) -> str:
        """Generate a response for the input text."""
        if self.model is not None:
            # Real model inference
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )

            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                num_beams=num_beams,
                do_sample=temperature > 0,
                early_stopping=True,
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        else:
            # Stub mode - return a simulated response
            return self._stub_response(text)

    def _stub_response(self, text: str) -> str:
        """Generate a stub response for testing."""
        text_lower = text.lower()

        # Simple keyword matching for common support queries
        if "password" in text_lower or "reset" in text_lower:
            return (
                "To reset your password: 1) Click 'Forgot Password' on the login page, "
                "2) Enter your email address, 3) Check your inbox for the reset link, "
                "4) Click the link and create a new password. The link expires in 24 hours."
            )
        elif "cancel" in text_lower and "subscription" in text_lower:
            return (
                "To cancel your subscription: 1) Log into your account, "
                "2) Go to Settings > Subscription, 3) Click 'Cancel Subscription', "
                "4) Follow the prompts to confirm. Your access continues until the end of your billing period."
            )
        elif "charged" in text_lower or "billing" in text_lower or "refund" in text_lower:
            return (
                "I apologize for any billing issues. Please provide your account email "
                "and the last 4 digits of the card charged. We'll investigate and process "
                "any necessary refunds within 24 hours."
            )
        elif "hours" in text_lower or "contact" in text_lower or "support" in text_lower:
            return (
                "Our customer support team is available Monday through Friday, 9 AM to 6 PM EST. "
                "You can reach us at support@company.com, via live chat on our website, "
                "or by phone at 1-800-SUPPORT."
            )
        else:
            return (
                "Thank you for your question. I'd be happy to help you with that. "
                "Could you please provide more details so I can assist you better? "
                "You can also check our Help Center at help.company.com for immediate answers."
            )


def create_app(model_loader: ModelLoader):
    """Create FastAPI application."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(
        title="Customer Support Bot API",
        description="AI-powered customer support chatbot",
        version="1.0.0",
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class PredictRequest(BaseModel):
        input: str
        max_length: int = 256
        temperature: float = 0.7

    class PredictResponse(BaseModel):
        output: str
        model: str
        input_length: int

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        model_name: str | None = None

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=model_loader.model is not None or model_loader.config is not None,
            model_name=(
                model_loader.config.get("model_name")
                if model_loader.config
                else "real_model"
            ),
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        """Generate a support response."""
        if not request.input or not request.input.strip():
            raise HTTPException(status_code=400, detail="Input text is required")

        try:
            response = model_loader.predict(
                text=request.input,
                max_length=request.max_length,
                temperature=request.temperature,
            )

            return PredictResponse(
                output=response,
                model=(
                    model_loader.config.get("model_name", "stub")
                    if model_loader.config
                    else "real_model"
                ),
                input_length=len(request.input),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        return {
            "name": "Customer Support Bot API",
            "version": "1.0.0",
            "endpoints": {
                "/health": "Health check",
                "/predict": "Generate support response (POST)",
                "/docs": "API documentation",
            },
        }

    return app


def run_server(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
):
    """Run the inference server."""
    import uvicorn

    print("=" * 60)
    print("Customer Support Bot - Inference Server")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model_loader = ModelLoader(model_path)

    # Create app
    print("Creating API server...")
    app = create_app(model_loader)

    # Run server
    print(f"\nStarting server at http://{host}:{port}")
    print(f"API docs available at http://{host}:{port}/docs")
    print("\nPress Ctrl+C to stop the server\n")

    uvicorn.run(app, host=host, port=port, reload=reload)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy Customer Support Bot inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Deploy locally
    python deploy.py --model ./output/support_bot --port 8000

    # Deploy for external access
    python deploy.py --model ./output/support_bot --port 8000 --host 0.0.0.0

    # Deploy with auto-reload (development)
    python deploy.py --model ./output/support_bot --port 8000 --reload
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Verify model path exists
    model_path = Path(args.model)
    if not model_path.exists():
        parser.error(f"Model path not found: {args.model}")

    run_server(
        model_path=args.model,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
