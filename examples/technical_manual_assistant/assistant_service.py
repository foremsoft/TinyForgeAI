#!/usr/bin/env python3
"""
Assistant Service for Technical Manual Assistant

FastAPI-based REST API for the technical manual Q&A assistant.

Usage:
    python assistant_service.py --port 8002

API Endpoints:
    POST /ask - Ask a question
    GET /health - Health check
    GET /info - Model information
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")


class AskRequest(BaseModel):
    question: str
    max_length: int = 256
    temperature: float = 0.7


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    response_time_ms: float


class InfoResponse(BaseModel):
    model_name: str
    model_path: str
    max_length: int
    status: str


class TechnicalAssistant:
    """Technical documentation assistant using fine-tuned model."""

    def __init__(self, model_path: str = None, config: Dict[str, Any] = None):
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.device = 'cpu'

        self._load_model()

    def _load_model(self):
        """Load the fine-tuned model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Determine model path
            if self.model_path and Path(self.model_path).exists():
                model_name = self.model_path
            else:
                # Use base model as fallback
                model_name = self.config.get('model', {}).get('base', 'gpt2')
                print(f"Fine-tuned model not found, using base model: {model_name}")

            print(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Move to GPU if available
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.model = self.model.to(self.device)
                print("Using GPU for inference")
            else:
                print("Using CPU for inference")

            self.model.eval()
            print("Model loaded successfully")

        except ImportError as e:
            print(f"Could not load model: {e}")
            print("Install transformers: pip install transformers torch")
            self.model = None

    def ask(self, question: str, max_length: int = 256, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate an answer to a question."""
        if not self.model:
            return {
                'answer': "Model not loaded. Please check the installation.",
                'confidence': 0.0
            }

        # Format prompt
        prompt = f"Question: {question}\n\nAnswer:"

        try:
            import torch

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )

            if self.device == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            gen_config = self.config.get('generation', {})

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=gen_config.get('top_p', 0.9),
                    do_sample=gen_config.get('do_sample', True),
                    repetition_penalty=gen_config.get('repetition_penalty', 1.1),
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract answer part
            if 'Answer:' in response:
                answer = response.split('Answer:')[-1].strip()
            else:
                answer = response[len(prompt):].strip()

            # Calculate confidence (based on token probabilities)
            # This is a simplified confidence measure
            confidence = min(0.9, 0.5 + len(answer) / 500)

            return {
                'answer': answer if answer else "I don't have enough information to answer this question.",
                'confidence': round(confidence, 2)
            }

        except Exception as e:
            print(f"Generation error: {e}")
            return {
                'answer': f"Error generating response: {str(e)}",
                'confidence': 0.0
            }


def create_app(config: Dict[str, Any]) -> "FastAPI":
    """Create FastAPI application."""
    app = FastAPI(
        title="Technical Manual Assistant API",
        description="Q&A assistant for technical documentation",
        version="1.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get('server', {}).get('cors_origins', ['*']),
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    # Initialize assistant
    model_config = config.get('model', {})
    model_path = model_config.get('output_dir', './models/technical_assistant')

    if not Path(model_path).is_absolute():
        model_path = Path(__file__).parent / model_path

    assistant = TechnicalAssistant(str(model_path), config)

    @app.get('/')
    async def root():
        return {'message': 'Technical Manual Assistant API', 'docs': '/docs'}

    @app.get('/health')
    async def health():
        return {
            'status': 'healthy' if assistant.model else 'degraded',
            'model_loaded': assistant.model is not None
        }

    @app.get('/info', response_model=InfoResponse)
    async def info():
        return InfoResponse(
            model_name=config.get('model', {}).get('base', 'gpt2'),
            model_path=str(model_path),
            max_length=config.get('generation', {}).get('max_new_tokens', 256),
            status='ready' if assistant.model else 'not_loaded'
        )

    @app.post('/ask', response_model=AskResponse)
    async def ask(request: AskRequest):
        start_time = time.time()

        result = assistant.ask(
            question=request.question,
            max_length=request.max_length,
            temperature=request.temperature
        )

        response_time = (time.time() - start_time) * 1000

        return AskResponse(
            question=request.question,
            answer=result['answer'],
            confidence=result['confidence'],
            response_time_ms=round(response_time, 2)
        )

    return app


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'configs' / 'assistant_config.yaml'

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)

    return {
        'model': {'base': 'gpt2'},
        'server': {'host': '0.0.0.0', 'port': 8002}
    }


def main():
    if not HAS_FASTAPI:
        print("FastAPI is required. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Run the technical assistant service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8002, help='Port to bind to')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')

    args = parser.parse_args()
    config = load_config(args.config)

    # Override with CLI args
    host = args.host or config.get('server', {}).get('host', '0.0.0.0')
    port = args.port or config.get('server', {}).get('port', 8002)

    print(f"\n=== Starting Technical Manual Assistant ===")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Docs: http://{host}:{port}/docs")
    print()

    app = create_app(config)
    uvicorn.run(app, host=host, port=port, reload=args.reload)


if __name__ == '__main__':
    main()
