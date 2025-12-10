#!/usr/bin/env python3
"""
Search Service for Internal Knowledge Search

FastAPI-based REST API for semantic search over indexed documents.

Usage:
    python search_service.py --port 8001

API Endpoints:
    POST /search - Search documents
    POST /answer - Generate answer with context
    GET /status - Index status
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import yaml
import numpy as np

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

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# Pydantic models for API
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.3
    filters: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float


class AnswerRequest(BaseModel):
    question: str
    include_sources: bool = True
    top_k: int = 3


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[List[SearchResult]] = None
    generation_time_ms: float


class StatusResponse(BaseModel):
    status: str
    index_path: str
    num_documents: int
    embedding_model: str
    last_updated: Optional[str] = None


class VectorIndex:
    """Vector index for semantic search."""

    def __init__(self, index_path: str, embedding_model: str = None):
        self.index_path = Path(index_path)
        self.embeddings = None
        self.documents = []
        self.metadata = {}
        self.model = None

        self._load_index()

        # Load embedding model
        model_name = embedding_model or self.metadata.get(
            "embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        if HAS_SENTENCE_TRANSFORMERS:
            print(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
        else:
            print("Warning: Using mock embeddings")

    def _load_index(self):
        """Load index from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found at {self.index_path}")

        # Load embeddings
        embeddings_path = self.index_path / "embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
        else:
            self.embeddings = np.array([])

        # Load documents
        docs_path = self.index_path / "documents.json"
        if docs_path.exists():
            with open(docs_path, encoding="utf-8") as f:
                self.documents = json.load(f)

        # Load metadata
        meta_path = self.index_path / "index_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)

        print(f"Loaded index with {len(self.documents)} documents")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        if self.model:
            return self.model.encode(text, convert_to_numpy=True)
        else:
            # Mock embedding
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(384).astype(np.float32)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        if len(self.embeddings) == 0:
            return []

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top results
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])

            if score < min_score:
                break

            doc = self.documents[idx]

            # Apply filters
            if filters:
                metadata = doc.get("metadata", {})
                skip = False
                for key, value in filters.items():
                    if metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            results.append({
                "text": doc["text"],
                "score": score,
                "metadata": doc.get("metadata", {})
            })

            if len(results) >= top_k:
                break

        return results


class AnswerGenerator:
    """Generate answers using retrieved context."""

    def __init__(self, model_path: str = None):
        self.model = None
        self.tokenizer = None

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = model_path or "gpt2"
            print(f"Loading answer generation model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except ImportError:
            print("Transformers not installed. Answer generation disabled.")
        except Exception as e:
            print(f"Could not load generation model: {e}")

    def generate(self, question: str, context: List[str], max_length: int = 256) -> str:
        """Generate answer based on context."""
        if not self.model:
            return self._simple_answer(question, context)

        # Format prompt
        context_text = "\n\n".join(context[:3])
        prompt = f"""Context information:
{context_text}

Based on the context above, answer the following question:
Question: {question}
Answer:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract answer part
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response[len(prompt):].strip()

            return answer if answer else self._simple_answer(question, context)

        except Exception as e:
            print(f"Generation error: {e}")
            return self._simple_answer(question, context)

    def _simple_answer(self, question: str, context: List[str]) -> str:
        """Simple answer extraction when model unavailable."""
        if not context:
            return "I couldn't find relevant information to answer your question."

        # Return most relevant context as answer
        return f"Based on the available information:\n\n{context[0]}"


def create_app(config: Dict[str, Any]) -> "FastAPI":
    """Create FastAPI application."""
    app = FastAPI(
        title="Internal Knowledge Search API",
        description="Semantic search over enterprise knowledge base",
        version="1.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("server", {}).get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize index
    index_path = config.get("index", {}).get("path", "./data/knowledge_index")
    try:
        index = VectorIndex(index_path)
    except FileNotFoundError:
        print(f"Warning: Index not found at {index_path}")
        print("Run index_documents.py first to create the index.")
        index = None

    # Initialize answer generator
    gen_config = config.get("generation", {})
    generator = None
    if gen_config.get("enabled", False):
        generator = AnswerGenerator(gen_config.get("model"))

    @app.get("/")
    async def root():
        return {"message": "Internal Knowledge Search API", "docs": "/docs"}

    @app.get("/status", response_model=StatusResponse)
    async def get_status():
        if not index:
            return StatusResponse(
                status="no_index",
                index_path=str(index_path),
                num_documents=0,
                embedding_model="none"
            )

        return StatusResponse(
            status="ready",
            index_path=str(index.index_path),
            num_documents=len(index.documents),
            embedding_model=index.metadata.get("embedding_model", "unknown"),
            last_updated=index.metadata.get("created_at")
        )

    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest):
        if not index:
            raise HTTPException(status_code=503, detail="Index not loaded")

        import time
        start = time.time()

        results = index.search(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score,
            filters=request.filters
        )

        search_time = (time.time() - start) * 1000

        return SearchResponse(
            query=request.query,
            results=[SearchResult(**r) for r in results],
            total_results=len(results),
            search_time_ms=round(search_time, 2)
        )

    @app.post("/answer", response_model=AnswerResponse)
    async def answer(request: AnswerRequest):
        if not index:
            raise HTTPException(status_code=503, detail="Index not loaded")

        import time
        start = time.time()

        # Search for relevant context
        results = index.search(
            query=request.question,
            top_k=request.top_k,
            min_score=0.2
        )

        context = [r["text"] for r in results]

        # Generate answer
        if generator:
            answer_text = generator.generate(request.question, context)
        else:
            if context:
                answer_text = f"Based on the available information:\n\n{context[0]}"
            else:
                answer_text = "I couldn't find relevant information to answer your question."

        gen_time = (time.time() - start) * 1000

        return AnswerResponse(
            question=request.question,
            answer=answer_text,
            sources=[SearchResult(**r) for r in results] if request.include_sources else None,
            generation_time_ms=round(gen_time, 2)
        )

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "search_config.yaml"

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)

    return {
        "index": {"path": "./data/knowledge_index"},
        "server": {"host": "0.0.0.0", "port": 8001}
    }


def main():
    if not HAS_FASTAPI:
        print("FastAPI is required. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run the knowledge search service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()
    config = load_config(args.config)

    # Override with CLI args
    host = args.host or config.get("server", {}).get("host", "0.0.0.0")
    port = args.port or config.get("server", {}).get("port", 8001)

    print(f"\n=== Starting Knowledge Search Service ===")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Docs: http://{host}:{port}/docs")
    print()

    app = create_app(config)
    uvicorn.run(app, host=host, port=port, reload=args.reload)


if __name__ == "__main__":
    main()
