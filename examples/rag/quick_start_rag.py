#!/usr/bin/env python3
"""
TinyForgeAI RAG Quick Start Example

This script demonstrates how to use the RAG (Retrieval-Augmented Generation)
document indexer to index documents and search for relevant content.

Prerequisites:
    pip install -e ".[rag]"

Usage:
    python examples/rag/quick_start_rag.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run the RAG quick start example."""
    print("=" * 60)
    print("TinyForgeAI RAG Quick Start")
    print("=" * 60)

    from connectors.indexer import DocumentIndexer, IndexerConfig
    from connectors import indexer as indexer_module

    # Check if RAG dependencies are available
    if not indexer_module.EMBEDDINGS_AVAILABLE:
        print("\n⚠️  RAG dependencies not installed!")
        print("Install with: pip install -e '.[rag]'")
        print("\nThis will install:")
        print("  - sentence-transformers (for embeddings)")
        print("  - numpy")
        print("\n💡 You can still use basic features without embeddings.")

    # Create sample documents
    print("\n📄 Creating sample documents...")
    documents_dir = project_root / "examples" / "rag" / "sample_docs"
    documents_dir.mkdir(parents=True, exist_ok=True)

    # Create sample documents
    sample_docs = {
        "getting_started.txt": """
        Getting Started with TinyForgeAI

        TinyForgeAI is a lightweight platform for fine-tuning language models
        and deploying them as inference microservices.

        Installation:
        Run 'pip install -e .' to install TinyForgeAI in development mode.
        For training features, use 'pip install -e ".[training]"'.

        First Steps:
        1. Prepare your training data in JSONL format
        2. Configure your training settings
        3. Run the training script
        4. Deploy your model as an API
        """,

        "training_guide.txt": """
        Training Models with TinyForgeAI

        TinyForgeAI supports training language models using HuggingFace Transformers
        and PEFT for efficient fine-tuning.

        Supported Models:
        - BERT and variants (DistilBERT, RoBERTa)
        - GPT-2 and variants
        - Llama 2 and other modern LLMs

        Training Methods:
        - Full fine-tuning: Update all model parameters
        - LoRA: Efficient fine-tuning with adapters
        - QLoRA: Quantized LoRA for even less memory

        Tips for Good Training:
        - Start with a small model for testing
        - Use at least 100 examples for simple tasks
        - Monitor validation loss to detect overfitting
        """,

        "deployment_guide.txt": """
        Deploying Models with TinyForgeAI

        After training, you can deploy your model as a REST API.

        Deployment Options:
        1. Local: Run uvicorn directly
        2. Docker: Use the provided Dockerfile
        3. Cloud: Deploy to AWS, GCP, or Azure

        API Endpoints:
        - POST /predict: Make predictions
        - GET /health: Check service health
        - GET /metrics: Prometheus metrics

        Best Practices:
        - Export to ONNX for faster inference
        - Use GPU for production workloads
        - Implement request batching for throughput
        """,

        "troubleshooting.txt": """
        Troubleshooting TinyForgeAI

        Common Issues and Solutions:

        1. Out of Memory (OOM) Errors:
           - Reduce batch size
           - Use gradient accumulation
           - Enable LoRA training
           - Use mixed precision (fp16)

        2. Training Not Converging:
           - Check your learning rate (try 1e-5 to 5e-5)
           - Verify your data format
           - Ensure enough training examples

        3. Slow Inference:
           - Export to ONNX format
           - Use GPU if available
           - Enable batching

        4. Import Errors:
           - Ensure all dependencies installed
           - Use correct Python version (3.10+)
        """
    }

    for filename, content in sample_docs.items():
        (documents_dir / filename).write_text(content.strip())
        print(f"   Created: {filename}")

    # Configure the indexer
    print("\n🔧 Configuring indexer...")
    config = IndexerConfig(
        chunk_size=200,           # Smaller chunks for demo
        chunk_overlap=20,         # Some overlap between chunks
        index_path=str(project_root / "examples" / "rag" / "index"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    print(f"   Chunk size: {config.chunk_size} characters")
    print(f"   Overlap: {config.chunk_overlap} characters")
    print(f"   Index path: {config.index_path}")

    # Create indexer
    indexer = DocumentIndexer(config)

    # Index documents
    print("\n📚 Indexing documents...")

    if indexer_module.EMBEDDINGS_AVAILABLE:
        for doc_file in documents_dir.glob("*.txt"):
            try:
                doc_ids = indexer.index_file(doc_file)
                print(f"   Indexed {doc_file.name}: {len(doc_ids)} chunks")
            except Exception as e:
                print(f"   ❌ Error indexing {doc_file.name}: {e}")

        # Save the index
        indexer.save_index()
        print(f"\n💾 Index saved to: {config.index_path}")

        # Show statistics
        stats = indexer.get_stats()
        print(f"\n📊 Index Statistics:")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Embedding dimension: {stats['embedding_dimension']}")
        print(f"   Embedding model: {stats['embedding_model']}")

        # Demo searches
        print("\n🔍 Demo Searches:")
        print("-" * 60)

        queries = [
            "How do I install TinyForgeAI?",
            "What is LoRA training?",
            "How do I fix memory errors?",
            "How to deploy as an API?",
        ]

        for query in queries:
            print(f"\n❓ Query: {query}")
            results = indexer.search(query, top_k=2)

            for i, result in enumerate(results, 1):
                print(f"\n   [{i}] Score: {result.score:.3f}")
                print(f"       Source: {result.document.metadata.get('source', 'unknown')}")
                # Show first 100 chars of content
                content_preview = result.document.content[:100].replace('\n', ' ')
                print(f"       Content: {content_preview}...")

    else:
        print("\n⚠️  Skipping embedding-based indexing (dependencies not installed)")
        print("   Install with: pip install -e '.[rag]'")

        # Demo chunking without embeddings
        print("\n📋 Demo: Text Chunking (no embeddings required)")
        from connectors.indexer import TextChunker

        chunker = TextChunker(chunk_size=200, overlap=20)

        sample_text = sample_docs["getting_started.txt"]
        chunks = chunker.chunk(sample_text)

        print(f"\n   Original text: {len(sample_text)} characters")
        print(f"   Chunks created: {len(chunks)}")
        print("\n   First chunk preview:")
        print(f"   '{chunks[0][:100]}...'")

    print("\n" + "=" * 60)
    print("RAG Quick Start Complete!")
    print("=" * 60)

    print("\n📖 Next steps:")
    print("   1. Index your own documents")
    print("   2. Integrate search results into prompts")
    print("   3. Build a RAG-powered chatbot")
    print("\n📚 See tutorials for more examples!")


if __name__ == "__main__":
    main()
