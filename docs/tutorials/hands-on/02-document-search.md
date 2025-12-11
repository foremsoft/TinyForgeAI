# Build a Document Search System

**Time needed:** 20 minutes
**Skill level:** Beginner
**What you'll build:** Search through your PDFs, Word docs, and text files using AI

---

## What We're Building

Imagine having hundreds of documents and being able to ask:

```
You: "What is the refund policy?"
System: Found in 'company_policies.pdf' (page 12):
        "Customers may request a full refund within 30 days..."

You: "How do I install the software?"
System: Found in 'user_manual.docx' (section 3):
        "To install, download the installer and run setup.exe..."
```

---

## Prerequisites

- Completed [00-quickstart.md](00-quickstart.md)
- TinyForgeAI installed

Install document support:
```bash
pip install python-docx PyMuPDF
```

---

## Step 1: Organize Your Documents

Create a folder structure:

```
my_documents/
├── policies/
│   ├── refund_policy.txt
│   └── privacy_policy.txt
├── manuals/
│   ├── user_guide.txt
│   └── quick_start.txt
└── faqs/
    └── common_questions.txt
```

Let's create sample documents:

```python
# create_sample_docs.py - Create sample documents for testing

import os

# Create folders
folders = ["my_documents/policies", "my_documents/manuals", "my_documents/faqs"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Sample documents
documents = {
    "my_documents/policies/refund_policy.txt": """
REFUND POLICY

1. Eligibility
All customers are eligible for a full refund within 30 days of purchase.
After 30 days, a partial refund of 50% may be issued at our discretion.

2. How to Request a Refund
- Contact support@example.com
- Include your order number
- Explain the reason for the refund

3. Processing Time
Refunds are processed within 5-7 business days.
Credit card refunds may take an additional 3-5 days to appear.

4. Non-Refundable Items
- Digital downloads after first use
- Gift cards
- Custom orders
""",

    "my_documents/policies/privacy_policy.txt": """
PRIVACY POLICY

1. Data We Collect
- Name and email address
- Payment information (processed securely)
- Usage data and preferences

2. How We Use Your Data
- To process your orders
- To send important updates
- To improve our services

3. Data Sharing
We never sell your personal data to third parties.
We may share data with service providers who help us operate.

4. Your Rights
- Request a copy of your data
- Request deletion of your data
- Opt out of marketing emails
""",

    "my_documents/manuals/user_guide.txt": """
USER GUIDE

Chapter 1: Getting Started
--------------------------
Welcome to our product! This guide will help you get up and running.

1.1 System Requirements
- Windows 10 or later / macOS 10.15 or later
- 4GB RAM minimum
- 500MB free disk space

1.2 Installation
1. Download the installer from our website
2. Run the installer and follow the prompts
3. Enter your license key when asked
4. Restart your computer

Chapter 2: Basic Features
-------------------------
2.1 Creating a New Project
Click File > New Project, enter a name, and click Create.

2.2 Saving Your Work
Your work is auto-saved every 5 minutes.
To manually save, press Ctrl+S (Windows) or Cmd+S (Mac).

2.3 Exporting
Go to File > Export and choose your format (PDF, Word, or HTML).
""",

    "my_documents/manuals/quick_start.txt": """
QUICK START GUIDE

5 Minutes to Your First Project!

Step 1: Launch the Application
Double-click the desktop icon or find it in your Start menu.

Step 2: Create Account
Click "Sign Up" and enter your email. Check your inbox for verification.

Step 3: New Project
Click the big "+" button in the center of the screen.

Step 4: Add Content
Drag and drop files, or click "Import" to browse.

Step 5: Share
Click "Share" in the top right. Enter email addresses to invite others.

That's it! You're ready to go.

Need help? Email support@example.com or visit help.example.com
""",

    "my_documents/faqs/common_questions.txt": """
FREQUENTLY ASKED QUESTIONS

Q: How do I reset my password?
A: Click "Forgot Password" on the login page. Enter your email and we'll send a reset link.

Q: Can I use the product on multiple computers?
A: Yes! Your license allows installation on up to 3 devices.

Q: How do I cancel my subscription?
A: Go to Account Settings > Subscription > Cancel. You'll keep access until the end of your billing period.

Q: Is my data secure?
A: Absolutely. We use industry-standard encryption and never share your data with third parties.

Q: Do you offer student discounts?
A: Yes! Students get 50% off. Email edu@example.com with your .edu email address.

Q: How do I contact support?
A: Email support@example.com or use the chat widget on our website. Response time is usually under 24 hours.
"""
}

# Create the files
for path, content in documents.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip())
    print(f"Created: {path}")

print(f"\nCreated {len(documents)} sample documents!")
```

Run it:

```bash
python create_sample_docs.py
```

---

## Step 2: Load and Index Documents

Now let's load all documents and make them searchable:

```python
# index_documents.py - Load and index all documents

import os
import json
from pathlib import Path
from connectors.file_ingest import ingest_file

def load_all_documents(folder_path):
    """Load all documents from a folder recursively."""
    documents = []
    supported_extensions = ['.txt', '.md', '.docx', '.pdf']

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext in supported_extensions:
                filepath = os.path.join(root, filename)
                try:
                    content = ingest_file(filepath)
                    documents.append({
                        "filename": filename,
                        "filepath": filepath,
                        "folder": os.path.basename(root),
                        "content": content
                    })
                    print(f"Loaded: {filepath}")
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")

    return documents

def split_into_chunks(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks for better search."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to end at a sentence or paragraph
        if end < len(text):
            # Look for paragraph break
            para_break = chunk.rfind('\n\n')
            if para_break > chunk_size // 2:
                end = start + para_break
                chunk = text[start:end]
            else:
                # Look for sentence end
                for punct in ['. ', '! ', '? ']:
                    sent_end = chunk.rfind(punct)
                    if sent_end > chunk_size // 2:
                        end = start + sent_end + 1
                        chunk = text[start:end]
                        break

        chunks.append(chunk.strip())
        start = end - overlap  # Overlap for context

    return chunks

def create_search_index(documents):
    """Create a searchable index from documents."""
    index = []

    for doc in documents:
        chunks = split_into_chunks(doc["content"])
        for i, chunk in enumerate(chunks):
            index.append({
                "chunk_id": f"{doc['filename']}_{i}",
                "filename": doc["filename"],
                "filepath": doc["filepath"],
                "folder": doc["folder"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk
            })

    return index

# Main execution
if __name__ == "__main__":
    print("Loading documents...")
    print("=" * 50)

    documents = load_all_documents("my_documents")

    print(f"\nLoaded {len(documents)} documents")
    print("=" * 50)

    print("\nCreating search index...")
    index = create_search_index(documents)

    print(f"Created {len(index)} searchable chunks")

    # Save the index
    with open("document_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"\nIndex saved to: document_index.json")

    # Show summary
    print("\nIndex Summary:")
    print("-" * 30)
    folders = {}
    for item in index:
        folder = item["folder"]
        folders[folder] = folders.get(folder, 0) + 1

    for folder, count in folders.items():
        print(f"  {folder}: {count} chunks")
```

Run it:

```bash
python index_documents.py
```

---

## Step 3: Build Simple Keyword Search

Let's start with a simple but effective keyword search:

```python
# simple_search.py - Keyword-based document search

import json
import re
from collections import Counter

class SimpleDocumentSearch:
    def __init__(self, index_file):
        """Load the search index."""
        with open(index_file, "r", encoding="utf-8") as f:
            self.index = json.load(f)
        print(f"Loaded index with {len(self.index)} chunks")

    def tokenize(self, text):
        """Convert text to lowercase words."""
        return re.findall(r'\b\w+\b', text.lower())

    def search(self, query, top_k=5):
        """Search for documents matching the query."""
        query_tokens = set(self.tokenize(query))

        results = []
        for chunk in self.index:
            chunk_tokens = set(self.tokenize(chunk["text"]))

            # Count matching tokens
            matches = query_tokens & chunk_tokens
            if matches:
                # Score based on percentage of query terms found
                score = len(matches) / len(query_tokens)

                # Boost exact phrase matches
                if query.lower() in chunk["text"].lower():
                    score += 0.5

                results.append({
                    "filename": chunk["filename"],
                    "folder": chunk["folder"],
                    "text": chunk["text"],
                    "score": score,
                    "matched_terms": list(matches)
                })

        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def highlight_matches(self, text, query, max_length=200):
        """Highlight matching terms in the text."""
        query_tokens = self.tokenize(query)

        # Find the best snippet containing query terms
        text_lower = text.lower()
        best_start = 0
        best_count = 0

        for i in range(0, len(text) - max_length, 50):
            snippet = text_lower[i:i + max_length]
            count = sum(1 for token in query_tokens if token in snippet)
            if count > best_count:
                best_count = count
                best_start = i

        snippet = text[best_start:best_start + max_length]

        # Add ellipsis if truncated
        if best_start > 0:
            snippet = "..." + snippet
        if best_start + max_length < len(text):
            snippet = snippet + "..."

        return snippet

    def interactive_search(self):
        """Run interactive search session."""
        print("\n" + "=" * 50)
        print("Document Search Ready!")
        print("Type your search query, or 'quit' to exit")
        print("=" * 50 + "\n")

        while True:
            query = input("Search: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            results = self.search(query)

            if not results:
                print("No results found.\n")
                continue

            print(f"\nFound {len(results)} results:\n")
            print("-" * 50)

            for i, result in enumerate(results, 1):
                snippet = self.highlight_matches(result["text"], query)
                print(f"{i}. [{result['folder']}] {result['filename']}")
                print(f"   Score: {result['score']:.2f}")
                print(f"   Matched: {', '.join(result['matched_terms'])}")
                print(f"   Preview: {snippet}")
                print()

            print("-" * 50 + "\n")


if __name__ == "__main__":
    search = SimpleDocumentSearch("document_index.json")
    search.interactive_search()
```

Run it:

```bash
python simple_search.py
```

**Try these searches:**
```
Search: refund policy
Search: how to install
Search: password reset
Search: student discount
Search: export PDF
```

---

## Step 4: Add Semantic Search (AI-Powered)

For better results, let's add AI-powered semantic search that understands meaning:

```python
# semantic_search.py - AI-powered document search

import json
import numpy as np

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Note: Install sentence-transformers for AI-powered search:")
    print("  pip install sentence-transformers")

class SemanticDocumentSearch:
    def __init__(self, index_file, model_name="all-MiniLM-L6-v2"):
        """Initialize with a sentence transformer model."""
        with open(index_file, "r", encoding="utf-8") as f:
            self.index = json.load(f)

        if SEMANTIC_AVAILABLE:
            print(f"Loading AI model: {model_name}")
            self.model = SentenceTransformer(model_name)
            print("Creating embeddings for all documents...")
            self._create_embeddings()
            print(f"Ready! Indexed {len(self.index)} chunks")
        else:
            self.model = None
            print("Running in keyword-only mode (install sentence-transformers for AI search)")

    def _create_embeddings(self):
        """Create vector embeddings for all chunks."""
        texts = [chunk["text"] for chunk in self.index]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(self, query, top_k=5):
        """Search using semantic similarity."""
        if not SEMANTIC_AVAILABLE or self.model is None:
            return self._keyword_search(query, top_k)

        # Get query embedding
        query_embedding = self.model.encode(query)

        # Calculate similarity with all chunks
        results = []
        for i, chunk in enumerate(self.index):
            similarity = self.cosine_similarity(query_embedding, self.embeddings[i])
            results.append({
                "filename": chunk["filename"],
                "folder": chunk["folder"],
                "text": chunk["text"],
                "score": float(similarity)
            })

        # Sort by similarity
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _keyword_search(self, query, top_k):
        """Fallback keyword search."""
        import re
        query_tokens = set(re.findall(r'\b\w+\b', query.lower()))

        results = []
        for chunk in self.index:
            chunk_tokens = set(re.findall(r'\b\w+\b', chunk["text"].lower()))
            matches = query_tokens & chunk_tokens
            if matches:
                score = len(matches) / len(query_tokens)
                results.append({
                    "filename": chunk["filename"],
                    "folder": chunk["folder"],
                    "text": chunk["text"],
                    "score": score
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def interactive_search(self):
        """Run interactive search."""
        mode = "Semantic (AI)" if SEMANTIC_AVAILABLE else "Keyword"
        print(f"\n{'=' * 50}")
        print(f"Document Search [{mode} Mode]")
        print(f"{'=' * 50}\n")

        while True:
            query = input("Search: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break
            if not query:
                continue

            results = self.search(query)

            if not results:
                print("No results found.\n")
                continue

            print(f"\nTop {len(results)} results:\n")
            for i, r in enumerate(results, 1):
                preview = r["text"][:150].replace("\n", " ") + "..."
                print(f"{i}. [{r['folder']}] {r['filename']} (score: {r['score']:.3f})")
                print(f"   {preview}\n")


if __name__ == "__main__":
    search = SemanticDocumentSearch("document_index.json")
    search.interactive_search()
```

Install and run:

```bash
pip install sentence-transformers
python semantic_search.py
```

**The difference:**
- Keyword search: "return item" only finds exact words
- Semantic search: "return item" also finds "refund policy" (same meaning!)

---

## Step 5: Create a Search API

Make your search available as a web service:

```python
# search_api.py - REST API for document search

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import json
import re

app = FastAPI(title="Document Search API")

# Load index at startup
with open("document_index.json", "r") as f:
    INDEX = json.load(f)

class SearchResult(BaseModel):
    filename: str
    folder: str
    preview: str
    score: float

class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResult]

def keyword_search(query: str, top_k: int = 5):
    """Simple keyword search."""
    query_tokens = set(re.findall(r'\b\w+\b', query.lower()))

    results = []
    for chunk in INDEX:
        chunk_tokens = set(re.findall(r'\b\w+\b', chunk["text"].lower()))
        matches = query_tokens & chunk_tokens
        if matches:
            score = len(matches) / len(query_tokens)
            if query.lower() in chunk["text"].lower():
                score += 0.5
            results.append({
                "filename": chunk["filename"],
                "folder": chunk["folder"],
                "preview": chunk["text"][:200] + "...",
                "score": round(score, 3)
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

@app.get("/")
def home():
    return {
        "service": "Document Search API",
        "documents_indexed": len(INDEX),
        "endpoints": {
            "/search?q=your+query": "Search documents",
            "/stats": "Index statistics"
        }
    }

@app.get("/stats")
def stats():
    folders = {}
    for chunk in INDEX:
        folder = chunk["folder"]
        folders[folder] = folders.get(folder, 0) + 1
    return {
        "total_chunks": len(INDEX),
        "by_folder": folders
    }

@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=20, description="Max results")
):
    results = keyword_search(q, limit)
    return SearchResponse(
        query=q,
        total_results=len(results),
        results=[SearchResult(**r) for r in results]
    )

# Run with: uvicorn search_api:app --reload
```

Run the API:

```bash
uvicorn search_api:app --reload
```

Test it:

```bash
# In browser or curl:
curl "http://localhost:8000/search?q=refund+policy"
curl "http://localhost:8000/search?q=installation+guide&limit=3"
```

---

## Step 6: Build a Simple Web Interface

Create a web page to search your documents:

```html
<!-- search.html - Simple search interface -->
<!DOCTYPE html>
<html>
<head>
    <title>Document Search</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        }
        h1 { color: #333; }
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        input[type="text"] {
            flex: 1;
            padding: 10px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover { background: #0056b3; }
        .result {
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        .filename { font-weight: bold; color: #333; }
        .folder { color: #666; font-size: 0.9em; }
        .score { color: #28a745; }
        .preview { color: #555; line-height: 1.5; }
        .no-results { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <h1>Document Search</h1>

    <div class="search-box">
        <input type="text" id="query" placeholder="Enter your search query..."
               onkeypress="if(event.key==='Enter') search()">
        <button onclick="search()">Search</button>
    </div>

    <div id="results"></div>

    <script>
        async function search() {
            const query = document.getElementById('query').value;
            if (!query.trim()) return;

            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = 'Searching...';

            try {
                const response = await fetch(
                    `http://localhost:8000/search?q=${encodeURIComponent(query)}`
                );
                const data = await response.json();

                if (data.results.length === 0) {
                    resultsDiv.innerHTML = '<p class="no-results">No results found.</p>';
                    return;
                }

                let html = `<p>Found ${data.total_results} results for "${data.query}"</p>`;

                for (const result of data.results) {
                    html += `
                        <div class="result">
                            <div class="result-header">
                                <span class="filename">${result.filename}</span>
                                <span class="score">Score: ${result.score}</span>
                            </div>
                            <div class="folder">Folder: ${result.folder}</div>
                            <div class="preview">${result.preview}</div>
                        </div>
                    `;
                }

                resultsDiv.innerHTML = html;
            } catch (error) {
                resultsDiv.innerHTML = `<p class="no-results">Error: ${error.message}</p>`;
            }
        }
    </script>
</body>
</html>
```

Open `search.html` in your browser (make sure the API is running).

---

## What You've Learned

1. Loading documents (TXT, PDF, DOCX) with TinyForgeAI
2. Creating a searchable index with chunks
3. Building keyword-based search
4. Adding AI-powered semantic search
5. Creating a REST API
6. Building a web interface

---

## What's Next?

| Tutorial | Description |
|----------|-------------|
| [03-train-your-model.md](03-train-your-model.md) | Train an AI model for Q&A |
| [04-deploy-your-project.md](04-deploy-your-project.md) | Deploy to production |

---

## Troubleshooting

### "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### "No module named 'PyMuPDF'" (for PDFs)
```bash
pip install PyMuPDF
```

### "CORS error" in browser
Add this to your FastAPI app:
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
```

---

**Excellent work!** You've built a complete document search system!
