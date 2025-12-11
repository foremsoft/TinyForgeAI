"""
Playground Exporter - Export models as standalone HTML playgrounds.

Creates self-contained HTML files that can run inference in the browser
using ONNX Runtime Web or TensorFlow.js.
"""

import base64
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Standalone HTML template with embedded model
STANDALONE_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - TinyForge Playground</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
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

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 2rem;
        }}

        .container {{
            max-width: 800px;
            margin: 0 auto;
        }}

        h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--primary), #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .subtitle {{
            color: var(--text-muted);
            margin-bottom: 2rem;
        }}

        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}

        label {{
            display: block;
            font-size: 0.875rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        textarea {{
            width: 100%;
            min-height: 100px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            color: var(--text);
            font-size: 1rem;
            font-family: inherit;
            resize: vertical;
        }}

        textarea:focus {{
            outline: none;
            border-color: var(--primary);
        }}

        .btn {{
            background: var(--primary);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            margin-top: 1rem;
            width: 100%;
        }}

        .btn:hover {{ background: var(--primary-dark); }}
        .btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}

        .output {{
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            min-height: 100px;
            white-space: pre-wrap;
        }}

        .output.loading {{
            color: var(--text-muted);
            font-style: italic;
        }}

        .metrics {{
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
            font-size: 0.875rem;
            color: var(--text-muted);
        }}

        .badge {{
            display: inline-block;
            background: var(--primary);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            margin-left: 1rem;
        }}

        .footer {{
            text-align: center;
            margin-top: 2rem;
            color: var(--text-muted);
            font-size: 0.875rem;
        }}

        .footer a {{
            color: var(--primary);
            text-decoration: none;
        }}

        .status {{
            padding: 0.5rem 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            font-size: 0.875rem;
        }}

        .status.loading {{
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid var(--primary);
            color: var(--primary);
        }}

        .status.ready {{
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid var(--success);
            color: var(--success);
        }}

        .status.error {{
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--error);
            color: var(--error);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title} <span class="badge">Offline</span></h1>
        <p class="subtitle">{description}</p>

        <div id="status" class="status loading">Loading model...</div>

        <div class="card">
            <label for="input">Input</label>
            <textarea id="input" placeholder="{placeholder}">{default_input}</textarea>
            <button class="btn" id="runBtn" onclick="runInference()" disabled>Run Model</button>
        </div>

        <div class="card">
            <label>Output</label>
            <div class="output loading" id="output">Waiting for model to load...</div>
            <div class="metrics" id="metrics" style="display: none;">
                <span>Latency: <strong id="latency">-</strong></span>
                <span>Model size: <strong>{model_size}</strong></span>
            </div>
        </div>

        <div class="footer">
            Powered by <a href="https://github.com/foremsoft/TinyForgeAI" target="_blank">TinyForgeAI</a>
            | Runs entirely in your browser - no server needed!
        </div>
    </div>

    <script>
        // Model configuration
        const MODEL_CONFIG = {model_config};

        // Vocabulary (for tokenization)
        const VOCAB = {vocab};

        let session = null;

        async function loadModel() {{
            const status = document.getElementById('status');
            const btn = document.getElementById('runBtn');
            const output = document.getElementById('output');

            try {{
                // Check if we have embedded model data or URL
                if (MODEL_CONFIG.model_data) {{
                    // Decode base64 model
                    const modelData = Uint8Array.from(atob(MODEL_CONFIG.model_data), c => c.charCodeAt(0));
                    session = await ort.InferenceSession.create(modelData.buffer);
                }} else if (MODEL_CONFIG.model_url) {{
                    session = await ort.InferenceSession.create(MODEL_CONFIG.model_url);
                }} else {{
                    throw new Error('No model data available');
                }}

                status.className = 'status ready';
                status.textContent = 'Model loaded! Ready for inference.';
                btn.disabled = false;
                output.className = 'output';
                output.textContent = 'Enter text and click "Run Model"';

            }} catch (error) {{
                console.error('Model loading error:', error);
                status.className = 'status error';
                status.textContent = 'Failed to load model: ' + error.message;
                output.textContent = 'Model loading failed. This playground requires WebAssembly support.';
            }}
        }}

        function tokenize(text) {{
            // Simple whitespace tokenization with vocab lookup
            const tokens = text.toLowerCase().split(/\\s+/);
            const ids = tokens.map(t => VOCAB[t] || VOCAB['[UNK]'] || 0);
            return ids;
        }}

        function detokenize(ids) {{
            // Reverse vocab lookup
            const reverseVocab = Object.fromEntries(
                Object.entries(VOCAB).map(([k, v]) => [v, k])
            );
            return ids.map(id => reverseVocab[id] || '[UNK]').join(' ');
        }}

        async function runInference() {{
            const input = document.getElementById('input').value.trim();
            if (!input || !session) return;

            const btn = document.getElementById('runBtn');
            const output = document.getElementById('output');
            const metrics = document.getElementById('metrics');

            btn.disabled = true;
            output.className = 'output loading';
            output.textContent = 'Running inference...';

            const startTime = performance.now();

            try {{
                // Tokenize input
                const inputIds = tokenize(input);

                // Pad/truncate to max length
                const maxLen = MODEL_CONFIG.max_length || 128;
                const paddedIds = inputIds.slice(0, maxLen);
                while (paddedIds.length < maxLen) paddedIds.push(0);

                // Create tensor
                const inputTensor = new ort.Tensor('int64', BigInt64Array.from(paddedIds.map(BigInt)), [1, maxLen]);
                const attentionMask = new ort.Tensor('int64', BigInt64Array.from(paddedIds.map(id => BigInt(id > 0 ? 1 : 0))), [1, maxLen]);

                // Run inference
                const feeds = {{
                    'input_ids': inputTensor,
                    'attention_mask': attentionMask,
                }};

                const results = await session.run(feeds);

                // Get output
                const outputTensor = results[Object.keys(results)[0]];
                const outputIds = Array.from(outputTensor.data).map(Number);
                const outputText = detokenize(outputIds);

                const endTime = performance.now();

                output.className = 'output';
                output.textContent = outputText;

                document.getElementById('latency').textContent = Math.round(endTime - startTime) + 'ms';
                metrics.style.display = 'flex';

            }} catch (error) {{
                console.error('Inference error:', error);
                output.className = 'output';
                output.textContent = 'Error: ' + error.message;
            }} finally {{
                btn.disabled = false;
            }}
        }}

        // Load model on page load
        loadModel();

        // Ctrl+Enter to run
        document.getElementById('input').addEventListener('keydown', function(e) {{
            if (e.ctrlKey && e.key === 'Enter') runInference();
        }});
    </script>
</body>
</html>
'''


@dataclass
class ExportConfig:
    """Configuration for playground export."""
    title: str = "TinyForge Playground"
    description: str = "Run AI inference directly in your browser"
    placeholder: str = "Enter your text here..."
    default_input: str = ""
    embed_model: bool = True  # Embed model in HTML (larger file, works offline)
    max_model_size_mb: int = 50  # Max model size for embedding


class PlaygroundExporter:
    """
    Export models as standalone HTML playgrounds.

    Creates self-contained HTML files that can:
    - Run entirely in the browser (no server needed)
    - Be hosted on static hosting (GitHub Pages, S3, etc.)
    - Work offline after initial load

    Usage:
        exporter = PlaygroundExporter()
        exporter.export(
            model_path="./my_model",
            output_path="./playground.html"
        )
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize the exporter.

        Args:
            config: Export configuration.
        """
        self.config = config or ExportConfig()

    def export(
        self,
        model_path: str,
        output_path: str,
        onnx_path: Optional[str] = None,
    ) -> Path:
        """
        Export a model as a standalone HTML playground.

        Args:
            model_path: Path to the trained model directory.
            output_path: Output HTML file path.
            onnx_path: Optional path to pre-exported ONNX model.

        Returns:
            Path to the exported HTML file.
        """
        model_path = Path(model_path)
        output_path = Path(output_path)

        # Find or create ONNX model
        if onnx_path:
            onnx_file = Path(onnx_path)
        else:
            onnx_file = self._find_or_export_onnx(model_path)

        if not onnx_file or not onnx_file.exists():
            logger.warning("No ONNX model available, creating stub playground")
            return self._create_stub_playground(output_path)

        # Load vocabulary
        vocab = self._load_vocabulary(model_path)

        # Check model size
        model_size = onnx_file.stat().st_size
        model_size_mb = model_size / (1024 * 1024)

        # Prepare model config
        model_config = {
            "max_length": 128,
            "model_type": "seq2seq",
        }

        # Embed model or use external URL
        if self.config.embed_model and model_size_mb <= self.config.max_model_size_mb:
            logger.info(f"Embedding model ({model_size_mb:.1f} MB)")
            with open(onnx_file, "rb") as f:
                model_data = base64.b64encode(f.read()).decode("ascii")
            model_config["model_data"] = model_data
        else:
            logger.info(f"Model too large to embed ({model_size_mb:.1f} MB)")
            # Suggest copying model alongside HTML
            model_config["model_url"] = onnx_file.name
            logger.info(f"Copy {onnx_file.name} alongside the HTML file")

        # Generate HTML
        html = STANDALONE_HTML_TEMPLATE.format(
            title=self.config.title,
            description=self.config.description,
            placeholder=self.config.placeholder,
            default_input=self.config.default_input,
            model_size=f"{model_size_mb:.1f} MB",
            model_config=json.dumps(model_config),
            vocab=json.dumps(vocab),
        )

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Exported playground to {output_path}")

        # Copy model file if not embedded
        if not self.config.embed_model or model_size_mb > self.config.max_model_size_mb:
            import shutil
            model_dest = output_path.parent / onnx_file.name
            shutil.copy(onnx_file, model_dest)
            logger.info(f"Copied model to {model_dest}")

        return output_path

    def _find_or_export_onnx(self, model_path: Path) -> Optional[Path]:
        """Find existing ONNX model or export one."""
        # Check for existing ONNX file
        onnx_files = list(model_path.glob("*.onnx"))
        if onnx_files:
            return onnx_files[0]

        # Try to export
        try:
            from backend.exporter.onnx_export import export_to_onnx

            onnx_path = model_path / "model.onnx"
            export_to_onnx(model_path, onnx_path)
            return onnx_path

        except ImportError:
            logger.warning("ONNX export not available")
            return None
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return None

    def _load_vocabulary(self, model_path: Path) -> Dict[str, int]:
        """Load tokenizer vocabulary."""
        # Try to load from tokenizer files
        vocab_files = [
            model_path / "vocab.json",
            model_path / "tokenizer.json",
            model_path / "vocab.txt",
        ]

        for vocab_file in vocab_files:
            if vocab_file.exists():
                try:
                    if vocab_file.suffix == ".json":
                        with open(vocab_file) as f:
                            data = json.load(f)
                            if "model" in data and "vocab" in data["model"]:
                                return data["model"]["vocab"]
                            return data
                    else:
                        vocab = {}
                        with open(vocab_file) as f:
                            for i, line in enumerate(f):
                                vocab[line.strip()] = i
                        return vocab
                except Exception as e:
                    logger.warning(f"Failed to load vocab from {vocab_file}: {e}")

        # Return basic vocab as fallback
        return {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4,
        }

    def _create_stub_playground(self, output_path: Path) -> Path:
        """Create a stub playground for models without ONNX export."""
        stub_html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title} - TinyForge Playground</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }}
        .card {{
            background: #1e293b;
            border-radius: 12px;
            padding: 2rem;
            max-width: 500px;
            text-align: center;
        }}
        h1 {{ color: #6366f1; margin-bottom: 1rem; }}
        p {{ color: #94a3b8; line-height: 1.6; }}
        code {{
            background: #0f172a;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Offline Export Not Available</h1>
        <p>
            This model doesn't have an ONNX export yet.
            To create a standalone playground, first export your model:
        </p>
        <p><code>foremforge export --model ./model --out ./service --export-onnx</code></p>
        <p>Then export the playground:</p>
        <p><code>foremforge playground --model ./service --export playground.html</code></p>
    </div>
</body>
</html>
'''
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(stub_html)

        return output_path


def export_playground(
    model_path: str,
    output_path: str = "./playground.html",
    title: str = "TinyForge Playground",
    embed_model: bool = True,
) -> Path:
    """
    Convenience function to export a model as HTML playground.

    Args:
        model_path: Path to model directory.
        output_path: Output HTML path.
        title: Playground title.
        embed_model: Whether to embed model in HTML.

    Returns:
        Path to exported file.
    """
    config = ExportConfig(
        title=title,
        embed_model=embed_model,
    )

    exporter = PlaygroundExporter(config)
    return exporter.export(model_path, output_path)
