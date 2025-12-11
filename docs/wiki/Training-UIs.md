# Training UIs

TinyForgeAI provides three different training interfaces to suit different users and use cases.

---

## Quick Comparison

| Interface | Best For | Technical Level | Key Features |
|-----------|----------|-----------------|--------------|
| **Gradio** | Demos, beginners | None | Shareable links, instant setup |
| **Streamlit** | Data scientists | Basic | Rich visualizations, batch testing |
| **React Dashboard** | Production | Any | Easy/Advanced modes, job management |

---

## 1. Gradio Interface

**Best for:** Quick demos, sharing with others, workshops

### Features
- Drag-and-drop file upload
- Instant shareable links (Gradio's public URL)
- Simple 4-step process: Upload → Train → Test → Download
- Model download as ZIP

### Quick Start

```bash
cd ui/gradio
pip install -r requirements.txt
python training_app.py
```

Opens at: `http://localhost:7860`

### Screenshot

```
┌─────────────────────────────────────────────────────────────┐
│  📁 Upload Data  │  🚀 Train Model  │  🧪 Test  │  📥 Download │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│     ┌─────────────────────────────────────┐                 │
│     │      📄 Drag & Drop Here            │                 │
│     │      .csv, .jsonl files             │                 │
│     └─────────────────────────────────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Streamlit Interface

**Best for:** Data exploration, research, detailed analysis

### Features
- Data statistics and visualizations (Plotly)
- Input/output length distributions
- Real-time training progress with callbacks
- Batch testing with metrics
- Export models and configurations

### Quick Start

```bash
cd ui/streamlit
pip install -r requirements.txt
streamlit run training_app.py
```

Opens at: `http://localhost:8501`

### Screenshot

```
┌─────────────────────────────────────────────────────────────┐
│  📁 Upload  │  📊 Explore  │  🚀 Train  │  🧪 Test  │  📥 Export │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 DATA EXPLORATION                                        │
│  ────────────────                                           │
│  • Total Examples: 100                                      │
│  • Avg Input Length: 45 chars                               │
│                                                             │
│  ┌─────────────────────────────────────────┐                │
│  │     Input Length Distribution           │                │
│  │     ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░             │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. React Dashboard

**Best for:** Production environments, enterprise use, full platform

### Features
- **Easy Mode**: Step-by-step wizard for beginners
- **Advanced Mode**: Full control for power users
- Real-time WebSocket updates
- Job management (view, cancel, delete)
- CLI command equivalents shown
- API connection status

### Quick Start

```bash
# Terminal 1: Start the API
python -m backend.api.main

# Terminal 2: Start the dashboard
cd dashboard
npm install
npm run dev
```

Opens at: `http://localhost:5173`

### Easy Mode (Wizard)

```
┌─────────────────────────────────────────────────────────────┐
│  ① Upload Data  │  ② Choose Model  │  ③ Settings  │  ④ Train │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  STEP 2: Choose Your AI Model                               │
│  ────────────────────────────                               │
│                                                             │
│  ┌───────────────────┐  ┌───────────────────┐               │
│  │ DistilBERT        │  │ BERT Base         │               │
│  │ [Recommended]     │  │ Industry standard │               │
│  │ Fast & efficient  │  │ Good balance      │               │
│  └───────────────────┘  └───────────────────┘               │
│                                                             │
│                              [← Back]  [Continue →]         │
└─────────────────────────────────────────────────────────────┘
```

### Advanced Mode

```
┌─────────────────────────────────────────────────────────────┐
│  Training Configuration                                     │
│  ─────────────────────                                      │
│                                                             │
│  Dataset Path:   [examples/data/demo_dataset.jsonl    ]     │
│  Output Dir:     [./tmp/model                         ]     │
│  Base Model:     [DistilBERT (66M)               ▼   ]     │
│  Epochs: [3]     Batch Size: [4]    Learning Rate: [0.0001] │
│                                                             │
│  ☐ Use LoRA Adapter (Parameter-Efficient Fine-Tuning)       │
│                                                             │
│  [Start Training]                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Choosing the Right Interface

```
Need to share quickly? ──────────────────────▶ Gradio
                │
                ├── Want to explore data? ──▶ Streamlit
                │
                └── Production use? ────────▶ React Dashboard
                        │
                        ├── Beginner? ────▶ Easy Mode
                        │
                        └── Expert? ──────▶ Advanced Mode
```

---

## Data Format

All interfaces accept the same formats:

### CSV
```csv
question,answer
"What is AI?","AI stands for Artificial Intelligence..."
```

### JSONL
```jsonl
{"input": "What is AI?", "output": "AI stands for..."}
```

---

## See Also

- [Beginner's Course](Beginners-Course) - Learn AI from scratch
- [Training Your First Model](Training-Your-First-Model) - CLI training guide
- [Data Formats](Data-Formats) - Detailed format documentation
