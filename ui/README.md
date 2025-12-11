# TinyForgeAI Training User Interfaces

Three training interfaces designed for different users and use cases. Choose the one that fits your needs!

---

## Quick Comparison

| Interface | Best For | Technical Level | Launch Time | Features |
|-----------|----------|-----------------|-------------|----------|
| **Gradio** | Demos, quick tests | Beginner | Instant | Shareable links |
| **Streamlit** | Data exploration | Intermediate | Instant | Rich visualizations |
| **React Dashboard** | Production | All levels | Requires setup | Full platform |

---

## 1. Gradio Interface

**Best for:** Quick demos, beginners, sharing with others

### Features
- Instant shareable links (Gradio's public URL feature)
- Simple step-by-step interface
- File upload with preview
- Model selection with recommendations
- Real-time training progress
- Download trained models as ZIP

### Quick Start

```bash
# Install dependencies
pip install gradio torch transformers

# Run the app
cd ui/gradio
python training_app.py
```

Opens at: `http://localhost:7860`

### Screenshots

```
┌─────────────────────────────────────────────────────────────┐
│  📁 1. Upload Data  │  🚀 2. Train Model  │  🧪 3. Test  │  📥 4. Download  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│     ┌─────────────────────────────────────┐                 │
│     │      📄 Drag & Drop Here            │                 │
│     │      or click to browse             │                 │
│     │      .csv, .jsonl files             │                 │
│     └─────────────────────────────────────┘                 │
│                                                             │
│     Data Preview:                                           │
│     ┌───────────────────────────────────────┐               │
│     │ question          │ answer            │               │
│     ├───────────────────┼───────────────────┤               │
│     │ What is AI?       │ Artificial...     │               │
│     └───────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Use Cases
- Live demos at conferences
- Quick model prototyping
- Sharing models with non-technical stakeholders
- Educational workshops

---

## 2. Streamlit Interface

**Best for:** Data scientists, exploratory work, detailed analysis

### Features
- Rich data exploration with statistics
- Interactive Plotly visualizations
- Training with real-time callbacks
- Batch testing with metrics
- Export models and configurations
- Session state for persistence

### Quick Start

```bash
# Install dependencies
pip install streamlit plotly torch transformers

# Run the app
cd ui/streamlit
streamlit run training_app.py
```

Opens at: `http://localhost:8501`

### Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│  📁 Upload Data  │  📊 Explore  │  🚀 Train  │  🧪 Test  │  📥 Export  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 DATA EXPLORATION                                        │
│  ────────────────                                           │
│                                                             │
│  Dataset Statistics:                                        │
│  • Total Examples: 100                                      │
│  • Avg Input Length: 45 chars                               │
│  • Avg Output Length: 120 chars                             │
│                                                             │
│  ┌─────────────────────────────────────────┐                │
│  │     Input Length Distribution           │                │
│  │     ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░             │                │
│  │     ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░             │                │
│  │     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░             │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Use Cases
- Data quality analysis before training
- Comparing different training configurations
- Research and experimentation
- Detailed model evaluation

---

## 3. React Dashboard

**Best for:** Production environments, enterprise use, full platform experience

### Features
- Easy Mode (wizard) for beginners
- Advanced Mode for power users
- Real-time WebSocket updates
- Job management (cancel, delete)
- CLI command equivalents
- API connection status
- Full platform integration

### Quick Start

```bash
# Install frontend dependencies
cd dashboard
npm install

# Start the dashboard
npm run dev

# In another terminal, start the API
cd ..
python -m backend.api.main
```

Opens at: `http://localhost:5173`

### Modes

#### Easy Mode (Wizard)
Step-by-step wizard for non-technical users:

```
┌─────────────────────────────────────────────────────────────┐
│  ① Upload Data  │  ② Choose Model  │  ③ Settings  │  ④ Train  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  STEP 1: Upload Your Training Data                          │
│  ──────────────────────────────                             │
│                                                             │
│  💡 Tip: Your data should have questions and answers        │
│                                                             │
│     ┌─────────────────────────────────────┐                 │
│     │      📄 Drag & Drop Here            │                 │
│     │      or click to browse             │                 │
│     └─────────────────────────────────────┘                 │
│                                                             │
│                              [← Back]  [Continue →]         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Advanced Mode
Full control for technical users:

```
┌─────────────────────────────────────────────────────────────┐
│  Train Model                           [🧙 Easy] [⚙️ Advanced]│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Training Configuration                                     │
│  ─────────────────────                                      │
│                                                             │
│  Dataset Path:        [examples/data/demo_dataset.jsonl  ]  │
│  Output Directory:    [./tmp/model                       ]  │
│  Base Model:          [Flan-T5 Small (77M)          ▼   ]  │
│  Epochs:              [3     ]  Batch Size:    [4     ]     │
│  Learning Rate:       [0.0001]                              │
│                                                             │
│  ☐ Use LoRA Adapter (Parameter-Efficient Fine-Tuning)       │
│                                                             │
│  [Start Training]                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Use Cases
- Enterprise deployments
- Multi-user environments
- Production model management
- Integration with existing infrastructure

---

## Choosing the Right Interface

### Decision Tree

```
Start Here
    │
    ├── Need to share quickly with others?
    │   └── YES → Gradio (shareable links)
    │
    ├── Want to explore/analyze data?
    │   └── YES → Streamlit (rich visualizations)
    │
    ├── Production environment?
    │   └── YES → React Dashboard (full platform)
    │
    └── Just getting started?
        └── Gradio or React Dashboard Easy Mode
```

### Feature Matrix

| Feature | Gradio | Streamlit | React Dashboard |
|---------|--------|-----------|-----------------|
| File Upload | ✅ | ✅ | ✅ |
| Data Preview | ✅ | ✅ | ✅ |
| Data Statistics | ❌ | ✅ | ❌ |
| Visualizations | ❌ | ✅ | ❌ |
| Model Selection | ✅ | ✅ | ✅ |
| Training Progress | ✅ | ✅ | ✅ |
| Batch Testing | ❌ | ✅ | ❌ |
| Model Download | ✅ | ✅ | ❌ |
| Job Management | ❌ | ❌ | ✅ |
| WebSocket Updates | ❌ | ❌ | ✅ |
| Beginner Wizard | ❌ | ❌ | ✅ |
| CLI Equivalent | ❌ | ❌ | ✅ |
| Shareable Links | ✅ | ❌ | ❌ |
| No Server Needed | ✅ | ✅ | ❌ |

---

## Installation

### All Interfaces

```bash
# Core dependencies
pip install torch transformers

# Gradio
pip install gradio

# Streamlit
pip install streamlit plotly

# React Dashboard
cd dashboard
npm install
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/foremsoft/TinyForgeAI.git
cd TinyForgeAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install all dependencies
pip install -e ".[dev]"
pip install gradio streamlit plotly

# Run tests
pytest tests/
```

---

## Data Format

All interfaces accept the same data formats:

### CSV Format
```csv
question,answer
"What is AI?","AI stands for Artificial Intelligence..."
"How does ML work?","Machine learning uses algorithms..."
```

### JSONL Format
```jsonl
{"input": "What is AI?", "output": "AI stands for Artificial Intelligence..."}
{"input": "How does ML work?", "output": "Machine learning uses algorithms..."}
```

### Sample Data
Sample training data is provided in `examples/tutorial_data/`:
- `sample_faqs.csv` - 15 FAQ pairs
- `sample_training_data.jsonl` - 25 training examples

---

## Architecture

```
ui/
├── gradio/
│   ├── training_app.py      # Gradio interface
│   └── requirements.txt     # Gradio dependencies
│
├── streamlit/
│   ├── training_app.py      # Streamlit interface
│   └── requirements.txt     # Streamlit dependencies
│
└── README.md                # This file

dashboard/
├── src/
│   ├── components/
│   │   └── TrainingWizard.jsx   # Easy mode wizard
│   ├── pages/
│   │   └── TrainPage.jsx        # Main training page
│   └── api/
│       └── client.js            # API client
└── package.json
```

---

## Contributing

We welcome contributions! Areas to help:

1. **New Visualizations** - Add charts to Gradio/Streamlit
2. **Accessibility** - Improve keyboard navigation
3. **Internationalization** - Add language support
4. **Mobile Support** - Responsive design improvements
5. **Testing** - Unit tests for components

---

## Support

- **Documentation:** [TinyForgeAI Wiki](https://github.com/foremsoft/TinyForgeAI/wiki)
- **Issues:** [GitHub Issues](https://github.com/foremsoft/TinyForgeAI/issues)
- **Discussions:** [GitHub Discussions](https://github.com/foremsoft/TinyForgeAI/discussions)

---

## License

Apache 2.0 - Free for personal and commercial use.
