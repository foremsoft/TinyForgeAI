"""
TinyForgeAI Training Interface - Gradio Version

A simple, shareable UI for training AI models.
Perfect for demos, beginners, and quick experiments.

Usage:
    pip install gradio torch transformers
    python training_app.py

Features:
    - Upload CSV/JSONL data
    - Select model and parameters
    - Train with visual progress
    - Test trained model
    - Download model files
"""

import gradio as gr
import json
import csv
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List
import threading

# Training imports (with fallback for demo mode)
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer
    )
    from torch.utils.data import Dataset
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    print("⚠️ PyTorch/Transformers not installed. Running in demo mode.")


# ============================================================
# Data Processing
# ============================================================

def parse_uploaded_file(file) -> Tuple[List[dict], str]:
    """Parse uploaded CSV or JSONL file."""
    if file is None:
        return [], "No file uploaded"

    file_path = file.name if hasattr(file, 'name') else str(file)
    examples = []

    try:
        if file_path.endswith('.csv'):
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try common column names
                    input_col = None
                    output_col = None

                    for col in ['input', 'question', 'text', 'query']:
                        if col in row:
                            input_col = col
                            break

                    for col in ['output', 'answer', 'response', 'label']:
                        if col in row:
                            output_col = col
                            break

                    if input_col and output_col:
                        examples.append({
                            'input': row[input_col],
                            'output': row[output_col]
                        })

        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if 'input' in data and 'output' in data:
                            examples.append(data)
        else:
            return [], f"Unsupported file format. Use .csv or .jsonl"

    except Exception as e:
        return [], f"Error reading file: {str(e)}"

    if not examples:
        return [], "No valid examples found. Check your file format."

    return examples, f"✅ Loaded {len(examples)} examples"


def preview_data(file) -> str:
    """Preview uploaded data."""
    examples, message = parse_uploaded_file(file)

    if not examples:
        return message

    preview = f"### Data Preview ({len(examples)} total examples)\n\n"
    preview += "| # | Input | Output |\n"
    preview += "|---|-------|--------|\n"

    for i, ex in enumerate(examples[:5]):
        input_text = ex['input'][:50] + "..." if len(ex['input']) > 50 else ex['input']
        output_text = ex['output'][:50] + "..." if len(ex['output']) > 50 else ex['output']
        preview += f"| {i+1} | {input_text} | {output_text} |\n"

    if len(examples) > 5:
        preview += f"\n*...and {len(examples) - 5} more examples*"

    # Data statistics
    preview += f"\n\n### Statistics\n"
    preview += f"- Total examples: {len(examples)}\n"
    preview += f"- Unique outputs: {len(set(ex['output'] for ex in examples))}\n"
    avg_input = sum(len(ex['input']) for ex in examples) / len(examples)
    avg_output = sum(len(ex['output']) for ex in examples) / len(examples)
    preview += f"- Avg input length: {avg_input:.0f} chars\n"
    preview += f"- Avg output length: {avg_output:.0f} chars\n"

    return preview


# ============================================================
# Training Logic
# ============================================================

class SimpleDataset(Dataset):
    """Simple dataset for training."""

    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Create label mapping
        self.labels = list(set(ex['output'] for ex in examples))
        self.label2id = {label: i for i, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        encoding = self.tokenizer(
            ex['input'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.label2id[ex['output']])
        }


# Global state for training
training_state = {
    'progress': 0,
    'status': 'idle',
    'log': [],
    'model_path': None
}


def train_model(
    file,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    progress=gr.Progress()
) -> str:
    """Train the model with uploaded data."""

    global training_state
    training_state['status'] = 'starting'
    training_state['log'] = []
    training_state['progress'] = 0

    def log(msg):
        training_state['log'].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    # Check if training is available
    if not TRAINING_AVAILABLE:
        return """### ⚠️ Demo Mode

PyTorch and Transformers are not installed.
Install them to enable training:

```bash
pip install torch transformers
```

In demo mode, you can:
- Upload and preview data
- See the training interface
- But actual training requires the libraries
"""

    # Parse data
    log("Loading data...")
    progress(0.1, desc="Loading data...")
    examples, message = parse_uploaded_file(file)

    if not examples:
        training_state['status'] = 'error'
        return f"### ❌ Error\n\n{message}"

    log(f"Loaded {len(examples)} examples")

    # Setup model
    log(f"Loading model: {model_name}")
    progress(0.2, desc="Loading model...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create dataset to get num_labels
        dataset = SimpleDataset(examples, tokenizer)
        num_labels = len(dataset.labels)

        log(f"Found {num_labels} unique classes")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        # Update model config
        model.config.label2id = dataset.label2id
        model.config.id2label = {v: k for k, v in dataset.label2id.items()}

    except Exception as e:
        training_state['status'] = 'error'
        return f"### ❌ Error loading model\n\n{str(e)}"

    # Create output directory
    output_dir = tempfile.mkdtemp(prefix="tinyforge_model_")
    log(f"Output directory: {output_dir}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        disable_tqdm=True
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    # Train
    log("Starting training...")
    training_state['status'] = 'training'

    try:
        for epoch in range(epochs):
            progress((0.3 + 0.5 * epoch / epochs), desc=f"Training epoch {epoch+1}/{epochs}...")
            log(f"Epoch {epoch + 1}/{epochs}")

        trainer.train()

        log("Training complete!")
        progress(0.9, desc="Saving model...")

        # Save model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save label mapping
        with open(os.path.join(output_dir, 'label_mapping.json'), 'w') as f:
            json.dump({
                'label2id': dataset.label2id,
                'id2label': {str(v): k for k, v in dataset.label2id.items()}
            }, f, indent=2)

        training_state['model_path'] = output_dir
        training_state['status'] = 'complete'

        progress(1.0, desc="Complete!")
        log(f"Model saved to: {output_dir}")

        return f"""### ✅ Training Complete!

**Model saved to:** `{output_dir}`

**Training Summary:**
- Examples: {len(examples)}
- Classes: {num_labels}
- Epochs: {epochs}
- Model: {model_name}

**Next Steps:**
1. Test your model in the "Test Model" tab
2. Download the model files
3. Deploy using TinyForgeAI

**Training Log:**
```
{chr(10).join(training_state['log'][-10:])}
```
"""

    except Exception as e:
        training_state['status'] = 'error'
        return f"### ❌ Training Error\n\n{str(e)}\n\n**Log:**\n```\n{chr(10).join(training_state['log'])}\n```"


# ============================================================
# Model Testing
# ============================================================

def test_model(question: str) -> str:
    """Test the trained model with a question."""

    if not TRAINING_AVAILABLE:
        return "⚠️ Training libraries not installed. Cannot test model."

    if training_state['model_path'] is None:
        return "⚠️ No model trained yet. Train a model first!"

    if not question.strip():
        return "Please enter a question."

    try:
        model_path = training_state['model_path']

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()

        # Load label mapping
        with open(os.path.join(model_path, 'label_mapping.json'), 'r') as f:
            mapping = json.load(f)
            id2label = {int(k): v for k, v in mapping['id2label'].items()}

        # Predict
        inputs = tokenizer(
            question,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        top_prob, top_idx = torch.max(probs, dim=-1)
        answer = id2label[top_idx.item()]
        confidence = top_prob.item()

        # Get top 3
        top_k = torch.topk(probs[0], k=min(3, len(id2label)))

        result = f"### 🤖 Answer\n\n**{answer}**\n\nConfidence: {confidence*100:.1f}%\n\n"
        result += "### Other Possibilities\n\n"

        for prob, idx in zip(top_k.values[1:], top_k.indices[1:]):
            result += f"- {id2label[idx.item()]}: {prob.item()*100:.1f}%\n"

        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"


def download_model():
    """Create a downloadable zip of the model."""

    if training_state['model_path'] is None:
        return None

    # Create zip file
    zip_path = tempfile.mktemp(suffix='.zip')
    shutil.make_archive(zip_path[:-4], 'zip', training_state['model_path'])

    return zip_path


# ============================================================
# Gradio Interface
# ============================================================

def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="TinyForgeAI Training",
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 1200px !important; }
            .main-header { text-align: center; margin-bottom: 20px; }
        """
    ) as app:

        gr.Markdown("""
        # 🤖 TinyForgeAI Training Interface

        Train your own AI model in minutes - no coding required!

        **Steps:** Upload Data → Configure → Train → Test → Download
        """)

        with gr.Tabs():
            # ==================== Tab 1: Upload Data ====================
            with gr.Tab("📁 1. Upload Data"):
                gr.Markdown("""
                ### Upload Your Training Data

                Supported formats:
                - **CSV**: Must have columns for questions and answers
                - **JSONL**: Each line should have `{"input": "...", "output": "..."}`
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        file_upload = gr.File(
                            label="Upload CSV or JSONL file",
                            file_types=[".csv", ".jsonl"]
                        )
                        preview_btn = gr.Button("Preview Data", variant="primary")

                    with gr.Column(scale=2):
                        data_preview = gr.Markdown("Upload a file to see preview")

                preview_btn.click(
                    fn=preview_data,
                    inputs=[file_upload],
                    outputs=[data_preview]
                )

                gr.Markdown("""
                ### Sample Data Format

                **CSV:**
                ```
                question,answer
                What are your hours?,We are open 9 AM to 6 PM.
                How do I contact you?,Email support@example.com
                ```

                **JSONL:**
                ```
                {"input": "What are your hours?", "output": "We are open 9 AM to 6 PM."}
                {"input": "How do I contact you?", "output": "Email support@example.com"}
                ```
                """)

            # ==================== Tab 2: Configure & Train ====================
            with gr.Tab("🚀 2. Train Model"):
                gr.Markdown("### Configure Training Parameters")

                with gr.Row():
                    with gr.Column():
                        model_dropdown = gr.Dropdown(
                            choices=[
                                ("DistilBERT (Fast, Good for beginners)", "distilbert-base-uncased"),
                                ("BERT Base (More accurate)", "bert-base-uncased"),
                                ("RoBERTa (Best quality)", "roberta-base")
                            ],
                            value="distilbert-base-uncased",
                            label="Select Model"
                        )

                        epochs_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Training Epochs (more = better but slower)"
                        )

                    with gr.Column():
                        batch_slider = gr.Slider(
                            minimum=2,
                            maximum=32,
                            value=8,
                            step=2,
                            label="Batch Size (lower if out of memory)"
                        )

                        lr_dropdown = gr.Dropdown(
                            choices=[
                                ("Slow & Careful (1e-5)", 1e-5),
                                ("Normal (5e-5)", 5e-5),
                                ("Fast (1e-4)", 1e-4)
                            ],
                            value=5e-5,
                            label="Learning Rate"
                        )

                train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                training_output = gr.Markdown("Configure settings and click 'Start Training'")

                train_btn.click(
                    fn=train_model,
                    inputs=[file_upload, model_dropdown, epochs_slider, batch_slider, lr_dropdown],
                    outputs=[training_output]
                )

            # ==================== Tab 3: Test Model ====================
            with gr.Tab("🧪 3. Test Model"):
                gr.Markdown("### Test Your Trained Model")

                test_input = gr.Textbox(
                    label="Enter a question to test",
                    placeholder="What are your business hours?",
                    lines=2
                )
                test_btn = gr.Button("Get Answer", variant="primary")
                test_output = gr.Markdown("Train a model first, then test it here.")

                test_btn.click(
                    fn=test_model,
                    inputs=[test_input],
                    outputs=[test_output]
                )

                gr.Markdown("""
                ### Testing Tips

                - Try questions similar to your training data
                - Try variations (different wording, same meaning)
                - Note which questions get low confidence
                - Add those to your training data and retrain
                """)

            # ==================== Tab 4: Download ====================
            with gr.Tab("📥 4. Download"):
                gr.Markdown("""
                ### Download Your Trained Model

                After training, download your model to use it anywhere!
                """)

                download_btn = gr.Button("📥 Download Model (ZIP)", variant="primary")
                download_file = gr.File(label="Model Download")

                download_btn.click(
                    fn=download_model,
                    outputs=[download_file]
                )

                gr.Markdown("""
                ### Using Your Downloaded Model

                ```python
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                # Load your model
                model = AutoModelForSequenceClassification.from_pretrained("./my_model")
                tokenizer = AutoTokenizer.from_pretrained("./my_model")

                # Use it
                inputs = tokenizer("Your question here", return_tensors="pt")
                outputs = model(**inputs)
                ```

                See the [TinyForgeAI documentation](https://github.com/foremsoft/TinyForgeAI) for more deployment options.
                """)

        gr.Markdown("""
        ---

        **TinyForgeAI** - Making AI Training Accessible to Everyone

        [GitHub](https://github.com/foremsoft/TinyForgeAI) | [Documentation](https://github.com/foremsoft/TinyForgeAI/wiki) | [Tutorials](https://github.com/foremsoft/TinyForgeAI/tree/main/docs/tutorials)
        """)

    return app


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TinyForgeAI Training Interface (Gradio)")
    print("=" * 60)

    if not TRAINING_AVAILABLE:
        print("\n⚠️  Running in DEMO MODE")
        print("   Install PyTorch and Transformers for full functionality:")
        print("   pip install torch transformers")

    print("\nStarting server...")

    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public link
        show_error=True
    )
