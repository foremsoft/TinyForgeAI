"""
TinyForgeAI Training Interface - Streamlit Version

A powerful, data-scientist-friendly UI for training AI models.
Great for experimentation and iterative development.

Usage:
    pip install streamlit torch transformers plotly pandas
    streamlit run training_app.py

Features:
    - Upload CSV/JSONL data
    - Data exploration and visualization
    - Model selection and configuration
    - Real-time training metrics
    - Interactive model testing
    - Export and deployment options
"""

import streamlit as st
import pandas as pd
import json
import csv
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import time

# Visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Training imports
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        TrainerCallback
    )
    from torch.utils.data import Dataset
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="TinyForgeAI Training",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        color: #155724;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 5px;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State Initialization
# ============================================================

if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'model_path' not in st.session_state:
    st.session_state.model_path = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'is_training' not in st.session_state:
    st.session_state.is_training = False


# ============================================================
# Data Processing Functions
# ============================================================

@st.cache_data
def load_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
    """Load and parse uploaded file."""
    if uploaded_file is None:
        return None, "No file uploaded"

    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)

            # Find input/output columns
            input_col = None
            output_col = None

            for col in ['input', 'question', 'text', 'query']:
                if col in df.columns:
                    input_col = col
                    break

            for col in ['output', 'answer', 'response', 'label']:
                if col in df.columns:
                    output_col = col
                    break

            if input_col and output_col:
                df = df[[input_col, output_col]].rename(
                    columns={input_col: 'input', output_col: 'output'}
                )
                return df, f"✅ Loaded {len(df)} examples from CSV"
            else:
                return None, f"❌ Could not find input/output columns. Found: {list(df.columns)}"

        elif uploaded_file.name.endswith('.jsonl'):
            examples = []
            for line in uploaded_file:
                if line.strip():
                    data = json.loads(line)
                    if 'input' in data and 'output' in data:
                        examples.append(data)

            if examples:
                df = pd.DataFrame(examples)
                return df, f"✅ Loaded {len(df)} examples from JSONL"
            else:
                return None, "❌ No valid examples found in JSONL"

        else:
            return None, "❌ Unsupported file format. Use .csv or .jsonl"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def get_data_stats(df: pd.DataFrame) -> Dict:
    """Calculate statistics about the data."""
    return {
        'total_examples': len(df),
        'unique_outputs': df['output'].nunique(),
        'avg_input_length': df['input'].str.len().mean(),
        'avg_output_length': df['output'].str.len().mean(),
        'min_input_length': df['input'].str.len().min(),
        'max_input_length': df['input'].str.len().max(),
        'output_distribution': df['output'].value_counts().to_dict()
    }


# ============================================================
# Training Functions
# ============================================================

class StreamlitCallback(TrainerCallback):
    """Callback to update Streamlit during training."""

    def __init__(self, progress_bar, status_text, epochs):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.epochs = epochs
        self.current_epoch = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch += 1
        self.status_text.text(f"Training epoch {self.current_epoch}/{self.epochs}...")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            progress = state.global_step / state.max_steps
            self.progress_bar.progress(progress)
            if 'loss' in logs:
                st.session_state.training_history.append({
                    'step': state.global_step,
                    'loss': logs['loss']
                })


class SimpleDataset(Dataset):
    """Simple dataset for training."""

    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.labels = df['output'].unique().tolist()
        self.label2id = {label: i for i, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encoding = self.tokenizer(
            row['input'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.label2id[row['output']])
        }


def train_model(
    df: pd.DataFrame,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    progress_bar,
    status_text
) -> Tuple[bool, str]:
    """Train the model."""

    if not TRAINING_AVAILABLE:
        return False, "Training libraries not available"

    st.session_state.training_history = []

    try:
        status_text.text("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        dataset = SimpleDataset(df, tokenizer)
        num_labels = len(dataset.labels)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        model.config.label2id = dataset.label2id
        model.config.id2label = {v: k for k, v in dataset.label2id.items()}

        # Output directory
        output_dir = tempfile.mkdtemp(prefix="tinyforge_")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=5,
            save_strategy="epoch",
            report_to="none",
            disable_tqdm=True
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[StreamlitCallback(progress_bar, status_text, epochs)]
        )

        # Train
        status_text.text("Training...")
        trainer.train()

        # Save
        status_text.text("Saving model...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, 'label_mapping.json'), 'w') as f:
            json.dump({
                'label2id': dataset.label2id,
                'id2label': {str(v): k for k, v in dataset.label2id.items()}
            }, f, indent=2)

        st.session_state.model_path = output_dir
        progress_bar.progress(1.0)
        status_text.text("✅ Training complete!")

        return True, output_dir

    except Exception as e:
        return False, str(e)


def test_model(question: str) -> Dict:
    """Test the trained model."""

    if st.session_state.model_path is None:
        return {'error': 'No model trained'}

    try:
        model_path = st.session_state.model_path

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()

        with open(os.path.join(model_path, 'label_mapping.json'), 'r') as f:
            mapping = json.load(f)
            id2label = {int(k): v for k, v in mapping['id2label'].items()}

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

        # Get all predictions
        all_preds = []
        for idx, prob in enumerate(probs[0]):
            all_preds.append({
                'answer': id2label[idx],
                'confidence': prob.item()
            })
        all_preds.sort(key=lambda x: x['confidence'], reverse=True)

        return {
            'answer': id2label[top_idx.item()],
            'confidence': top_prob.item(),
            'all_predictions': all_preds[:5]
        }

    except Exception as e:
        return {'error': str(e)}


# ============================================================
# Main Application
# ============================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 TinyForgeAI Training Studio</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Train custom AI models with your data - no coding required</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📋 Quick Guide")
        st.markdown("""
        1. **Upload** your training data
        2. **Explore** your data statistics
        3. **Configure** training parameters
        4. **Train** your model
        5. **Test** with sample questions
        6. **Download** your trained model
        """)

        st.divider()

        st.header("⚙️ System Status")
        st.write(f"Training Available: {'✅' if TRAINING_AVAILABLE else '❌'}")
        st.write(f"Visualization: {'✅' if PLOTLY_AVAILABLE else '❌'}")

        if st.session_state.model_path:
            st.success(f"Model trained! ✅")

        st.divider()
        st.markdown("[📚 Documentation](https://github.com/foremsoft/TinyForgeAI)")
        st.markdown("[🐛 Report Issue](https://github.com/foremsoft/TinyForgeAI/issues)")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Upload Data",
        "📊 Explore Data",
        "🚀 Train Model",
        "🧪 Test Model",
        "📥 Export"
    ])

    # ==================== Tab 1: Upload Data ====================
    with tab1:
        st.header("Upload Training Data")

        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Choose a CSV or JSONL file",
                type=['csv', 'jsonl'],
                help="CSV: columns 'question' and 'answer' or 'input' and 'output'\nJSONL: each line has 'input' and 'output' fields"
            )

            if uploaded_file:
                df, message = load_data(uploaded_file)
                st.write(message)

                if df is not None:
                    st.session_state.training_data = df
                    st.success(f"Data loaded: {len(df)} examples")

        with col2:
            st.markdown("""
            ### Supported Formats

            **CSV Example:**
            ```csv
            question,answer
            What are your hours?,We are open 9-5.
            How do I contact you?,Email us at help@example.com
            ```

            **JSONL Example:**
            ```json
            {"input": "What are your hours?", "output": "We are open 9-5."}
            {"input": "How do I contact you?", "output": "Email us at help@example.com"}
            ```
            """)

        # Preview data
        if st.session_state.training_data is not None:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.training_data.head(10), use_container_width=True)

    # ==================== Tab 2: Explore Data ====================
    with tab2:
        st.header("Explore Your Data")

        if st.session_state.training_data is None:
            st.warning("Please upload data first")
        else:
            df = st.session_state.training_data
            stats = get_data_stats(df)

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Examples", stats['total_examples'])
            with col2:
                st.metric("Unique Outputs", stats['unique_outputs'])
            with col3:
                st.metric("Avg Input Length", f"{stats['avg_input_length']:.0f}")
            with col4:
                st.metric("Avg Output Length", f"{stats['avg_output_length']:.0f}")

            # Visualizations
            if PLOTLY_AVAILABLE:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Output Distribution")
                    output_counts = df['output'].value_counts().head(15)
                    fig = px.bar(
                        x=output_counts.values,
                        y=[o[:40] + "..." if len(o) > 40 else o for o in output_counts.index],
                        orientation='h',
                        labels={'x': 'Count', 'y': 'Output'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Input Length Distribution")
                    fig = px.histogram(
                        df,
                        x=df['input'].str.len(),
                        nbins=30,
                        labels={'x': 'Character Count'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

            # Search data
            st.subheader("Search Data")
            search_term = st.text_input("Search in inputs")
            if search_term:
                filtered = df[df['input'].str.contains(search_term, case=False)]
                st.write(f"Found {len(filtered)} matches")
                st.dataframe(filtered, use_container_width=True)

    # ==================== Tab 3: Train Model ====================
    with tab3:
        st.header("Train Your Model")

        if st.session_state.training_data is None:
            st.warning("Please upload data first")
        elif not TRAINING_AVAILABLE:
            st.error("Training libraries not installed. Run: `pip install torch transformers`")
        else:
            col1, col2 = st.columns(2)

            with col1:
                model_name = st.selectbox(
                    "Select Base Model",
                    options=[
                        ("DistilBERT (Fast)", "distilbert-base-uncased"),
                        ("BERT Base (Balanced)", "bert-base-uncased"),
                        ("RoBERTa (Best Quality)", "roberta-base")
                    ],
                    format_func=lambda x: x[0]
                )[1]

                epochs = st.slider("Training Epochs", 1, 10, 3)
                batch_size = st.slider("Batch Size", 2, 32, 8)

            with col2:
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[1e-5, 2e-5, 5e-5, 1e-4],
                    value=5e-5,
                    format_func=lambda x: f"{x:.0e}"
                )

                st.markdown(f"""
                ### Training Configuration
                - **Model:** {model_name}
                - **Examples:** {len(st.session_state.training_data)}
                - **Classes:** {st.session_state.training_data['output'].nunique()}
                - **Epochs:** {epochs}
                - **Batch Size:** {batch_size}
                - **Learning Rate:** {learning_rate}
                """)

            st.divider()

            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                success, result = train_model(
                    st.session_state.training_data,
                    model_name,
                    epochs,
                    batch_size,
                    learning_rate,
                    progress_bar,
                    status_text
                )

                if success:
                    st.success(f"✅ Model trained and saved to: {result}")

                    # Show training curve
                    if st.session_state.training_history and PLOTLY_AVAILABLE:
                        history_df = pd.DataFrame(st.session_state.training_history)
                        fig = px.line(history_df, x='step', y='loss', title='Training Loss')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"❌ Training failed: {result}")

    # ==================== Tab 4: Test Model ====================
    with tab4:
        st.header("Test Your Model")

        if st.session_state.model_path is None:
            st.warning("Please train a model first")
        else:
            test_question = st.text_input(
                "Enter a question to test",
                placeholder="What are your business hours?"
            )

            if st.button("Get Answer", type="primary"):
                if test_question:
                    result = test_model(test_question)

                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        st.success(f"**Answer:** {result['answer']}")
                        st.info(f"**Confidence:** {result['confidence']*100:.1f}%")

                        st.subheader("All Predictions")
                        for pred in result['all_predictions']:
                            confidence = pred['confidence'] * 100
                            st.progress(pred['confidence'])
                            st.write(f"{confidence:.1f}% - {pred['answer'][:100]}...")

            # Batch testing
            st.divider()
            st.subheader("Batch Testing")

            test_questions = st.text_area(
                "Enter multiple questions (one per line)",
                placeholder="What are your hours?\nHow do I contact support?\nDo you offer refunds?"
            )

            if st.button("Test All"):
                questions = [q.strip() for q in test_questions.split('\n') if q.strip()]
                if questions:
                    results = []
                    for q in questions:
                        r = test_model(q)
                        if 'error' not in r:
                            results.append({
                                'Question': q,
                                'Answer': r['answer'][:50] + "...",
                                'Confidence': f"{r['confidence']*100:.1f}%"
                            })

                    if results:
                        st.dataframe(pd.DataFrame(results), use_container_width=True)

    # ==================== Tab 5: Export ====================
    with tab5:
        st.header("Export Your Model")

        if st.session_state.model_path is None:
            st.warning("Please train a model first")
        else:
            st.success(f"Model ready at: `{st.session_state.model_path}`")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Download Model")

                if st.button("📦 Create ZIP Download"):
                    zip_path = tempfile.mktemp(suffix='.zip')
                    shutil.make_archive(zip_path[:-4], 'zip', st.session_state.model_path)

                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            "⬇️ Download Model ZIP",
                            data=f,
                            file_name="tinyforge_model.zip",
                            mime="application/zip"
                        )

            with col2:
                st.subheader("Usage Code")
                st.code("""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model
model = AutoModelForSequenceClassification.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")

# Predict
inputs = tokenizer("Your question", return_tensors="pt")
outputs = model(**inputs)
                """, language="python")


if __name__ == "__main__":
    main()
