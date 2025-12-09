"""
Model Zoo Registry

Central registry for all pre-configured model configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class TaskType(str, Enum):
    """Supported NLP task types."""
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    CODE_GENERATION = "code_generation"
    CONVERSATION = "conversation"
    SENTIMENT = "sentiment"
    NER = "named_entity_recognition"
    TRANSLATION = "translation"
    TEXT_GENERATION = "text_generation"


@dataclass
class ModelConfig:
    """Pre-configured model configuration."""
    # Identification
    name: str
    display_name: str
    description: str
    task_type: TaskType

    # Model settings
    base_model: str
    model_type: str  # seq2seq, causal, encoder

    # Training defaults
    default_epochs: int = 3
    default_batch_size: int = 8
    default_learning_rate: float = 2e-5
    max_input_length: int = 512
    max_output_length: int = 128

    # LoRA settings
    lora_recommended: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(default_factory=list)

    # Resource requirements
    min_gpu_memory_gb: float = 4.0
    cpu_compatible: bool = True
    estimated_training_time_per_epoch: str = "5-10 min"

    # Data format
    input_field: str = "input"
    output_field: str = "output"
    data_format_example: Dict[str, str] = field(default_factory=dict)

    # Tags and metadata
    tags: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    paper_url: Optional[str] = None
    model_url: Optional[str] = None

    def to_training_config(self) -> Dict[str, Any]:
        """Convert to training configuration dict."""
        return {
            "model_name": self.base_model,
            "num_epochs": self.default_epochs,
            "batch_size": self.default_batch_size,
            "learning_rate": self.default_learning_rate,
            "max_length": self.max_input_length,
            "max_new_tokens": self.max_output_length,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "task_type": self.task_type.value,
            "base_model": self.base_model,
            "model_type": self.model_type,
            "training_defaults": {
                "epochs": self.default_epochs,
                "batch_size": self.default_batch_size,
                "learning_rate": self.default_learning_rate,
                "max_input_length": self.max_input_length,
                "max_output_length": self.max_output_length,
            },
            "lora": {
                "recommended": self.lora_recommended,
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "target_modules": self.lora_target_modules,
            },
            "resources": {
                "min_gpu_memory_gb": self.min_gpu_memory_gb,
                "cpu_compatible": self.cpu_compatible,
                "estimated_time_per_epoch": self.estimated_training_time_per_epoch,
            },
            "data_format": {
                "input_field": self.input_field,
                "output_field": self.output_field,
                "example": self.data_format_example,
            },
            "metadata": {
                "tags": self.tags,
                "use_cases": self.use_cases,
                "paper_url": self.paper_url,
                "model_url": self.model_url,
            },
        }


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY: Dict[str, ModelConfig] = {}


def register_model(config: ModelConfig) -> None:
    """Register a model configuration."""
    MODEL_REGISTRY[config.name] = config


def load_model_config(name: str) -> ModelConfig:
    """Load a model configuration by name."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}' not found. Available: {available}")
    return MODEL_REGISTRY[name]


def list_models(task_type: Optional[TaskType] = None) -> List[Dict[str, Any]]:
    """List all available models, optionally filtered by task type."""
    models = []
    for config in MODEL_REGISTRY.values():
        if task_type is None or config.task_type == task_type:
            models.append({
                "name": config.name,
                "display_name": config.display_name,
                "task_type": config.task_type.value,
                "base_model": config.base_model,
                "description": config.description,
                "tags": config.tags,
            })
    return models


def get_model_info(name: str) -> Dict[str, Any]:
    """Get detailed information about a model."""
    config = load_model_config(name)
    return config.to_dict()


# =============================================================================
# Register Pre-configured Models
# =============================================================================

# Question Answering - Flan-T5 Small
register_model(ModelConfig(
    name="qa_flan_t5_small",
    display_name="Q&A (Flan-T5 Small)",
    description="Question answering model using Flan-T5 Small. Great for FAQ bots, customer support, and knowledge base applications.",
    task_type=TaskType.QUESTION_ANSWERING,
    base_model="google/flan-t5-small",
    model_type="seq2seq",
    default_epochs=3,
    default_batch_size=8,
    default_learning_rate=3e-4,
    max_input_length=512,
    max_output_length=128,
    lora_recommended=False,
    min_gpu_memory_gb=2.0,
    cpu_compatible=True,
    estimated_training_time_per_epoch="3-5 min",
    data_format_example={
        "input": "What is your return policy?",
        "output": "You can return items within 30 days of purchase with a receipt."
    },
    tags=["qa", "faq", "small", "fast"],
    use_cases=["FAQ bots", "Customer support", "Knowledge bases", "Help desk"],
    model_url="https://huggingface.co/google/flan-t5-small",
))

# Question Answering - Flan-T5 Base
register_model(ModelConfig(
    name="qa_flan_t5_base",
    display_name="Q&A (Flan-T5 Base)",
    description="Question answering model using Flan-T5 Base. Better accuracy than small variant for complex Q&A tasks.",
    task_type=TaskType.QUESTION_ANSWERING,
    base_model="google/flan-t5-base",
    model_type="seq2seq",
    default_epochs=3,
    default_batch_size=4,
    default_learning_rate=2e-4,
    max_input_length=512,
    max_output_length=256,
    lora_recommended=True,
    lora_rank=8,
    lora_target_modules=["q", "v"],
    min_gpu_memory_gb=4.0,
    cpu_compatible=True,
    estimated_training_time_per_epoch="10-15 min",
    data_format_example={
        "input": "Explain quantum computing",
        "output": "Quantum computing uses quantum mechanics principles like superposition and entanglement to perform calculations."
    },
    tags=["qa", "medium", "accurate"],
    use_cases=["Technical Q&A", "Educational chatbots", "Expert systems"],
    model_url="https://huggingface.co/google/flan-t5-base",
))

# Summarization - T5 Small
register_model(ModelConfig(
    name="summarization_t5_small",
    display_name="Summarization (T5 Small)",
    description="Text summarization model using T5 Small. Ideal for summarizing articles, documents, and long-form content.",
    task_type=TaskType.SUMMARIZATION,
    base_model="t5-small",
    model_type="seq2seq",
    default_epochs=4,
    default_batch_size=4,
    default_learning_rate=3e-4,
    max_input_length=1024,
    max_output_length=150,
    lora_recommended=False,
    min_gpu_memory_gb=2.0,
    cpu_compatible=True,
    estimated_training_time_per_epoch="5-8 min",
    input_field="document",
    output_field="summary",
    data_format_example={
        "document": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is often used for typography testing.",
        "summary": "A pangram sentence used for typography testing."
    },
    tags=["summarization", "small", "fast"],
    use_cases=["News summarization", "Document condensation", "Meeting notes", "Report generation"],
    model_url="https://huggingface.co/t5-small",
))

# Summarization - BART
register_model(ModelConfig(
    name="summarization_bart",
    display_name="Summarization (BART)",
    description="High-quality text summarization using BART. Produces more fluent and coherent summaries.",
    task_type=TaskType.SUMMARIZATION,
    base_model="facebook/bart-base",
    model_type="seq2seq",
    default_epochs=3,
    default_batch_size=2,
    default_learning_rate=2e-5,
    max_input_length=1024,
    max_output_length=256,
    lora_recommended=True,
    lora_rank=16,
    lora_target_modules=["q_proj", "v_proj"],
    min_gpu_memory_gb=6.0,
    cpu_compatible=False,
    estimated_training_time_per_epoch="15-20 min",
    input_field="document",
    output_field="summary",
    data_format_example={
        "document": "Long article text here...",
        "summary": "Concise summary of the main points."
    },
    tags=["summarization", "bart", "high-quality"],
    use_cases=["Professional summaries", "Legal documents", "Academic papers"],
    model_url="https://huggingface.co/facebook/bart-base",
))

# Text Classification - DistilBERT
register_model(ModelConfig(
    name="classification_distilbert",
    display_name="Classification (DistilBERT)",
    description="Fast text classification using DistilBERT. Perfect for sentiment analysis, topic classification, and spam detection.",
    task_type=TaskType.CLASSIFICATION,
    base_model="distilbert-base-uncased",
    model_type="encoder",
    default_epochs=3,
    default_batch_size=16,
    default_learning_rate=2e-5,
    max_input_length=512,
    max_output_length=1,
    lora_recommended=False,
    min_gpu_memory_gb=2.0,
    cpu_compatible=True,
    estimated_training_time_per_epoch="2-4 min",
    input_field="text",
    output_field="label",
    data_format_example={
        "text": "This product is amazing! Best purchase ever.",
        "label": "positive"
    },
    tags=["classification", "fast", "efficient"],
    use_cases=["Sentiment analysis", "Spam detection", "Topic classification", "Intent detection"],
    model_url="https://huggingface.co/distilbert-base-uncased",
))

# Sentiment Analysis - RoBERTa
register_model(ModelConfig(
    name="sentiment_roberta",
    display_name="Sentiment (RoBERTa)",
    description="Sentiment analysis using RoBERTa. More accurate than DistilBERT for nuanced sentiment detection.",
    task_type=TaskType.SENTIMENT,
    base_model="roberta-base",
    model_type="encoder",
    default_epochs=3,
    default_batch_size=8,
    default_learning_rate=1e-5,
    max_input_length=512,
    max_output_length=1,
    lora_recommended=True,
    lora_rank=8,
    lora_target_modules=["query", "value"],
    min_gpu_memory_gb=4.0,
    cpu_compatible=True,
    estimated_training_time_per_epoch="5-8 min",
    input_field="text",
    output_field="sentiment",
    data_format_example={
        "text": "The service was okay, but I expected more for the price.",
        "sentiment": "mixed"
    },
    tags=["sentiment", "roberta", "accurate"],
    use_cases=["Product reviews", "Social media monitoring", "Customer feedback", "Brand analysis"],
    model_url="https://huggingface.co/roberta-base",
))

# Code Generation - CodeGen Small
register_model(ModelConfig(
    name="code_gen_small",
    display_name="Code Generation (CodeGen Small)",
    description="Code generation using Salesforce CodeGen. Great for code completion, generation, and documentation.",
    task_type=TaskType.CODE_GENERATION,
    base_model="Salesforce/codegen-350M-mono",
    model_type="causal",
    default_epochs=2,
    default_batch_size=2,
    default_learning_rate=5e-5,
    max_input_length=512,
    max_output_length=256,
    lora_recommended=True,
    lora_rank=16,
    lora_target_modules=["c_attn", "c_proj"],
    min_gpu_memory_gb=4.0,
    cpu_compatible=False,
    estimated_training_time_per_epoch="10-15 min",
    input_field="prompt",
    output_field="code",
    data_format_example={
        "prompt": "# Function to calculate factorial",
        "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
    },
    tags=["code", "generation", "python"],
    use_cases=["Code completion", "Function generation", "Code documentation", "Boilerplate generation"],
    model_url="https://huggingface.co/Salesforce/codegen-350M-mono",
))

# Conversational AI - GPT-2 Small
register_model(ModelConfig(
    name="chat_gpt2_small",
    display_name="Chat (GPT-2 Small)",
    description="Conversational AI using GPT-2 Small. Fast and lightweight for basic chatbot applications.",
    task_type=TaskType.CONVERSATION,
    base_model="gpt2",
    model_type="causal",
    default_epochs=3,
    default_batch_size=4,
    default_learning_rate=5e-5,
    max_input_length=512,
    max_output_length=128,
    lora_recommended=False,
    min_gpu_memory_gb=2.0,
    cpu_compatible=True,
    estimated_training_time_per_epoch="5-8 min",
    input_field="context",
    output_field="response",
    data_format_example={
        "context": "User: Hello, how are you today?",
        "response": "I'm doing great, thank you for asking! How can I help you?"
    },
    tags=["chat", "conversational", "gpt2", "small"],
    use_cases=["Simple chatbots", "Customer service", "Virtual assistants", "Interactive applications"],
    model_url="https://huggingface.co/gpt2",
))

# Conversational AI - DialoGPT
register_model(ModelConfig(
    name="chat_dialogpt",
    display_name="Chat (DialoGPT)",
    description="Conversational AI using Microsoft DialoGPT. Specifically trained for dialogue, produces more natural conversations.",
    task_type=TaskType.CONVERSATION,
    base_model="microsoft/DialoGPT-small",
    model_type="causal",
    default_epochs=3,
    default_batch_size=4,
    default_learning_rate=3e-5,
    max_input_length=512,
    max_output_length=128,
    lora_recommended=True,
    lora_rank=8,
    lora_target_modules=["c_attn"],
    min_gpu_memory_gb=2.0,
    cpu_compatible=True,
    estimated_training_time_per_epoch="5-8 min",
    input_field="context",
    output_field="response",
    data_format_example={
        "context": "I'm planning a trip to Paris next month.",
        "response": "That sounds exciting! Paris is beautiful in spring. Have you decided which attractions you want to visit?"
    },
    tags=["chat", "dialogue", "natural"],
    use_cases=["Conversational bots", "Social chatbots", "Entertainment", "Companions"],
    model_url="https://huggingface.co/microsoft/DialoGPT-small",
))

# Named Entity Recognition - BERT
register_model(ModelConfig(
    name="ner_bert",
    display_name="NER (BERT)",
    description="Named Entity Recognition using BERT. Identifies names, organizations, locations, and other entities in text.",
    task_type=TaskType.NER,
    base_model="bert-base-uncased",
    model_type="encoder",
    default_epochs=3,
    default_batch_size=16,
    default_learning_rate=3e-5,
    max_input_length=512,
    max_output_length=512,
    lora_recommended=False,
    min_gpu_memory_gb=4.0,
    cpu_compatible=True,
    estimated_training_time_per_epoch="5-10 min",
    input_field="text",
    output_field="entities",
    data_format_example={
        "text": "John Smith works at Google in Mountain View.",
        "entities": "[{\"text\": \"John Smith\", \"label\": \"PERSON\"}, {\"text\": \"Google\", \"label\": \"ORG\"}, {\"text\": \"Mountain View\", \"label\": \"LOC\"}]"
    },
    tags=["ner", "entities", "bert"],
    use_cases=["Information extraction", "Document processing", "Search enhancement", "Data mining"],
    model_url="https://huggingface.co/bert-base-uncased",
))

# Translation - MarianMT (English to Spanish)
register_model(ModelConfig(
    name="translation_en_es",
    display_name="Translation (EN→ES)",
    description="English to Spanish translation using MarianMT. Fast and accurate for common translation tasks.",
    task_type=TaskType.TRANSLATION,
    base_model="Helsinki-NLP/opus-mt-en-es",
    model_type="seq2seq",
    default_epochs=3,
    default_batch_size=8,
    default_learning_rate=2e-5,
    max_input_length=512,
    max_output_length=512,
    lora_recommended=False,
    min_gpu_memory_gb=2.0,
    cpu_compatible=True,
    estimated_training_time_per_epoch="5-8 min",
    input_field="source",
    output_field="target",
    data_format_example={
        "source": "Hello, how are you?",
        "target": "Hola, ¿cómo estás?"
    },
    tags=["translation", "en-es", "marian"],
    use_cases=["Document translation", "Localization", "Multilingual support"],
    model_url="https://huggingface.co/Helsinki-NLP/opus-mt-en-es",
))

# Translation - MarianMT (English to French)
register_model(ModelConfig(
    name="translation_en_fr",
    display_name="Translation (EN→FR)",
    description="English to French translation using MarianMT.",
    task_type=TaskType.TRANSLATION,
    base_model="Helsinki-NLP/opus-mt-en-fr",
    model_type="seq2seq",
    default_epochs=3,
    default_batch_size=8,
    default_learning_rate=2e-5,
    max_input_length=512,
    max_output_length=512,
    lora_recommended=False,
    min_gpu_memory_gb=2.0,
    cpu_compatible=True,
    estimated_training_time_per_epoch="5-8 min",
    input_field="source",
    output_field="target",
    data_format_example={
        "source": "The weather is nice today.",
        "target": "Le temps est beau aujourd'hui."
    },
    tags=["translation", "en-fr", "marian"],
    use_cases=["Document translation", "Localization", "Multilingual support"],
    model_url="https://huggingface.co/Helsinki-NLP/opus-mt-en-fr",
))

# Text Generation - GPT-2 Medium (with LoRA)
register_model(ModelConfig(
    name="text_gen_gpt2_medium",
    display_name="Text Generation (GPT-2 Medium)",
    description="Creative text generation using GPT-2 Medium with LoRA. For story generation, content creation, and creative writing.",
    task_type=TaskType.TEXT_GENERATION,
    base_model="gpt2-medium",
    model_type="causal",
    default_epochs=2,
    default_batch_size=2,
    default_learning_rate=3e-5,
    max_input_length=512,
    max_output_length=256,
    lora_recommended=True,
    lora_rank=16,
    lora_alpha=32,
    lora_target_modules=["c_attn", "c_proj"],
    min_gpu_memory_gb=6.0,
    cpu_compatible=False,
    estimated_training_time_per_epoch="15-20 min",
    input_field="prompt",
    output_field="continuation",
    data_format_example={
        "prompt": "Once upon a time in a distant galaxy,",
        "continuation": "there lived a young explorer who dreamed of discovering new worlds."
    },
    tags=["text-generation", "creative", "gpt2", "lora"],
    use_cases=["Story generation", "Content creation", "Marketing copy", "Creative writing assistance"],
    model_url="https://huggingface.co/gpt2-medium",
))
