"""
TinyForgeAI Model Zoo

Pre-configured model configurations for common NLP tasks.
Each configuration includes optimized hyperparameters and sample data.

Available Models:
- qa_flan_t5: Question answering with Flan-T5
- summarization_t5: Text summarization with T5
- classification_distilbert: Text classification with DistilBERT
- code_gen_codegen: Code generation with Salesforce CodeGen
- chat_gpt2: Conversational AI with GPT-2
- sentiment_roberta: Sentiment analysis with RoBERTa
- ner_bert: Named Entity Recognition with BERT
- translation_marian: Translation with MarianMT

Usage:
    from model_zoo import load_model_config, list_models, get_model_info

    # List available models
    models = list_models()

    # Get info about a model
    info = get_model_info("qa_flan_t5")

    # Load configuration
    config = load_model_config("qa_flan_t5")
"""

from model_zoo.registry import (
    load_model_config,
    list_models,
    get_model_info,
    MODEL_REGISTRY,
    ModelConfig,
)

__all__ = [
    "load_model_config",
    "list_models",
    "get_model_info",
    "MODEL_REGISTRY",
    "ModelConfig",
]
