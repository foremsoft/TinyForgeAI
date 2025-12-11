"""
MCP Model Zoo Tools

Tools for browsing and using pre-configured models.
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("tinyforge-mcp.model_zoo")

# Pre-configured models (mirrors model_zoo/registry.py)
MODEL_REGISTRY = {
    "qa_flan_t5_small": {
        "name": "Q&A Flan-T5 Small",
        "task_type": "qa",
        "base_model": "google/flan-t5-small",
        "description": "Question answering model based on Flan-T5 Small. Good for extractive and abstractive Q&A.",
        "parameters": "80M",
        "use_lora": True,
        "recommended_epochs": 3,
        "recommended_batch_size": 8
    },
    "qa_flan_t5_base": {
        "name": "Q&A Flan-T5 Base",
        "task_type": "qa",
        "base_model": "google/flan-t5-base",
        "description": "Larger Q&A model with better accuracy. Requires more resources.",
        "parameters": "250M",
        "use_lora": True,
        "recommended_epochs": 3,
        "recommended_batch_size": 4
    },
    "summarization_t5_small": {
        "name": "Summarization T5 Small",
        "task_type": "summarization",
        "base_model": "t5-small",
        "description": "Text summarization model. Condenses long text into concise summaries.",
        "parameters": "60M",
        "use_lora": True,
        "recommended_epochs": 5,
        "recommended_batch_size": 8
    },
    "summarization_bart": {
        "name": "Summarization BART",
        "task_type": "summarization",
        "base_model": "facebook/bart-base",
        "description": "BART-based summarization. Better for longer documents.",
        "parameters": "140M",
        "use_lora": True,
        "recommended_epochs": 3,
        "recommended_batch_size": 4
    },
    "classification_distilbert": {
        "name": "Classification DistilBERT",
        "task_type": "classification",
        "base_model": "distilbert-base-uncased",
        "description": "Fast text classification model. Great for sentiment, topic, intent classification.",
        "parameters": "66M",
        "use_lora": False,
        "recommended_epochs": 5,
        "recommended_batch_size": 16
    },
    "sentiment_roberta": {
        "name": "Sentiment RoBERTa",
        "task_type": "sentiment",
        "base_model": "cardiffnlp/twitter-roberta-base-sentiment",
        "description": "Pre-trained sentiment analysis model. Fine-tune for domain-specific sentiment.",
        "parameters": "125M",
        "use_lora": True,
        "recommended_epochs": 3,
        "recommended_batch_size": 16
    },
    "code_gen_small": {
        "name": "Code Generation Small",
        "task_type": "code_generation",
        "base_model": "Salesforce/codegen-350M-mono",
        "description": "Code generation model for Python. Generate code from natural language.",
        "parameters": "350M",
        "use_lora": True,
        "recommended_epochs": 3,
        "recommended_batch_size": 4
    },
    "chat_gpt2_small": {
        "name": "Chat GPT-2 Small",
        "task_type": "conversation",
        "base_model": "gpt2",
        "description": "Conversational model based on GPT-2. Good for chatbots and dialogue.",
        "parameters": "124M",
        "use_lora": True,
        "recommended_epochs": 3,
        "recommended_batch_size": 8
    },
    "chat_dialogpt": {
        "name": "Chat DialoGPT",
        "task_type": "conversation",
        "base_model": "microsoft/DialoGPT-small",
        "description": "Dialogue-optimized GPT model. Better conversational flow.",
        "parameters": "124M",
        "use_lora": True,
        "recommended_epochs": 3,
        "recommended_batch_size": 8
    },
    "ner_bert": {
        "name": "NER BERT",
        "task_type": "ner",
        "base_model": "bert-base-cased",
        "description": "Named Entity Recognition model. Extract names, organizations, locations.",
        "parameters": "110M",
        "use_lora": False,
        "recommended_epochs": 5,
        "recommended_batch_size": 16
    },
    "translation_en_es": {
        "name": "Translation EN-ES",
        "task_type": "translation",
        "base_model": "Helsinki-NLP/opus-mt-en-es",
        "description": "English to Spanish translation model.",
        "parameters": "74M",
        "use_lora": True,
        "recommended_epochs": 3,
        "recommended_batch_size": 16
    },
    "translation_en_fr": {
        "name": "Translation EN-FR",
        "task_type": "translation",
        "base_model": "Helsinki-NLP/opus-mt-en-fr",
        "description": "English to French translation model.",
        "parameters": "74M",
        "use_lora": True,
        "recommended_epochs": 3,
        "recommended_batch_size": 16
    },
    "text_gen_gpt2_medium": {
        "name": "Text Generation GPT-2 Medium",
        "task_type": "text_generation",
        "base_model": "gpt2-medium",
        "description": "General text generation model. Creative writing, content generation.",
        "parameters": "355M",
        "use_lora": True,
        "recommended_epochs": 3,
        "recommended_batch_size": 4
    }
}


class ModelZooTools:
    """Model Zoo browsing and management tools."""

    async def list_models(
        self,
        task_type: str = "all"
    ) -> dict[str, Any]:
        """
        List available models in the Model Zoo.

        Args:
            task_type: Filter by task type

        Returns:
            List of available models
        """
        models = []

        for model_id, info in MODEL_REGISTRY.items():
            if task_type == "all" or info["task_type"] == task_type:
                models.append({
                    "model_id": model_id,
                    "name": info["name"],
                    "task_type": info["task_type"],
                    "parameters": info["parameters"],
                    "description": info["description"][:100] + "..." if len(info["description"]) > 100 else info["description"]
                })

        # Group by task type
        task_types = list(set(m["task_type"] for m in models))

        return {
            "success": True,
            "total_models": len(models),
            "task_types": task_types,
            "models": models
        }

    async def get_model_info(
        self,
        model_id: str
    ) -> dict[str, Any]:
        """
        Get detailed information about a model.

        Args:
            model_id: Model identifier

        Returns:
            Detailed model information
        """
        if model_id not in MODEL_REGISTRY:
            # Try to find similar models
            similar = [m for m in MODEL_REGISTRY.keys() if model_id.lower() in m.lower()]
            return {
                "success": False,
                "error": f"Model not found: {model_id}",
                "similar_models": similar[:5] if similar else None
            }

        info = MODEL_REGISTRY[model_id]

        return {
            "success": True,
            "model_id": model_id,
            "name": info["name"],
            "task_type": info["task_type"],
            "base_model": info["base_model"],
            "description": info["description"],
            "parameters": info["parameters"],
            "training_config": {
                "use_lora": info["use_lora"],
                "recommended_epochs": info["recommended_epochs"],
                "recommended_batch_size": info["recommended_batch_size"]
            },
            "usage_example": f"""
# Train this model using TinyForgeAI CLI:
foremforge train --model {info['base_model']} --data your_data.jsonl --epochs {info['recommended_epochs']}

# Or using Python:
from backend.training.real_trainer import RealTrainer, TrainingConfig

config = TrainingConfig(
    model_name="{info['base_model']}",
    num_epochs={info['recommended_epochs']},
    batch_size={info['recommended_batch_size']},
    use_lora={info['use_lora']}
)

trainer = RealTrainer(config)
trainer.train("your_data.jsonl")
"""
        }

    async def export_model(
        self,
        model_path: str,
        output_path: str,
        quantize: bool = False,
    ) -> dict[str, Any]:
        """
        Export a trained model to ONNX format.

        Args:
            model_path: Path to trained model
            output_path: Path for exported model
            quantize: Apply quantization

        Returns:
            Export result
        """
        model_dir = Path(model_path)
        output_dir = Path(output_path)

        if not model_dir.exists():
            return {
                "success": False,
                "error": f"Model not found: {model_path}"
            }

        try:
            from backend.model_exporter.onnx_exporter import ONNXExporter

            exporter = ONNXExporter()

            output_dir.mkdir(parents=True, exist_ok=True)
            onnx_path = output_dir / "model.onnx"

            exporter.export(
                model_path=str(model_dir),
                output_path=str(onnx_path),
                quantize=quantize
            )

            return {
                "success": True,
                "model_path": str(model_path),
                "output_path": str(onnx_path),
                "quantized": quantize,
                "message": "Model exported to ONNX format successfully"
            }

        except ImportError:
            return {
                "success": False,
                "error": "ONNX exporter not available. Install with: pip install onnx onnxruntime"
            }
        except Exception as e:
            logger.error(f"Export error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
