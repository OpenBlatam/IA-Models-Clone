"""
Advanced AI Models for Document Classification
============================================

State-of-the-art AI models including transformers, diffusion models, and LLMs
for comprehensive document processing and classification.

Modules:
- transformer_models: Advanced transformer architectures with LoRA, AdaLoRA, and quantization
- diffusion_models: Document generation using Stable Diffusion and ControlNet
- llm_models: Large language model integration for classification and generation
- gradio_interface: Interactive web interface for model management and testing
"""

from .transformer_models import (
    ModelFactory, ModelConfig, TransformerTrainer,
    CustomTransformerClassifier, MultiTaskTransformer,
    DocumentTransformer, LoRATransformer, AdaLoRATransformer,
    QuantizedTransformer, PositionalEncoding, MultiHeadAttention,
    TransformerBlock
)

from .diffusion_models import (
    DiffusionModelFactory, DiffusionConfig,
    DocumentDiffusionPipeline, ControlNetDocumentGenerator,
    DocumentInpaintingPipeline, MultimodalDocumentGenerator,
    DocumentStyleTransfer, DocumentUpscaler
)

from .llm_models import (
    LLMModelFactory, LLMConfig,
    DocumentLLMClassifier, DocumentGenerator,
    DocumentAnalyzer, LocalLLMModel, CustomStoppingCriteria
)

from .gradio_interface import GradioInterface

__all__ = [
    # Transformer Models
    "ModelFactory", "ModelConfig", "TransformerTrainer",
    "CustomTransformerClassifier", "MultiTaskTransformer",
    "DocumentTransformer", "LoRATransformer", "AdaLoRATransformer",
    "QuantizedTransformer", "PositionalEncoding", "MultiHeadAttention",
    "TransformerBlock",
    
    # Diffusion Models
    "DiffusionModelFactory", "DiffusionConfig",
    "DocumentDiffusionPipeline", "ControlNetDocumentGenerator",
    "DocumentInpaintingPipeline", "MultimodalDocumentGenerator",
    "DocumentStyleTransfer", "DocumentUpscaler",
    
    # LLM Models
    "LLMModelFactory", "LLMConfig",
    "DocumentLLMClassifier", "DocumentGenerator",
    "DocumentAnalyzer", "LocalLLMModel", "CustomStoppingCriteria",
    
    # Gradio Interface
    "GradioInterface"
]
























