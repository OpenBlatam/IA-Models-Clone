"""
Transformer Fine-Tuner Module

Advanced fine-tuning utilities for transformer models.
"""

from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoModel,
        AutoTokenizer,
        Wav2Vec2Model,
        Wav2Vec2Processor
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available")


class TransformerFineTuner:
    """
    Advanced fine-tuning utilities for transformer models.
    
    Args:
        model_name: HuggingFace model name.
        task_type: Task type ("classification", etc.).
        num_labels: Number of labels for classification.
    """
    
    def __init__(
        self,
        model_name: str,
        task_type: str = "classification",
        num_labels: int = 10
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required")
        
        self.model_name = model_name
        self.task_type = task_type
        self.num_labels = num_labels
        
        # Load model and tokenizer
        try:
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            # Try audio model
            try:
                self.model = Wav2Vec2Model.from_pretrained(model_name)
                self.processor = Wav2Vec2Processor.from_pretrained(model_name)
                self.tokenizer = None
            except Exception as e:
                logger.error(f"Could not load model {model_name}: {str(e)}")
                raise
    
    def add_classification_head(self, hidden_size: Optional[int] = None):
        """
        Add classification head to transformer.
        
        Args:
            hidden_size: Hidden size (auto-detect if None).
        
        Returns:
            Classification head module.
        """
        if hidden_size is None:
            hidden_size = self.model.config.hidden_size
        
        classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, self.num_labels)
        )
        
        return classifier
    
    def freeze_base_model(self, freeze_layers: Optional[List[int]] = None):
        """
        Freeze base transformer model.
        
        Args:
            freeze_layers: Optional list of layer indices to freeze (None = freeze all).
        """
        if freeze_layers is None:
            # Freeze all
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Freeze specific layers
            for layer_idx in freeze_layers:
                if hasattr(self.model, 'encoder'):
                    if layer_idx < len(self.model.encoder.layer):
                        for param in self.model.encoder.layer[layer_idx].parameters():
                            param.requires_grad = False



