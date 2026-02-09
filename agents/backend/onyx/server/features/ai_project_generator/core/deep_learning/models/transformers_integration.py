"""
Transformers Library Integration
================================

Integration with Hugging Face Transformers library for:
- Pre-trained models
- Tokenizers
- Fine-tuning with LoRA/P-tuning
"""

import logging
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import (
        AutoModel, AutoModelForSequenceClassification,
        AutoTokenizer, AutoConfig,
        PreTrainedModel, PreTrainedTokenizer
    )
    from transformers import Trainer as HFTrainer, TrainingArguments
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Install with: pip install transformers")

# Try to import PEFT for efficient fine-tuning
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library not available. Install with: pip install peft")


class TransformersModelWrapper(nn.Module):
    """
    Wrapper for Hugging Face Transformers models.
    
    Provides a unified interface for using pre-trained models.
    """
    
    def __init__(
        self,
        model_name: str,
        num_labels: Optional[int] = None,
        use_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize transformers model wrapper.
        
        Args:
            model_name: Hugging Face model name or path
            num_labels: Number of labels for classification
            use_lora: Use LoRA for efficient fine-tuning
            lora_config: LoRA configuration
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required")
        
        self.model_name = model_name
        
        # Load model
        if num_labels is not None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        else:
            self.model = AutoModel.from_pretrained(model_name)
        
        # Apply LoRA if requested
        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT library is required for LoRA")
            
            if lora_config is None:
                lora_config = {
                    'r': 8,
                    'lora_alpha': 16,
                    'target_modules': ['q_proj', 'v_proj'],
                    'lora_dropout': 0.1,
                    'bias': 'none',
                    'task_type': TaskType.FEATURE_EXTRACTION
                }
            
            peft_config = LoraConfig(**lora_config)
            self.model = get_peft_model(self.model, peft_config)
            logger.info(f"Applied LoRA to model: {model_name}")
    
    def forward(self, **kwargs) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            **kwargs: Model inputs (input_ids, attention_mask, etc.)
            
        Returns:
            Model outputs
        """
        return self.model(**kwargs)
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """
        Get associated tokenizer.
        
        Returns:
            Pre-trained tokenizer
        """
        return AutoTokenizer.from_pretrained(self.model_name)
    
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save model to directory.
        
        Args:
            save_directory: Directory to save model
        """
        self.model.save_pretrained(save_directory)


def create_transformers_model(
    model_name: str,
    task: str = 'classification',
    num_labels: Optional[int] = None,
    use_lora: bool = False,
    lora_config: Optional[Dict[str, Any]] = None
) -> TransformersModelWrapper:
    """
    Create a transformers model wrapper.
    
    Args:
        model_name: Hugging Face model name
        task: Task type ('classification', 'feature_extraction')
        num_labels: Number of labels (for classification)
        use_lora: Use LoRA for fine-tuning
        lora_config: LoRA configuration
        
    Returns:
        TransformersModelWrapper instance
    """
    if task == 'classification' and num_labels is None:
        num_labels = 2  # Default binary classification
    
    return TransformersModelWrapper(
        model_name=model_name,
        num_labels=num_labels if task == 'classification' else None,
        use_lora=use_lora,
        lora_config=lora_config
    )



