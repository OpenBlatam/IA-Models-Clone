"""
LoRA Fine-tuning for Transformers
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available. LoRA fine-tuning disabled.")


class LoRATrainer:
    """LoRA fine-tuning trainer for transformers"""
    
    def __init__(
        self,
        model: nn.Module,
        lora_config: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize LoRA trainer
        
        Args:
            model: Base transformer model
            lora_config: LoRA configuration
            device: Device to use
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library is required for LoRA")
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default LoRA config
        default_config = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["query", "value"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": TaskType.FEATURE_EXTRACTION
        }
        
        if lora_config:
            default_config.update(lora_config)
        
        # Create LoRA config
        self.lora_config = LoraConfig(**default_config)
        
        # Apply LoRA to model
        self.model = get_peft_model(model, self.lora_config)
        self.model = self.model.to(self.device)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("LoRATrainer initialized")
    
    def get_model(self) -> nn.Module:
        """Get LoRA model"""
        return self.model
    
    def save_lora_weights(self, path: str):
        """Save LoRA weights"""
        self.model.save_pretrained(path)
        logger.info(f"LoRA weights saved to {path}")
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights"""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, path)
        logger.info(f"LoRA weights loaded from {path}")


def create_lora_model(
    base_model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[list] = None,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Create LoRA model from base model
    
    Args:
        base_model: Base transformer model
        r: LoRA rank
        lora_alpha: LoRA alpha
        target_modules: Target modules for LoRA
        device: Device to use
        
    Returns:
        LoRA model
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT library is required")
    
    if target_modules is None:
        target_modules = ["query", "value"]
    
    trainer = LoRATrainer(
        base_model,
        lora_config={
            "r": r,
            "lora_alpha": lora_alpha,
            "target_modules": target_modules
        },
        device=device
    )
    
    return trainer.get_model()

