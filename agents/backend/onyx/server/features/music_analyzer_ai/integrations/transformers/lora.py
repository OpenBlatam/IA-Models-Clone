"""
LoRA Integration Module

Implements LoRA (Low-Rank Adaptation) for efficient fine-tuning.
"""

import logging
from typing import Optional, Dict, Any, List
import torch

logger = logging.getLogger(__name__)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library not available for LoRA")


class LoRATransformerWrapper:
    """
    LoRA wrapper for efficient fine-tuning of transformer models.
    
    Args:
        base_model: Base transformer model.
        r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        target_modules: Target modules for LoRA (None for auto-detect).
        lora_dropout: LoRA dropout probability.
    """
    
    def __init__(
        self,
        base_model,
        r: int = 8,
        lora_alpha: int = 16,
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.1
    ):
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library required for LoRA")
        
        self.base_model = base_model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Auto-detect target modules if not provided
        if target_modules is None:
            target_modules = self._auto_detect_modules()
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        
        # Apply LoRA
        self.model = get_peft_model(base_model, lora_config)
        logger.info(f"Applied LoRA with r={r}, alpha={lora_alpha}, target_modules={target_modules}")
    
    def _auto_detect_modules(self) -> List[str]:
        """Auto-detect target modules for LoRA."""
        # Common patterns for attention layers
        modules = []
        for name, module in self.base_model.named_modules():
            if "q_proj" in name or "k_proj" in name or "v_proj" in name or "out_proj" in name:
                modules.append(name.split(".")[-1])
            elif "query" in name.lower() or "key" in name.lower() or "value" in name.lower():
                modules.append(name.split(".")[-1])
        
        if not modules:
            # Fallback to common names
            modules = ["q_proj", "v_proj"]
        
        return list(set(modules))
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def save_pretrained(self, path: str):
        """Save LoRA adapters."""
        self.model.save_pretrained(path)
        logger.info(f"Saved LoRA adapters to {path}")

