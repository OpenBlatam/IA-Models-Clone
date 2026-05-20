"""
LoRA Fine-tuning for Transformer Models
Efficient fine-tuning using Low-Rank Adaptation
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library not available, LoRA features disabled")


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation Layer
    Efficient fine-tuning with minimal parameters
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation"""
        # LoRA: W' = W + (B @ A) * scaling
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return lora_output * self.scaling


def apply_lora_to_linear(
    linear_layer: nn.Linear,
    rank: int = 8,
    alpha: float = 16.0
) -> nn.Module:
    """Apply LoRA to a linear layer"""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for LoRA")
    
    lora = LoRALayer(
        linear_layer.in_features,
        linear_layer.out_features,
        rank=rank,
        alpha=alpha
    )
    
    return lora


class LoRATransformerFineTuner:
    """
    Fine-tune transformer models using LoRA
    Efficient fine-tuning with minimal parameter updates
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_modules: Optional[List[str]] = None,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library required for LoRA fine-tuning")
        
        self.base_model = model
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "out_proj"]
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules or ["q_proj", "v_proj"]
        )
        
        # Apply LoRA
        self.model = get_peft_model(model, lora_config)
        
        # Enable training
        self.model.train()
        
        logger.info(f"Applied LoRA with rank={rank}, alpha={alpha}")
        logger.info(f"Trainable parameters: {self.model.num_parameters('trainable')}")
        logger.info(f"Total parameters: {self.model.num_parameters()}")
    
    def get_model(self) -> nn.Module:
        """Get the LoRA-enhanced model"""
        return self.model
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights (much smaller than full model)"""
        self.model.save_pretrained(path)
        logger.info(f"Saved LoRA weights to {path}")
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights"""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base_model, path)
        logger.info(f"Loaded LoRA weights from {path}")
    
    def merge_and_unload(self) -> nn.Module:
        """Merge LoRA weights into base model and return"""
        merged_model = self.model.merge_and_unload()
        return merged_model


def create_lora_config(
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create LoRA configuration"""
    return {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": target_modules or ["q_proj", "v_proj", "k_proj", "out_proj"],
        "bias": "none",
        "task_type": "FEATURE_EXTRACTION"
    }

