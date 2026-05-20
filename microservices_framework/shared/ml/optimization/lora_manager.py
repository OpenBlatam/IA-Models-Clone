"""
LoRA Manager
Advanced LoRA configuration and management for efficient fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig,
)
import logging

logger = logging.getLogger(__name__)


class LoRAManager:
    """
    Manager for LoRA (Low-Rank Adaptation) operations.
    """
    
    TASK_TYPE_MAP = {
        "causal_lm": TaskType.CAUSAL_LM,
        "seq2seq": TaskType.SEQ_2_SEQ_LM,
        "classification": TaskType.SEQ_CLS,
        "token_classification": TaskType.TOKEN_CLS,
        "question_answering": TaskType.QUESTION_ANS,
        "feature_extraction": TaskType.FEATURE_EXTRACTION,
    }
    
    def __init__(
        self,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        bias: str = "none",
        target_modules: Optional[List[str]] = None,
        task_type: str = "CAUSAL_LM",
    ):
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.bias = bias
        self.target_modules = target_modules
        self.task_type = self._get_task_type(task_type)
    
    def _get_task_type(self, task_type: str) -> TaskType:
        """Convert string task type to TaskType enum."""
        if isinstance(task_type, TaskType):
            return task_type
        
        task_type_upper = task_type.upper()
        if task_type_upper in self.TASK_TYPE_MAP:
            return self.TASK_TYPE_MAP[task_type_upper]
        
        # Try direct mapping
        try:
            return TaskType[task_type_upper]
        except KeyError:
            logger.warning(f"Unknown task type {task_type}, using CAUSAL_LM")
            return TaskType.CAUSAL_LM
    
    def create_config(
        self,
        r: Optional[int] = None,
        alpha: Optional[int] = None,
        dropout: Optional[float] = None,
        bias: Optional[str] = None,
        target_modules: Optional[List[str]] = None,
        task_type: Optional[str] = None,
    ) -> LoraConfig:
        """
        Create LoRA configuration.
        
        Args:
            r: LoRA rank
            alpha: LoRA alpha
            dropout: LoRA dropout
            bias: Bias type
            target_modules: Target modules for LoRA
            task_type: Task type
            
        Returns:
            LoRA configuration
        """
        return LoraConfig(
            r=r or self.r,
            lora_alpha=alpha or self.alpha,
            target_modules=target_modules or self.target_modules,
            lora_dropout=dropout or self.dropout,
            bias=bias or self.bias,
            task_type=task_type or self.task_type,
        )
    
    def apply_lora(
        self,
        model: nn.Module,
        config: Optional[LoraConfig] = None,
        **config_kwargs
    ) -> PeftModel:
        """
        Apply LoRA to model.
        
        Args:
            model: Base model
            config: LoRA configuration (optional)
            **config_kwargs: Configuration overrides
            
        Returns:
            Model with LoRA applied
        """
        if config is None:
            config = self.create_config(**config_kwargs)
        
        # Auto-detect target modules if not specified
        if config.target_modules is None:
            config.target_modules = self._auto_detect_target_modules(model)
            logger.info(f"Auto-detected target modules: {config.target_modules}")
        
        peft_model = get_peft_model(model, config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        logger.info(
            f"LoRA applied: {trainable_params:,} trainable / {total_params:,} total parameters "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
        
        return peft_model
    
    def _auto_detect_target_modules(self, model: nn.Module) -> List[str]:
        """
        Auto-detect target modules for LoRA.
        
        Args:
            model: Model to analyze
            
        Returns:
            List of target module names
        """
        # Common patterns for transformer models
        target_patterns = [
            "q_proj", "v_proj", "k_proj", "out_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",  # MLP
            "query", "key", "value",  # Alternative attention names
            "dense", "attention",  # Generic
        ]
        
        module_names = []
        for name, module in model.named_modules():
            for pattern in target_patterns:
                if pattern in name.lower() and isinstance(module, nn.Linear):
                    module_names.append(name)
                    break
        
        # If no matches, use all Linear layers
        if not module_names:
            module_names = [
                name for name, module in model.named_modules()
                if isinstance(module, nn.Linear)
            ][:8]  # Limit to first 8 to avoid too many
        
        return module_names
    
    def merge_and_unload(self, peft_model: PeftModel) -> nn.Module:
        """
        Merge LoRA weights into base model and unload.
        
        Args:
            peft_model: PEFT model
            
        Returns:
            Base model with merged weights
        """
        return peft_model.merge_and_unload()
    
    def save_lora_weights(self, model: PeftModel, path: str):
        """
        Save LoRA weights.
        
        Args:
            model: PEFT model
            path: Save path
        """
        model.save_pretrained(path)
        logger.info(f"LoRA weights saved to {path}")
    
    def load_lora_weights(
        self,
        base_model: nn.Module,
        path: str,
        device: Optional[str] = None
    ) -> PeftModel:
        """
        Load LoRA weights onto base model.
        
        Args:
            base_model: Base model
            path: Path to LoRA weights
            device: Device to load on
            
        Returns:
            PEFT model with loaded weights
        """
        if device:
            base_model = base_model.to(device)
        
        peft_model = PeftModel.from_pretrained(base_model, path)
        logger.info(f"LoRA weights loaded from {path}")
        
        return peft_model



