"""
LoRA (Low-Rank Adaptation) and P-tuning for Efficient LLM Fine-tuning
Implements parameter-efficient fine-tuning techniques for transformers
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import logging
import math

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoConfig,
        GPT2LMHeadModel,
        GPT2Config
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
        prepare_model_for_kbit_training
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library not available. Install with: pip install peft")


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer
    Efficient fine-tuning by learning low-rank decomposition of weight updates
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        """
        Initialize LoRA layer
        
        Args:
            in_features: Input features
            out_features: Output features
            rank: Rank of low-rank decomposition
            alpha: Scaling factor
            dropout: Dropout rate
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # LoRA computation: x @ A^T @ B^T
        x = self.dropout(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return lora_output


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation
    """
    
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        """
        Initialize LoRA linear layer
        
        Args:
            linear_layer: Original linear layer
            rank: LoRA rank
            alpha: Scaling factor
            dropout: Dropout rate
        """
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA"""
        base_output = self.linear(x)
        lora_output = self.lora(x)
        return base_output + lora_output


class LoRAFineTuner:
    """
    LoRA fine-tuning wrapper for transformer models
    Uses PEFT library when available, falls back to custom implementation
    """
    
    def __init__(
        self,
        model_name: str,
        target_modules: Optional[List[str]] = None,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
        use_peft: bool = True
    ):
        """
        Initialize LoRA fine-tuner
        
        Args:
            model_name: HuggingFace model name
            target_modules: Modules to apply LoRA (e.g., ['q_proj', 'v_proj'])
            rank: LoRA rank
            alpha: LoRA alpha (scaling factor)
            dropout: LoRA dropout
            device: Device to use
            use_peft: Use PEFT library if available
        """
        if not PEFT_AVAILABLE and use_peft:
            logger.warning("PEFT not available, using custom LoRA implementation")
            use_peft = False
        
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_peft = use_peft and PEFT_AVAILABLE
        
        # Default target modules for common architectures
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
        
        try:
            if self.use_peft:
                # Load base model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Configure LoRA
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=rank,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    target_modules=target_modules,
                    bias="none"
                )
                
                # Apply LoRA
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()
                
                logger.info(f"LoRA applied using PEFT library (rank={rank}, alpha={alpha})")
            else:
                # Custom LoRA implementation
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._apply_custom_lora(target_modules, rank, alpha, dropout)
                
                logger.info(f"LoRA applied using custom implementation (rank={rank}, alpha={alpha})")
            
            self.model = self.model.to(self.device)
            self.model.train()
            
        except Exception as e:
            logger.error(f"Failed to initialize LoRA fine-tuner: {e}")
            raise
    
    def _apply_custom_lora(
        self,
        target_modules: List[str],
        rank: int,
        alpha: float,
        dropout: float
    ):
        """Apply custom LoRA to model"""
        for name, module in self.model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with LoRA linear
                    lora_linear = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                    # Set in parent module
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = self.model.get_submodule(parent_name)
                    setattr(parent, child_name, lora_linear)
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable parameters (only LoRA parameters)"""
        if self.use_peft:
            return list(self.model.parameters())
        else:
            # Return only LoRA parameters
            trainable_params = []
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower():
                    trainable_params.append(param)
            return trainable_params
    
    def save_lora_weights(self, path: str):
        """Save LoRA weights"""
        if self.use_peft:
            self.model.save_pretrained(path)
        else:
            # Save only LoRA parameters
            lora_state_dict = {}
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower():
                    lora_state_dict[name] = param.cpu()
            torch.save(lora_state_dict, path)
            logger.info(f"LoRA weights saved to {path}")
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights"""
        if self.use_peft:
            self.model = PeftModel.from_pretrained(self.model, path)
        else:
            lora_state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(lora_state_dict, strict=False)
            logger.info(f"LoRA weights loaded from {path}")


class PTuningFineTuner:
    """
    P-tuning (Prompt Tuning) for efficient fine-tuning
    Learns continuous prompt embeddings instead of fine-tuning all parameters
    """
    
    def __init__(
        self,
        model_name: str,
        num_virtual_tokens: int = 10,
        prompt_embedding_dim: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize P-tuning fine-tuner
        
        Args:
            model_name: HuggingFace model name
            num_virtual_tokens: Number of virtual prompt tokens
            prompt_embedding_dim: Dimension of prompt embeddings
            device: Device to use
        """
        self.model_name = model_name
        self.num_virtual_tokens = num_virtual_tokens
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            if PEFT_AVAILABLE:
                from peft import PromptTuningConfig, get_peft_model, TaskType
                
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Configure P-tuning
                prompt_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=num_virtual_tokens,
                    prompt_tuning_init="TEXT",
                    prompt_tuning_init_text="Classify the following text:"
                )
                
                self.model = get_peft_model(self.model, prompt_config)
                self.model.print_trainable_parameters()
                
                logger.info(f"P-tuning applied (num_virtual_tokens={num_virtual_tokens})")
            else:
                logger.warning("PEFT not available for P-tuning, using basic model")
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            self.model = self.model.to(self.device)
            self.model.train()
            
        except Exception as e:
            logger.error(f"Failed to initialize P-tuning fine-tuner: {e}")
            raise


def create_lora_finetuner(
    model_name: str = "gpt2",
    rank: int = 8,
    alpha: float = 16.0,
    device: Optional[torch.device] = None
) -> LoRAFineTuner:
    """Factory function for LoRA fine-tuner"""
    return LoRAFineTuner(
        model_name=model_name,
        rank=rank,
        alpha=alpha,
        device=device
    )


def create_ptuning_finetuner(
    model_name: str = "gpt2",
    num_virtual_tokens: int = 10,
    device: Optional[torch.device] = None
) -> PTuningFineTuner:
    """Factory function for P-tuning fine-tuner"""
    return PTuningFineTuner(
        model_name=model_name,
        num_virtual_tokens=num_virtual_tokens,
        device=device
    )

