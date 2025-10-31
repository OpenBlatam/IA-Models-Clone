from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Efficient Fine-tuning Techniques for Transformer Models
Implementation of LoRA, P-tuning, AdaLoRA, and other parameter-efficient fine-tuning methods
"""


logger = logging.getLogger(__name__)

@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)"""
    r: int = 16  # Rank of the low-rank adaptation
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.1  # Dropout rate for LoRA layers
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"  # none, all, lora_only
    task_type: str = "CAUSAL_LM"  # CAUSAL_LM, SEQ_2_SEQ_LM, SEQUENCE_CLASSIFICATION
    inference_mode: bool = False
    fan_in_fan_out: bool = False
    modules_to_save: Optional[List[str]] = None
    init_lora_weights: bool = True
    layers_to_transform: Optional[List[int]] = None
    layers_pattern: Optional[str] = None

@dataclass
class PEFTConfig:
    """Configuration for Parameter-Efficient Fine-Tuning"""
    peft_type: str = "LORA"  # LORA, P_TUNING, PREFIX_TUNING, ADALORA, etc.
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False
    
    # LoRA specific
    lora_config: Optional[LoRAConfig] = None
    
    # P-tuning specific
    num_virtual_tokens: int = 20
    encoder_hidden_size: int = 128
    encoder_num_layers: int = 2
    encoder_dropout: float = 0.1
    
    # Prefix tuning specific
    num_prefix_tokens: int = 20
    prefix_projection: bool = False
    
    # AdaLoRA specific
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    init_r: int = 12
    target_r: int = 8
    beta1: float = 0.85
    beta2: float = 0.85
    tinit: int = 200
    tfinal: int = 1000
    deltaT: int = 10
    orth_reg_weight: float = 0.5

class LoRALayer(nn.Module):
    """LoRA layer implementation"""
    
    def __init__(self, in_features: int, out_features: int, r: int = 16, 
                 lora_alpha: int = 32, lora_dropout: float = 0.1, 
                 bias: str = "none", fan_in_fan_out: bool = False):
        
    """__init__ function."""
super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.fan_in_fan_out = fan_in_fan_out
        
        # LoRA layers
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = lora_alpha / r
        
        # Optional bias
        if bias == "all":
            self.lora_bias = nn.Parameter(torch.zeros(out_features))
        elif bias == "lora_only":
            self.lora_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.lora_bias = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> Any:
        """Initialize LoRA weights"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        if self.lora_bias is not None:
            nn.init.zeros_(self.lora_bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer"""
        if self.lora_dropout > 0:
            x = F.dropout(x, p=self.lora_dropout, training=self.training)
        
        # LoRA computation
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T
        lora_output = lora_output * self.scaling
        
        if self.lora_bias is not None:
            lora_output = lora_output + self.lora_bias
        
        return lora_output

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    
    def __init__(self, linear_layer: nn.Linear, config: LoRAConfig):
        
    """__init__ function."""
super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            fan_in_fan_out=config.fan_in_fan_out
        )
        self.config = config
        
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation"""
        base_output = self.linear(x)
        lora_output = self.lora(x)
        return base_output + lora_output

class P_TuningEmbedding(nn.Module):
    """P-tuning virtual token embeddings"""
    
    def __init__(self, num_virtual_tokens: int, encoder_hidden_size: int, 
                 encoder_num_layers: int, encoder_dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.encoder_hidden_size = encoder_hidden_size
        
        # Virtual token embeddings
        self.virtual_tokens = nn.Parameter(torch.randn(num_virtual_tokens, encoder_hidden_size))
        
        # Encoder for virtual tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_hidden_size,
            nhead=8,
            dim_feedforward=encoder_hidden_size * 4,
            dropout=encoder_dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_num_layers)
        
        # Projection to model hidden size
        self.projection = nn.Linear(encoder_hidden_size, encoder_hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> Any:
        """Initialize P-tuning weights"""
        nn.init.normal_(self.virtual_tokens, std=0.02)
        nn.init.normal_(self.projection.weight, std=0.02)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """Generate virtual token embeddings"""
        # Expand virtual tokens to batch size
        virtual_tokens = self.virtual_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Encode virtual tokens
        encoded_tokens = self.encoder(virtual_tokens)
        
        # Project to model hidden size
        projected_tokens = self.projection(encoded_tokens)
        
        return projected_tokens

class PrefixTuningEmbedding(nn.Module):
    """Prefix tuning implementation"""
    
    def __init__(self, num_prefix_tokens: int, hidden_size: int, 
                 num_layers: int, num_heads: int, prefix_projection: bool = False):
        
    """__init__ function."""
super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_projection = prefix_projection
        
        # Prefix embeddings for each layer
        self.prefix_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(num_prefix_tokens, hidden_size))
            for _ in range(num_layers)
        ])
        
        # Optional projection
        if prefix_projection:
            self.prefix_projection = nn.Linear(hidden_size, hidden_size)
        else:
            self.prefix_projection = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> Any:
        """Initialize prefix tuning weights"""
        for prefix_emb in self.prefix_embeddings:
            nn.init.normal_(prefix_emb, std=0.02)
        
        if self.prefix_projection is not None:
            nn.init.normal_(self.prefix_projection.weight, std=0.02)
            nn.init.zeros_(self.prefix_projection.bias)
    
    def get_prefix_embeddings(self, layer_idx: int, batch_size: int) -> torch.Tensor:
        """Get prefix embeddings for a specific layer"""
        prefix_emb = self.prefix_embeddings[layer_idx]
        
        if self.prefix_projection is not None:
            prefix_emb = self.prefix_projection(prefix_emb)
        
        # Expand to batch size
        prefix_emb = prefix_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        return prefix_emb

class AdaLoRALayer(nn.Module):
    """AdaLoRA layer with adaptive rank allocation"""
    
    def __init__(self, in_features: int, out_features: int, init_r: int = 12, 
                 target_r: int = 8, beta1: float = 0.85, beta2: float = 0.85,
                 tinit: int = 200, tfinal: int = 1000, deltaT: int = 10,
                 orth_reg_weight: float = 0.5):
        
    """__init__ function."""
super().__init__()
        self.init_r = init_r
        self.target_r = target_r
        self.beta1 = beta1
        self.beta2 = beta2
        self.tinit = tinit
        self.tfinal = tfinal
        self.deltaT = deltaT
        self.orth_reg_weight = orth_reg_weight
        
        # AdaLoRA layers
        self.lora_A = nn.Parameter(torch.zeros(init_r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, init_r))
        self.lora_E = nn.Parameter(torch.zeros(init_r, init_r))  # Importance matrix
        
        # Initialize weights
        self._init_weights()
        
        # Training state
        self.register_buffer('step', torch.tensor(0))
        self.register_buffer('importance_history', torch.zeros(init_r))
    
    def _init_weights(self) -> Any:
        """Initialize AdaLoRA weights"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.eye_(self.lora_E)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AdaLoRA layer"""
        # Compute importance scores
        importance = torch.diag(self.lora_E).abs()
        
        # Update importance history
        if self.training:
            self.importance_history = self.beta1 * self.importance_history + (1 - self.beta1) * importance
        
        # Compute LoRA output
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T
        
        # Apply importance scaling
        lora_output = lora_output * importance.unsqueeze(0).unsqueeze(0)
        
        return lora_output
    
    def update_rank(self) -> Any:
        """Update rank allocation based on importance"""
        if not self.training or self.step < self.tinit:
            return
        
        if self.step % self.deltaT == 0:
            # Sort importance scores
            importance = torch.diag(self.lora_E).abs()
            sorted_indices = torch.argsort(importance, descending=True)
            
            # Keep top target_r components
            keep_indices = sorted_indices[:self.target_r]
            
            # Update parameters
            self.lora_A.data = self.lora_A.data[keep_indices]
            self.lora_B.data = self.lora_B.data[:, keep_indices]
            self.lora_E.data = self.lora_E.data[keep_indices][:, keep_indices]
            
            # Update importance history
            self.importance_history = self.importance_history[keep_indices]
    
    def get_orthogonal_regularization(self) -> torch.Tensor:
        """Compute orthogonal regularization loss"""
        if self.lora_A.size(0) <= 1:
            return torch.tensor(0.0, device=self.lora_A.device)
        
        # Compute correlation matrix
        A_norm = F.normalize(self.lora_A, dim=1)
        correlation = torch.mm(A_norm, A_norm.T)
        
        # Remove diagonal
        mask = torch.eye(correlation.size(0), device=correlation.device)
        correlation = correlation * (1 - mask)
        
        # Orthogonal regularization
        orth_reg = torch.norm(correlation, p='fro') ** 2
        
        return orth_reg * self.orth_reg_weight

class EfficientFineTuningManager:
    """Manager for efficient fine-tuning techniques"""
    
    def __init__(self, model: nn.Module, config: PEFTConfig):
        
    """__init__ function."""
self.model = model
        self.config = config
        self.peft_layers = {}
        self.original_layers = {}
        
        # Apply PEFT based on configuration
        if config.peft_type == "LORA":
            self._apply_lora()
        elif config.peft_type == "P_TUNING":
            self._apply_p_tuning()
        elif config.peft_type == "PREFIX_TUNING":
            self._apply_prefix_tuning()
        elif config.peft_type == "ADALORA":
            self._apply_adalora()
        else:
            raise ValueError(f"Unsupported PEFT type: {config.peft_type}")
    
    def _apply_lora(self) -> Any:
        """Apply LoRA to the model"""
        if self.config.lora_config is None:
            self.config.lora_config = LoRAConfig()
        
        for name, module in self.model.named_modules():
            if any(target in name for target in self.config.lora_config.target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with LoRA layer
                    lora_layer = LoRALinear(module, self.config.lora_config)
                    self._replace_module(name, lora_layer)
                    self.peft_layers[name] = lora_layer
    
    def _apply_p_tuning(self) -> Any:
        """Apply P-tuning to the model"""
        # Create P-tuning embeddings
        self.p_tuning_embeddings = P_TuningEmbedding(
            num_virtual_tokens=self.config.num_virtual_tokens,
            encoder_hidden_size=self.config.encoder_hidden_size,
            encoder_num_layers=self.config.encoder_num_layers,
            encoder_dropout=self.config.encoder_dropout
        )
        
        # Add to model
        self.model.register_parameter('p_tuning_embeddings', 
                                    self.p_tuning_embeddings.virtual_tokens)
    
    def _apply_prefix_tuning(self) -> Any:
        """Apply prefix tuning to the model"""
        # Find transformer layers
        num_layers = 0
        hidden_size = 768  # Default
        num_heads = 12  # Default
        
        for name, module in self.model.named_modules():
            if "layer" in name and isinstance(module, nn.Module):
                num_layers += 1
            if "hidden_size" in name:
                hidden_size = getattr(module, "hidden_size", hidden_size)
            if "num_heads" in name:
                num_heads = getattr(module, "num_heads", num_heads)
        
        # Create prefix tuning embeddings
        self.prefix_embeddings = PrefixTuningEmbedding(
            num_prefix_tokens=self.config.num_prefix_tokens,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            prefix_projection=self.config.prefix_projection
        )
    
    def _apply_adalora(self) -> Any:
        """Apply AdaLoRA to the model"""
        for name, module in self.model.named_modules():
            if any(target in name for target in self.config.target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with AdaLoRA layer
                    adalora_layer = AdaLoRALayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        init_r=self.config.init_r,
                        target_r=self.config.target_r,
                        beta1=self.config.beta1,
                        beta2=self.config.beta2,
                        tinit=self.config.tinit,
                        tfinal=self.config.tfinal,
                        deltaT=self.config.deltaT,
                        orth_reg_weight=self.config.orth_reg_weight
                    )
                    self._replace_module(name, adalora_layer)
                    self.peft_layers[name] = adalora_layer
    
    def _replace_module(self, name: str, new_module: nn.Module):
        """Replace a module in the model"""
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent = self.model.get_submodule(parent_name)
            setattr(parent, child_name, new_module)
        else:
            setattr(self.model, child_name, new_module)
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get trainable parameters"""
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count statistics"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_parameters())
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100
        }
    
    def save_pretrained(self, save_directory: str):
        """Save PEFT model"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        config_dict = {
            "peft_type": self.config.peft_type,
            "task_type": self.config.task_type,
            "inference_mode": self.config.inference_mode
        }
        
        if self.config.lora_config:
            config_dict["lora_config"] = {
                "r": self.config.lora_config.r,
                "lora_alpha": self.config.lora_config.lora_alpha,
                "lora_dropout": self.config.lora_config.lora_dropout,
                "target_modules": self.config.lora_config.target_modules,
                "bias": self.config.lora_config.bias
            }
        
        with open(os.path.join(save_directory, "config.json"), "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(config_dict, f, indent=2)
        
        # Save PEFT weights
        peft_state_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                peft_state_dict[name] = param.data
        
        torch.save(peft_state_dict, os.path.join(save_directory, "adapter_model.bin"))
    
    def load_pretrained(self, load_directory: str):
        """Load PEFT model"""
        # Load configuration
        config_path = os.path.join(load_directory, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                config_dict = json.load(f)
        
        # Load PEFT weights
        weights_path = os.path.join(load_directory, "adapter_model.bin")
        if os.path.exists(weights_path):
            peft_state_dict = torch.load(weights_path, map_location='cpu')
            
            # Load weights
            model_state_dict = self.model.state_dict()
            for name, param in peft_state_dict.items():
                if name in model_state_dict:
                    model_state_dict[name] = param
            
            self.model.load_state_dict(model_state_dict)

class PEFTTrainer:
    """Trainer for PEFT models"""
    
    def __init__(self, model: nn.Module, peft_config: PEFTConfig, 
                 optimizer_config: Dict[str, Any] = None):
        
    """__init__ function."""
self.model = model
        self.peft_config = peft_config
        self.peft_manager = EfficientFineTuningManager(model, peft_config)
        
        # Setup optimizer
        if optimizer_config is None:
            optimizer_config = {
                "lr": 1e-4,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999)
            }
        
        self.optimizer = torch.optim.AdamW(
            self.peft_manager.get_trainable_parameters(),
            **optimizer_config
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.peft_manager.get_trainable_parameters(), 
            max_norm=1.0
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Update AdaLoRA if applicable
        if self.peft_config.peft_type == "ADALORA":
            for layer in self.peft_manager.peft_layers.values():
                if isinstance(layer, AdaLoRALayer):
                    layer.step += 1
                    layer.update_rank()
        
        return {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}
    
    def get_orthogonal_regularization_loss(self) -> torch.Tensor:
        """Get orthogonal regularization loss for AdaLoRA"""
        if self.peft_config.peft_type != "ADALORA":
            return torch.tensor(0.0)
        
        orth_reg_loss = torch.tensor(0.0)
        for layer in self.peft_manager.peft_layers.values():
            if isinstance(layer, AdaLoRALayer):
                orth_reg_loss += layer.get_orthogonal_regularization()
        
        return orth_reg_loss
    
    def save_checkpoint(self, save_path: str):
        """Save training checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "peft_config": self.peft_config,
            "parameter_stats": self.peft_manager.get_parameter_count()
        }
        
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

def create_peft_config(peft_type: str, **kwargs) -> PEFTConfig:
    """Create PEFT configuration"""
    config = PEFTConfig(peft_type=peft_type)
    
    if peft_type == "LORA":
        config.lora_config = LoRAConfig(**kwargs)
    elif peft_type == "P_TUNING":
        config.num_virtual_tokens = kwargs.get("num_virtual_tokens", 20)
        config.encoder_hidden_size = kwargs.get("encoder_hidden_size", 128)
        config.encoder_num_layers = kwargs.get("encoder_num_layers", 2)
        config.encoder_dropout = kwargs.get("encoder_dropout", 0.1)
    elif peft_type == "PREFIX_TUNING":
        config.num_prefix_tokens = kwargs.get("num_prefix_tokens", 20)
        config.prefix_projection = kwargs.get("prefix_projection", False)
    elif peft_type == "ADALORA":
        config.target_modules = kwargs.get("target_modules", ["q_proj", "v_proj"])
        config.init_r = kwargs.get("init_r", 12)
        config.target_r = kwargs.get("target_r", 8)
        config.beta1 = kwargs.get("beta1", 0.85)
        config.beta2 = kwargs.get("beta2", 0.85)
        config.tinit = kwargs.get("tinit", 200)
        config.tfinal = kwargs.get("tfinal", 1000)
        config.deltaT = kwargs.get("deltaT", 10)
        config.orth_reg_weight = kwargs.get("orth_reg_weight", 0.5)
    
    return config

def apply_peft_to_model(model: nn.Module, peft_type: str, **kwargs) -> EfficientFineTuningManager:
    """Apply PEFT to a model"""
    config = create_peft_config(peft_type, **kwargs)
    return EfficientFineTuningManager(model, config) 