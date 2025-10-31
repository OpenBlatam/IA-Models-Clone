"""
Advanced Transformer Architectures Module

This module contains advanced transformer architectures including
Mixture of Experts, Switch Transformer, and other cutting-edge models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List
from .transformer_config import TransformerConfig
from .attention_mechanisms import MultiHeadAttention, SparseAttention, LinearAttention


class MixtureOfExperts(nn.Module):
    """Mixture of Experts (MoE) layer for efficient scaling."""
    
    def __init__(self, 
                 hidden_size: int, 
                 intermediate_size: int, 
                 num_experts: int = 8,
                 top_k: int = 2,
                 expert_capacity_factor: float = 1.25):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.ReLU(),
                nn.Linear(intermediate_size, hidden_size)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(hidden_size, num_experts)
        
        # Load balancing loss
        self.load_balancing_loss = 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MoE layer."""
        batch_size, seq_len, hidden_size = x.size()
        
        # Compute gating scores
        gate_scores = self.gate(x)  # [batch_size, seq_len, num_experts]
        gate_probs = F.softmax(gate_scores, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_probs, dim=-1)
        
        # Compute expert capacity
        expert_capacity = int(self.expert_capacity_factor * seq_len * batch_size / self.num_experts)
        
        # Process through experts
        output = torch.zeros_like(x)
        
        for i in range(self.num_experts):
            # Find positions where expert i is selected
            expert_mask = (top_k_indices == i).any(dim=-1)  # [batch_size, seq_len]
            
            if expert_mask.any():
                # Get inputs for this expert
                expert_inputs = x[expert_mask]  # [num_selected, hidden_size]
                
                # Process through expert
                expert_output = self.experts[i](expert_inputs)
                
                # Get weights for this expert
                expert_weights = top_k_probs[expert_mask]  # [num_selected, top_k]
                expert_weight = expert_weights[:, (top_k_indices[expert_mask] == i).nonzero(as_tuple=True)[1]]
                
                # Weighted output
                weighted_output = expert_output * expert_weight.unsqueeze(-1)
                
                # Add to output
                output[expert_mask] += weighted_output.squeeze(1)
        
        # Compute load balancing loss
        expert_usage = gate_probs.mean(dim=(0, 1))  # [num_experts]
        expert_usage_std = expert_usage.std()
        self.load_balancing_loss = self.num_experts * expert_usage_std
        
        return output


class SwitchTransformerBlock(nn.Module):
    """Switch Transformer block with routing."""
    
    def __init__(self, config: TransformerConfig, num_experts: int = 8):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Attention layer
        self.attention = MultiHeadAttention(config)
        
        # Switch layer (MoE)
        self.switch = MixtureOfExperts(
            config.hidden_size,
            config.intermediate_size,
            num_experts=num_experts,
            top_k=1  # Switch uses top-1
        )
        
        # Layer normalization
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass of Switch Transformer block."""
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, attention_mask)
        x = self.attention_norm(x + self.dropout(attn_output))
        
        # Switch layer with residual connection
        switch_output = self.switch(x)
        x = self.ffn_norm(x + switch_output)
        
        return x, {
            'attention_weights': attn_weights,
            'load_balancing_loss': self.switch.load_balancing_loss
        }


class SparseTransformerBlock(nn.Module):
    """Sparse Transformer block with sparse attention."""
    
    def __init__(self, config: TransformerConfig, attention_type: str = "strided"):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Sparse attention layer
        self.attention = SparseAttention(
            config.hidden_size,
            config.num_attention_heads,
            attention_type=attention_type
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of Sparse Transformer block."""
        # Sparse attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, attention_mask)
        x = self.attention_norm(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output)
        
        return x, attn_weights


class AdaptiveTransformerBlock(nn.Module):
    """Adaptive Transformer block with dynamic scaling."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Adaptive attention
        self.attention = MultiHeadAttention(config)
        
        # Adaptive feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # Adaptive scaling factors
        self.attention_scale = nn.Parameter(torch.ones(1))
        self.ffn_scale = nn.Parameter(torch.ones(1))
        
        # Layer normalization
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of Adaptive Transformer block."""
        # Adaptive attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, attention_mask)
        x = self.attention_norm(x + self.dropout(attn_output) * self.attention_scale)
        
        # Adaptive feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output * self.ffn_scale)
        
        return x, attn_weights


class DynamicLayerScaling(nn.Module):
    """Dynamic layer scaling for adaptive depth."""
    
    def __init__(self, hidden_size: int, max_layers: int = 12):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_layers = max_layers
        
        # Layer importance predictors
        self.layer_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            ) for _ in range(max_layers)
        ])
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            AdaptiveTransformerBlock(TransformerConfig(hidden_size=hidden_size))
            for _ in range(max_layers)
        ])
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[float]]:
        """Forward pass with dynamic layer scaling."""
        layer_importances = []
        
        for i, (predictor, block) in enumerate(zip(self.layer_predictors, self.transformer_blocks)):
            # Predict layer importance
            importance = predictor(x.mean(dim=1))  # [batch_size, 1]
            layer_importances.append(importance.mean().item())
            
            # Apply layer if important enough
            if importance.mean() > 0.5:  # Threshold for layer activation
                x, _ = block(x, attention_mask)
        
        return x, layer_importances


class NeuralArchitectureSearch(nn.Module):
    """Neural Architecture Search for optimal transformer design."""
    
    def __init__(self, config: TransformerConfig, search_space: Dict[str, List] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Default search space
        if search_space is None:
            search_space = {
                'num_heads': [8, 12, 16],
                'intermediate_size': [2048, 3072, 4096],
                'attention_type': ['sparse', 'linear', 'standard']
            }
        
        self.search_space = search_space
        
        # Architecture parameters
        self.architecture_params = nn.ParameterDict({
            'num_heads': nn.Parameter(torch.tensor(12.0)),
            'intermediate_size': nn.Parameter(torch.tensor(3072.0)),
            'attention_type': nn.Parameter(torch.tensor(0.0))  # 0=sparse, 1=linear, 2=standard
        })
        
        # Create architecture
        self._create_architecture()
    
    def _create_architecture(self):
        """Create architecture based on current parameters."""
        # Get discrete choices
        num_heads = int(self.architecture_params['num_heads'].item())
        intermediate_size = int(self.architecture_params['intermediate_size'].item())
        attention_type_idx = int(self.architecture_params['attention_type'].item())
        
        attention_types = ['sparse', 'linear', 'standard']
        attention_type = attention_types[attention_type_idx % len(attention_types)]
        
        # Create attention layer
        if attention_type == 'sparse':
            self.attention = SparseAttention(self.hidden_size, num_heads)
        elif attention_type == 'linear':
            self.attention = LinearAttention(self.hidden_size, num_heads)
        else:
            self.attention = MultiHeadAttention(TransformerConfig(
                hidden_size=self.hidden_size,
                num_attention_heads=num_heads
            ))
        
        # Create feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_size, self.hidden_size),
            nn.Dropout(0.1)
        )
        
        # Layer normalization
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.ffn_norm = nn.LayerNorm(self.hidden_size)
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with architecture search."""
        # Update architecture if needed
        self._create_architecture()
        
        # Attention
        attn_output, attn_weights = self.attention(x, x, x, attention_mask)
        x = self.attention_norm(x + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output)
        
        return x, {
            'attention_weights': attn_weights,
            'architecture_params': {
                'num_heads': int(self.architecture_params['num_heads'].item()),
                'intermediate_size': int(self.architecture_params['intermediate_size'].item()),
                'attention_type': int(self.architecture_params['attention_type'].item())
            }
        }


class ModelEnsemble(nn.Module):
    """Model ensemble for improved performance."""
    
    def __init__(self, models: List[nn.Module], ensemble_method: str = "average"):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.num_models = len(models)
        
        if ensemble_method == "weighted":
            # Learnable ensemble weights
            self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass of ensemble."""
        outputs = []
        attention_weights = []
        
        for model in self.models:
            if hasattr(model, 'forward'):
                output = model(x, **kwargs)
                if isinstance(output, dict):
                    outputs.append(output['logits'] if 'logits' in output else output['hidden_states'])
                    if 'attention_weights' in output:
                        attention_weights.append(output['attention_weights'])
                else:
                    outputs.append(output)
        
        # Ensemble outputs
        if self.ensemble_method == "average":
            ensemble_output = torch.stack(outputs).mean(dim=0)
        elif self.ensemble_method == "weighted":
            weights = F.softmax(self.ensemble_weights, dim=0)
            ensemble_output = sum(w * output for w, output in zip(weights, outputs))
        elif self.ensemble_method == "voting":
            # For classification tasks
            ensemble_output = torch.stack(outputs).mode(dim=0)[0]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        result = {'ensemble_output': ensemble_output}
        
        if attention_weights:
            result['attention_weights'] = attention_weights
        
        return result


