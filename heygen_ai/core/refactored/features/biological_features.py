"""
Biological Features for Enhanced Transformer Models

This module contains biologically-inspired features and components
for the refactored transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List
from ..base import BaseFeatureModule, BaseCoordinator
from ...transformer_config import TransformerConfig


class NeuralPlasticity(nn.Module):
    """Neural plasticity mechanism inspired by biological neural networks."""
    
    def __init__(self, hidden_size: int, plasticity_rate: float = 0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.plasticity_rate = plasticity_rate
        
        # Plasticity parameters
        self.synaptic_weights = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.plasticity_threshold = nn.Parameter(torch.tensor(0.5))
        self.learning_rate = nn.Parameter(torch.tensor(0.01))
        
        # Activity tracking
        self.register_buffer('activity_history', torch.zeros(hidden_size))
        self.register_buffer('plasticity_state', torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply neural plasticity."""
        # Calculate activity level
        activity = torch.norm(x, dim=-1).mean(dim=0)
        
        # Update activity history
        self.activity_history = 0.9 * self.activity_history + 0.1 * activity
        
        # Calculate plasticity based on activity
        plasticity_signal = torch.sigmoid(activity - self.plasticity_threshold)
        
        # Update synaptic weights
        weight_update = self.plasticity_rate * plasticity_signal.unsqueeze(0) * x.mean(dim=0).unsqueeze(1)
        self.synaptic_weights.data += weight_update
        
        # Apply plasticity
        plastic_output = torch.matmul(x, self.synaptic_weights)
        
        # Update plasticity state
        self.plasticity_state = 0.9 * self.plasticity_state + 0.1 * plasticity_signal
        
        return plastic_output


class SynapticScaling(nn.Module):
    """Synaptic scaling mechanism for maintaining network stability."""
    
    def __init__(self, hidden_size: int, scaling_factor: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.scaling_factor = scaling_factor
        
        # Scaling parameters
        self.target_activity = nn.Parameter(torch.tensor(1.0))
        self.scaling_rate = nn.Parameter(torch.tensor(0.01))
        
        # Activity tracking
        self.register_buffer('mean_activity', torch.tensor(1.0))
        self.register_buffer('scaling_weights', torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply synaptic scaling."""
        # Calculate current activity
        current_activity = torch.norm(x, dim=-1).mean()
        
        # Update mean activity
        self.mean_activity = 0.9 * self.mean_activity + 0.1 * current_activity
        
        # Calculate scaling factor
        scaling_ratio = self.target_activity / (self.mean_activity + 1e-8)
        
        # Update scaling weights
        self.scaling_weights = 0.9 * self.scaling_weights + 0.1 * scaling_ratio
        
        # Apply scaling
        scaled_output = x * self.scaling_weights.unsqueeze(0).unsqueeze(0)
        
        return scaled_output


class HomeostaticMechanism(nn.Module):
    """Homeostatic mechanism for maintaining network balance."""
    
    def __init__(self, hidden_size: int, target_rate: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_rate = target_rate
        
        # Homeostatic parameters
        self.homeostatic_gain = nn.Parameter(torch.tensor(1.0))
        self.adaptation_rate = nn.Parameter(torch.tensor(0.01))
        
        # Activity tracking
        self.register_buffer('firing_rate', torch.zeros(hidden_size))
        self.register_buffer('homeostatic_bias', torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply homeostatic mechanism."""
        # Calculate firing rate
        firing_rate = torch.sigmoid(x).mean(dim=0)
        
        # Update firing rate history
        self.firing_rate = 0.9 * self.firing_rate + 0.1 * firing_rate
        
        # Calculate homeostatic adjustment
        rate_error = self.target_rate - self.firing_rate
        homeostatic_adjustment = self.homeostatic_gain * rate_error
        
        # Update homeostatic bias
        self.homeostatic_bias += self.adaptation_rate * homeostatic_adjustment
        
        # Apply homeostatic adjustment
        homeostatic_output = x + self.homeostatic_bias.unsqueeze(0).unsqueeze(0)
        
        return homeostatic_output


class AdaptiveThreshold(nn.Module):
    """Adaptive threshold mechanism for neural activation."""
    
    def __init__(self, hidden_size: int, initial_threshold: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.initial_threshold = initial_threshold
        
        # Threshold parameters
        self.threshold = nn.Parameter(torch.full((hidden_size,), initial_threshold))
        self.adaptation_rate = nn.Parameter(torch.tensor(0.01))
        
        # Activity tracking
        self.register_buffer('activity_level', torch.zeros(hidden_size))
        self.register_buffer('threshold_history', torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive threshold."""
        # Calculate activity level
        activity = torch.abs(x).mean(dim=0)
        
        # Update activity level
        self.activity_level = 0.9 * self.activity_level + 0.1 * activity
        
        # Adapt threshold based on activity
        threshold_error = self.activity_level - self.threshold
        threshold_update = self.adaptation_rate * threshold_error
        
        # Update threshold
        self.threshold.data += threshold_update
        
        # Apply threshold
        thresholded_output = torch.where(
            torch.abs(x) > self.threshold.unsqueeze(0).unsqueeze(0),
            x,
            torch.zeros_like(x)
        )
        
        # Update threshold history
        self.threshold_history = 0.9 * self.threshold_history + 0.1 * self.threshold.data
        
        return thresholded_output


class MemoryConsolidation(nn.Module):
    """Memory consolidation mechanism for long-term storage."""
    
    def __init__(self, hidden_size: int, consolidation_rate: float = 0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.consolidation_rate = consolidation_rate
        
        # Memory parameters
        self.memory_trace = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.consolidation_threshold = nn.Parameter(torch.tensor(0.5))
        self.forgetting_rate = nn.Parameter(torch.tensor(0.001))
        
        # Consolidation tracking
        self.register_buffer('consolidation_state', torch.zeros(hidden_size))
        self.register_buffer('memory_strength', torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply memory consolidation."""
        # Calculate memory strength
        memory_strength = torch.norm(x, dim=-1).mean(dim=0)
        
        # Update memory strength
        self.memory_strength = 0.9 * self.memory_strength + 0.1 * memory_strength
        
        # Determine consolidation
        consolidation_signal = torch.sigmoid(memory_strength - self.consolidation_threshold)
        
        # Update memory trace
        memory_update = self.consolidation_rate * consolidation_signal.unsqueeze(0) * x.mean(dim=0).unsqueeze(1)
        self.memory_trace.data += memory_update
        
        # Apply forgetting
        self.memory_trace.data *= (1 - self.forgetting_rate)
        
        # Retrieve consolidated memory
        consolidated_output = torch.matmul(x, self.memory_trace)
        
        # Update consolidation state
        self.consolidation_state = 0.9 * self.consolidation_state + 0.1 * consolidation_signal
        
        return consolidated_output


class BiologicalAttention(BaseFeatureModule):
    """Biologically-inspired attention mechanism."""
    
    def __init__(self, 
                 hidden_size: int, 
                 attention_dim: int = 512,
                 biological_level: float = 0.8):
        super().__init__(hidden_size, attention_dim, biological_level)
        
        # Biological attention components
        self.biological_query = nn.Linear(hidden_size, attention_dim)
        self.biological_key = nn.Linear(hidden_size, attention_dim)
        self.biological_value = nn.Linear(hidden_size, attention_dim)
        self.biological_output = nn.Linear(attention_dim, hidden_size)
        
        # Biological mechanisms
        self.neural_plasticity = NeuralPlasticity(attention_dim)
        self.synaptic_scaling = SynapticScaling(attention_dim)
        self.homeostatic_mechanism = HomeostaticMechanism(attention_dim)
        self.adaptive_threshold = AdaptiveThreshold(attention_dim)
        
        # Attention scaling
        self.attention_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of biological attention."""
        # Project to biological attention space
        q = self.biological_query(x)
        k = self.biological_key(x)
        v = self.biological_value(x)
        
        # Apply biological mechanisms
        q = self.neural_plasticity(q)
        k = self.synaptic_scaling(k)
        v = self.homeostatic_mechanism(v)
        
        # Compute biological attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attention_scale
        
        # Apply adaptive threshold
        scores = self.adaptive_threshold(scores)
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        
        # Output projection
        output = self.biological_output(context)
        
        # Apply biological level scaling
        output = output * self.feature_level
        
        return output


class BiologicalNeuralNetwork(BaseFeatureModule):
    """Biologically-inspired neural network feature module."""
    
    def __init__(self, 
                 hidden_size: int, 
                 biological_dim: int = 1024,
                 biological_level: float = 0.8):
        super().__init__(hidden_size, biological_dim, biological_level)
        
        # Biological mechanisms
        self.neural_plasticity = NeuralPlasticity(hidden_size)
        self.synaptic_scaling = SynapticScaling(hidden_size)
        self.homeostatic_mechanism = HomeostaticMechanism(hidden_size)
        self.adaptive_threshold = AdaptiveThreshold(hidden_size)
        self.memory_consolidation = MemoryConsolidation(hidden_size)
        
        # Biological processing network
        self.biological_network = nn.Sequential(
            nn.Linear(hidden_size, biological_dim),
            nn.ReLU(),
            nn.Linear(biological_dim, biological_dim),
            nn.ReLU(),
            nn.Linear(biological_dim, hidden_size),
            nn.Tanh()
        )
        
        # Biological state
        self.register_buffer('biological_state', torch.zeros(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of biological neural network."""
        # Apply biological mechanisms
        x = self.neural_plasticity(x)
        x = self.synaptic_scaling(x)
        x = self.homeostatic_mechanism(x)
        x = self.adaptive_threshold(x)
        x = self.memory_consolidation(x)
        
        # Process through biological network
        biological_output = self.biological_network(x)
        
        # Apply biological level scaling
        biological_output = biological_output * self.feature_level
        
        # Update biological state
        self.biological_state = 0.9 * self.biological_state + 0.1 * biological_output.mean(dim=0)
        
        return biological_output


class BiologicalTransformerBlock(BaseFeatureModule):
    """Biologically-enhanced transformer block."""
    
    def __init__(self, 
                 config: TransformerConfig, 
                 biological_level: float = 0.8):
        super().__init__(config.hidden_size, biological_level=biological_level)
        self.config = config
        
        # Biological components
        self.biological_attention = BiologicalAttention(config.hidden_size, biological_level=biological_level)
        self.biological_ffn = BiologicalNeuralNetwork(config.hidden_size, biological_level=biological_level)
        
        # Standard components
        from ..core.attention_mechanisms import EnhancedMultiHeadAttention
        self.attention = EnhancedMultiHeadAttention(config)
        
        # Layer normalization
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of biological transformer block."""
        # Biological-enhanced attention
        biological_attn = self.biological_attention(x)
        attn_output, _ = self.attention(x, x, x)
        x = self.attention_norm(x + self.dropout(attn_output + biological_attn))
        
        # Biological-enhanced feed-forward
        biological_ffn = self.biological_ffn(x)
        ffn_output = self.biological_ffn(x)
        x = self.ffn_norm(x + ffn_output + biological_ffn)
        
        return x


class BiologicalCoordinator(BaseCoordinator):
    """Coordinates all biological modules."""
    
    def __init__(self, 
                 hidden_size: int, 
                 biological_level: float = 0.8):
        super().__init__(hidden_size, biological_level)
        
        # Biological modules
        self.biological_neural_network = BiologicalNeuralNetwork(hidden_size, biological_level=biological_level)
        self.biological_attention = BiologicalAttention(hidden_size, biological_level=biological_level)
        
        # Add to feature modules
        self.add_feature_module(self.biological_neural_network)
        self.add_feature_module(self.biological_attention)
        
        # Biological integration
        self.biological_integration = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
    
    def integrate_features(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate biological features."""
        # Get biological outputs
        biological_nn_output = self.biological_neural_network(x)
        biological_attn_output = self.biological_attention(x)
        
        # Combine biological outputs
        combined = torch.cat([biological_nn_output, biological_attn_output], dim=-1)
        
        # Integrate
        integrated = self.biological_integration(combined)
        
        return integrated

