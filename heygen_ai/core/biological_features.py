"""
Biological Neural Network Inspired Features for Transformer Models

This module implements biological neural network concepts including
neural plasticity, synaptic scaling, homeostatic mechanisms, and more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List
from .transformer_config import TransformerConfig


class NeuralPlasticity(nn.Module):
    """Neural plasticity mechanism inspired by biological neurons."""
    
    def __init__(self, 
                 hidden_size: int, 
                 plasticity_rate: float = 0.01,
                 decay_rate: float = 0.95):
        super().__init__()
        self.hidden_size = hidden_size
        self.plasticity_rate = plasticity_rate
        self.decay_rate = decay_rate
        
        # Synaptic weights (learnable)
        self.synaptic_weights = nn.Parameter(
            torch.randn(hidden_size, hidden_size) * 0.1
        )
        
        # Activity-dependent plasticity
        self.activity_tracker = nn.Parameter(
            torch.zeros(hidden_size), requires_grad=False
        )
        
        # Long-term potentiation (LTP) and depression (LTD)
        self.ltp_threshold = 0.7
        self.ltd_threshold = 0.3
        
        # Spike-timing dependent plasticity (STDP)
        self.std_learning_rate = 0.001
        self.tau_plus = 20.0  # LTP time constant
        self.tau_minus = 20.0  # LTD time constant
        
        # Pre and post synaptic traces
        self.register_buffer('pre_trace', torch.zeros(hidden_size))
        self.register_buffer('post_trace', torch.zeros(hidden_size))
        self.register_buffer('last_spike_time', torch.zeros(hidden_size))
    
    def update_activity(self, x: torch.Tensor, timestep: int):
        """Update activity-dependent plasticity."""
        # Update activity tracker
        self.activity_tracker.data = self.decay_rate * self.activity_tracker.data + (1 - self.decay_rate) * x.mean(dim=0)
        
        # Update spike traces
        current_time = timestep
        time_diff = current_time - self.last_spike_time
        
        # Update traces with exponential decay
        self.pre_trace *= torch.exp(-time_diff / self.tau_plus)
        self.post_trace *= torch.exp(-time_diff / self.tau_minus)
        
        # Update last spike time
        self.last_spike_time.fill_(current_time)
    
    def apply_stdp(self, pre_activity: torch.Tensor, post_activity: torch.Tensor):
        """Apply spike-timing dependent plasticity."""
        # Calculate STDP weight changes
        ltp_effect = torch.outer(post_activity, self.pre_trace) * self.std_learning_rate
        ltd_effect = torch.outer(self.post_trace, pre_activity) * self.std_learning_rate
        
        # Update synaptic weights
        weight_change = ltp_effect - ltd_effect
        self.synaptic_weights.data += weight_change
        
        # Update traces
        self.pre_trace += pre_activity
        self.post_trace += post_activity
    
    def forward(self, x: torch.Tensor, timestep: int = 0) -> torch.Tensor:
        """Forward pass with neural plasticity."""
        # Update activity
        self.update_activity(x, timestep)
        
        # Apply synaptic weights
        output = torch.matmul(x, self.synaptic_weights)
        
        # Apply activity-dependent scaling
        activity_scale = torch.sigmoid(self.activity_tracker)
        output = output * activity_scale.unsqueeze(0)
        
        return output


class SynapticScaling(nn.Module):
    """Synaptic scaling mechanism for maintaining neural activity."""
    
    def __init__(self, 
                 hidden_size: int, 
                 target_activity: float = 0.1,
                 scaling_rate: float = 0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_activity = target_activity
        self.scaling_rate = scaling_rate
        
        # Scaling factors for each synapse
        self.scaling_factors = nn.Parameter(torch.ones(hidden_size))
        
        # Activity history for scaling
        self.register_buffer('activity_history', torch.zeros(100))
        self.register_buffer('history_index', torch.tensor(0))
    
    def update_scaling(self, activity: torch.Tensor):
        """Update synaptic scaling based on activity."""
        # Update activity history
        self.activity_history[self.history_index] = activity.mean().item()
        self.history_index = (self.history_index + 1) % 100
        
        # Calculate current activity level
        current_activity = self.activity_history.mean()
        
        # Calculate scaling adjustment
        activity_ratio = self.target_activity / (current_activity + 1e-8)
        scaling_adjustment = self.scaling_rate * (activity_ratio - 1.0)
        
        # Update scaling factors
        self.scaling_factors.data *= (1 + scaling_adjustment)
        self.scaling_factors.data = torch.clamp(self.scaling_factors.data, 0.1, 10.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with synaptic scaling."""
        # Apply scaling factors
        scaled_x = x * self.scaling_factors.unsqueeze(0)
        
        # Update scaling based on current activity
        self.update_scaling(scaled_x)
        
        return scaled_x


class HomeostaticMechanism(nn.Module):
    """Homeostatic mechanism for maintaining neural stability."""
    
    def __init__(self, 
                 hidden_size: int, 
                 target_firing_rate: float = 0.1,
                 homeostatic_rate: float = 0.001):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_firing_rate = target_firing_rate
        self.homeostatic_rate = homeostatic_rate
        
        # Homeostatic scaling factors
        self.homeostatic_scaling = nn.Parameter(torch.ones(hidden_size))
        
        # Firing rate tracking
        self.register_buffer('firing_rate', torch.zeros(hidden_size))
        self.register_buffer('firing_count', torch.zeros(hidden_size))
        self.register_buffer('total_steps', torch.tensor(0))
    
    def update_firing_rate(self, x: torch.Tensor):
        """Update firing rate statistics."""
        # Count spikes (activations above threshold)
        spike_threshold = 0.5
        spikes = (x > spike_threshold).float()
        
        # Update firing count
        self.firing_count += spikes.sum(dim=0)
        self.total_steps += x.size(0)
        
        # Calculate firing rate
        self.firing_rate = self.firing_count / (self.total_steps + 1e-8)
    
    def apply_homeostatic_scaling(self):
        """Apply homeostatic scaling to maintain target firing rate."""
        # Calculate scaling adjustment
        firing_ratio = self.target_firing_rate / (self.firing_rate + 1e-8)
        scaling_adjustment = self.homeostatic_rate * (firing_ratio - 1.0)
        
        # Update homeostatic scaling
        self.homeostatic_scaling.data *= (1 + scaling_adjustment)
        self.homeostatic_scaling.data = torch.clamp(self.homeostatic_scaling.data, 0.01, 100.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with homeostatic mechanism."""
        # Update firing rate
        self.update_firing_rate(x)
        
        # Apply homeostatic scaling
        self.apply_homeostatic_scaling()
        
        # Scale output
        output = x * self.homeostatic_scaling.unsqueeze(0)
        
        return output


class AdaptiveThreshold(nn.Module):
    """Adaptive threshold mechanism for neural activation."""
    
    def __init__(self, 
                 hidden_size: int, 
                 initial_threshold: float = 0.5,
                 adaptation_rate: float = 0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.adaptation_rate = adaptation_rate
        
        # Adaptive thresholds for each neuron
        self.thresholds = nn.Parameter(torch.full((hidden_size,), initial_threshold))
        
        # Threshold adaptation history
        self.register_buffer('threshold_history', torch.zeros(100, hidden_size))
        self.register_buffer('history_index', torch.tensor(0))
    
    def update_thresholds(self, x: torch.Tensor):
        """Update adaptive thresholds based on input activity."""
        # Calculate current activity level
        current_activity = x.mean(dim=0)
        
        # Update threshold history
        self.threshold_history[self.history_index] = self.thresholds.data
        self.history_index = (self.history_index + 1) % 100
        
        # Calculate threshold adjustment
        activity_ratio = current_activity / (self.thresholds.data + 1e-8)
        threshold_adjustment = self.adaptation_rate * (activity_ratio - 1.0)
        
        # Update thresholds
        self.thresholds.data += threshold_adjustment
        self.thresholds.data = torch.clamp(self.thresholds.data, 0.01, 2.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive threshold."""
        # Update thresholds
        self.update_thresholds(x)
        
        # Apply adaptive threshold
        output = torch.where(x > self.thresholds.unsqueeze(0), x, torch.zeros_like(x))
        
        return output


class MemoryConsolidation(nn.Module):
    """Memory consolidation mechanism for long-term learning."""
    
    def __init__(self, 
                 hidden_size: int, 
                 consolidation_rate: float = 0.001,
                 memory_capacity: int = 1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.consolidation_rate = consolidation_rate
        self.memory_capacity = memory_capacity
        
        # Memory buffer for consolidation
        self.register_buffer('memory_buffer', torch.zeros(memory_capacity, hidden_size))
        self.register_buffer('memory_importance', torch.zeros(memory_capacity))
        self.register_buffer('memory_index', torch.tensor(0))
        self.register_buffer('memory_count', torch.tensor(0))
        
        # Consolidation weights
        self.consolidation_weights = nn.Parameter(
            torch.randn(hidden_size, hidden_size) * 0.1
        )
        
        # Memory retrieval network
        self.retrieval_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()
        )
    
    def store_memory(self, x: torch.Tensor, importance: torch.Tensor = None):
        """Store memory in the consolidation buffer."""
        if importance is None:
            importance = torch.ones(x.size(0))
        
        # Store memories
        for i in range(x.size(0)):
            idx = self.memory_index % self.memory_capacity
            self.memory_buffer[idx] = x[i]
            self.memory_importance[idx] = importance[i]
            self.memory_index += 1
            self.memory_count += 1
    
    def consolidate_memories(self):
        """Consolidate memories into long-term storage."""
        if self.memory_count == 0:
            return
        
        # Get recent memories
        recent_memories = self.memory_buffer[:min(self.memory_count, self.memory_capacity)]
        recent_importance = self.memory_importance[:min(self.memory_count, self.memory_capacity)]
        
        # Weighted consolidation
        weights = F.softmax(recent_importance, dim=0)
        consolidated_memory = torch.sum(weights.unsqueeze(1) * recent_memories, dim=0)
        
        # Update consolidation weights
        weight_update = self.consolidation_rate * torch.outer(consolidated_memory, consolidated_memory)
        self.consolidation_weights.data += weight_update
    
    def retrieve_memory(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve consolidated memories."""
        # Calculate similarity with stored memories
        similarities = torch.matmul(query, self.memory_buffer.T)
        attention_weights = F.softmax(similarities, dim=-1)
        
        # Retrieve weighted memories
        retrieved_memory = torch.matmul(attention_weights, self.memory_buffer)
        
        # Apply retrieval network
        output = self.retrieval_network(retrieved_memory)
        
        return output
    
    def forward(self, x: torch.Tensor, importance: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with memory consolidation."""
        # Store current input as memory
        self.store_memory(x, importance)
        
        # Consolidate memories
        self.consolidate_memories()
        
        # Retrieve consolidated memories
        consolidated_output = self.retrieve_memory(x)
        
        # Combine current input with consolidated memory
        output = x + consolidated_output
        
        return output


class BiologicalAttention(nn.Module):
    """Biological-inspired attention mechanism."""
    
    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int,
                 plasticity_rate: float = 0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Biological mechanisms
        self.neural_plasticity = NeuralPlasticity(self.head_dim, plasticity_rate)
        self.synaptic_scaling = SynapticScaling(self.head_dim)
        self.homeostatic_mechanism = HomeostaticMechanism(self.head_dim)
        self.adaptive_threshold = AdaptiveThreshold(self.head_dim)
        self.memory_consolidation = MemoryConsolidation(self.head_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of biological attention."""
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply biological mechanisms to each head
        Q_biological = []
        K_biological = []
        V_biological = []
        
        for head in range(self.num_heads):
            Q_head = Q[:, head, :, :]
            K_head = K[:, head, :, :]
            V_head = V[:, head, :, :]
            
            # Apply neural plasticity
            Q_head = self.neural_plasticity(Q_head, timestep=0)
            K_head = self.neural_plasticity(K_head, timestep=0)
            V_head = self.neural_plasticity(V_head, timestep=0)
            
            # Apply synaptic scaling
            Q_head = self.synaptic_scaling(Q_head)
            K_head = self.synaptic_scaling(K_head)
            V_head = self.synaptic_scaling(V_head)
            
            # Apply homeostatic mechanism
            Q_head = self.homeostatic_mechanism(Q_head)
            K_head = self.homeostatic_mechanism(K_head)
            V_head = self.homeostatic_mechanism(V_head)
            
            # Apply adaptive threshold
            Q_head = self.adaptive_threshold(Q_head)
            K_head = self.adaptive_threshold(K_head)
            V_head = self.adaptive_threshold(V_head)
            
            # Apply memory consolidation
            Q_head = self.memory_consolidation(Q_head)
            K_head = self.memory_consolidation(K_head)
            V_head = self.memory_consolidation(V_head)
            
            Q_biological.append(Q_head)
            K_biological.append(K_head)
            V_biological.append(V_head)
        
        # Stack biological-transformed tensors
        Q = torch.stack(Q_biological, dim=1)
        K = torch.stack(K_biological, dim=1)
        V = torch.stack(V_biological, dim=1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        output = self.out_proj(context)
        
        return output, attn_weights


class BiologicalTransformerBlock(nn.Module):
    """Biological-inspired transformer block."""
    
    def __init__(self, config: TransformerConfig, plasticity_rate: float = 0.01):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Biological attention
        self.biological_attention = BiologicalAttention(
            config.hidden_size,
            config.num_attention_heads,
            plasticity_rate
        )
        
        # Biological feed-forward network
        self.biological_ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # Apply biological mechanisms to FFN
        self.ffn_plasticity = NeuralPlasticity(config.hidden_size, plasticity_rate)
        self.ffn_scaling = SynapticScaling(config.hidden_size)
        self.ffn_homeostasis = HomeostaticMechanism(config.hidden_size)
        self.ffn_threshold = AdaptiveThreshold(config.hidden_size)
        self.ffn_memory = MemoryConsolidation(config.hidden_size)
        
        # Layer normalization
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of biological transformer block."""
        # Biological attention with residual connection
        attn_output, attn_weights = self.biological_attention(x, x, x, attention_mask)
        x = self.attention_norm(x + self.dropout(attn_output))
        
        # Biological feed-forward with residual connection
        ffn_output = self.biological_ffn(x)
        
        # Apply biological mechanisms to FFN output
        ffn_output = self.ffn_plasticity(ffn_output, timestep=0)
        ffn_output = self.ffn_scaling(ffn_output)
        ffn_output = self.ffn_homeostasis(ffn_output)
        ffn_output = self.ffn_threshold(ffn_output)
        ffn_output = self.ffn_memory(ffn_output)
        
        x = self.ffn_norm(x + ffn_output)
        
        return x, attn_weights


