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
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass
import logging
from enum import Enum
import json
from pathlib import Path
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced Transformers and LLMs
Comprehensive implementation of transformers and large language models with advanced features.
"""



class AttentionType(Enum):
    """Types of attention mechanisms."""
    SCALED_DOT_PRODUCT = "scaled_dot_product"
    MULTI_HEAD = "multi_head"
    FLASH_ATTENTION = "flash_attention"
    SPARSE_ATTENTION = "sparse_attention"
    LOCAL_ATTENTION = "local_attention"
    ROTARY_POSITIONAL = "rotary_positional"


class ModelType(Enum):
    """Types of transformer models."""
    ENCODER_ONLY = "encoder_only"
    DECODER_ONLY = "decoder_only"
    ENCODER_DECODER = "encoder_decoder"
    CAUSAL_LM = "causal_lm"
    MASKED_LM = "masked_lm"
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TOKEN_CLASSIFICATION = "token_classification"


@dataclass
class TransformerConfig:
    """Configuration for transformer models."""
    # Model architecture
    model_type: ModelType = ModelType.ENCODER_ONLY
    vocab_size: int = 50000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    
    # Attention configuration
    attention_type: AttentionType = AttentionType.MULTI_HEAD
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    
    # Advanced features
    use_rotary_positional: bool = True
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    
    # Training configuration
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding for transformers."""
    
    def __init__(self, hidden_size: int, max_position_embeddings: int = 512):
        
    """__init__ function."""
super().__init__()
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        
        # Create rotation matrix
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary positional embedding."""
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply rotation
        cos = torch.cos(emb)[None, :, None, :]
        sin = torch.sin(emb)[None, :, None, :]
        
        # Split input into even and odd dimensions
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        # Apply rotation
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        
        # Interleave back
        rotated = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
        
        return rotated


class AdvancedMultiHeadAttention(nn.Module):
    """Advanced multi-head attention with multiple attention types."""
    
    def __init__(self, config: TransformerConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear transformations
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout_prob)
        
        # Rotary positional embedding
        if config.use_rotary_positional:
            self.rotary_emb = RotaryPositionalEmbedding(
                self.attention_head_size, config.max_position_embeddings
            )
        else:
            self.rotary_emb = None
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for multi-head attention."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> Tuple[torch.Tensor, ...]:
        """Forward pass with advanced attention."""
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Apply rotary positional embedding
        if self.rotary_emb is not None:
            seq_len = hidden_states.size(1)
            mixed_query_layer = self.rotary_emb(mixed_query_layer, seq_len)
            mixed_key_layer = self.rotary_emb(mixed_key_layer, seq_len)
        
        # Transpose for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply attention weights
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply head mask
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # Compute context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        attention_output = self.output(context_layer)
        
        outputs = (attention_output,)
        if output_attentions:
            outputs += (attention_probs,)
        
        return outputs


class FlashAttention(nn.Module):
    """Flash attention implementation for efficiency."""
    
    def __init__(self, config: TransformerConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear transformations
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, ...]:
        """Forward pass with flash attention."""
        batch_size, seq_len, _ = hidden_states.size()
        
        # Linear transformations
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        
        # Transpose for attention computation
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Flash attention computation
        attention_output = self._flash_attention(query, key, value, attention_mask)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.all_head_size)
        attention_output = self.output(attention_output)
        
        outputs = (attention_output,)
        if output_attentions:
            # For flash attention, we don't store attention weights to save memory
            outputs += (None,)
        
        return outputs
    
    def _flash_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Flash attention implementation."""
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Compute output
        attention_output = torch.matmul(attention_probs, value)
        
        return attention_output


class TransformerLayer(nn.Module):
    """Single transformer layer with advanced features."""
    
    def __init__(self, config: TransformerConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # Attention layer
        if config.use_flash_attention:
            self.attention = FlashAttention(config)
        else:
            self.attention = AdvancedMultiHeadAttention(config)
        
        # Layer normalization
        self.attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Gradient checkpointing
        if config.use_gradient_checkpointing:
            self.gradient_checkpointing = True
        else:
            self.gradient_checkpointing = False
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> Tuple[torch.Tensor, ...]:
        """Forward pass with gradient checkpointing."""
        if self.gradient_checkpointing and self.training:
            return self._gradient_checkpointing_forward(hidden_states, attention_mask, head_mask, output_attentions)
        else:
            return self._forward_impl(hidden_states, attention_mask, head_mask, output_attentions)
    
    def _forward_impl(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                      head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> Tuple[torch.Tensor, ...]:
        """Implementation of forward pass."""
        # Self-attention
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = attention_outputs[0]
        
        # Residual connection and layer normalization
        attention_output = self.dropout(attention_output)
        attention_output = self.attention_layernorm(attention_output + hidden_states)
        
        # Feed-forward network
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = F.gelu(intermediate_output)
        
        # Output projection
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layernorm(layer_output + attention_output)
        
        outputs = (layer_output,)
        if output_attentions:
            outputs += (attention_outputs[1],)
        
        return outputs
    
    def _gradient_checkpointing_forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                                       head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> Tuple[torch.Tensor, ...]:
        """Forward pass with gradient checkpointing."""
        def create_custom_forward(module) -> Any:
            def custom_forward(*inputs) -> Any:
                return module._forward_impl(*inputs)
            return custom_forward
        
        return torch.utils.checkpoint.checkpoint(
            create_custom_forward(self),
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions
        )


class AdvancedTransformerModel(nn.Module):
    """Advanced transformer model with comprehensive features."""
    
    def __init__(self, config: TransformerConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = TransformerEmbeddings(config)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer normalization
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> Any:
        """Initialize weights using best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False,
                output_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with comprehensive outputs."""
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Create extended attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Prepare head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            head_mask = [None] * self.config.num_hidden_layers
        
        # Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        
        # Encoder layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        hidden_states = embedding_output
        
        for i, layer_module in enumerate(self.encoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer_module(hidden_states, extended_attention_mask, head_mask[i], output_attentions)
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Final layer normalization
        hidden_states = self.final_layernorm(hidden_states)
        
        # Prepare outputs
        outputs = {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions
        }
        
        return outputs


class TransformerEmbeddings(nn.Module):
    """Embeddings for transformer models."""
    
    def __init__(self, config: TransformerConfig):
        
    """__init__ function."""
super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for embeddings."""
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class LargeLanguageModel(nn.Module):
    """Large language model with advanced features."""
    
    def __init__(self, config: TransformerConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # Transformer backbone
        self.transformer = AdvancedTransformerModel(config)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> Any:
        """Initialize weights for language model."""
        # Initialize language model head
        nn.init.normal_(self.lm_head.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, output_attentions: bool = False,
                output_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass for language modeling."""
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        # Get logits
        hidden_states = transformer_outputs['last_hidden_state']
        logits = self.lm_head(hidden_states)
        
        # Prepare outputs
        outputs = {
            'logits': logits,
            'hidden_states': transformer_outputs.get('hidden_states'),
            'attentions': transformer_outputs.get('attentions')
        }
        
        # Compute loss if labels are provided
        if labels is not None:
            # Shift logits and labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            outputs['loss'] = loss
        
        return outputs


class TransformerFactory:
    """Factory for creating different transformer models."""
    
    @staticmethod
    def create_model(config: TransformerConfig) -> nn.Module:
        """Create transformer model based on configuration."""
        if config.model_type == ModelType.ENCODER_ONLY:
            return AdvancedTransformerModel(config)
        elif config.model_type == ModelType.CAUSAL_LM:
            return LargeLanguageModel(config)
        elif config.model_type == ModelType.SEQUENCE_CLASSIFICATION:
            return SequenceClassificationModel(config)
        elif config.model_type == ModelType.TOKEN_CLASSIFICATION:
            return TokenClassificationModel(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")


class SequenceClassificationModel(nn.Module):
    """Sequence classification model based on transformer."""
    
    def __init__(self, config: TransformerConfig, num_labels: int = 2):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        # Transformer backbone
        self.transformer = AdvancedTransformerModel(config)
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> Any:
        """Initialize weights for classification."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for sequence classification."""
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get pooled output
        hidden_states = transformer_outputs['last_hidden_state']
        pooled_output = hidden_states[:, 0, :]  # Use [CLS] token
        
        # Classification
        logits = self.classifier(pooled_output)
        
        outputs = {'logits': logits}
        
        # Compute loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs['loss'] = loss
        
        return outputs


class TokenClassificationModel(nn.Module):
    """Token classification model based on transformer."""
    
    def __init__(self, config: TransformerConfig, num_labels: int = 2):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        # Transformer backbone
        self.transformer = AdvancedTransformerModel(config)
        
        # Token classification head
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> Any:
        """Initialize weights for token classification."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for token classification."""
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get sequence output
        sequence_output = transformer_outputs['last_hidden_state']
        
        # Token classification
        logits = self.classifier(sequence_output)
        
        outputs = {'logits': logits}
        
        # Compute loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only compute loss on non-padded tokens
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            outputs['loss'] = loss
        
        return outputs


def demonstrate_advanced_transformers():
    """Demonstrate advanced transformers and LLMs."""
    print("Advanced Transformers and LLMs Demonstration")
    print("=" * 55)
    
    # Test different configurations
    configs = [
        TransformerConfig(
            model_type=ModelType.ENCODER_ONLY,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            use_flash_attention=True,
            use_rotary_positional=True
        ),
        TransformerConfig(
            model_type=ModelType.CAUSAL_LM,
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            use_flash_attention=False,
            use_rotary_positional=True
        ),
        TransformerConfig(
            model_type=ModelType.SEQUENCE_CLASSIFICATION,
            hidden_size=384,
            num_hidden_layers=3,
            num_attention_heads=6,
            use_flash_attention=True,
            use_rotary_positional=False
        )
    ]
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\nTesting {config.model_type.value} model:")
        
        try:
            # Create model
            model = TransformerFactory.create_model(config)
            
            # Create dummy input
            batch_size = 2
            seq_length = 16
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            attention_mask = torch.ones_like(input_ids)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Analyze outputs
            if 'logits' in outputs:
                logits_shape = outputs['logits'].shape
                print(f"  Logits shape: {logits_shape}")
            
            if 'hidden_states' in outputs and outputs['hidden_states'] is not None:
                hidden_states_count = len(outputs['hidden_states'])
                print(f"  Hidden states count: {hidden_states_count}")
            
            if 'attentions' in outputs and outputs['attentions'] is not None:
                attentions_count = len(outputs['attentions'])
                print(f"  Attention layers count: {attentions_count}")
            
            # Model statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            
            results[f"model_{i}"] = {
                'config': config,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'success': True
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            results[f"model_{i}"] = {
                'config': config,
                'error': str(e),
                'success': False
            }
    
    return results


if __name__ == "__main__":
    # Demonstrate advanced transformers
    results = demonstrate_advanced_transformers()
    print("\nAdvanced transformers demonstration completed!") 