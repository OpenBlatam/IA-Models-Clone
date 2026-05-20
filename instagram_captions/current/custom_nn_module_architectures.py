"""
Custom nn.Module Architectures System
Comprehensive implementation of custom PyTorch nn.Module classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging


@dataclass
class ModelConfig:
    """Configuration for custom model architectures."""
    
    # Model dimensions
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    
    # Regularization
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Initialization
    initializer_range: float = 0.02
    activation_type: str = "gelu"
    
    # Training specific
    use_bias: bool = True
    tie_word_embeddings: bool = True


class MultiHeadAttention(nn.Module):
    """Custom multi-head attention implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear transformations
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        
        # Dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout_rate)
        self.output_dropout = nn.Dropout(config.dropout_rate)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        nn.init.normal_(self.query.weight, std=self.config.initializer_range)
        nn.init.normal_(self.key.weight, std=self.config.initializer_range)
        nn.init.normal_(self.value.weight, std=self.config.initializer_range)
        nn.init.normal_(self.output.weight, std=self.config.initializer_range)
        
        if self.config.use_bias:
            nn.init.zeros_(self.query.bias)
            nn.init.zeros_(self.key.bias)
            nn.init.zeros_(self.value.bias)
            nn.init.zeros_(self.output.bias)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for multi-head attention."""
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Linear transformations
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply head mask
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # Context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        attention_output = self.output(context_layer)
        attention_output = self.output_dropout(attention_output)
        
        return attention_output, attention_probs


class FeedForwardNetwork(nn.Module):
    """Custom feed-forward network implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Linear layers
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Activation function
        if config.activation_type == "gelu":
            self.activation = nn.GELU()
        elif config.activation_type == "relu":
            self.activation = nn.ReLU()
        elif config.activation_type == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation type: {config.activation_type}")
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize feed-forward weights."""
        nn.init.normal_(self.intermediate.weight, std=self.config.initializer_range)
        nn.init.normal_(self.output.weight, std=self.config.initializer_range)
        
        if self.config.use_bias:
            nn.init.zeros_(self.intermediate.bias)
            nn.init.zeros_(self.output.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for feed-forward network."""
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = self.activation(intermediate_output)
        intermediate_output = self.dropout(intermediate_output)
        
        output = self.output(intermediate_output)
        output = self.dropout(output)
        
        return output


class TransformerLayer(nn.Module):
    """Custom transformer layer implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Attention layer
        self.attention = MultiHeadAttention(config)
        
        # Feed-forward layer
        self.feed_forward = FeedForwardNetwork(config)
        
        # Layer normalization
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize transformer layer weights."""
        nn.init.ones_(self.attention_layer_norm.weight)
        nn.init.zeros_(self.attention_layer_norm.bias)
        nn.init.ones_(self.feed_forward_layer_norm.weight)
        nn.init.zeros_(self.feed_forward_layer_norm.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for transformer layer."""
        # Self-attention with residual connection
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask, head_mask
        )
        attention_output = self.attention_layer_norm(hidden_states + attention_output)
        
        # Feed-forward with residual connection
        feed_forward_output = self.feed_forward(attention_output)
        output = self.feed_forward_layer_norm(attention_output + feed_forward_output)
        
        return output, attention_probs


class PositionalEmbedding(nn.Module):
    """Custom positional embedding implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize positional embedding weights."""
        nn.init.normal_(self.position_embeddings.weight, std=self.config.initializer_range)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for positional embedding."""
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings = self.dropout(position_embeddings)
        
        return position_embeddings


class TokenEmbedding(nn.Module):
    """Custom token embedding implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize token embedding weights."""
        nn.init.normal_(self.word_embeddings.weight, std=self.config.initializer_range)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for token embedding."""
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class CustomTransformerModel(nn.Module):
    """Custom transformer model implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embeddings = TokenEmbedding(config)
        self.position_embeddings = PositionalEmbedding(config)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.word_embeddings.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.ones_(self.final_layer_norm.weight)
        nn.init.zeros_(self.final_layer_norm.bias)
        
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, std=self.config.initializer_range)
    
    def get_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate attention mask for input sequence."""
        batch_size, seq_length = input_ids.shape
        attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        
        # Create causal mask for language modeling
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=input_ids.device), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask = attention_mask.masked_fill(causal_mask == 1, float('-inf'))
        
        return attention_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for transformer model."""
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(input_ids)
        
        # Combine embeddings
        hidden_states = token_embeddings + position_embeddings
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        
        # Apply transformer layers
        all_attention_probs = []
        for layer in self.layers:
            hidden_states, attention_probs = layer(hidden_states, attention_mask)
            all_attention_probs.append(attention_probs)
        
        # Final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Language model head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states,
            'attention_probs': all_attention_probs
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.9,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for _ in range(max_length - input_ids.size(1)):
                # Forward pass
                outputs = self(current_ids)
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                
                # Check for EOS token
                if eos_token_id is not None and (current_ids == eos_token_id).any():
                    break
            
            return current_ids


class CustomClassificationHead(nn.Module):
    """Custom classification head for downstream tasks."""
    
    def __init__(self, config: ModelConfig, num_labels: int):
        super().__init__()
        self.config = config
        
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classification head weights."""
        nn.init.normal_(self.dense.weight, std=self.config.initializer_range)
        nn.init.normal_(self.out_proj.weight, std=self.config.initializer_range)
        nn.init.zeros_(self.dense.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification head."""
        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        
        return x


class CustomSequenceClassificationModel(nn.Module):
    """Custom sequence classification model."""
    
    def __init__(self, config: ModelConfig, num_labels: int):
        super().__init__()
        self.config = config
        
        # Base transformer
        self.transformer = CustomTransformerModel(config)
        
        # Classification head
        self.classifier = CustomClassificationHead(config, num_labels)
        
        # Initialize classification head
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Classification head is initialized in its constructor
        pass
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for sequence classification."""
        # Get transformer outputs
        transformer_outputs = self.transformer(input_ids, attention_mask)
        hidden_states = transformer_outputs['hidden_states']
        
        # Get sequence representation (use [CLS] token or mean pooling)
        if attention_mask is not None:
            # Mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            sequence_output = sum_hidden / sum_mask
        else:
            # Use first token
            sequence_output = hidden_states[:, 0, :]
        
        # Classification
        logits = self.classifier(sequence_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        }


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Model configuration
    config = ModelConfig(
        vocab_size=50257,
        hidden_size=768,
        num_layers=6,  # Smaller for demo
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=1024
    )
    
    # Initialize custom model
    model = CustomTransformerModel(config)
    
    # Sample input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Forward pass
    outputs = model(input_ids)
    
    print(f"Model output keys: {outputs.keys()}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Hidden states shape: {outputs['hidden_states'].shape}")
    
    # Test generation
    test_input = torch.randint(0, config.vocab_size, (1, 5))
    generated_ids = model.generate(test_input, max_length=20, temperature=0.7)
    print(f"Generated sequence shape: {generated_ids.shape}")
    
    # Test classification model
    num_labels = 3
    classification_model = CustomSequenceClassificationModel(config, num_labels)
    
    # Classification forward pass
    class_outputs = classification_model(input_ids)
    print(f"Classification logits shape: {class_outputs['logits'].shape}")
    
    logging.info("Custom nn.Module architectures tested successfully!")





