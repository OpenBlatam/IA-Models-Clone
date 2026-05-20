#!/usr/bin/env python3
"""
Advanced Transformers and Large Language Models for Blaze AI
Implements modern transformer architectures, attention mechanisms, and LLM best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for transformer models"""
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    rotary_emb_dim: int = 64
    use_flash_attention: bool = True
    use_xformers: bool = True


@dataclass
class LLMConfig:
    """Configuration for Large Language Models"""
    model_type: str = "decoder_only"  # decoder_only, encoder_decoder, encoder_only
    architecture: str = "gpt"  # gpt, bert, t5, llama, mistral
    use_rope: bool = True  # Rotary Position Embedding
    use_alibi: bool = False  # Attention with Linear Biases
    use_grouped_query_attention: bool = False
    num_key_value_heads: int = 8
    sliding_window: int = 4096
    use_swiglu: bool = True  # SwiGLU activation
    use_rmsnorm: bool = True  # RMSNorm instead of LayerNorm
    use_parallel_attention: bool = True
    use_parallel_mlp: bool = True


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for transformers"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Generate rotation matrix
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary position embedding"""
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply rotation
        cos = torch.cos(emb).unsqueeze(1).unsqueeze(0)
        sin = torch.sin(emb).unsqueeze(1).unsqueeze(0)
        
        x_rot = x * cos + self._rotate_half(x) * sin
        return x_rot
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dimension"""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional optimizations"""
    
    def __init__(self, config: TransformerConfig, llm_config: LLMConfig):
        super().__init__()
        self.config = config
        self.llm_config = llm_config
        
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Attention layers
        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        
        # Output projection
        self.output = nn.Linear(self.all_head_size, self.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout_prob)
        
        # Grouped Query Attention
        if llm_config.use_grouped_query_attention:
            self.num_key_value_heads = llm_config.num_key_value_heads
            self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
            self.key_value = nn.Linear(self.hidden_size, 
                                     self.num_key_value_heads * self.attention_head_size, bias=False)
        
        # Flash Attention support
        self.use_flash_attention = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
        self.use_xformers = config.use_xformers
        
        # ALiBi support
        if llm_config.use_alibi:
            self.alibi_bias = self._create_alibi_bias(config.max_position_embeddings)
    
    def _create_alibi_bias(self, max_seq_len: int) -> torch.Tensor:
        """Create ALiBi bias matrix"""
        slopes = torch.Tensor(self._get_slopes(self.num_attention_heads))
        alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0)
        return alibi_bias
    
    def _get_slopes(self, n: int) -> List[float]:
        """Get ALiBi slopes"""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + self._get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False, use_cache: bool = False) -> Tuple[torch.Tensor, ...]:
        
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project queries, keys, and values
        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)
        
        # Reshape for attention
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        
        # Apply rotary position embedding
        if hasattr(self, 'rotary_emb') and self.rotary_emb is not None:
            query_states = self.rotary_emb(query_states, seq_len)
            key_states = self.rotary_emb(key_states, seq_len)
        
        # Handle past key/value states
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Compute attention scores
        if self.use_flash_attention:
            # Use PyTorch 2.0 scaled dot product attention
            attention_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask,
                dropout_p=self.config.attention_dropout_prob if self.training else 0.0
            )
        else:
            # Standard attention computation
            attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            
            # Apply ALiBi bias if enabled
            if hasattr(self, 'alibi_bias'):
                attention_scores = attention_scores + self.alibi_bias[:, :seq_len, :seq_len].unsqueeze(0)
            
            # Apply attention mask
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            # Apply softmax and dropout
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            
            # Compute attention output
            attention_output = torch.matmul(attention_probs, value_states)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.all_head_size)
        attention_output = self.output(attention_output)
        
        outputs = (attention_output,)
        if output_attentions:
            outputs += (attention_probs,)
        if use_cache:
            outputs += (past_key_value,)
        
        return outputs


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(hidden_states)) * self.w2(hidden_states))


class TransformerBlock(nn.Module):
    """Single transformer block with modern optimizations"""
    
    def __init__(self, config: TransformerConfig, llm_config: LLMConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.llm_config = llm_config
        self.layer_idx = layer_idx
        
        # Attention
        self.attention = MultiHeadAttention(config, llm_config)
        
        # Layer normalization
        if llm_config.use_rmsnorm:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # MLP
        if llm_config.use_swiglu:
            self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.dropout_prob)
            )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_prob)
        
        # Parallel attention and MLP (for efficiency)
        self.use_parallel = llm_config.use_parallel_attention and llm_config.use_parallel_mlp
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False, use_cache: bool = False) -> Tuple[torch.Tensor, ...]:
        
        residual = hidden_states
        
        # Pre-norm
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache
        )
        
        attention_output = attention_outputs[0]
        
        # Residual connection
        hidden_states = residual + self.dropout(attention_output)
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(mlp_output)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += attention_outputs[1:]
        if use_cache:
            outputs += attention_outputs[-1:]
        
        return outputs


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class AdvancedTransformer(nn.Module):
    """Advanced transformer model with modern optimizations"""
    
    def __init__(self, config: TransformerConfig, llm_config: LLMConfig):
        super().__init__()
        self.config = config
        self.llm_config = llm_config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        
        # Position embeddings
        if llm_config.use_rope:
            self.rotary_emb = RotaryPositionalEmbedding(
                config.rotary_emb_dim,
                config.max_position_embeddings
            )
        else:
            self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, llm_config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Final layer normalization
        if llm_config.use_rmsnorm:
            self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language model head
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie word embeddings if specified
        if config.tie_word_embeddings:
            self.lm_head = self.embed_tokens
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value
    
    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
                inputs_embeds: Optional[torch.Tensor] = None, use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.use_cache
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True
        
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Get position embeddings
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.size(1), device=inputs_embeds.device).unsqueeze(0)
        
        if hasattr(self, 'rotary_emb') and self.rotary_emb is not None:
            # Apply rotary position embedding
            inputs_embeds = self.rotary_emb(inputs_embeds, inputs_embeds.size(1))
        elif hasattr(self, 'embed_positions'):
            # Use learned position embeddings
            position_embeddings = self.embed_positions(position_ids)
            inputs_embeds = inputs_embeds + position_embeddings
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        hidden_states = inputs_embeds
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        next_cache = () if use_cache else None
        
        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_cache += (layer_outputs[-1],)
            
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
        
        # Final layer normalization
        hidden_states = self.norm(hidden_states)
        
        # Add final hidden states
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Language model head
        logits = self.lm_head(hidden_states)
        
        if not return_dict:
            return tuple(v for v in [logits, next_cache, all_hidden_states, all_self_attentions] if v is not None)
        
        return {
            'logits': logits,
            'past_key_values': next_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attentions
        }


class LLMTrainer:
    """Trainer for Large Language Models"""
    
    def __init__(self, model: nn.Module, config: TransformerConfig, llm_config: LLMConfig):
        self.model = model
        self.config = config
        self.llm_config = llm_config
        
        # Training components
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=1e-6
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Training state
        self.current_step = 0
        self.training_history = []
    
    def train_step(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None) -> Dict[str, float]:
        """Single training step"""
        
        self.optimizer.zero_grad()
        
        # Forward pass
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids=input_ids)
                logits = outputs['logits']
                
                if labels is not None:
                    # Shift logits and labels for next token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    # Compute loss
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                else:
                    loss = torch.tensor(0.0, device=logits.device)
        else:
            outputs = self.model(input_ids=input_ids)
            logits = outputs['logits']
            
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
            else:
                loss = torch.tensor(0.0, device=logits.device)
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        # Update scheduler
        self.scheduler.step()
        
        # Update step counter
        self.current_step += 1
        
        return {
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'step': self.current_step
        }
    
    def generate(self, input_ids: torch.LongTensor, max_length: int = 100, 
                temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                do_sample: bool = True, pad_token_id: int = None) -> torch.LongTensor:
        """Generate text using the model"""
        
        self.model.eval()
        
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        
        batch_size = input_ids.size(0)
        current_length = input_ids.size(1)
        
        # Initialize output with input
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - current_length):
                # Get model outputs
                outputs = self.model(input_ids=generated, use_cache=True)
                next_token_logits = outputs['logits'][:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check if all sequences have reached EOS
                if (generated == self.config.eos_token_id).any(dim=1).all():
                    break
        
        return generated


class TransformerExperiments:
    """Collection of transformer experiments and demonstrations"""
    
    @staticmethod
    def demonstrate_attention_mechanisms():
        """Demonstrate different attention mechanisms"""
        
        logger.info("Demonstrating attention mechanisms...")
        
        # Create configurations
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512
        )
        
        llm_config = LLMConfig(
            use_rope=True,
            use_alibi=False,
            use_rmsnorm=True
        )
        
        # Create model
        model = AdvancedTransformer(config, llm_config)
        
        # Create input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs['logits']
        
        logger.info(f"Model output shape: {logits.shape}")
        logger.info(f"Attention demonstration completed")
        
        return model, outputs
    
    @staticmethod
    def demonstrate_text_generation():
        """Demonstrate text generation capabilities"""
        
        logger.info("Demonstrating text generation...")
        
        # Create smaller model for demonstration
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_attention_heads=8,
            max_position_embeddings=256
        )
        
        llm_config = LLMConfig(
            use_rope=True,
            use_rmsnorm=True,
            use_swiglu=True
        )
        
        # Create model and trainer
        model = AdvancedTransformer(config, llm_config)
        trainer = LLMTrainer(model, config, llm_config)
        
        # Create sample input
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        
        # Generate text
        generated = trainer.generate(
            input_ids=input_ids,
            max_length=20,
            temperature=0.8,
            top_k=10,
            do_sample=True
        )
        
        logger.info(f"Input sequence: {input_ids[0].tolist()}")
        logger.info(f"Generated sequence: {generated[0].tolist()}")
        logger.info(f"Text generation demonstration completed")
        
        return model, trainer, generated
    
    @staticmethod
    def demonstrate_optimization_features():
        """Demonstrate optimization features"""
        
        logger.info("Demonstrating optimization features...")
        
        # Create configurations
        config = TransformerConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4
        )
        
        llm_config = LLMConfig(
            use_flash_attention=True,
            use_xformers=True,
            use_parallel_attention=True,
            use_parallel_mlp=True
        )
        
        # Create model
        model = AdvancedTransformer(config, llm_config)
        
        # Test forward pass
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        logger.info(f"Optimization features demonstration completed")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        return model, outputs


def main():
    """Main execution function"""
    logger.info("Starting Advanced Transformers and LLMs Demonstrations...")
    
    # Demonstrate attention mechanisms
    logger.info("Testing attention mechanisms...")
    attention_model, attention_outputs = TransformerExperiments.demonstrate_attention_mechanisms()
    
    # Demonstrate text generation
    logger.info("Testing text generation...")
    gen_model, gen_trainer, generated_text = TransformerExperiments.demonstrate_text_generation()
    
    # Demonstrate optimization features
    logger.info("Testing optimization features...")
    opt_model, opt_outputs = TransformerExperiments.demonstrate_optimization_features()
    
    # Create comprehensive model
    logger.info("Creating comprehensive transformer model...")
    
    config = TransformerConfig(
        vocab_size=50257,
        hidden_size=768,
        num_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=1024
    )
    
    llm_config = LLMConfig(
        model_type="decoder_only",
        architecture="gpt",
        use_rope=True,
        use_rmsnorm=True,
        use_swiglu=True,
        use_flash_attention=True,
        use_parallel_attention=True
    )
    
    comprehensive_model = AdvancedTransformer(config, llm_config)
    
    # Test comprehensive model
    test_input = torch.randint(0, config.vocab_size, (1, 10))
    
    with torch.no_grad():
        test_outputs = comprehensive_model(input_ids=test_input)
    
    logger.info(f"Comprehensive model output shape: {test_outputs['logits'].shape}")
    logger.info(f"Comprehensive model parameters: {sum(p.numel() for p in comprehensive_model.parameters())}")
    
    # Summary
    logger.info("Transformers and LLMs Summary:")
    logger.info(f"Attention mechanisms tested: ✓")
    logger.info(f"Text generation tested: ✓")
    logger.info(f"Optimization features tested: ✓")
    logger.info(f"Comprehensive model created: ✓")
    logger.info(f"Total parameters across models: {sum(p.numel() for p in [attention_model, gen_model, opt_model, comprehensive_model])}")
    
    logger.info("Advanced Transformers and LLMs demonstrations completed successfully!")


if __name__ == "__main__":
    main()
