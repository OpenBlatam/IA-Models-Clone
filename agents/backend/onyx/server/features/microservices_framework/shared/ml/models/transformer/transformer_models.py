"""
Transformer Model Implementations
Specialized transformer model classes.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from ...models.base_model import BaseLLMModel


class TransformerBlock(nn.Module):
    """Transformer block with attention and feedforward."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class CausalTransformerModel(BaseLLMModel):
    """Causal transformer language model."""
    
    def _build_model(self):
        """Build transformer architecture."""
        # Embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Embedding(
            self.max_seq_length,
            self.hidden_size
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.hidden_size * 4,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(self.hidden_size)
        self.head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        batch_size, seq_length = input_ids.shape
        
        # Embeddings
        token_embeds = self.embedding(input_ids)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        x = token_embeds + position_embeds
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask)
        
        # Output
        x = self.ln_f(x)
        logits = self.head(x)
        
        result = {"logits": logits}
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            result["loss"] = loss
        
        return result

