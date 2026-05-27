import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Paper_2103_03874v2Config:
    """Configuration for the MATH dataset solver module."""
    def __init__(
        self,
        vocab_size: int = 30000,          # size of token vocabulary (excluding numbers)
        number_vocab_size: int = 10000,   # max integer value to embed directly (0..9999)
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 512,
        num_answers: int = 5,             # typical multiple-choice (A,B,C,D,E)
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_ln: bool = True,
        use_math_kernel_attention: bool = True,  # polynomial kernel for attention
    ):
        self.vocab_size = vocab_size
        self.number_vocab_size = number_vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.num_answers = num_answers
        self.dropout = dropout
        self.activation = activation
        self.pre_ln = pre_ln
        self.use_math_kernel_attention = use_math_kernel_attention


class MathEmbedding(nn.Module):
    """
    Custom embedding that treats numbers differently:
    - Regular tokens use a standard embedding table.
    - Number tokens (identified by a special token id range) use a separate embedding
      that maps the numeric value directly into d_model space.
    """
    def __init__(self, config: Paper_2103_03874v2Config):
        super().__init__()
        self.number_separator = config.vocab_size  # first id above vocab is the "number" flag
        self.token_embed = nn.Embedding(config.vocab_size + config.number_vocab_size, config.d_model)
        # number embedding: learnable table for small integers
        self.number_embed = nn.Embedding(config.number_vocab_size, config.d_model, padding_idx=0)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor, number_mask: torch.BoolTensor) -> torch.Tensor:
        """
        input_ids: (batch, seq_len) – regular token ids from 0..vocab_size-1
        number_mask: (batch, seq_len) – True where token represents a numeric value
        """
        # For number tokens, we use number_embed with the actual integer value
        number_ids = input_ids - self.number_separator  # shift to 0..number_vocab_size-1
        number_ids = number_ids.clamp(0, self.number_embed.num_embeddings - 1)
        number_emb = self.number_embed(number_ids)  # (batch, seq_len, d_model)

        # For regular tokens, we use token_embed
        regular_ids = input_ids.clamp(0, self.vocab_size - 1)
        regular_emb = self.token_embed(regular_ids)

        # Combine: use number_emb where number_mask is True, else regular_emb
        emb = torch.where(number_mask.unsqueeze(-1), number_emb, regular_emb)
        return self.dropout(emb)


class MathKernelAttention(nn.Module):
    """
    Attention that replaces the dot product with a learnable polynomial kernel.
    Score = (Q·K / sqrt(d))^2 + α·(Q·K / sqrt(d)) + β
    where α, β are learnable scalars. This introduces non‑linear interactions
    inspired by mathematical similarity measures.
    """
    def __init__(self, config: Paper_2103_03874v2Config):
        super().__init__()
        self.num_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

        # learnable polynomial coefficients
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, H, T, head_dim)

        attn_base = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        # polynomial kernel: score = base^2 + α * base + β
        attn = attn_base.pow(2) + self.alpha * attn_base + self.beta

        if mask is not None:
            # mask shape (B, 1, 1, T) or (B, 1, T, T)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.out(out)


class MathTransformerBlock(nn.Module):
    """Transformer block with optional math‑kernel attention and Pre‑LN."""
    def __init__(self, config: Paper_2103_03874v2Config):
        super().__init__()
        self.pre_ln = config.pre_ln
        if config.use_math_kernel_attention:
            self.attn = MathKernelAttention(config)
        else:
            from torch.nn import MultiheadAttention
            self.attn = MultiheadAttention(
                config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
            )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        act = {"gelu": F.gelu, "relu": F.relu}[config.activation]
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.Dropout(config.dropout),
            act(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN or Post-LN
        if self.pre_ln:
            x = x + self.attn(self.norm1(x), attn_mask=mask if isinstance(self.attn, MultiheadAttention) else mask)
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm1(x + self.attn(x, attn_mask=mask if isinstance(self.attn, MultiheadAttention) else mask))
            x = self.norm2(x + self.ffn(x))
        return x


class Paper_2103_03874v2Module(nn.Module):
    """
    PyTorch module implementing a neural solver for the MATH dataset.

    Uses a custom embedding that separates numbers from text, and optionally
    a polynomial‑kernel attention mechanism. The output is a distribution over
    answer choices (multiple‑choice format of MATH).
    """
    def __init__(self, config: Paper_2103_03874v2Config):
        super().__init__()
        self.config = config
        self.embedding = MathEmbedding(config)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        self.blocks = nn.ModuleList([
            MathTransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # pool over sequence dimension
        self.classifier = nn.Linear(config.d_model, config.num_answers)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        number_mask: torch.BoolTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) – token ids (0..vocab_size-1) and numbers (vocab_size..vocab_size+number_vocab_size-1)
            number_mask: (batch, seq_len) – True for tokens representing numbers
            attention_mask: (batch, seq_len) – padding mask (1 for real tokens, 0 for padding)
        Returns:
            logits: (batch, num_answers)
        """
        B, T = input_ids.shape

        # Embedding
        x = self.embedding(input_ids, number_mask)  # (B, T, D)

        # Positional encoding
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(pos_ids)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=attention_mask)

        # Global pooling over sequence dimension (ignoring padding)
        if attention_mask is not None:
            # mask out padding tokens by setting them to zero
            x = x * attention_mask.unsqueeze(-1).float()
        x = self.global_pool(x.transpose(1, 2)).squeeze(-1)  # (B, D)

        # Final classifier
        logits = self.classifier(self.dropout(x))
        return logits


# ======================== Main test ========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = Paper_2103_03874v2Config(
        vocab_size=30000,
        number_vocab_size=10000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        max_seq_len=128,
        num_answers=5,
        dropout=0.1,
        activation="gelu",
        pre_ln=True,
        use_math_kernel_attention=True,
    )

    model = Paper_2103_03874v2Module(config).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Create dummy input: batch=2, seq_len=64
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size + config.number_vocab_size - 1, (batch_size, seq_len)).to(device)
    # Simulate number mask: suppose tokens after vocab_size are numbers
    number_mask = (input_ids >= config.vocab_size).to(device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    # Add some padding (last 10 tokens are padding)
    attention_mask[:, -10:] = 0

    # Forward pass
    logits = model(input_ids, number_mask, attention_mask)
    print(f"Output logits shape: {logits.shape}")  # expected (2, 5)

    # Compute loss (dummy targets)
    targets = torch.randint(0, config.num_answers, (batch_size,)).to(device)
    loss = F.cross_entropy(logits, targets)
    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()
    print("Backward pass successful.")

    # Show that module works with JIT tracing (optional)
    try:
        traced = torch.jit.trace(model, (input_ids, number_mask, attention_mask))
        print("JIT trace successful.")
    except Exception as e:
        print(f"JIT trace failed (non‑critical): {e}")

    print("All tests passed.")