"""
Deep Learning Templates
========================

Plantillas para diferentes tipos de modelos de deep learning.
"""

from typing import Dict, List, Any
from enum import Enum


class ModelTemplate:
    """Base class para plantillas de modelos."""
    
    @staticmethod
    def generate_model_code(config: Dict[str, Any]) -> str:
        """Genera código del modelo."""
        raise NotImplementedError
    
    @staticmethod
    def generate_config() -> Dict[str, Any]:
        """Genera configuración por defecto."""
        raise NotImplementedError


class TransformerTemplate(ModelTemplate):
    """Plantilla para modelos Transformer."""
    
    @staticmethod
    def generate_model_code(config: Dict[str, Any]) -> str:
        """Genera código para modelo Transformer."""
        vocab_size = config.get('vocab_size', 50257)
        d_model = config.get('d_model', 768)
        nhead = config.get('nhead', 12)
        num_layers = config.get('num_layers', 12)
        dim_feedforward = config.get('dim_feedforward', 3072)
        max_seq_length = config.get('max_seq_length', 512)
        dropout = config.get('dropout', 0.1)
        
        return f'''"""
Transformer Model
=================

Modelo Transformer completo con codificación posicional y atención multi-head.
"""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Codificación posicional para Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Modelo Transformer completo."""
    
    def __init__(
        self,
        vocab_size: int = {vocab_size},
        d_model: int = {d_model},
        nhead: int = {nhead},
        num_layers: int = {num_layers},
        dim_feedforward: int = {dim_feedforward},
        max_seq_length: int = {max_seq_length},
        dropout: float = {dropout}
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embedding de tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Codificación posicional
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Capas de encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # Capa de salida
        self.decoder = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa pesos del modelo."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: Tensor de entrada, shape [seq_len, batch_size]
            src_mask: Máscara de atención opcional
        Returns:
            Tensor de salida, shape [seq_len, batch_size, vocab_size]
        """
        # Embedding + posicional
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Transformer encoder
        output = self.transformer_encoder(src, src_mask)
        
        # Decoder
        output = self.decoder(output)
        
        return output
    
    def generate_mask(self, sz: int) -> torch.Tensor:
        """Genera máscara causal para autoregresión."""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
'''
    
    @staticmethod
    def generate_config() -> Dict[str, Any]:
        return {
            'vocab_size': 50257,
            'd_model': 768,
            'nhead': 12,
            'num_layers': 12,
            'dim_feedforward': 3072,
            'max_seq_length': 512,
            'dropout': 0.1
        }


class DiffusionTemplate(ModelTemplate):
    """Plantilla para modelos de difusión."""
    
    @staticmethod
    def generate_model_code(config: Dict[str, Any]) -> str:
        """Genera código para modelo de difusión."""
        in_channels = config.get('in_channels', 3)
        out_channels = config.get('out_channels', 3)
        model_channels = config.get('model_channels', 128)
        num_res_blocks = config.get('num_res_blocks', 2)
        attention_resolutions = config.get('attention_resolutions', [16, 8])
        dropout = config.get('dropout', 0.0)
        
        return f'''"""
Diffusion Model
===============

Modelo UNet para difusión con bloques residuales y atención.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ResBlock(nn.Module):
    """Bloque residual para modelo de difusión."""
    
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        return x + h


class AttentionBlock(nn.Module):
    """Bloque de atención para modelo de difusión."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.norm(x)
        x = x.view(B, C, H * W)
        
        q, k, v = self.qkv(x).chunk(3, dim=1)
        scale = (C // 3) ** -0.5
        
        attn = torch.softmax(q.transpose(-2, -1) @ k * scale, dim=-1)
        h = (v @ attn.transpose(-2, -1)).view(B, C, H, W)
        
        return x.view(B, C, H, W) + self.proj(h)


class DiffusionUNet(nn.Module):
    """UNet para modelo de difusión."""
    
    def __init__(
        self,
        in_channels: int = {in_channels},
        out_channels: int = {out_channels},
        model_channels: int = {model_channels},
        num_res_blocks: int = {num_res_blocks},
        attention_resolutions: List[int] = {attention_resolutions},
        dropout: float = {dropout}
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        
        # Embedding de tiempo
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels)
        )
        
        # Entrada
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        for i, res in enumerate([32, 16, 8, 4]):
            self.down_blocks.append(self._make_down_block(ch, ch * 2, num_res_blocks, dropout, res in attention_resolutions))
            ch *= 2
        
        # Middle
        self.middle_block = ResBlock(ch, dropout)
        self.middle_attn = AttentionBlock(ch)
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i, res in enumerate([4, 8, 16, 32]):
            self.up_blocks.append(self._make_up_block(ch, ch // 2, num_res_blocks, dropout, res in attention_resolutions))
            ch //= 2
        
        # Salida
        self.output_norm = nn.GroupNorm(32, model_channels)
        self.output_conv = nn.Conv2d(model_channels, out_channels, 3, padding=1)
    
    def _make_down_block(self, in_ch: int, out_ch: int, num_res: int, dropout: float, use_attn: bool):
        blocks = [nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)]
        for _ in range(num_res):
            blocks.append(ResBlock(out_ch, dropout))
        if use_attn:
            blocks.append(AttentionBlock(out_ch))
        return nn.Sequential(*blocks)
    
    def _make_up_block(self, in_ch: int, out_ch: int, num_res: int, dropout: float, use_attn: bool):
        blocks = []
        for _ in range(num_res):
            blocks.append(ResBlock(in_ch, dropout))
        if use_attn:
            blocks.append(AttentionBlock(in_ch))
        blocks.append(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1))
        return nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de entrada [B, C, H, W]
            timestep: Tensor de timesteps [B]
        """
        # Time embedding
        t_emb = self.time_embed(timestep.float()[:, None])
        
        # Downsampling
        h = self.input_conv(x)
        skip_connections = []
        for block in self.down_blocks:
            h = block(h)
            skip_connections.append(h)
        
        # Middle
        h = self.middle_block(h)
        h = self.middle_attn(h)
        
        # Upsampling
        for block in self.up_blocks:
            h = torch.cat([h, skip_connections.pop()], dim=1)
            h = block(h)
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        return self.output_conv(h)
'''
    
    @staticmethod
    def generate_config() -> Dict[str, Any]:
        return {
            'in_channels': 3,
            'out_channels': 3,
            'model_channels': 128,
            'num_res_blocks': 2,
            'attention_resolutions': [16, 8],
            'dropout': 0.0
        }

