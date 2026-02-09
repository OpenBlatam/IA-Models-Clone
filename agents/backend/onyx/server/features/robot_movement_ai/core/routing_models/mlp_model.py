"""
MLP Route Predictor Model
=========================

Modelo MLP (Multi-Layer Perceptron) con capas residuales y atención
para predicción de métricas de rutas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base_model import BaseRouteModel, ModelConfig


class MLPRoutePredictor(BaseRouteModel):
    """
    MLP con conexiones residuales y atención para predicción de rutas.
    """
    
    def __init__(self, config: ModelConfig, use_attention: bool = True):
        """
        Inicializar MLP.
        
        Args:
            config: Configuración del modelo
            use_attention: Usar mecanismo de atención
        """
        super(MLPRoutePredictor, self).__init__(config)
        
        self.use_attention = use_attention
        hidden_dims = config.hidden_dims
        
        # Capa de entrada
        self.input_layer = nn.Linear(config.input_dim, hidden_dims[0])
        self.input_norm = nn.LayerNorm(hidden_dims[0]) if config.use_layer_norm else nn.BatchNorm1d(hidden_dims[0]) if config.use_batch_norm else nn.Identity()
        
        # Capas ocultas con conexiones residuales
        self.hidden_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            )
            if config.use_layer_norm:
                self.norm_layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            elif config.use_batch_norm:
                self.norm_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            else:
                self.norm_layers.append(nn.Identity())
            self.dropout_layers.append(nn.Dropout(config.dropout))
        
        # Mecanismo de atención (self-attention)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dims[-1],
                num_heads=4,
                dropout=config.dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dims[-1])
        
        # Capa de salida
        self.output_layer = nn.Linear(hidden_dims[-1], config.output_dim)
        
        # Activación
        self.activation = self.get_activation(config.activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch_size, input_dim]
            
        Returns:
            Tensor de salida [batch_size, output_dim]
        """
        # Capa de entrada
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = self.activation(x)
        
        # Capas ocultas con conexiones residuales
        for i, (hidden, norm, dropout) in enumerate(
            zip(self.hidden_layers, self.norm_layers, self.dropout_layers)
        ):
            residual = x if i == 0 and x.shape[-1] == hidden.out_features else None
            x = hidden(x)
            x = norm(x)
            x = self.activation(x)
            x = dropout(x)
            
            # Conexión residual (si las dimensiones coinciden)
            if residual is not None and x.shape == residual.shape:
                x = x + residual
        
        # Mecanismo de atención
        if self.use_attention:
            # Reshape para atención: [batch_size, 1, hidden_dim]
            x_attn = x.unsqueeze(1)
            attn_out, _ = self.attention(x_attn, x_attn, x_attn)
            x = x + attn_out.squeeze(1)
            x = self.attention_norm(x)
        
        # Capa de salida
        output = self.output_layer(x)
        
        # Aplicar activaciones apropiadas
        # Tiempo y costo: ReLU (valores positivos)
        # Carga y probabilidad: Sigmoid (0-1)
        output = torch.cat([
            F.relu(output[:, :2]),  # tiempo, costo
            torch.sigmoid(output[:, 2:])  # carga, probabilidad
        ], dim=1)
        
        return output




