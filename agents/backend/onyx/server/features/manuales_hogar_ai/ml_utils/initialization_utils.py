"""
Initialization Utils - Utilidades de Inicialización de Pesos
==============================================================

Utilidades para inicialización de pesos en modelos.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.init as init
import math
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class WeightInitializer:
    """
    Inicializador de pesos.
    """
    
    @staticmethod
    def xavier_uniform(module: nn.Module):
        """
        Inicializar con Xavier uniform.
        
        Args:
            module: Módulo
        """
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param)
    
    @staticmethod
    def xavier_normal(module: nn.Module):
        """
        Inicializar con Xavier normal.
        
        Args:
            module: Módulo
        """
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.xavier_normal_(param)
    
    @staticmethod
    def kaiming_uniform(module: nn.Module, mode: str = 'fan_in', nonlinearity: str = 'relu'):
        """
        Inicializar con Kaiming uniform.
        
        Args:
            module: Módulo
            mode: Modo ('fan_in', 'fan_out')
            nonlinearity: No linealidad
        """
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.kaiming_uniform_(param, mode=mode, nonlinearity=nonlinearity)
    
    @staticmethod
    def kaiming_normal(module: nn.Module, mode: str = 'fan_in', nonlinearity: str = 'relu'):
        """
        Inicializar con Kaiming normal.
        
        Args:
            module: Módulo
            mode: Modo
            nonlinearity: No linealidad
        """
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.kaiming_normal_(param, mode=mode, nonlinearity=nonlinearity)
    
    @staticmethod
    def orthogonal(module: nn.Module, gain: float = 1.0):
        """
        Inicializar con ortogonal.
        
        Args:
            module: Módulo
            gain: Ganancia
        """
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param, gain=gain)
    
    @staticmethod
    def sparse(module: nn.Module, sparsity: float = 0.1, std: float = 0.01):
        """
        Inicializar sparse.
        
        Args:
            module: Módulo
            sparsity: Esparsidad
            std: Desviación estándar
        """
        for param in module.parameters():
            if len(param.shape) >= 2:
                init.normal_(param, mean=0, std=std)
                # Hacer sparse
                mask = torch.rand_like(param) > sparsity
                param.data *= mask.float()
    
    @staticmethod
    def initialize_model(
        model: nn.Module,
        method: str = "kaiming_uniform",
        **kwargs
    ):
        """
        Inicializar modelo completo.
        
        Args:
            model: Modelo
            method: Método de inicialización
            **kwargs: Argumentos adicionales
        """
        if method == "xavier_uniform":
            WeightInitializer.xavier_uniform(model)
        elif method == "xavier_normal":
            WeightInitializer.xavier_normal(model)
        elif method == "kaiming_uniform":
            WeightInitializer.kaiming_uniform(model, **kwargs)
        elif method == "kaiming_normal":
            WeightInitializer.kaiming_normal(model, **kwargs)
        elif method == "orthogonal":
            WeightInitializer.orthogonal(model, **kwargs)
        elif method == "sparse":
            WeightInitializer.sparse(model, **kwargs)
        else:
            raise ValueError(f"Unknown initialization method: {method}")


class LayerInitializer:
    """
    Inicializador específico por tipo de capa.
    """
    
    @staticmethod
    def initialize_linear(layer: nn.Linear, method: str = "kaiming_uniform"):
        """
        Inicializar capa lineal.
        
        Args:
            layer: Capa lineal
            method: Método
        """
        if method == "kaiming_uniform":
            init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        elif method == "xavier_uniform":
            init.xavier_uniform_(layer.weight)
        elif method == "orthogonal":
            init.orthogonal_(layer.weight)
        
        if layer.bias is not None:
            init.constant_(layer.bias, 0.0)
    
    @staticmethod
    def initialize_conv2d(layer: nn.Conv2d, method: str = "kaiming_uniform"):
        """
        Inicializar capa convolucional.
        
        Args:
            layer: Capa Conv2d
            method: Método
        """
        if method == "kaiming_uniform":
            init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='relu')
        elif method == "xavier_uniform":
            init.xavier_uniform_(layer.weight)
        
        if layer.bias is not None:
            init.constant_(layer.bias, 0.0)
    
    @staticmethod
    def initialize_batch_norm(layer: nn.BatchNorm2d):
        """
        Inicializar BatchNorm.
        
        Args:
            layer: Capa BatchNorm
        """
        if layer.weight is not None:
            init.constant_(layer.weight, 1.0)
        if layer.bias is not None:
            init.constant_(layer.bias, 0.0)
    
    @staticmethod
    def initialize_lstm(layer: nn.LSTM):
        """
        Inicializar LSTM.
        
        Args:
            layer: Capa LSTM
        """
        for name, param in layer.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.orthogonal_(param)
            elif 'bias' in name:
                init.constant_(param, 0.0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1.0)




