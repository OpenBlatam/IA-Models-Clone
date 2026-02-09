"""
Weight Initialization - Sistema de inicialización de pesos
===========================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class InitializationMethod(Enum):
    """Métodos de inicialización"""
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    KAIMING_NORMAL = "kaiming_normal"
    ORTHOGONAL = "orthogonal"
    ZERO = "zero"
    UNIFORM = "uniform"
    NORMAL = "normal"


@dataclass
class InitializationConfig:
    """Configuración de inicialización"""
    method: InitializationMethod = InitializationMethod.XAVIER_UNIFORM
    gain: float = 1.0
    a: float = 0.0  # Para LeakyReLU negative slope
    mode: str = "fan_in"  # Para Kaiming
    nonlinearity: str = "relu"  # Para Kaiming


class WeightInitializer:
    """Inicializador de pesos"""
    
    def __init__(self, config: InitializationConfig):
        self.config = config
    
    def initialize_model(self, model: nn.Module):
        """Inicializa todos los pesos de un modelo"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                self._initialize_linear(module)
            elif isinstance(module, nn.Conv2d):
                self._initialize_conv2d(module)
            elif isinstance(module, nn.Conv1d):
                self._initialize_conv1d(module)
            elif isinstance(module, nn.Embedding):
                self._initialize_embedding(module)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                self._initialize_batchnorm(module)
            elif isinstance(module, nn.LSTM):
                self._initialize_lstm(module)
        
        logger.info(f"Modelo inicializado con método {self.config.method.value}")
    
    def _initialize_linear(self, module: nn.Linear):
        """Inicializa capa Linear"""
        if self.config.method == InitializationMethod.XAVIER_UNIFORM:
            init.xavier_uniform_(module.weight, gain=self.config.gain)
        elif self.config.method == InitializationMethod.XAVIER_NORMAL:
            init.xavier_normal_(module.weight, gain=self.config.gain)
        elif self.config.method == InitializationMethod.KAIMING_UNIFORM:
            init.kaiming_uniform_(
                module.weight,
                a=self.config.a,
                mode=self.config.mode,
                nonlinearity=self.config.nonlinearity
            )
        elif self.config.method == InitializationMethod.KAIMING_NORMAL:
            init.kaiming_normal_(
                module.weight,
                a=self.config.a,
                mode=self.config.mode,
                nonlinearity=self.config.nonlinearity
            )
        elif self.config.method == InitializationMethod.ORTHOGONAL:
            init.orthogonal_(module.weight, gain=self.config.gain)
        elif self.config.method == InitializationMethod.ZERO:
            init.zeros_(module.weight)
        elif self.config.method == InitializationMethod.UNIFORM:
            init.uniform_(module.weight, -0.1, 0.1)
        elif self.config.method == InitializationMethod.NORMAL:
            init.normal_(module.weight, 0.0, 0.02)
        
        if module.bias is not None:
            init.zeros_(module.bias)
    
    def _initialize_conv2d(self, module: nn.Conv2d):
        """Inicializa capa Conv2d"""
        if self.config.method == InitializationMethod.KAIMING_UNIFORM:
            init.kaiming_uniform_(
                module.weight,
                a=self.config.a,
                mode=self.config.mode,
                nonlinearity=self.config.nonlinearity
            )
        elif self.config.method == InitializationMethod.KAIMING_NORMAL:
            init.kaiming_normal_(
                module.weight,
                a=self.config.a,
                mode=self.config.mode,
                nonlinearity=self.config.nonlinearity
            )
        elif self.config.method == InitializationMethod.XAVIER_UNIFORM:
            init.xavier_uniform_(module.weight, gain=self.config.gain)
        elif self.config.method == InitializationMethod.XAVIER_NORMAL:
            init.xavier_normal_(module.weight, gain=self.config.gain)
        else:
            init.normal_(module.weight, 0.0, 0.02)
        
        if module.bias is not None:
            init.zeros_(module.bias)
    
    def _initialize_conv1d(self, module: nn.Conv1d):
        """Inicializa capa Conv1d"""
        self._initialize_conv2d(module)  # Similar logic
    
    def _initialize_embedding(self, module: nn.Embedding):
        """Inicializa capa Embedding"""
        if self.config.method == InitializationMethod.NORMAL:
            init.normal_(module.weight, 0.0, 0.02)
        elif self.config.method == InitializationMethod.UNIFORM:
            init.uniform_(module.weight, -0.1, 0.1)
        else:
            init.normal_(module.weight, 0.0, 1.0)
    
    def _initialize_batchnorm(self, module: nn.BatchNorm2d):
        """Inicializa BatchNorm"""
        if module.weight is not None:
            init.ones_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    
    def _initialize_lstm(self, module: nn.LSTM):
        """Inicializa LSTM"""
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.orthogonal_(param)
            elif 'bias' in name:
                init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)




