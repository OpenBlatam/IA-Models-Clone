"""
Model Generator - Generador de arquitecturas de modelos
========================================================

Genera arquitecturas de modelos PyTorch siguiendo mejores prácticas.
Soporta diferentes tipos de modelos: CNN, RNN, Transformer, Diffusion, etc.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from .base_generator import BaseGenerator
from ..utils.code_templates import get_transformer_model_template
from ..utils.detectors import ModelType, detect_model_type
from ...shared_utils import safe_write_file

logger = logging.getLogger(__name__)


def _should_generate_model(keywords: Dict[str, Any]) -> bool:
    """
    Determina si se debe generar modelo (función pura).
    
    Args:
        keywords: Keywords del proyecto
        
    Returns:
        True si se debe generar, False en caso contrario
    """
    return keywords.get("is_deep_learning", False) or keywords.get("requires_model", True)


def _get_framework(project_info: Dict[str, Any]) -> str:
    """
    Obtiene el framework del proyecto (función pura).
    
    Args:
        project_info: Información del proyecto
        
    Returns:
        Framework a usar
    """
    return project_info.get("framework", "pytorch")


class ModelGenerator(BaseGenerator):
    """
    Generador de arquitecturas de modelos.
    
    Genera código para diferentes tipos de modelos de deep learning
    siguiendo mejores prácticas de PyTorch.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(self) -> None:
        """Inicializar generador de modelos."""
        super().__init__(
            name="model",
            description="Generates PyTorch model architectures"
        )
    
    def should_generate(self, keywords: Dict[str, Any]) -> bool:
        """
        Determinar si se debe generar modelo.
        
        Args:
            keywords: Keywords del proyecto
            
        Returns:
            True si se debe generar, False en caso contrario
        """
        return _should_generate_model(keywords)
    
    def generate(
        self,
        target_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> None:
        """
        Generar arquitectura de modelo.
        
        Args:
            target_dir: Directorio destino
            keywords: Keywords extraídos
            project_info: Información del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        model_type = self._detect_model_type(keywords)
        framework = _get_framework(project_info)
        
        if model_type == ModelType.TRANSFORMER:
            self._generate_transformer_model(target_dir, keywords, project_info, framework)
        elif model_type == ModelType.DIFFUSION:
            self._generate_diffusion_model(target_dir, keywords, project_info)
        elif model_type == ModelType.CNN:
            self._generate_cnn_model(target_dir, keywords, project_info)
        elif model_type == ModelType.RNN:
            self._generate_rnn_model(target_dir, keywords, project_info)
        else:
            self._generate_base_model(target_dir, keywords, project_info)
    
    def _detect_model_type(self, keywords: Dict[str, Any]) -> ModelType:
        """
        Detectar tipo de modelo desde keywords.
        
        Args:
            keywords: Keywords extraídos
            
        Returns:
            Tipo de modelo detectado
        """
        return detect_model_type(keywords)
    
    def _generate_transformer_model(
        self,
        target_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
        framework: str = "pytorch"
    ) -> None:
        """
        Generar modelo Transformer usando plantillas.
        
        Args:
            target_dir: Directorio destino
            keywords: Keywords extraídos
            project_info: Información del proyecto
            framework: Framework a usar
        """
        model_file = target_dir / "transformer_model.py"
        code = get_transformer_model_template(framework)
        
        safe_write_file(model_file, code, logger=self.logger)
        self.logger.info(f"Generated Transformer model at {model_file}")
        
        if project_info.get("requires_training", True):
            config_file = target_dir / "model_config.py"
            config_code = '''"""
Model Configuration
===================

Configuración del modelo Transformer.
"""

from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    """Configuración del modelo"""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
'''
            safe_write_file(config_file, config_code, logger=self.logger)
            self.logger.info(f"Generated model config at {config_file}")
    
    def _generate_diffusion_model(
        self,
        target_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> None:
        """
        Generar modelo Diffusion.
        
        Args:
            target_dir: Directorio destino
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        self.logger.info("Diffusion model generation not yet implemented")
    
    def _generate_cnn_model(
        self,
        target_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> None:
        """
        Generar modelo CNN.
        
        Args:
            target_dir: Directorio destino
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        self.logger.info("CNN model generation not yet implemented")
    
    def _generate_rnn_model(
        self,
        target_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> None:
        """
        Generar modelo RNN.
        
        Args:
            target_dir: Directorio destino
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        self.logger.info("RNN model generation not yet implemented")
    
    def _generate_base_model(
        self,
        target_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> None:
        """
        Generar modelo base genérico.
        
        Args:
            target_dir: Directorio destino
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        model_file = target_dir / "model.py"
        
        code = '''"""
Base Model Architecture
========================

Arquitectura de modelo base siguiendo mejores prácticas de PyTorch.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class BaseModel(nn.Module):
    """
    Modelo base.
    
    Proporciona estructura base para modelos de deep learning.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Inicializar modelo base.
        
        Args:
            input_size: Tamaño de entrada
            hidden_size: Tamaño de capas ocultas
            output_size: Tamaño de salida
            num_layers: Número de capas
            dropout: Tasa de dropout
        """
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.model = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Inicializar pesos"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)
'''
        
        safe_write_file(model_file, code, logger=self.logger)
        self.logger.info(f"Generated base model at {model_file}")
