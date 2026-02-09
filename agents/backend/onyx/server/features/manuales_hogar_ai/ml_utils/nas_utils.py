"""
NAS Utils - Neural Architecture Search Utilities
=================================================

Utilidades para búsqueda automática de arquitecturas.
"""

import logging
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureConfig:
    """Configuración de arquitectura."""
    num_layers: int
    hidden_sizes: List[int]
    activation: str = "relu"
    dropout: float = 0.1


class ArchitectureSearch:
    """
    Búsqueda de arquitecturas.
    """
    
    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        objective_fn: Callable,
        strategy: str = "random"
    ):
        """
        Inicializar búsqueda.
        
        Args:
            search_space: Espacio de búsqueda
            objective_fn: Función objetivo
            strategy: Estrategia ('random', 'grid')
        """
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.strategy = strategy
        self.results: List[Dict[str, Any]] = []
    
    def search(
        self,
        num_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Buscar mejor arquitectura.
        
        Args:
            num_trials: Número de trials
            
        Returns:
            Mejor configuración
        """
        best_score = float('-inf')
        best_config = None
        
        for trial in range(num_trials):
            # Generar configuración
            config = self._generate_config()
            
            # Evaluar
            score = self.objective_fn(config)
            
            self.results.append({
                'config': config,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_config = config
            
            logger.info(f"Trial {trial + 1}/{num_trials}, Score: {score:.4f}")
        
        return {
            'best_config': best_config,
            'best_score': best_score,
            'all_results': self.results
        }
    
    def _generate_config(self) -> Dict[str, Any]:
        """
        Generar configuración aleatoria.
        
        Returns:
            Configuración
        """
        config = {}
        for key, values in self.search_space.items():
            config[key] = random.choice(values)
        return config


class SuperNet(nn.Module):
    """
    SuperNet para one-shot NAS.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        max_layers: int = 5,
        max_hidden_size: int = 512
    ):
        """
        Inicializar SuperNet.
        
        Args:
            input_size: Tamaño de entrada
            output_size: Tamaño de salida
            max_layers: Máximo número de capas
            max_hidden_size: Máximo tamaño oculto
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.max_layers = max_layers
        self.max_hidden_size = max_hidden_size
        
        # Capas opcionales
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for i in range(max_layers):
            # Múltiples opciones de tamaño
            layer_options = nn.ModuleList([
                nn.Linear(prev_size, hidden_size)
                for hidden_size in [64, 128, 256, 512]
                if hidden_size <= max_hidden_size
            ])
            self.layers.append(layer_options)
            prev_size = max_hidden_size
    
    def forward(
        self,
        x: torch.Tensor,
        architecture: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input
            architecture: Arquitectura específica (opcional)
            
        Returns:
            Output
        """
        if architecture is None:
            # Usar todas las capas
            for layer_options in self.layers:
                x = layer_options[0](x)  # Default
                x = torch.relu(x)
        else:
            # Usar arquitectura específica
            for i, layer_idx in enumerate(architecture):
                if i < len(self.layers):
                    x = self.layers[i][layer_idx](x)
                    x = torch.relu(x)
        
        return x


class WeightSharing:
    """
    Weight sharing para NAS eficiente.
    """
    
    def __init__(self, supernet: SuperNet):
        """
        Inicializar weight sharing.
        
        Args:
            supernet: SuperNet
        """
        self.supernet = supernet
    
    def sample_architecture(self) -> List[int]:
        """
        Muestrear arquitectura.
        
        Returns:
            Arquitectura
        """
        architecture = []
        for layer_options in self.supernet.layers:
            architecture.append(random.randint(0, len(layer_options) - 1))
        return architecture
    
    def evaluate_architecture(
        self,
        architecture: List[int],
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda"
    ) -> float:
        """
        Evaluar arquitectura.
        
        Args:
            architecture: Arquitectura
            dataloader: DataLoader
            device: Dispositivo
            
        Returns:
            Score
        """
        self.supernet.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, None
                
                inputs = inputs.to(device)
                outputs = self.supernet(inputs, architecture)
                
                if targets is not None:
                    targets = targets.to(device)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += len(targets)
        
        return correct / total if total > 0 else 0.0




