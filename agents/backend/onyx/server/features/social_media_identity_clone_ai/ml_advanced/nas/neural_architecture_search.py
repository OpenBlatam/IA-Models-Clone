"""
Neural Architecture Search (NAS)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import logging
import random

logger = logging.getLogger(__name__)


class SearchSpace:
    """Espacio de búsqueda para NAS"""
    
    def __init__(self):
        self.layer_types = ["linear", "conv1d", "conv2d"]
        self.activation_types = ["relu", "gelu", "tanh", "swish"]
        self.normalization_types = ["batch_norm", "layer_norm", "group_norm"]
        self.dropout_values = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    def sample_architecture(self) -> Dict[str, Any]:
        """Muestra arquitectura aleatoria del espacio de búsqueda"""
        return {
            "num_layers": random.randint(2, 6),
            "hidden_dims": [random.choice([128, 256, 512, 768]) for _ in range(random.randint(2, 4))],
            "activation": random.choice(self.activation_types),
            "normalization": random.choice(self.normalization_types),
            "dropout": random.choice(self.dropout_values)
        }


class ArchitectureBuilder:
    """Constructor de arquitecturas"""
    
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def build_from_config(self, config: Dict[str, Any]) -> nn.Module:
        """Construye modelo desde configuración"""
        layers = []
        hidden_dims = config.get("hidden_dims", [256, 128])
        activation = config.get("activation", "relu")
        normalization = config.get("normalization", "batch_norm")
        dropout = config.get("dropout", 0.1)
        
        # Primera capa
        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        
        # Capas intermedias
        for i in range(len(hidden_dims) - 1):
            # Normalization
            if normalization == "batch_norm":
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
            elif normalization == "layer_norm":
                layers.append(nn.LayerNorm(hidden_dims[i]))
            
            # Activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "swish":
                layers.append(nn.SiLU())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            # Linear
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        # Última capa
        layers.append(nn.Linear(hidden_dims[-1], self.output_dim))
        
        return nn.Sequential(*layers)


class NAS:
    """Neural Architecture Search básico"""
    
    def __init__(
        self,
        search_space: SearchSpace,
        architecture_builder: ArchitectureBuilder,
        num_trials: int = 50
    ):
        self.search_space = search_space
        self.architecture_builder = architecture_builder
        self.num_trials = num_trials
        self.trials = []
    
    def search(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Busca mejor arquitectura
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            device: Device
            
        Returns:
            Mejor arquitectura y métricas
        """
        best_architecture = None
        best_score = float('-inf')
        
        for trial in range(self.num_trials):
            # Muestrear arquitectura
            config = self.search_space.sample_architecture()
            
            # Construir modelo
            model = self.architecture_builder.build_from_config(config).to(device)
            
            # Entrenar rápidamente
            score = self._quick_train_eval(model, train_loader, val_loader, device)
            
            self.trials.append({
                "config": config,
                "score": score
            })
            
            if score > best_score:
                best_score = score
                best_architecture = config
            
            logger.info(f"Trial {trial+1}/{self.num_trials}, Score: {score:.4f}")
        
        return {
            "best_architecture": best_architecture,
            "best_score": best_score,
            "all_trials": self.trials
        }
    
    def _quick_train_eval(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: str,
        epochs: int = 3
    ) -> float:
        """Entrenamiento rápido y evaluación"""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Entrenar
        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                inputs = batch["input_ids"].to(device) if "input_ids" in batch else batch[0].to(device)
                labels = batch["labels"].to(device) if "labels" in batch else batch[1].to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Evaluar
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_ids"].to(device) if "input_ids" in batch else batch[0].to(device)
                labels = batch["labels"].to(device) if "labels" in batch else batch[1].to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy




