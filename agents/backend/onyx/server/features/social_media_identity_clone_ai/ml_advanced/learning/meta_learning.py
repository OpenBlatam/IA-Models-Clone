"""
Meta-Learning / Few-Shot Learning
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
import logging
import copy

logger = logging.getLogger(__name__)


class MAML:
    """Model-Agnostic Meta-Learning (MAML)"""
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 1
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
    
    def meta_update(
        self,
        support_set: List[Dict[str, torch.Tensor]],
        query_set: List[Dict[str, torch.Tensor]],
        meta_optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Meta-update step
        
        Args:
            support_set: Support set para adaptación rápida
            query_set: Query set para meta-loss
            meta_optimizer: Optimizador meta
            
        Returns:
            Meta-loss
        """
        # Copiar modelo para adaptación rápida
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Inner loop: adaptación rápida en support set
        for _ in range(self.inner_steps):
            grads = self._compute_gradients(support_set, fast_weights)
            fast_weights = {
                name: fast_weights[name] - self.inner_lr * grads[name]
                for name in fast_weights.keys()
            }
        
        # Outer loop: calcular meta-loss en query set
        meta_loss = self._compute_meta_loss(query_set, fast_weights)
        
        # Meta-update
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
        
        return meta_loss.item()
    
    def _compute_gradients(
        self,
        support_set: List[Dict[str, torch.Tensor]],
        weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Calcula gradientes con pesos específicos"""
        # Implementación simplificada
        # En producción usaría functional API de PyTorch
        grads = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grads[name] = torch.zeros_like(param)
        return grads
    
    def _compute_meta_loss(
        self,
        query_set: List[Dict[str, torch.Tensor]],
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calcula meta-loss"""
        # Implementación simplificada
        return torch.tensor(0.0, requires_grad=True)


class PrototypicalNetwork:
    """Prototypical Networks para few-shot learning"""
    
    def __init__(self, encoder: nn.Module):
        self.encoder = encoder
    
    def compute_prototypes(
        self,
        support_set: List[Dict[str, torch.Tensor]],
        labels: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Calcula prototipos por clase
        
        Args:
            support_set: Support set
            labels: Labels de support set
            
        Returns:
            Prototipos por clase
        """
        # Codificar support set
        embeddings = []
        for sample in support_set:
            with torch.no_grad():
                emb = self.encoder(**sample)
                if hasattr(emb, 'last_hidden_state'):
                    emb = emb.last_hidden_state.mean(dim=1)
                embeddings.append(emb)
        
        embeddings = torch.stack(embeddings)
        
        # Calcular prototipos por clase
        prototypes = {}
        unique_labels = torch.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            class_embeddings = embeddings[mask]
            prototypes[int(label)] = class_embeddings.mean(dim=0)
        
        return prototypes
    
    def predict(
        self,
        query: Dict[str, torch.Tensor],
        prototypes: Dict[int, torch.Tensor]
    ) -> Tuple[int, torch.Tensor]:
        """
        Predice clase usando prototipos
        
        Args:
            query: Query sample
            prototypes: Prototipos por clase
            
        Returns:
            Clase predicha y distancias
        """
        # Codificar query
        with torch.no_grad():
            query_emb = self.encoder(**query)
            if hasattr(query_emb, 'last_hidden_state'):
                query_emb = query_emb.last_hidden_state.mean(dim=1)
        
        # Calcular distancias a prototipos
        distances = {}
        for label, prototype in prototypes.items():
            distance = torch.norm(query_emb - prototype, dim=-1)
            distances[label] = distance
        
        # Clase más cercana
        predicted_class = min(distances, key=distances.get)
        
        return predicted_class, distances




