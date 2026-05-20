"""
Few-Shot Learning Utils - Utilidades de Few-Shot Learning
==========================================================

Utilidades para few-shot learning y meta-learning.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network para few-shot learning.
    
    Paper: https://arxiv.org/abs/1703.05175
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int
    ):
        """
        Inicializar Prototypical Network.
        
        Args:
            encoder: Encoder de features
            embedding_dim: Dimensión de embeddings
        """
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
    
    def compute_prototypes(
        self,
        support_set: torch.Tensor,
        support_labels: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Calcular prototipos por clase.
        
        Args:
            support_set: Support set [N, ...]
            support_labels: Labels del support set [N]
            
        Returns:
            Diccionario de prototipos por clase
        """
        embeddings = self.encoder(support_set)
        prototypes = {}
        
        for class_label in support_labels.unique():
            class_mask = support_labels == class_label
            class_embeddings = embeddings[class_mask]
            prototypes[int(class_label)] = class_embeddings.mean(dim=0)
        
        return prototypes
    
    def forward(
        self,
        query_set: torch.Tensor,
        support_set: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query_set: Query set [Q, ...]
            support_set: Support set [N, ...]
            support_labels: Labels del support set [N]
            
        Returns:
            Logits de distancia a prototipos
        """
        query_embeddings = self.encoder(query_set)
        prototypes = self.compute_prototypes(support_set, support_labels)
        
        # Calcular distancias euclidianas
        num_classes = len(prototypes)
        num_queries = query_embeddings.shape[0]
        
        distances = torch.zeros(num_queries, num_classes)
        
        for i, (class_label, prototype) in enumerate(prototypes.items()):
            distances[:, i] = torch.norm(
                query_embeddings - prototype.unsqueeze(0),
                dim=1
            )
        
        # Convertir distancias a logits (distancia negativa)
        logits = -distances
        
        return logits


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML).
    
    Paper: https://arxiv.org/abs/1703.03400
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 1
    ):
        """
        Inicializar MAML.
        
        Args:
            model: Modelo base
            inner_lr: Learning rate interno
            num_inner_steps: Número de pasos internos
        """
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
    
    def inner_update(
        self,
        support_set: torch.Tensor,
        support_labels: torch.Tensor,
        loss_fn: Callable
    ) -> Dict[str, torch.Tensor]:
        """
        Actualización interna (adaptación rápida).
        
        Args:
            support_set: Support set
            support_labels: Labels del support set
            loss_fn: Función de pérdida
            
        Returns:
            Parámetros actualizados
        """
        # Guardar parámetros originales
        original_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }
        
        # Actualización interna
        for _ in range(self.num_inner_steps):
            outputs = self.model(support_set)
            loss = loss_fn(outputs, support_labels)
            
            # Calcular gradientes
            grads = torch.autograd.grad(
                loss,
                self.model.parameters(),
                create_graph=True
            )
            
            # Actualizar parámetros
            for (name, param), grad in zip(self.model.named_parameters(), grads):
                param.data = param.data - self.inner_lr * grad
        
        # Retornar parámetros actualizados
        updated_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }
        
        # Restaurar parámetros originales
        for name, param in self.model.named_parameters():
            param.data = original_params[name]
        
        return updated_params
    
    def meta_update(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        loss_fn: Callable,
        meta_optimizer: torch.optim.Optimizer
    ):
        """
        Actualización meta (entrenamiento).
        
        Args:
            tasks: Lista de tareas (support_x, support_y, query_x, query_y)
            loss_fn: Función de pérdida
            meta_optimizer: Optimizador meta
        """
        meta_loss = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Adaptación rápida
            updated_params = self.inner_update(support_x, support_y, loss_fn)
            
            # Evaluar en query set con parámetros actualizados
            # Guardar parámetros originales
            original_params = {
                name: param.clone()
                for name, param in self.model.named_parameters()
            }
            
            # Aplicar parámetros actualizados
            for name, param in self.model.named_parameters():
                param.data = updated_params[name]
            
            # Forward en query set
            query_outputs = self.model(query_x)
            query_loss = loss_fn(query_outputs, query_y)
            
            meta_loss += query_loss
            
            # Restaurar parámetros
            for name, param in self.model.named_parameters():
                param.data = original_params[name]
        
        # Actualización meta
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()


class FewShotDataset:
    """
    Dataset para few-shot learning.
    """
    
    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15
    ):
        """
        Inicializar dataset few-shot.
        
        Args:
            data: Datos
            labels: Labels
            n_way: Número de clases
            k_shot: Número de muestras por clase en support
            n_query: Número de muestras por clase en query
        """
        self.data = data
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        
        # Agrupar por clase
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[int(label)].append(idx)
    
    def sample_episode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Muestrear un episodio.
        
        Returns:
            Tupla (support_x, support_y, query_x, query_y)
        """
        # Seleccionar clases aleatorias
        selected_classes = np.random.choice(
            list(self.class_indices.keys()),
            self.n_way,
            replace=False
        )
        
        support_x = []
        support_y = []
        query_x = []
        query_y = []
        
        for class_idx, class_label in enumerate(selected_classes):
            class_indices = self.class_indices[class_label]
            
            # Muestrear support y query
            sampled_indices = np.random.choice(
                class_indices,
                self.k_shot + self.n_query,
                replace=False
            )
            
            support_indices = sampled_indices[:self.k_shot]
            query_indices = sampled_indices[self.k_shot:]
            
            # Agregar a support
            support_x.append(self.data[support_indices])
            support_y.extend([class_idx] * self.k_shot)
            
            # Agregar a query
            query_x.append(self.data[query_indices])
            query_y.extend([class_idx] * self.n_query)
        
        support_x = torch.cat(support_x, dim=0)
        support_y = torch.tensor(support_y)
        query_x = torch.cat(query_x, dim=0)
        query_y = torch.tensor(query_y)
        
        return support_x, support_y, query_x, query_y




