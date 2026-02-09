"""
Continual Learning / Lifelong Learning
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ElasticWeightConsolidation:
    """Elastic Weight Consolidation para continual learning"""
    
    def __init__(self, model: nn.Module, lambda_reg: float = 1.0):
        self.model = model
        self.lambda_reg = lambda_reg
        self.fisher_information = {}
        self.optimal_params = {}
    
    def compute_fisher_information(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 100
    ):
        """Calcula Fisher Information Matrix"""
        self.model.eval()
        fisher = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            self.model.zero_grad()
            inputs = {k: v.to(next(self.model.parameters()).device) 
                     if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Calcular gradientes
            for i in range(logits.size(0)):
                log_prob = torch.log_softmax(logits[i], dim=-1)
                sample_log_prob = log_prob[torch.argmax(log_prob)]
                sample_log_prob.backward(retain_graph=True)
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher[name] += param.grad ** 2
                
                self.model.zero_grad()
                sample_count += 1
        
        # Normalizar
        for name in fisher:
            fisher[name] /= sample_count
        
        self.fisher_information = fisher
    
    def save_optimal_params(self):
        """Guarda parámetros óptimos"""
        self.optimal_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
    
    def ewc_loss(self) -> torch.Tensor:
        """Calcula EWC loss"""
        ewc_loss = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information and name in self.optimal_params:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
        
        return self.lambda_reg * ewc_loss


class ContinualLearner:
    """Aprendizaje continuo"""
    
    def __init__(self, model: nn.Module, method: str = "ewc"):
        self.model = model
        self.method = method
        self.ewc = ElasticWeightConsolidation(model) if method == "ewc" else None
    
    def learn_task(
        self,
        task_data: torch.utils.data.DataLoader,
        num_epochs: int = 5,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """Aprende nueva tarea"""
        if self.method == "ewc" and self.ewc:
            # Calcular Fisher Information de tarea anterior
            if hasattr(self, 'previous_task_data'):
                self.ewc.compute_fisher_information(self.previous_task_data)
                self.ewc.save_optimal_params()
        
        # Entrenar en nueva tarea
        self.model.train()
        for epoch in range(num_epochs):
            for batch in task_data:
                optimizer.zero_grad()
                
                inputs = {k: v.to(next(self.model.parameters()).device)
                         if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                
                outputs = self.model(**inputs)
                loss = self._compute_loss(outputs, batch)
                
                # Agregar EWC loss
                if self.method == "ewc" and self.ewc:
                    ewc_loss = self.ewc.ewc_loss()
                    loss = loss + ewc_loss
                
                loss.backward()
                optimizer.step()
        
        # Guardar datos de tarea para siguiente iteración
        self.previous_task_data = task_data
    
    def _compute_loss(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        """Calcula loss"""
        if hasattr(outputs, 'loss'):
            return outputs.loss
        
        # Loss por defecto
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        labels = batch.get('labels')
        
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            return criterion(logits, labels)
        
        return torch.tensor(0.0, requires_grad=True)




