"""
Federated Learning básico
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import logging
import copy

logger = logging.getLogger(__name__)


class FederatedAveraging:
    """Federated Averaging (FedAvg)"""
    
    def __init__(self):
        pass
    
    def aggregate_models(
        self,
        client_models: List[nn.Module],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Agrega modelos de clientes
        
        Args:
            client_models: Lista de modelos de clientes
            client_weights: Pesos de cada cliente (None para promedio uniforme)
            
        Returns:
            Parámetros agregados
        """
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        
        # Normalizar pesos
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Obtener parámetros de todos los modelos
        all_params = []
        for model in client_models:
            params = {name: param.data.clone() for name, param in model.named_parameters()}
            all_params.append(params)
        
        # Promedio ponderado
        aggregated_params = {}
        for name in all_params[0].keys():
            aggregated_params[name] = sum(
                client_weights[i] * all_params[i][name]
                for i in range(len(client_models))
            )
        
        return aggregated_params
    
    def update_global_model(
        self,
        global_model: nn.Module,
        aggregated_params: Dict[str, torch.Tensor]
    ):
        """Actualiza modelo global con parámetros agregados"""
        for name, param in global_model.named_parameters():
            if name in aggregated_params:
                param.data = aggregated_params[name].clone()


class FederatedTrainer:
    """Trainer para Federated Learning"""
    
    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int = 5,
        rounds: int = 10,
        epochs_per_round: int = 1
    ):
        self.global_model = global_model
        self.num_clients = num_clients
        self.rounds = rounds
        self.epochs_per_round = epochs_per_round
        self.fed_avg = FederatedAveraging()
    
    def train_round(
        self,
        client_data: List[torch.utils.data.DataLoader],
        optimizer_fn: callable
    ) -> Dict[str, Any]:
        """
        Entrena una ronda de federated learning
        
        Args:
            client_data: Lista de dataloaders por cliente
            optimizer_fn: Función que crea optimizador
            
        Returns:
            Métricas de la ronda
        """
        # Distribuir modelo global a clientes
        client_models = []
        for _ in range(self.num_clients):
            client_model = copy.deepcopy(self.global_model)
            client_models.append(client_model)
        
        # Entrenar en cada cliente
        client_weights = []
        for i, (client_model, data) in enumerate(zip(client_models, client_data)):
            optimizer = optimizer_fn(client_model.parameters())
            
            # Entrenar localmente
            for epoch in range(self.epochs_per_round):
                client_model.train()
                for batch in data:
                    optimizer.zero_grad()
                    # Forward y backward (simplificado)
                    # ... entrenamiento local ...
                    pass
            
            # Peso basado en tamaño de datos
            client_weights.append(len(data.dataset))
        
        # Agregar modelos
        aggregated_params = self.fed_avg.aggregate_models(client_models, client_weights)
        
        # Actualizar modelo global
        self.fed_avg.update_global_model(self.global_model, aggregated_params)
        
        return {
            "round_completed": True,
            "num_clients": self.num_clients
        }




