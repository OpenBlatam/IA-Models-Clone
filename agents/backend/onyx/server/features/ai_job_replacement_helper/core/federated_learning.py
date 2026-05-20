"""
Federated Learning Service - Aprendizaje federado
==================================================

Sistema para aprendizaje federado distribuido.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class AggregationMethod(str):
    """Métodos de agregación"""
    FED_AVG = "fed_avg"  # Federated Averaging
    FED_SGD = "fed_sgd"  # Federated SGD
    WEIGHTED_AVG = "weighted_avg"  # Weighted Average


@dataclass
class FederatedConfig:
    """Configuración de aprendizaje federado"""
    aggregation_method: AggregationMethod = AggregationMethod.FED_AVG
    num_clients: int = 10
    rounds: int = 100
    clients_per_round: int = 5
    local_epochs: int = 1
    learning_rate: float = 0.01


@dataclass
class ClientUpdate:
    """Actualización de cliente"""
    client_id: str
    model_state: Dict[str, torch.Tensor]
    num_samples: int
    loss: float
    round_number: int


class FederatedLearningService:
    """Servicio de aprendizaje federado"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.client_updates: Dict[int, List[ClientUpdate]] = {}
        self.global_model_state: Optional[Dict[str, torch.Tensor]] = None
        logger.info("FederatedLearningService initialized")
    
    def aggregate_updates(
        self,
        updates: List[ClientUpdate],
        method: AggregationMethod = AggregationMethod.FED_AVG
    ) -> Dict[str, torch.Tensor]:
        """
        Agregar actualizaciones de clientes.
        
        Args:
            updates: Lista de actualizaciones de clientes
            method: Método de agregación
        
        Returns:
            Estado del modelo global agregado
        """
        if not updates:
            return {}
        
        if method == AggregationMethod.FED_AVG:
            # Federated Averaging: promedio ponderado por número de muestras
            total_samples = sum(update.num_samples for update in updates)
            
            aggregated = {}
            for key in updates[0].model_state.keys():
                weighted_sum = torch.zeros_like(updates[0].model_state[key])
                
                for update in updates:
                    weight = update.num_samples / total_samples
                    weighted_sum += weight * update.model_state[key]
                
                aggregated[key] = weighted_sum
        
        elif method == AggregationMethod.WEIGHTED_AVG:
            # Similar a FedAvg pero con pesos personalizados
            total_weight = sum(update.num_samples for update in updates)
            aggregated = {}
            
            for key in updates[0].model_state.keys():
                weighted_sum = torch.zeros_like(updates[0].model_state[key])
                
                for update in updates:
                    weight = update.num_samples / total_weight
                    weighted_sum += weight * update.model_state[key]
                
                aggregated[key] = weighted_sum
        
        else:
            # Simple average
            aggregated = {}
            for key in updates[0].model_state.keys():
                avg = torch.stack([u.model_state[key] for u in updates]).mean(0)
                aggregated[key] = avg
        
        return aggregated
    
    def receive_client_update(
        self,
        round_number: int,
        client_id: str,
        model_state: Dict[str, torch.Tensor],
        num_samples: int,
        loss: float
    ) -> None:
        """Recibir actualización de cliente"""
        update = ClientUpdate(
            client_id=client_id,
            model_state=model_state,
            num_samples=num_samples,
            loss=loss,
            round_number=round_number
        )
        
        if round_number not in self.client_updates:
            self.client_updates[round_number] = []
        
        self.client_updates[round_number].append(update)
        logger.info(f"Received update from client {client_id} for round {round_number}")
    
    def get_round_updates(self, round_number: int) -> List[ClientUpdate]:
        """Obtener actualizaciones de una ronda"""
        return self.client_updates.get(round_number, [])
    
    def update_global_model(
        self,
        model: nn.Module,
        updates: List[ClientUpdate],
        method: AggregationMethod = AggregationMethod.FED_AVG
    ) -> None:
        """Actualizar modelo global con agregación"""
        aggregated_state = self.aggregate_updates(updates, method)
        
        if aggregated_state:
            model.load_state_dict(aggregated_state)
            self.global_model_state = aggregated_state
            logger.info("Global model updated")




