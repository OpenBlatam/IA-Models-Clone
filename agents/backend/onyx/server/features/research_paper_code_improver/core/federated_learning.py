"""
Federated Learning Framework - Framework de aprendizaje federado
==================================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Métodos de agregación"""
    FED_AVG = "fed_avg"  # Federated Averaging
    FED_PROX = "fed_prox"
    WEIGHTED_AVG = "weighted_avg"


@dataclass
class ClientUpdate:
    """Actualización de cliente"""
    client_id: str
    model_state: Dict[str, torch.Tensor]
    num_samples: int
    loss: float = 0.0


class FederatedLearningFramework:
    """Framework de aprendizaje federado"""
    
    def __init__(self, aggregation_method: AggregationMethod = AggregationMethod.FED_AVG):
        self.aggregation_method = aggregation_method
        self.client_updates: List[ClientUpdate] = []
        self.global_model: Optional[nn.Module] = None
    
    def aggregate_updates(
        self,
        client_updates: List[ClientUpdate],
        global_model: nn.Module
    ) -> nn.Module:
        """Agrega actualizaciones de clientes"""
        if not client_updates:
            return global_model
        
        if self.aggregation_method == AggregationMethod.FED_AVG:
            return self._federated_averaging(client_updates, global_model)
        elif self.aggregation_method == AggregationMethod.WEIGHTED_AVG:
            return self._weighted_averaging(client_updates, global_model)
        else:
            return self._federated_averaging(client_updates, global_model)
    
    def _federated_averaging(
        self,
        client_updates: List[ClientUpdate],
        global_model: nn.Module
    ) -> nn.Module:
        """Federated Averaging"""
        total_samples = sum(update.num_samples for update in client_updates)
        
        # Inicializar parámetros agregados
        aggregated_state = {}
        for name, param in global_model.named_parameters():
            aggregated_state[name] = torch.zeros_like(param.data)
        
        # Agregar actualizaciones ponderadas
        for update in client_updates:
            weight = update.num_samples / total_samples
            for name, param_data in update.model_state.items():
                if name in aggregated_state:
                    aggregated_state[name] += weight * param_data
        
        # Actualizar modelo global
        global_model.load_state_dict(aggregated_state, strict=False)
        logger.info(f"Modelo global actualizado con {len(client_updates)} clientes")
        
        return global_model
    
    def _weighted_averaging(
        self,
        client_updates: List[ClientUpdate],
        global_model: nn.Module
    ) -> nn.Module:
        """Promedio ponderado"""
        total_samples = sum(update.num_samples for update in client_updates)
        
        aggregated_state = {}
        for name, param in global_model.named_parameters():
            aggregated_state[name] = torch.zeros_like(param.data)
        
        for update in client_updates:
            weight = update.num_samples / total_samples
            for name, param_data in update.model_state.items():
                if name in aggregated_state:
                    aggregated_state[name] += weight * param_data
        
        global_model.load_state_dict(aggregated_state, strict=False)
        return global_model
    
    def add_client_update(
        self,
        client_id: str,
        model: nn.Module,
        num_samples: int,
        loss: float = 0.0
    ):
        """Agrega actualización de cliente"""
        update = ClientUpdate(
            client_id=client_id,
            model_state=model.state_dict(),
            num_samples=num_samples,
            loss=loss
        )
        self.client_updates.append(update)
        logger.info(f"Actualización de cliente agregada: {client_id}")




