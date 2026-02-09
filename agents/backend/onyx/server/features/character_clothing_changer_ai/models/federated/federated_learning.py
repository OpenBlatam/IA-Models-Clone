"""
Federated Learning System
==========================
Sistema de aprendizaje federado para entrenamiento distribuido
"""

import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ClientStatus(Enum):
    """Estados de cliente"""
    IDLE = "idle"
    TRAINING = "training"
    UPDATING = "updating"
    OFFLINE = "offline"


class AggregationMethod(Enum):
    """Métodos de agregación"""
    FED_AVG = "fed_avg"
    FED_SGD = "fed_sgd"
    FED_PROX = "fed_prox"
    WEIGHTED_AVG = "weighted_avg"


@dataclass
class FederatedClient:
    """Cliente federado"""
    id: str
    name: str
    data_size: int
    status: ClientStatus
    model_version: int = 0
    last_update: float = 0.0
    training_rounds: int = 0


@dataclass
class TrainingRound:
    """Ronda de entrenamiento"""
    round_number: int
    started_at: float
    completed_at: Optional[float] = None
    participants: List[str] = None
    aggregated_model: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.participants is None:
            self.participants = []


class FederatedLearning:
    """
    Sistema de aprendizaje federado
    """
    
    def __init__(self):
        self.clients: Dict[str, FederatedClient] = {}
        self.training_rounds: List[TrainingRound] = []
        self.global_model: Optional[Dict[str, Any]] = None
        self.aggregation_method: AggregationMethod = AggregationMethod.FED_AVG
        self.min_clients_per_round: int = 3
    
    def register_client(
        self,
        name: str,
        data_size: int
    ) -> FederatedClient:
        """
        Registrar cliente federado
        
        Args:
            name: Nombre del cliente
            data_size: Tamaño del dataset del cliente
        """
        client_id = f"client_{int(time.time())}"
        
        client = FederatedClient(
            id=client_id,
            name=name,
            data_size=data_size,
            status=ClientStatus.IDLE,
            last_update=time.time()
        )
        
        self.clients[client_id] = client
        return client
    
    def initialize_global_model(self, model_config: Dict[str, Any]):
        """Inicializar modelo global"""
        self.global_model = {
            'config': model_config,
            'version': 0,
            'weights': self._initialize_weights(model_config),
            'created_at': time.time()
        }
    
    def _initialize_weights(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Inicializar pesos del modelo (simplificado)"""
        # En implementación real, inicializar según arquitectura
        return {
            'layer1': [0.0] * config.get('layer1_size', 128),
            'layer2': [0.0] * config.get('layer2_size', 64),
            'output': [0.0] * config.get('output_size', 10)
        }
    
    def start_training_round(self) -> TrainingRound:
        """Iniciar ronda de entrenamiento"""
        round_number = len(self.training_rounds) + 1
        
        # Seleccionar clientes disponibles
        available_clients = [
            client for client in self.clients.values()
            if client.status == ClientStatus.IDLE
        ]
        
        if len(available_clients) < self.min_clients_per_round:
            raise ValueError(f"Not enough clients. Need {self.min_clients_per_round}, have {len(available_clients)}")
        
        # Seleccionar clientes para esta ronda
        selected_clients = available_clients[:self.min_clients_per_round]
        
        round_obj = TrainingRound(
            round_number=round_number,
            started_at=time.time(),
            participants=[c.id for c in selected_clients]
        )
        
        # Marcar clientes como entrenando
        for client in selected_clients:
            client.status = ClientStatus.TRAINING
        
        self.training_rounds.append(round_obj)
        return round_obj
    
    def submit_client_update(
        self,
        client_id: str,
        model_weights: Dict[str, Any],
        training_samples: int
    ):
        """
        Recibir actualización de cliente
        
        Args:
            client_id: ID del cliente
            model_weights: Pesos del modelo entrenado
            training_samples: Número de muestras usadas
        """
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not found")
        
        client = self.clients[client_id]
        client.status = ClientStatus.UPDATING
        client.last_update = time.time()
        client.training_rounds += 1
        
        # En implementación real, almacenar actualización
        # y agregar a la ronda actual
        
        client.status = ClientStatus.IDLE
    
    def aggregate_updates(self, round_number: int) -> Dict[str, Any]:
        """
        Agregar actualizaciones de clientes
        
        Args:
            round_number: Número de ronda
        """
        if round_number > len(self.training_rounds):
            raise ValueError(f"Round {round_number} not found")
        
        round_obj = self.training_rounds[round_number - 1]
        
        if not round_obj.participants:
            raise ValueError("No participants in round")
        
        # Obtener actualizaciones de clientes (simplificado)
        # En implementación real, obtener pesos de cada cliente
        client_updates = []
        for client_id in round_obj.participants:
            client = self.clients[client_id]
            # Simular actualización
            update = {
                'client_id': client_id,
                'weights': self._simulate_client_weights(),
                'data_size': client.data_size
            }
            client_updates.append(update)
        
        # Agregar según método
        if self.aggregation_method == AggregationMethod.FED_AVG:
            aggregated = self._federated_average(client_updates)
        elif self.aggregation_method == AggregationMethod.WEIGHTED_AVG:
            aggregated = self._weighted_average(client_updates)
        else:
            aggregated = self._federated_average(client_updates)
        
        round_obj.aggregated_model = aggregated
        round_obj.completed_at = time.time()
        
        # Actualizar modelo global
        if self.global_model:
            self.global_model['weights'] = aggregated
            self.global_model['version'] += 1
        
        return aggregated
    
    def _federated_average(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Promedio federado simple"""
        if not updates:
            return {}
        
        # Promedio simple de pesos
        # En implementación real, promediar tensores
        return {
            'layer1': [0.5] * 128,  # Simplificado
            'layer2': [0.5] * 64,
            'output': [0.5] * 10
        }
    
    def _weighted_average(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Promedio ponderado por tamaño de datos"""
        total_size = sum(u['data_size'] for u in updates)
        
        if total_size == 0:
            return self._federated_average(updates)
        
        # Promedio ponderado
        # En implementación real, calcular según pesos
        return self._federated_average(updates)
    
    def _simulate_client_weights(self) -> Dict[str, Any]:
        """Simular pesos de cliente (placeholder)"""
        return {
            'layer1': [0.5] * 128,
            'layer2': [0.5] * 64,
            'output': [0.5] * 10
        }
    
    def get_global_model(self) -> Optional[Dict[str, Any]]:
        """Obtener modelo global"""
        return self.global_model
    
    def distribute_model(self, client_ids: Optional[List[str]] = None):
        """Distribuir modelo global a clientes"""
        if not self.global_model:
            raise ValueError("Global model not initialized")
        
        target_clients = client_ids or list(self.clients.keys())
        
        for client_id in target_clients:
            if client_id in self.clients:
                client = self.clients[client_id]
                client.model_version = self.global_model['version']
                client.status = ClientStatus.IDLE
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de federated learning"""
        return {
            'total_clients': len(self.clients),
            'active_clients': len([c for c in self.clients.values() if c.status != ClientStatus.OFFLINE]),
            'total_rounds': len(self.training_rounds),
            'completed_rounds': len([r for r in self.training_rounds if r.completed_at]),
            'global_model_version': self.global_model['version'] if self.global_model else 0,
            'aggregation_method': self.aggregation_method.value,
            'average_training_rounds_per_client': (
                sum(c.training_rounds for c in self.clients.values()) / len(self.clients)
                if self.clients else 0
            )
        }


# Instancia global
federated_learning = FederatedLearning()

