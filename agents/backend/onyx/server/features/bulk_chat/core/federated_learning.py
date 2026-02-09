"""
Federated Learning - Aprendizaje Federado
==========================================

Sistema de aprendizaje federado para entrenar modelos distribuidos sin compartir datos.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class LearningRoundStatus(Enum):
    """Estado de ronda de aprendizaje."""
    INITIALIZING = "initializing"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ModelUpdate:
    """Actualización del modelo."""
    update_id: str
    client_id: str
    round_id: str
    weights: Dict[str, Any]
    sample_count: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningRound:
    """Ronda de aprendizaje."""
    round_id: str
    status: LearningRoundStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    model_version: str = "1.0"
    participants: List[str] = field(default_factory=list)
    updates_received: int = 0
    updates_expected: int = 0
    aggregated_weights: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FederatedLearning:
    """Sistema de aprendizaje federado."""
    
    def __init__(
        self,
        min_clients: int = 3,
        aggregation_method: str = "federated_averaging",
    ):
        self.min_clients = min_clients
        self.aggregation_method = aggregation_method
        self.clients: Dict[str, Dict[str, Any]] = {}
        self.learning_rounds: Dict[str, LearningRound] = {}
        self.model_updates: Dict[str, List[ModelUpdate]] = defaultdict(list)
        self.global_model: Dict[str, Any] = {}
        self.model_version: str = "1.0"
        self._lock = asyncio.Lock()
    
    def register_client(
        self,
        client_id: str,
        capabilities: Optional[Dict[str, Any]] = None,
    ):
        """Registrar cliente para aprendizaje federado."""
        self.clients[client_id] = {
            "capabilities": capabilities or {},
            "registered_at": datetime.now(),
            "status": "active",
            "participation_count": 0,
        }
        
        logger.info(f"Registered federated learning client: {client_id}")
    
    async def start_learning_round(
        self,
        round_id: Optional[str] = None,
        client_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Iniciar ronda de aprendizaje.
        
        Args:
            round_id: ID de ronda (se genera si no se proporciona)
            client_ids: IDs de clientes participantes (todos si None)
        
        Returns:
            ID de la ronda
        """
        if round_id is None:
            round_id = f"round_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if client_ids is None:
            client_ids = [c for c in self.clients.keys() if self.clients[c]["status"] == "active"]
        
        if len(client_ids) < self.min_clients:
            raise ValueError(f"Need at least {self.min_clients} clients, got {len(client_ids)}")
        
        round_obj = LearningRound(
            round_id=round_id,
            status=LearningRoundStatus.INITIALIZING,
            start_time=datetime.now(),
            participants=client_ids,
            updates_expected=len(client_ids),
            model_version=self.model_version,
        )
        
        async with self._lock:
            self.learning_rounds[round_id] = round_obj
        
        # Cambiar estado a training
        round_obj.status = LearningRoundStatus.TRAINING
        
        logger.info(f"Started learning round: {round_id} with {len(client_ids)} clients")
        return round_id
    
    async def submit_model_update(
        self,
        round_id: str,
        client_id: str,
        weights: Dict[str, Any],
        sample_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Enviar actualización de modelo.
        
        Args:
            round_id: ID de ronda
            client_id: ID del cliente
            weights: Pesos del modelo
            sample_count: Número de muestras usadas
            metadata: Metadatos adicionales
        
        Returns:
            ID de actualización
        """
        round_obj = self.learning_rounds.get(round_id)
        if not round_obj:
            raise ValueError(f"Learning round not found: {round_id}")
        
        if client_id not in round_obj.participants:
            raise ValueError(f"Client {client_id} not in round {round_id}")
        
        update_id = f"update_{client_id}_{datetime.now().timestamp()}"
        
        update = ModelUpdate(
            update_id=update_id,
            client_id=client_id,
            round_id=round_id,
            weights=weights,
            sample_count=sample_count,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.model_updates[round_id].append(update)
            round_obj.updates_received += 1
        
        logger.info(
            f"Received model update from {client_id} for round {round_id} "
            f"({round_obj.updates_received}/{round_obj.updates_expected})"
        )
        
        # Verificar si podemos agregar
        if round_obj.updates_received >= round_obj.updates_expected:
            await self._aggregate_updates(round_id)
        
        return update_id
    
    async def _aggregate_updates(self, round_id: str):
        """Agregar actualizaciones de modelo."""
        round_obj = self.learning_rounds.get(round_id)
        if not round_obj:
            return
        
        updates = self.model_updates.get(round_id, [])
        if len(updates) < self.min_clients:
            return
        
        round_obj.status = LearningRoundStatus.AGGREGATING
        
        # Federated Averaging
        if self.aggregation_method == "federated_averaging":
            total_samples = sum(update.sample_count for update in updates)
            
            aggregated_weights = {}
            
            # Agregar pesos ponderados por número de muestras
            for update in updates:
                weight_factor = update.sample_count / total_samples
                
                for key, value in update.weights.items():
                    if key not in aggregated_weights:
                        aggregated_weights[key] = 0.0
                    
                    # Asumir que los valores son numéricos
                    if isinstance(value, (int, float)):
                        aggregated_weights[key] += value * weight_factor
                    elif isinstance(value, dict):
                        # Recursivo para estructuras anidadas
                        if key not in aggregated_weights:
                            aggregated_weights[key] = {}
                        
                        for sub_key, sub_value in value.items():
                            if sub_key not in aggregated_weights[key]:
                                aggregated_weights[key][sub_key] = 0.0
                            if isinstance(sub_value, (int, float)):
                                aggregated_weights[key][sub_key] += sub_value * weight_factor
            
            round_obj.aggregated_weights = aggregated_weights
            
            # Actualizar modelo global
            async with self._lock:
                self.global_model = aggregated_weights
                self.model_version = f"{float(self.model_version) + 0.1:.1f}"
        
        round_obj.status = LearningRoundStatus.COMPLETED
        round_obj.end_time = datetime.now()
        
        # Actualizar estadísticas de clientes
        for update in updates:
            if update.client_id in self.clients:
                self.clients[update.client_id]["participation_count"] += 1
        
        logger.info(f"Aggregated updates for round {round_id}")
    
    def get_global_model(self) -> Dict[str, Any]:
        """Obtener modelo global."""
        return {
            "model_version": self.model_version,
            "weights": self.global_model,
            "last_updated": datetime.now().isoformat(),
        }
    
    def get_round_status(self, round_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de una ronda."""
        round_obj = self.learning_rounds.get(round_id)
        if not round_obj:
            return None
        
        return {
            "round_id": round_id,
            "status": round_obj.status.value,
            "start_time": round_obj.start_time.isoformat(),
            "end_time": round_obj.end_time.isoformat() if round_obj.end_time else None,
            "model_version": round_obj.model_version,
            "participants_count": len(round_obj.participants),
            "updates_received": round_obj.updates_received,
            "updates_expected": round_obj.updates_expected,
            "progress": round_obj.updates_received / round_obj.updates_expected if round_obj.updates_expected > 0 else 0.0,
        }
    
    def get_federated_learning_summary(self) -> Dict[str, Any]:
        """Obtener resumen de aprendizaje federado."""
        completed_rounds = sum(
            1 for r in self.learning_rounds.values()
            if r.status == LearningRoundStatus.COMPLETED
        )
        
        active_rounds = sum(
            1 for r in self.learning_rounds.values()
            if r.status in [
                LearningRoundStatus.INITIALIZING,
                LearningRoundStatus.TRAINING,
                LearningRoundStatus.AGGREGATING,
            ]
        )
        
        return {
            "total_clients": len(self.clients),
            "active_clients": sum(
                1 for c in self.clients.values() if c["status"] == "active"
            ),
            "total_rounds": len(self.learning_rounds),
            "completed_rounds": completed_rounds,
            "active_rounds": active_rounds,
            "model_version": self.model_version,
            "aggregation_method": self.aggregation_method,
        }
















