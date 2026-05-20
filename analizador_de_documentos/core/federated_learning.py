"""
Sistema de Aprendizaje Federado
=================================

Sistema para aprendizaje federado distribuido.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FederatedRoundStatus(Enum):
    """Estado de ronda federada"""
    INITIALIZING = "initializing"
    COLLECTING = "collecting"
    AGGREGATING = "aggregating"
    UPDATING = "updating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FederatedClient:
    """Cliente en aprendizaje federado"""
    client_id: str
    url: str
    active: bool = True
    last_update: Optional[str] = None
    rounds_participated: int = 0


@dataclass
class FederatedRound:
    """Ronda de aprendizaje federado"""
    round_id: str
    status: FederatedRoundStatus
    clients: List[str]
    model_updates: Dict[str, Any]
    aggregated_model: Optional[Dict[str, Any]] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class FederatedLearningSystem:
    """
    Sistema de aprendizaje federado
    
    Proporciona:
    - Coordinación de clientes federados
    - Agregación de modelos
    - Rondas de entrenamiento
    - Seguridad y privacidad
    - Monitoreo de progreso
    """
    
    def __init__(self):
        """Inicializar sistema"""
        self.clients: Dict[str, FederatedClient] = {}
        self.rounds: Dict[str, FederatedRound] = {}
        self.current_round: Optional[str] = None
        logger.info("FederatedLearningSystem inicializado")
    
    def register_client(
        self,
        client_id: str,
        url: str
    ) -> FederatedClient:
        """Registrar cliente federado"""
        client = FederatedClient(
            client_id=client_id,
            url=url,
            last_update=datetime.now().isoformat()
        )
        
        self.clients[client_id] = client
        logger.info(f"Cliente federado registrado: {client_id}")
        
        return client
    
    def start_round(
        self,
        round_id: Optional[str] = None,
        client_ids: Optional[List[str]] = None
    ) -> FederatedRound:
        """
        Iniciar ronda de aprendizaje federado
        
        Args:
            round_id: ID de ronda (auto-generado si None)
            client_ids: IDs de clientes participantes (todos si None)
        
        Returns:
            Ronda creada
        """
        if round_id is None:
            round_id = f"round_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if client_ids is None:
            client_ids = [cid for cid, c in self.clients.items() if c.active]
        
        round_obj = FederatedRound(
            round_id=round_id,
            status=FederatedRoundStatus.INITIALIZING,
            clients=client_ids,
            model_updates={}
        )
        
        self.rounds[round_id] = round_obj
        self.current_round = round_id
        
        logger.info(f"Ronda federada iniciada: {round_id} con {len(client_ids)} clientes")
        
        return round_obj
    
    def submit_model_update(
        self,
        round_id: str,
        client_id: str,
        model_update: Dict[str, Any]
    ) -> bool:
        """Enviar actualización de modelo desde cliente"""
        if round_id not in self.rounds:
            logger.error(f"Ronda no encontrada: {round_id}")
            return False
        
        round_obj = self.rounds[round_id]
        
        if client_id not in round_obj.clients:
            logger.error(f"Cliente no participa en ronda: {client_id}")
            return False
        
        round_obj.model_updates[client_id] = model_update
        
        # Actualizar estado
        if len(round_obj.model_updates) == len(round_obj.clients):
            round_obj.status = FederatedRoundStatus.COLLECTING
        
        logger.info(f"Actualización recibida de {client_id} para ronda {round_id}")
        
        return True
    
    def aggregate_models(
        self,
        round_id: str
    ) -> Dict[str, Any]:
        """
        Agregar modelos de clientes
        
        Args:
            round_id: ID de ronda
        
        Returns:
            Modelo agregado
        """
        if round_id not in self.rounds:
            raise ValueError(f"Ronda no encontrada: {round_id}")
        
        round_obj = self.rounds[round_id]
        
        if not round_obj.model_updates:
            raise ValueError("No hay actualizaciones para agregar")
        
        # Agregación promedio simple (FedAvg)
        # En producción, se usaría agregación más sofisticada
        aggregated = {}
        
        # Simulación de agregación
        # En producción, aquí se haría la agregación real de parámetros del modelo
        aggregated["aggregation_method"] = "fedavg"
        aggregated["num_clients"] = len(round_obj.model_updates)
        aggregated["timestamp"] = datetime.now().isoformat()
        
        round_obj.aggregated_model = aggregated
        round_obj.status = FederatedRoundStatus.COMPLETED
        
        logger.info(f"Modelos agregados para ronda {round_id}")
        
        return aggregated
    
    def get_round_status(self, round_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de ronda"""
        if round_id not in self.rounds:
            return None
        
        round_obj = self.rounds[round_id]
        
        return {
            "round_id": round_id,
            "status": round_obj.status.value,
            "total_clients": len(round_obj.clients),
            "updates_received": len(round_obj.model_updates),
            "created_at": round_obj.created_at
        }


# Instancia global
_federated_learning: Optional[FederatedLearningSystem] = None


def get_federated_learning() -> FederatedLearningSystem:
    """Obtener instancia global del sistema"""
    global _federated_learning
    if _federated_learning is None:
        _federated_learning = FederatedLearningSystem()
    return _federated_learning














