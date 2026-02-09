"""
Model Lifecycle Manager - Gestor de ciclo de vida de modelos
=============================================================
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class LifecycleStage(Enum):
    """Etapas del ciclo de vida"""
    DEVELOPMENT = "development"
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class LifecycleEvent:
    """Evento del ciclo de vida"""
    stage: LifecycleStage
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class ModelLifecycle:
    """Ciclo de vida de modelo"""
    model_id: str
    model_name: str
    current_stage: LifecycleStage
    events: List[LifecycleEvent] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class ModelLifecycleManager:
    """Gestor de ciclo de vida de modelos"""
    
    def __init__(self):
        self.lifecycles: Dict[str, ModelLifecycle] = {}
    
    def create_lifecycle(
        self,
        model_id: str,
        model_name: str,
        initial_stage: LifecycleStage = LifecycleStage.DEVELOPMENT
    ) -> ModelLifecycle:
        """Crea ciclo de vida para modelo"""
        lifecycle = ModelLifecycle(
            model_id=model_id,
            model_name=model_name,
            current_stage=initial_stage
        )
        
        # Agregar evento inicial
        lifecycle.events.append(LifecycleEvent(
            stage=initial_stage,
            notes=f"Modelo creado en etapa {initial_stage.value}"
        ))
        
        self.lifecycles[model_id] = lifecycle
        logger.info(f"Ciclo de vida creado para modelo {model_id}")
        return lifecycle
    
    def transition_stage(
        self,
        model_id: str,
        new_stage: LifecycleStage,
        notes: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Transiciona a nueva etapa"""
        if model_id not in self.lifecycles:
            logger.error(f"Modelo {model_id} no encontrado")
            return False
        
        lifecycle = self.lifecycles[model_id]
        
        # Validar transición
        if not self._is_valid_transition(lifecycle.current_stage, new_stage):
            logger.warning(f"Transición inválida: {lifecycle.current_stage.value} -> {new_stage.value}")
            return False
        
        # Crear evento
        event = LifecycleEvent(
            stage=new_stage,
            notes=notes,
            metadata=metadata or {}
        )
        
        lifecycle.events.append(event)
        lifecycle.current_stage = new_stage
        lifecycle.updated_at = datetime.now()
        
        logger.info(f"Modelo {model_id} transicionado a {new_stage.value}")
        return True
    
    def _is_valid_transition(
        self,
        current: LifecycleStage,
        new: LifecycleStage
    ) -> bool:
        """Verifica si transición es válida"""
        valid_transitions = {
            LifecycleStage.DEVELOPMENT: [LifecycleStage.TRAINING, LifecycleStage.ARCHIVED],
            LifecycleStage.TRAINING: [LifecycleStage.VALIDATION, LifecycleStage.DEVELOPMENT],
            LifecycleStage.VALIDATION: [LifecycleStage.TESTING, LifecycleStage.TRAINING],
            LifecycleStage.TESTING: [LifecycleStage.STAGING, LifecycleStage.VALIDATION],
            LifecycleStage.STAGING: [LifecycleStage.PRODUCTION, LifecycleStage.TESTING],
            LifecycleStage.PRODUCTION: [LifecycleStage.DEPRECATED, LifecycleStage.STAGING],
            LifecycleStage.DEPRECATED: [LifecycleStage.ARCHIVED, LifecycleStage.PRODUCTION],
            LifecycleStage.ARCHIVED: []  # Final
        }
        
        return new in valid_transitions.get(current, [])
    
    def get_lifecycle(self, model_id: str) -> Optional[ModelLifecycle]:
        """Obtiene ciclo de vida de modelo"""
        return self.lifecycles.get(model_id)
    
    def get_models_by_stage(self, stage: LifecycleStage) -> List[ModelLifecycle]:
        """Obtiene modelos por etapa"""
        return [
            lifecycle
            for lifecycle in self.lifecycles.values()
            if lifecycle.current_stage == stage
        ]
    
    def get_lifecycle_history(self, model_id: str) -> List[LifecycleEvent]:
        """Obtiene historial de ciclo de vida"""
        if model_id not in self.lifecycles:
            return []
        
        return self.lifecycles[model_id].events




