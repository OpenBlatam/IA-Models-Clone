"""
Workflow - Sistema de workflow y estados
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class WorkflowState(Enum):
    """Estados del workflow"""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    REJECTED = "rejected"


@dataclass
class WorkflowTransition:
    """Transición de estado"""
    from_state: WorkflowState
    to_state: WorkflowState
    required_permission: Optional[str] = None
    auto: bool = False


@dataclass
class WorkflowInstance:
    """Instancia de workflow"""
    id: str
    content_id: str
    current_state: WorkflowState
    history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class WorkflowManager:
    """Gestor de workflows"""

    def __init__(self):
        """Inicializar gestor de workflows"""
        self.workflows: Dict[str, WorkflowInstance] = {}
        self.transitions: Dict[str, List[WorkflowTransition]] = {}
        self.default_workflow = self._create_default_workflow()

    def _create_default_workflow(self) -> str:
        """Crear workflow por defecto"""
        workflow_id = "default"
        self.transitions[workflow_id] = [
            WorkflowTransition(WorkflowState.DRAFT, WorkflowState.REVIEW),
            WorkflowTransition(WorkflowState.REVIEW, WorkflowState.APPROVED),
            WorkflowTransition(WorkflowState.REVIEW, WorkflowState.REJECTED),
            WorkflowTransition(WorkflowState.APPROVED, WorkflowState.PUBLISHED),
            WorkflowTransition(WorkflowState.PUBLISHED, WorkflowState.ARCHIVED),
        ]
        return workflow_id

    def create_workflow(
        self,
        content_id: str,
        workflow_id: Optional[str] = None
    ) -> WorkflowInstance:
        """
        Crear instancia de workflow.

        Args:
            content_id: ID del contenido
            workflow_id: ID del workflow (opcional)

        Returns:
            Instancia de workflow
        """
        instance_id = str(uuid.uuid4())
        workflow_id = workflow_id or self.default_workflow
        
        instance = WorkflowInstance(
            id=instance_id,
            content_id=content_id,
            current_state=WorkflowState.DRAFT
        )
        
        instance.history.append({
            "state": WorkflowState.DRAFT.value,
            "timestamp": datetime.utcnow().isoformat(),
            "action": "created"
        })
        
        self.workflows[content_id] = instance
        logger.info(f"Workflow creado para contenido: {content_id}")
        return instance

    def transition(
        self,
        content_id: str,
        to_state: WorkflowState,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Transicionar a un nuevo estado.

        Args:
            content_id: ID del contenido
            to_state: Estado destino
            user_id: ID del usuario

        Returns:
            True si la transición fue exitosa
        """
        instance = self.workflows.get(content_id)
        if not instance:
            return False
        
        # Verificar si la transición es válida
        workflow_id = self.default_workflow
        valid_transitions = self.transitions.get(workflow_id, [])
        
        valid = any(
            t.from_state == instance.current_state and t.to_state == to_state
            for t in valid_transitions
        )
        
        if not valid:
            logger.warning(f"Transición inválida: {instance.current_state} -> {to_state}")
            return False
        
        # Realizar transición
        instance.current_state = to_state
        instance.updated_at = datetime.utcnow()
        instance.history.append({
            "from_state": instance.current_state.value,
            "to_state": to_state.value,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": "transition"
        })
        
        logger.info(f"Transición realizada: {content_id} -> {to_state.value}")
        return True

    def get_workflow(self, content_id: str) -> Optional[WorkflowInstance]:
        """
        Obtener workflow de un contenido.

        Args:
            content_id: ID del contenido

        Returns:
            Instancia de workflow o None
        """
        return self.workflows.get(content_id)

    def get_available_transitions(self, content_id: str) -> List[WorkflowState]:
        """
        Obtener transiciones disponibles.

        Args:
            content_id: ID del contenido

        Returns:
            Lista de estados disponibles
        """
        instance = self.workflows.get(content_id)
        if not instance:
            return []
        
        workflow_id = self.default_workflow
        transitions = self.transitions.get(workflow_id, [])
        
        available = [
            t.to_state
            for t in transitions
            if t.from_state == instance.current_state
        ]
        
        return available






