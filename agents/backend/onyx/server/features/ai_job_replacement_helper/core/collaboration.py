"""
Collaboration Service - Sistema de colaboración
================================================

Sistema para colaborar con otros usuarios en proyectos y objetivos.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CollaborationType(str, Enum):
    """Tipos de colaboración"""
    STUDY_GROUP = "study_group"
    PROJECT = "project"
    MENTORSHIP = "mentorship"
    ACCOUNTABILITY_PARTNER = "accountability_partner"


class CollaborationStatus(str, Enum):
    """Estado de colaboración"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class Collaboration:
    """Colaboración"""
    id: str
    creator_id: str
    collaboration_type: CollaborationType
    title: str
    description: str
    members: List[str] = field(default_factory=list)
    status: CollaborationStatus = CollaborationStatus.ACTIVE
    goals: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class CollaborationService:
    """Servicio de colaboración"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.collaborations: Dict[str, Collaboration] = {}
        logger.info("CollaborationService initialized")
    
    def create_collaboration(
        self,
        creator_id: str,
        collaboration_type: CollaborationType,
        title: str,
        description: str,
        goals: Optional[List[str]] = None
    ) -> Collaboration:
        """Crear colaboración"""
        collaboration = Collaboration(
            id=f"collab_{creator_id}_{int(datetime.now().timestamp())}",
            creator_id=creator_id,
            collaboration_type=collaboration_type,
            title=title,
            description=description,
            members=[creator_id],
            goals=goals or [],
        )
        
        self.collaborations[collaboration.id] = collaboration
        
        logger.info(f"Collaboration created: {collaboration.id}")
        return collaboration
    
    def join_collaboration(self, collaboration_id: str, user_id: str) -> bool:
        """Unirse a colaboración"""
        collaboration = self.collaborations.get(collaboration_id)
        if not collaboration:
            return False
        
        if user_id not in collaboration.members:
            collaboration.members.append(user_id)
            collaboration.updated_at = datetime.now()
            logger.info(f"User {user_id} joined collaboration {collaboration_id}")
            return True
        
        return False
    
    def get_user_collaborations(self, user_id: str) -> List[Collaboration]:
        """Obtener colaboraciones del usuario"""
        return [
            collab for collab in self.collaborations.values()
            if user_id in collab.members
        ]
    
    def get_collaboration(self, collaboration_id: str) -> Optional[Collaboration]:
        """Obtener colaboración específica"""
        return self.collaborations.get(collaboration_id)




