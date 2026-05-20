"""
Learning Path Service - Sistema de rutas de aprendizaje
=========================================================

Sistema para crear y gestionar rutas de aprendizaje personalizadas.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class LearningPathStatus(str, Enum):
    """Estado de la ruta de aprendizaje"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"


@dataclass
class LearningModule:
    """Módulo de aprendizaje"""
    id: str
    title: str
    description: str
    content_type: str  # video, article, course, exercise
    duration_minutes: int
    resources: List[Dict[str, str]] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    completed: bool = False


@dataclass
class LearningPath:
    """Ruta de aprendizaje"""
    id: str
    user_id: str
    title: str
    description: str
    skill_focus: str
    modules: List[LearningModule] = field(default_factory=list)
    status: LearningPathStatus = LearningPathStatus.NOT_STARTED
    progress_percentage: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration_days: int = 30
    created_at: datetime = field(default_factory=datetime.now)


class LearningPathService:
    """Servicio de rutas de aprendizaje"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.learning_paths: Dict[str, List[LearningPath]] = {}  # user_id -> [paths]
        logger.info("LearningPathService initialized")
    
    def create_learning_path(
        self,
        user_id: str,
        title: str,
        description: str,
        skill_focus: str,
        modules: Optional[List[Dict[str, Any]]] = None,
        estimated_duration_days: int = 30
    ) -> LearningPath:
        """Crear ruta de aprendizaje"""
        learning_modules = []
        
        if modules:
            for module_data in modules:
                module = LearningModule(
                    id=module_data.get("id", f"module_{len(learning_modules) + 1}"),
                    title=module_data.get("title", ""),
                    description=module_data.get("description", ""),
                    content_type=module_data.get("content_type", "article"),
                    duration_minutes=module_data.get("duration_minutes", 60),
                    resources=module_data.get("resources", []),
                    prerequisites=module_data.get("prerequisites", []),
                )
                learning_modules.append(module)
        
        path = LearningPath(
            id=f"path_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            title=title,
            description=description,
            skill_focus=skill_focus,
            modules=learning_modules,
            estimated_duration_days=estimated_duration_days,
        )
        
        if user_id not in self.learning_paths:
            self.learning_paths[user_id] = []
        
        self.learning_paths[user_id].append(path)
        
        logger.info(f"Learning path created for user {user_id}: {title}")
        return path
    
    def start_learning_path(self, user_id: str, path_id: str) -> LearningPath:
        """Iniciar ruta de aprendizaje"""
        path = self._get_path(user_id, path_id)
        if not path:
            raise ValueError(f"Learning path {path_id} not found")
        
        path.status = LearningPathStatus.IN_PROGRESS
        path.started_at = datetime.now()
        
        return path
    
    def complete_module(self, user_id: str, path_id: str, module_id: str) -> LearningPath:
        """Completar módulo"""
        path = self._get_path(user_id, path_id)
        if not path:
            raise ValueError(f"Learning path {path_id} not found")
        
        module = next((m for m in path.modules if m.id == module_id), None)
        if module:
            module.completed = True
        
        # Actualizar progreso
        completed_modules = sum(1 for m in path.modules if m.completed)
        path.progress_percentage = (
            completed_modules / len(path.modules) * 100
            if path.modules else 0
        )
        
        # Verificar si está completo
        if path.progress_percentage >= 100:
            path.status = LearningPathStatus.COMPLETED
            path.completed_at = datetime.now()
        
        return path
    
    def get_user_paths(self, user_id: str) -> List[LearningPath]:
        """Obtener rutas de aprendizaje del usuario"""
        return self.learning_paths.get(user_id, [])
    
    def _get_path(self, user_id: str, path_id: str) -> Optional[LearningPath]:
        """Obtener ruta por ID"""
        paths = self.learning_paths.get(user_id, [])
        return next((p for p in paths if p.id == path_id), None)




