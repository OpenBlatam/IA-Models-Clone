"""
Collaboration System
====================

Sistema de colaboración y trabajo en equipo.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Estado de tarea."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class UserRole(Enum):
    """Rol de usuario."""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    DEVELOPER = "developer"


@dataclass
class User:
    """Usuario."""
    user_id: str
    username: str
    email: str
    role: UserRole
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Task:
    """Tarea."""
    task_id: str
    title: str
    description: str
    assigned_to: Optional[str] = None
    created_by: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1-10
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Comment:
    """Comentario."""
    comment_id: str
    task_id: str
    user_id: str
    content: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class CollaborationSystem:
    """
    Sistema de colaboración.
    
    Gestiona usuarios, tareas y colaboración.
    """
    
    def __init__(self):
        """Inicializar sistema de colaboración."""
        self.users: Dict[str, User] = {}
        self.tasks: Dict[str, Task] = {}
        self.comments: Dict[str, List[Comment]] = {}  # task_id -> comments
        self.activity_log: List[Dict[str, Any]] = []
    
    def create_user(
        self,
        user_id: str,
        username: str,
        email: str,
        role: UserRole = UserRole.VIEWER,
        metadata: Optional[Dict[str, Any]] = None
    ) -> User:
        """
        Crear usuario.
        
        Args:
            user_id: ID único del usuario
            username: Nombre de usuario
            email: Email
            role: Rol
            metadata: Metadata adicional
            
        Returns:
            Usuario creado
        """
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            metadata=metadata or {}
        )
        
        self.users[user_id] = user
        self._log_activity("user_created", {"user_id": user_id})
        logger.info(f"Created user: {username} ({user_id})")
        
        return user
    
    def create_task(
        self,
        task_id: str,
        title: str,
        description: str,
        created_by: str,
        assigned_to: Optional[str] = None,
        priority: int = 5,
        tags: Optional[List[str]] = None
    ) -> Task:
        """
        Crear tarea.
        
        Args:
            task_id: ID único de la tarea
            title: Título
            description: Descripción
            created_by: ID del creador
            assigned_to: ID del asignado (opcional)
            priority: Prioridad (1-10)
            tags: Tags
            
        Returns:
            Tarea creada
        """
        task = Task(
            task_id=task_id,
            title=title,
            description=description,
            created_by=created_by,
            assigned_to=assigned_to,
            priority=priority,
            tags=tags or []
        )
        
        self.tasks[task_id] = task
        self.comments[task_id] = []
        self._log_activity("task_created", {"task_id": task_id, "created_by": created_by})
        logger.info(f"Created task: {title} ({task_id})")
        
        return task
    
    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        user_id: str
    ) -> Optional[Task]:
        """
        Actualizar estado de tarea.
        
        Args:
            task_id: ID de la tarea
            status: Nuevo estado
            user_id: ID del usuario que actualiza
            
        Returns:
            Tarea actualizada o None
        """
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        old_status = task.status
        task.status = status
        task.updated_at = datetime.now().isoformat()
        
        self._log_activity(
            "task_status_updated",
            {
                "task_id": task_id,
                "old_status": old_status.value,
                "new_status": status.value,
                "user_id": user_id
            }
        )
        
        return task
    
    def assign_task(
        self,
        task_id: str,
        user_id: str,
        assigned_by: str
    ) -> Optional[Task]:
        """
        Asignar tarea a usuario.
        
        Args:
            task_id: ID de la tarea
            user_id: ID del usuario
            assigned_by: ID del usuario que asigna
            
        Returns:
            Tarea actualizada o None
        """
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        task.assigned_to = user_id
        task.updated_at = datetime.now().isoformat()
        
        self._log_activity(
            "task_assigned",
            {
                "task_id": task_id,
                "assigned_to": user_id,
                "assigned_by": assigned_by
            }
        )
        
        return task
    
    def add_comment(
        self,
        task_id: str,
        user_id: str,
        content: str
    ) -> Optional[Comment]:
        """
        Agregar comentario a tarea.
        
        Args:
            task_id: ID de la tarea
            user_id: ID del usuario
            content: Contenido del comentario
            
        Returns:
            Comentario creado o None
        """
        if task_id not in self.tasks:
            return None
        
        comment_id = f"comment_{len(self.comments.get(task_id, []))}"
        comment = Comment(
            comment_id=comment_id,
            task_id=task_id,
            user_id=user_id,
            content=content
        )
        
        if task_id not in self.comments:
            self.comments[task_id] = []
        self.comments[task_id].append(comment)
        
        self._log_activity(
            "comment_added",
            {
                "task_id": task_id,
                "user_id": user_id,
                "comment_id": comment_id
            }
        )
        
        return comment
    
    def get_user_tasks(
        self,
        user_id: str,
        status: Optional[TaskStatus] = None
    ) -> List[Task]:
        """
        Obtener tareas de usuario.
        
        Args:
            user_id: ID del usuario
            status: Filtrar por estado (opcional)
            
        Returns:
            Lista de tareas
        """
        tasks = [
            t for t in self.tasks.values()
            if t.assigned_to == user_id or t.created_by == user_id
        ]
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        return tasks
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Obtener tarea por ID."""
        return self.tasks.get(task_id)
    
    def get_task_comments(self, task_id: str) -> List[Comment]:
        """Obtener comentarios de tarea."""
        return self.comments.get(task_id, [])
    
    def _log_activity(self, activity_type: str, data: Dict[str, Any]) -> None:
        """Registrar actividad."""
        self.activity_log.append({
            "type": activity_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })


# Instancia global
_collaboration_system: Optional[CollaborationSystem] = None


def get_collaboration_system() -> CollaborationSystem:
    """Obtener instancia global del sistema de colaboración."""
    global _collaboration_system
    if _collaboration_system is None:
        _collaboration_system = CollaborationSystem()
    return _collaboration_system






