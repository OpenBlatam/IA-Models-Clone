"""
Collaboration System - Sistema de Colaboración
==============================================

Gestiona colaboración en proyectos.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class CollaborationSystem:
    """Sistema de colaboración"""

    def __init__(self, data_dir: Path = None):
        """
        Inicializa el sistema de colaboración.

        Args:
            data_dir: Directorio para almacenar datos
        """
        if data_dir is None:
            data_dir = Path("collaboration")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.project_collaborators: Dict[str, List[str]] = defaultdict(list)
        self.project_permissions: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.comments: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_collaborator(
        self,
        project_id: str,
        user_id: str,
        role: str = "viewer",
    ) -> bool:
        """
        Agrega un colaborador a un proyecto.

        Args:
            project_id: ID del proyecto
            user_id: ID del usuario
            role: Rol (owner, editor, viewer)

        Returns:
            True si se agregó exitosamente
        """
        if user_id not in self.project_collaborators[project_id]:
            self.project_collaborators[project_id].append(user_id)
            self.project_permissions[project_id][role].append(user_id)
            logger.info(f"Colaborador {user_id} agregado a proyecto {project_id} con rol {role}")
            return True
        return False

    def remove_collaborator(
        self,
        project_id: str,
        user_id: str,
    ) -> bool:
        """Elimina un colaborador de un proyecto"""
        if user_id in self.project_collaborators[project_id]:
            self.project_collaborators[project_id].remove(user_id)
            # Remover de todos los roles
            for role_users in self.project_permissions[project_id].values():
                if user_id in role_users:
                    role_users.remove(user_id)
            logger.info(f"Colaborador {user_id} eliminado de proyecto {project_id}")
            return True
        return False

    def get_collaborators(
        self,
        project_id: str,
    ) -> List[Dict[str, Any]]:
        """Obtiene lista de colaboradores de un proyecto"""
        collaborators = []
        for role, users in self.project_permissions[project_id].items():
            for user_id in users:
                collaborators.append({
                    "user_id": user_id,
                    "role": role,
                })
        return collaborators

    def add_comment(
        self,
        project_id: str,
        user_id: str,
        comment: str,
        parent_comment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Agrega un comentario a un proyecto.

        Args:
            project_id: ID del proyecto
            user_id: ID del usuario
            comment: Texto del comentario
            parent_comment_id: ID del comentario padre (para respuestas)

        Returns:
            Información del comentario creado
        """
        comment_data = {
            "id": f"comment_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "project_id": project_id,
            "user_id": user_id,
            "comment": comment,
            "parent_comment_id": parent_comment_id,
            "created_at": datetime.now().isoformat(),
            "replies": [],
        }

        if parent_comment_id:
            # Agregar como respuesta
            for c in self.comments[project_id]:
                if c["id"] == parent_comment_id:
                    c["replies"].append(comment_data)
                    break
        else:
            # Agregar como comentario principal
            self.comments[project_id].append(comment_data)

        logger.info(f"Comentario agregado a proyecto {project_id}")
        return comment_data

    def get_comments(
        self,
        project_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Obtiene comentarios de un proyecto"""
        comments = self.comments.get(project_id, [])
        return comments[:limit]

    def has_permission(
        self,
        project_id: str,
        user_id: str,
        permission: str,
    ) -> bool:
        """
        Verifica si un usuario tiene un permiso.

        Args:
            project_id: ID del proyecto
            user_id: ID del usuario
            permission: Permiso a verificar (read, write, delete)

        Returns:
            True si tiene permiso
        """
        # Owner tiene todos los permisos
        if user_id in self.project_permissions[project_id].get("owner", []):
            return True

        # Editor tiene read y write
        if user_id in self.project_permissions[project_id].get("editor", []):
            return permission in ["read", "write"]

        # Viewer solo tiene read
        if user_id in self.project_permissions[project_id].get("viewer", []):
            return permission == "read"

        return False


