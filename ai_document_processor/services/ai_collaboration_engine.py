"""
Motor de Colaboración AI
========================

Motor para colaboración en tiempo real, gestión de equipos y trabajo colaborativo con IA.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from pathlib import Path
import hashlib
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

class CollaborationType(str, Enum):
    """Tipos de colaboración"""
    DOCUMENT_REVIEW = "document_review"
    TEAM_WORKSPACE = "team_workspace"
    REAL_TIME_EDITING = "real_time_editing"
    COMMENT_SYSTEM = "comment_system"
    VERSION_CONTROL = "version_control"
    TASK_ASSIGNMENT = "task_assignment"
    MEETING_COORDINATION = "meeting_coordination"
    KNOWLEDGE_SHARING = "knowledge_sharing"

class UserRole(str, Enum):
    """Roles de usuario"""
    ADMIN = "admin"
    EDITOR = "editor"
    REVIEWER = "reviewer"
    VIEWER = "viewer"
    GUEST = "guest"

class ActivityType(str, Enum):
    """Tipos de actividad"""
    DOCUMENT_CREATED = "document_created"
    DOCUMENT_EDITED = "document_edited"
    DOCUMENT_SHARED = "document_shared"
    COMMENT_ADDED = "comment_added"
    COMMENT_RESOLVED = "comment_resolved"
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"

@dataclass
class User:
    """Usuario del sistema"""
    id: str
    username: str
    email: str
    display_name: str
    role: UserRole
    avatar_url: Optional[str] = None
    is_online: bool = False
    last_seen: Optional[datetime] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Workspace:
    """Espacio de trabajo colaborativo"""
    id: str
    name: str
    description: str
    owner_id: str
    members: Dict[str, UserRole] = field(default_factory=dict)
    documents: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Document:
    """Documento colaborativo"""
    id: str
    title: str
    content: str
    workspace_id: str
    owner_id: str
    collaborators: Dict[str, UserRole] = field(default_factory=dict)
    version: int = 1
    is_locked: bool = False
    locked_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Comment:
    """Comentario en documento"""
    id: str
    document_id: str
    author_id: str
    content: str
    position: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    is_resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Task:
    """Tarea colaborativa"""
    id: str
    title: str
    description: str
    workspace_id: str
    document_id: Optional[str] = None
    assigned_to: Optional[str] = None
    assigned_by: str
    status: str = "pending"
    priority: str = "medium"
    due_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Activity:
    """Actividad colaborativa"""
    id: str
    type: ActivityType
    user_id: str
    workspace_id: str
    document_id: Optional[str] = None
    task_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class AICollaborationEngine:
    """Motor de colaboración AI"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.workspaces: Dict[str, Workspace] = {}
        self.documents: Dict[str, Document] = {}
        self.comments: Dict[str, Comment] = {}
        self.tasks: Dict[str, Task] = {}
        self.activities: List[Activity] = []
        
        # Colaboración en tiempo real
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.document_locks: Dict[str, Dict[str, Any]] = {}
        self.real_time_subscribers: Dict[str, List[str]] = defaultdict(list)
        
        # Configuración
        self.max_workspace_members = 100
        self.max_document_versions = 50
        self.activity_retention_days = 30
        
    async def initialize(self):
        """Inicializa el motor de colaboración"""
        logger.info("Inicializando motor de colaboración AI...")
        
        # Cargar datos existentes
        await self._load_collaboration_data()
        
        # Iniciar workers de colaboración
        await self._start_collaboration_workers()
        
        logger.info("Motor de colaboración AI inicializado")
    
    async def _load_collaboration_data(self):
        """Carga datos de colaboración"""
        try:
            # Cargar usuarios
            users_file = Path("data/collaboration_users.json")
            if users_file.exists():
                with open(users_file, 'r', encoding='utf-8') as f:
                    users_data = json.load(f)
                
                for user_data in users_data:
                    user = User(
                        id=user_data["id"],
                        username=user_data["username"],
                        email=user_data["email"],
                        display_name=user_data["display_name"],
                        role=UserRole(user_data["role"]),
                        avatar_url=user_data.get("avatar_url"),
                        is_online=user_data.get("is_online", False),
                        last_seen=datetime.fromisoformat(user_data["last_seen"]) if user_data.get("last_seen") else None,
                        preferences=user_data.get("preferences", {}),
                        created_at=datetime.fromisoformat(user_data["created_at"])
                    )
                    self.users[user.id] = user
                
                logger.info(f"Cargados {len(self.users)} usuarios")
            
            # Cargar espacios de trabajo
            workspaces_file = Path("data/collaboration_workspaces.json")
            if workspaces_file.exists():
                with open(workspaces_file, 'r', encoding='utf-8') as f:
                    workspaces_data = json.load(f)
                
                for workspace_data in workspaces_data:
                    workspace = Workspace(
                        id=workspace_data["id"],
                        name=workspace_data["name"],
                        description=workspace_data["description"],
                        owner_id=workspace_data["owner_id"],
                        members={k: UserRole(v) for k, v in workspace_data["members"].items()},
                        documents=workspace_data["documents"],
                        settings=workspace_data.get("settings", {}),
                        created_at=datetime.fromisoformat(workspace_data["created_at"]),
                        updated_at=datetime.fromisoformat(workspace_data["updated_at"])
                    )
                    self.workspaces[workspace.id] = workspace
                
                logger.info(f"Cargados {len(self.workspaces)} espacios de trabajo")
            
            # Cargar documentos
            documents_file = Path("data/collaboration_documents.json")
            if documents_file.exists():
                with open(documents_file, 'r', encoding='utf-8') as f:
                    documents_data = json.load(f)
                
                for document_data in documents_data:
                    document = Document(
                        id=document_data["id"],
                        title=document_data["title"],
                        content=document_data["content"],
                        workspace_id=document_data["workspace_id"],
                        owner_id=document_data["owner_id"],
                        collaborators={k: UserRole(v) for k, v in document_data["collaborators"].items()},
                        version=document_data["version"],
                        is_locked=document_data.get("is_locked", False),
                        locked_by=document_data.get("locked_by"),
                        created_at=datetime.fromisoformat(document_data["created_at"]),
                        updated_at=datetime.fromisoformat(document_data["updated_at"])
                    )
                    self.documents[document.id] = document
                
                logger.info(f"Cargados {len(self.documents)} documentos")
            
        except Exception as e:
            logger.error(f"Error cargando datos de colaboración: {e}")
    
    async def _start_collaboration_workers(self):
        """Inicia workers de colaboración"""
        try:
            # Worker de limpieza de sesiones
            asyncio.create_task(self._session_cleanup_worker())
            
            # Worker de notificaciones en tiempo real
            asyncio.create_task(self._realtime_notification_worker())
            
            logger.info("Workers de colaboración iniciados")
            
        except Exception as e:
            logger.error(f"Error iniciando workers de colaboración: {e}")
    
    async def _session_cleanup_worker(self):
        """Worker de limpieza de sesiones"""
        while True:
            try:
                await asyncio.sleep(300)  # Cada 5 minutos
                
                # Limpiar sesiones inactivas
                current_time = datetime.now()
                inactive_sessions = []
                
                for session_id, session_data in self.active_sessions.items():
                    last_activity = session_data.get("last_activity")
                    if last_activity and (current_time - last_activity).seconds > 1800:  # 30 minutos
                        inactive_sessions.append(session_id)
                
                for session_id in inactive_sessions:
                    await self._cleanup_session(session_id)
                
            except Exception as e:
                logger.error(f"Error en worker de limpieza de sesiones: {e}")
                await asyncio.sleep(60)
    
    async def _realtime_notification_worker(self):
        """Worker de notificaciones en tiempo real"""
        while True:
            try:
                await asyncio.sleep(1)  # Cada segundo
                
                # Procesar notificaciones pendientes
                # En implementación real, usar WebSockets
                
            except Exception as e:
                logger.error(f"Error en worker de notificaciones: {e}")
                await asyncio.sleep(5)
    
    async def create_user(
        self,
        username: str,
        email: str,
        display_name: str,
        role: UserRole = UserRole.VIEWER
    ) -> str:
        """Crea un nuevo usuario"""
        try:
            user_id = f"user_{uuid.uuid4().hex[:8]}"
            
            user = User(
                id=user_id,
                username=username,
                email=email,
                display_name=display_name,
                role=role
            )
            
            self.users[user_id] = user
            
            # Registrar actividad
            await self._log_activity(
                ActivityType.USER_JOINED,
                user_id,
                None,
                {"username": username, "display_name": display_name}
            )
            
            # Guardar datos
            await self._save_collaboration_data()
            
            logger.info(f"Usuario creado: {user_id}")
            return user_id
            
        except Exception as e:
            logger.error(f"Error creando usuario: {e}")
            raise
    
    async def create_workspace(
        self,
        name: str,
        description: str,
        owner_id: str
    ) -> str:
        """Crea un nuevo espacio de trabajo"""
        try:
            if owner_id not in self.users:
                raise ValueError(f"Usuario no encontrado: {owner_id}")
            
            workspace_id = f"workspace_{uuid.uuid4().hex[:8]}"
            
            workspace = Workspace(
                id=workspace_id,
                name=name,
                description=description,
                owner_id=owner_id
            )
            
            # Agregar propietario como miembro
            workspace.members[owner_id] = UserRole.ADMIN
            
            self.workspaces[workspace_id] = workspace
            
            # Registrar actividad
            await self._log_activity(
                ActivityType.DOCUMENT_CREATED,
                owner_id,
                workspace_id,
                {"workspace_name": name}
            )
            
            # Guardar datos
            await self._save_collaboration_data()
            
            logger.info(f"Espacio de trabajo creado: {workspace_id}")
            return workspace_id
            
        except Exception as e:
            logger.error(f"Error creando espacio de trabajo: {e}")
            raise
    
    async def add_workspace_member(
        self,
        workspace_id: str,
        user_id: str,
        role: UserRole,
        added_by: str
    ) -> bool:
        """Agrega miembro al espacio de trabajo"""
        try:
            if workspace_id not in self.workspaces:
                return False
            
            if user_id not in self.users:
                return False
            
            workspace = self.workspaces[workspace_id]
            
            # Verificar permisos
            if added_by not in workspace.members or workspace.members[added_by] not in [UserRole.ADMIN, UserRole.EDITOR]:
                return False
            
            # Agregar miembro
            workspace.members[user_id] = role
            workspace.updated_at = datetime.now()
            
            # Registrar actividad
            await self._log_activity(
                ActivityType.USER_JOINED,
                added_by,
                workspace_id,
                {"added_user": user_id, "role": role.value}
            )
            
            # Guardar datos
            await self._save_collaboration_data()
            
            logger.info(f"Miembro agregado al espacio de trabajo: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error agregando miembro al espacio de trabajo: {e}")
            return False
    
    async def create_document(
        self,
        title: str,
        content: str,
        workspace_id: str,
        owner_id: str
    ) -> str:
        """Crea un nuevo documento"""
        try:
            if workspace_id not in self.workspaces:
                raise ValueError(f"Espacio de trabajo no encontrado: {workspace_id}")
            
            if owner_id not in self.users:
                raise ValueError(f"Usuario no encontrado: {owner_id}")
            
            document_id = f"document_{uuid.uuid4().hex[:8]}"
            
            document = Document(
                id=document_id,
                title=title,
                content=content,
                workspace_id=workspace_id,
                owner_id=owner_id
            )
            
            # Agregar propietario como colaborador
            document.collaborators[owner_id] = UserRole.EDITOR
            
            self.documents[document_id] = document
            
            # Agregar documento al espacio de trabajo
            workspace = self.workspaces[workspace_id]
            workspace.documents.append(document_id)
            workspace.updated_at = datetime.now()
            
            # Registrar actividad
            await self._log_activity(
                ActivityType.DOCUMENT_CREATED,
                owner_id,
                workspace_id,
                document_id,
                {"title": title}
            )
            
            # Guardar datos
            await self._save_collaboration_data()
            
            logger.info(f"Documento creado: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error creando documento: {e}")
            raise
    
    async def update_document(
        self,
        document_id: str,
        content: str,
        user_id: str
    ) -> bool:
        """Actualiza un documento"""
        try:
            if document_id not in self.documents:
                return False
            
            document = self.documents[document_id]
            
            # Verificar permisos
            if user_id not in document.collaborators or document.collaborators[user_id] not in [UserRole.ADMIN, UserRole.EDITOR]:
                return False
            
            # Verificar si está bloqueado por otro usuario
            if document.is_locked and document.locked_by != user_id:
                return False
            
            # Actualizar documento
            document.content = content
            document.version += 1
            document.updated_at = datetime.now()
            
            # Registrar actividad
            await self._log_activity(
                ActivityType.DOCUMENT_EDITED,
                user_id,
                document.workspace_id,
                document_id,
                {"version": document.version}
            )
            
            # Guardar datos
            await self._save_collaboration_data()
            
            logger.info(f"Documento actualizado: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando documento: {e}")
            return False
    
    async def add_comment(
        self,
        document_id: str,
        author_id: str,
        content: str,
        position: Dict[str, Any] = None,
        parent_id: str = None
    ) -> str:
        """Agrega comentario a documento"""
        try:
            if document_id not in self.documents:
                raise ValueError(f"Documento no encontrado: {document_id}")
            
            if author_id not in self.users:
                raise ValueError(f"Usuario no encontrado: {author_id}")
            
            comment_id = f"comment_{uuid.uuid4().hex[:8]}"
            
            comment = Comment(
                id=comment_id,
                document_id=document_id,
                author_id=author_id,
                content=content,
                position=position or {},
                parent_id=parent_id
            )
            
            self.comments[comment_id] = comment
            
            # Registrar actividad
            document = self.documents[document_id]
            await self._log_activity(
                ActivityType.COMMENT_ADDED,
                author_id,
                document.workspace_id,
                document_id,
                {"comment_id": comment_id}
            )
            
            # Guardar datos
            await self._save_collaboration_data()
            
            logger.info(f"Comentario agregado: {comment_id}")
            return comment_id
            
        except Exception as e:
            logger.error(f"Error agregando comentario: {e}")
            raise
    
    async def resolve_comment(
        self,
        comment_id: str,
        resolved_by: str
    ) -> bool:
        """Resuelve un comentario"""
        try:
            if comment_id not in self.comments:
                return False
            
            comment = self.comments[comment_id]
            
            # Verificar permisos
            document = self.documents[comment.document_id]
            if resolved_by not in document.collaborators or document.collaborators[resolved_by] not in [UserRole.ADMIN, UserRole.EDITOR]:
                return False
            
            # Resolver comentario
            comment.is_resolved = True
            comment.resolved_by = resolved_by
            comment.resolved_at = datetime.now()
            
            # Registrar actividad
            await self._log_activity(
                ActivityType.COMMENT_RESOLVED,
                resolved_by,
                document.workspace_id,
                comment.document_id,
                {"comment_id": comment_id}
            )
            
            # Guardar datos
            await self._save_collaboration_data()
            
            logger.info(f"Comentario resuelto: {comment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolviendo comentario: {e}")
            return False
    
    async def create_task(
        self,
        title: str,
        description: str,
        workspace_id: str,
        assigned_by: str,
        assigned_to: str = None,
        document_id: str = None,
        priority: str = "medium",
        due_date: datetime = None
    ) -> str:
        """Crea una nueva tarea"""
        try:
            if workspace_id not in self.workspaces:
                raise ValueError(f"Espacio de trabajo no encontrado: {workspace_id}")
            
            if assigned_by not in self.users:
                raise ValueError(f"Usuario no encontrado: {assigned_by}")
            
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            
            task = Task(
                id=task_id,
                title=title,
                description=description,
                workspace_id=workspace_id,
                document_id=document_id,
                assigned_to=assigned_to,
                assigned_by=assigned_by,
                priority=priority,
                due_date=due_date
            )
            
            self.tasks[task_id] = task
            
            # Registrar actividad
            await self._log_activity(
                ActivityType.TASK_ASSIGNED,
                assigned_by,
                workspace_id,
                document_id,
                task_id,
                {"title": title, "assigned_to": assigned_to}
            )
            
            # Guardar datos
            await self._save_collaboration_data()
            
            logger.info(f"Tarea creada: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error creando tarea: {e}")
            raise
    
    async def complete_task(
        self,
        task_id: str,
        completed_by: str
    ) -> bool:
        """Completa una tarea"""
        try:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            # Verificar permisos
            if completed_by != task.assigned_to and completed_by not in self.workspaces[task.workspace_id].members:
                return False
            
            # Completar tarea
            task.status = "completed"
            task.updated_at = datetime.now()
            
            # Registrar actividad
            await self._log_activity(
                ActivityType.TASK_COMPLETED,
                completed_by,
                task.workspace_id,
                task.document_id,
                task_id,
                {"title": task.title}
            )
            
            # Guardar datos
            await self._save_collaboration_data()
            
            logger.info(f"Tarea completada: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error completando tarea: {e}")
            return False
    
    async def lock_document(
        self,
        document_id: str,
        user_id: str
    ) -> bool:
        """Bloquea documento para edición"""
        try:
            if document_id not in self.documents:
                return False
            
            document = self.documents[document_id]
            
            # Verificar permisos
            if user_id not in document.collaborators or document.collaborators[user_id] not in [UserRole.ADMIN, UserRole.EDITOR]:
                return False
            
            # Verificar si ya está bloqueado
            if document.is_locked and document.locked_by != user_id:
                return False
            
            # Bloquear documento
            document.is_locked = True
            document.locked_by = user_id
            
            # Guardar datos
            await self._save_collaboration_data()
            
            logger.info(f"Documento bloqueado: {document_id} por {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error bloqueando documento: {e}")
            return False
    
    async def unlock_document(
        self,
        document_id: str,
        user_id: str
    ) -> bool:
        """Desbloquea documento"""
        try:
            if document_id not in self.documents:
                return False
            
            document = self.documents[document_id]
            
            # Verificar si el usuario puede desbloquear
            if document.locked_by != user_id and user_id not in self.workspaces[document.workspace_id].members:
                return False
            
            # Desbloquear documento
            document.is_locked = False
            document.locked_by = None
            
            # Guardar datos
            await self._save_collaboration_data()
            
            logger.info(f"Documento desbloqueado: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error desbloqueando documento: {e}")
            return False
    
    async def _log_activity(
        self,
        activity_type: ActivityType,
        user_id: str,
        workspace_id: str,
        document_id: str = None,
        task_id: str = None,
        details: Dict[str, Any] = None
    ):
        """Registra actividad"""
        try:
            activity = Activity(
                id=f"activity_{uuid.uuid4().hex[:8]}",
                type=activity_type,
                user_id=user_id,
                workspace_id=workspace_id,
                document_id=document_id,
                task_id=task_id,
                details=details or {}
            )
            
            self.activities.append(activity)
            
            # Mantener solo las últimas actividades
            if len(self.activities) > 10000:
                self.activities = self.activities[-5000:]
            
        except Exception as e:
            logger.error(f"Error registrando actividad: {e}")
    
    async def _cleanup_session(self, session_id: str):
        """Limpia sesión inactiva"""
        try:
            if session_id in self.active_sessions:
                session_data = self.active_sessions[session_id]
                
                # Desbloquear documentos si es necesario
                user_id = session_data.get("user_id")
                if user_id:
                    for document in self.documents.values():
                        if document.locked_by == user_id:
                            document.is_locked = False
                            document.locked_by = None
                
                # Remover de suscriptores
                for workspace_id in self.real_time_subscribers:
                    if session_id in self.real_time_subscribers[workspace_id]:
                        self.real_time_subscribers[workspace_id].remove(session_id)
                
                # Eliminar sesión
                del self.active_sessions[session_id]
                
                logger.info(f"Sesión limpiada: {session_id}")
            
        except Exception as e:
            logger.error(f"Error limpiando sesión: {e}")
    
    async def _save_collaboration_data(self):
        """Guarda datos de colaboración"""
        try:
            # Crear directorio de datos
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Guardar usuarios
            users_data = []
            for user in self.users.values():
                users_data.append({
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "display_name": user.display_name,
                    "role": user.role.value,
                    "avatar_url": user.avatar_url,
                    "is_online": user.is_online,
                    "last_seen": user.last_seen.isoformat() if user.last_seen else None,
                    "preferences": user.preferences,
                    "created_at": user.created_at.isoformat()
                })
            
            users_file = data_dir / "collaboration_users.json"
            with open(users_file, 'w', encoding='utf-8') as f:
                json.dump(users_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Guardar espacios de trabajo
            workspaces_data = []
            for workspace in self.workspaces.values():
                workspaces_data.append({
                    "id": workspace.id,
                    "name": workspace.name,
                    "description": workspace.description,
                    "owner_id": workspace.owner_id,
                    "members": {k: v.value for k, v in workspace.members.items()},
                    "documents": workspace.documents,
                    "settings": workspace.settings,
                    "created_at": workspace.created_at.isoformat(),
                    "updated_at": workspace.updated_at.isoformat()
                })
            
            workspaces_file = data_dir / "collaboration_workspaces.json"
            with open(workspaces_file, 'w', encoding='utf-8') as f:
                json.dump(workspaces_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Guardar documentos
            documents_data = []
            for document in self.documents.values():
                documents_data.append({
                    "id": document.id,
                    "title": document.title,
                    "content": document.content,
                    "workspace_id": document.workspace_id,
                    "owner_id": document.owner_id,
                    "collaborators": {k: v.value for k, v in document.collaborators.items()},
                    "version": document.version,
                    "is_locked": document.is_locked,
                    "locked_by": document.locked_by,
                    "created_at": document.created_at.isoformat(),
                    "updated_at": document.updated_at.isoformat()
                })
            
            documents_file = data_dir / "collaboration_documents.json"
            with open(documents_file, 'w', encoding='utf-8') as f:
                json.dump(documents_data, f, indent=2, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"Error guardando datos de colaboración: {e}")
    
    async def get_workspace_activity(
        self,
        workspace_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Obtiene actividad del espacio de trabajo"""
        try:
            workspace_activities = [
                activity for activity in self.activities
                if activity.workspace_id == workspace_id
            ]
            
            # Ordenar por timestamp descendente
            workspace_activities.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Limitar resultados
            workspace_activities = workspace_activities[:limit]
            
            # Convertir a formato serializable
            activities_data = []
            for activity in workspace_activities:
                user = self.users.get(activity.user_id)
                activities_data.append({
                    "id": activity.id,
                    "type": activity.type.value,
                    "user": {
                        "id": user.id if user else activity.user_id,
                        "username": user.username if user else "Unknown",
                        "display_name": user.display_name if user else "Unknown User"
                    },
                    "workspace_id": activity.workspace_id,
                    "document_id": activity.document_id,
                    "task_id": activity.task_id,
                    "details": activity.details,
                    "timestamp": activity.timestamp.isoformat()
                })
            
            return activities_data
            
        except Exception as e:
            logger.error(f"Error obteniendo actividad del espacio de trabajo: {e}")
            return []
    
    async def get_collaboration_dashboard_data(self) -> Dict[str, Any]:
        """Obtiene datos para dashboard de colaboración"""
        try:
            # Estadísticas generales
            total_users = len(self.users)
            online_users = len([u for u in self.users.values() if u.is_online])
            total_workspaces = len(self.workspaces)
            total_documents = len(self.documents)
            total_tasks = len(self.tasks)
            active_tasks = len([t for t in self.tasks.values() if t.status == "pending"])
            
            # Actividad reciente
            recent_activities = self.activities[-20:] if self.activities else []
            
            # Distribución de roles
            role_distribution = {}
            for user in self.users.values():
                role = user.role.value
                role_distribution[role] = role_distribution.get(role, 0) + 1
            
            # Documentos más activos
            document_activity = {}
            for activity in self.activities:
                if activity.document_id:
                    document_activity[activity.document_id] = document_activity.get(activity.document_id, 0) + 1
            
            most_active_documents = sorted(
                document_activity.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return {
                "total_users": total_users,
                "online_users": online_users,
                "total_workspaces": total_workspaces,
                "total_documents": total_documents,
                "total_tasks": total_tasks,
                "active_tasks": active_tasks,
                "completed_tasks": total_tasks - active_tasks,
                "role_distribution": role_distribution,
                "most_active_documents": [
                    {
                        "document_id": doc_id,
                        "title": self.documents[doc_id].title if doc_id in self.documents else "Unknown",
                        "activity_count": count
                    }
                    for doc_id, count in most_active_documents
                ],
                "recent_activities": [
                    {
                        "id": activity.id,
                        "type": activity.type.value,
                        "user_id": activity.user_id,
                        "workspace_id": activity.workspace_id,
                        "timestamp": activity.timestamp.isoformat()
                    }
                    for activity in recent_activities
                ],
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo datos del dashboard de colaboración: {e}")
            return {"error": str(e)}

