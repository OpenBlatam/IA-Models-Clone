"""
Chat Session Management
=======================

Maneja el estado de las sesiones de chat, incluyendo control de pausa/continuación.
"""

import asyncio
import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


class ChatState(Enum):
    """Estado de la sesión de chat."""
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ChatMessage:
    """Mensaje en la conversación."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = "user"  # user, assistant, system
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSession:
    """Sesión de chat con control de estado."""
    
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    state: ChatState = ChatState.IDLE
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Control de pausa
    is_paused: bool = False
    pause_reason: Optional[str] = None
    auto_continue: bool = True  # Si True, continúa automáticamente después de responder
    
    # Configuración
    max_messages: int = 1000
    auto_respond_interval: float = 2.0  # Segundos entre respuestas automáticas
    
    # Locks y control
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _pause_event: asyncio.Event = field(default_factory=asyncio.Event)
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    
    def __post_init__(self):
        """Inicializar eventos después de la creación."""
        self._pause_event.set()  # Iniciar sin pausa
        self._stop_event.clear()  # No detenido
    
    async def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> ChatMessage:
        """Agregar mensaje a la sesión."""
        async with self._lock:
            if len(self.messages) >= self.max_messages:
                # Mantener solo los últimos mensajes
                self.messages = self.messages[-self.max_messages + 1:]
            
            message = ChatMessage(
                role=role,
                content=content,
                metadata=metadata or {}
            )
            self.messages.append(message)
            self.updated_at = datetime.now()
            return message
    
    async def pause(self, reason: Optional[str] = None):
        """Pausar la sesión de chat."""
        async with self._lock:
            self.is_paused = True
            self.pause_reason = reason
            self.state = ChatState.PAUSED
            self._pause_event.clear()  # Bloquear ejecución
            self.updated_at = datetime.now()
    
    async def resume(self):
        """Reanudar la sesión de chat."""
        async with self._lock:
            self.is_paused = False
            self.pause_reason = None
            if self.state == ChatState.PAUSED:
                self.state = ChatState.ACTIVE
            self._pause_event.set()  # Permitir ejecución
            self.updated_at = datetime.now()
    
    async def stop(self):
        """Detener la sesión de chat."""
        async with self._lock:
            self.state = ChatState.STOPPED
            self._stop_event.set()  # Señal de parada
            self._pause_event.set()  # Liberar cualquier bloqueo
            self.updated_at = datetime.now()
    
    async def activate(self):
        """Activar la sesión."""
        async with self._lock:
            self.state = ChatState.ACTIVE
            self.is_paused = False
            self._pause_event.set()
            self.updated_at = datetime.now()
    
    async def wait_if_paused(self):
        """Esperar si la sesión está pausada."""
        await self._pause_event.wait()
    
    def is_stopped(self) -> bool:
        """Verificar si la sesión está detenida."""
        return self._stop_event.is_set() or self.state == ChatState.STOPPED
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Obtener historial de conversación en formato para LLM."""
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in self.messages
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir sesión a diccionario."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "state": self.state.value,
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
            "message_count": len(self.messages),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "auto_continue": self.auto_continue,
        }
































