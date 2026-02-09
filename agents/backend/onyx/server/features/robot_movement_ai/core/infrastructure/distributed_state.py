"""
Distributed State System
========================

Sistema de estado distribuido.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class StateStatus(Enum):
    """Estado del estado."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    LOCKED = "locked"
    ERROR = "error"


@dataclass
class StateEntry:
    """Entrada de estado."""
    key: str
    value: Any
    version: int = 1
    status: StateStatus = StateStatus.ACTIVE
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class DistributedState:
    """
    Estado distribuido.
    
    Gestiona estado compartido entre instancias.
    """
    
    def __init__(self, name: str):
        """
        Inicializar estado distribuido.
        
        Args:
            name: Nombre del estado
        """
        self.name = name
        self.state: Dict[str, StateEntry] = {}
        self.locks: Dict[str, str] = {}  # key -> lock_owner
        self.change_history: List[Dict[str, Any]] = []
        self.max_history = 10000
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del estado.
        
        Args:
            key: Clave
            
        Returns:
            Valor o None
        """
        if key in self.state:
            entry = self.state[key]
            if entry.status == StateStatus.ACTIVE:
                return entry.value
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StateEntry:
        """
        Establecer valor en estado.
        
        Args:
            key: Clave
            value: Valor
            metadata: Metadata adicional
            
        Returns:
            Entrada de estado
        """
        now = datetime.now().isoformat()
        
        if key in self.state:
            entry = self.state[key]
            entry.value = value
            entry.version += 1
            entry.updated_at = now
            if metadata:
                entry.metadata.update(metadata)
        else:
            entry = StateEntry(
                key=key,
                value=value,
                metadata=metadata or {}
            )
            self.state[key] = entry
        
        # Registrar cambio
        self._record_change(key, "set", entry.version)
        
        return entry
    
    def update(
        self,
        key: str,
        updater: callable,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[StateEntry]:
        """
        Actualizar valor con función.
        
        Args:
            key: Clave
            updater: Función que recibe valor actual y retorna nuevo valor
            metadata: Metadata adicional
            
        Returns:
            Entrada actualizada o None
        """
        current_value = self.get(key)
        if current_value is None:
            return None
        
        new_value = updater(current_value)
        return self.set(key, new_value, metadata)
    
    def delete(self, key: str) -> bool:
        """Eliminar entrada del estado."""
        if key in self.state:
            entry = self.state[key]
            entry.status = StateStatus.INACTIVE
            self._record_change(key, "delete", entry.version)
            return True
        return False
    
    def lock(self, key: str, owner: str) -> bool:
        """
        Bloquear entrada.
        
        Args:
            key: Clave
            owner: Propietario del lock
            
        Returns:
            True si se bloqueó, False si ya está bloqueado
        """
        if key in self.locks:
            return False
        
        self.locks[key] = owner
        if key in self.state:
            self.state[key].status = StateStatus.LOCKED
        return True
    
    def unlock(self, key: str, owner: str) -> bool:
        """
        Desbloquear entrada.
        
        Args:
            key: Clave
            owner: Propietario del lock
            
        Returns:
            True si se desbloqueó, False si no es el propietario
        """
        if key not in self.locks:
            return False
        
        if self.locks[key] != owner:
            return False
        
        del self.locks[key]
        if key in self.state:
            self.state[key].status = StateStatus.ACTIVE
        return True
    
    def _record_change(
        self,
        key: str,
        action: str,
        version: int
    ) -> None:
        """Registrar cambio en historial."""
        self.change_history.append({
            "key": key,
            "action": action,
            "version": version,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.change_history) > self.max_history:
            self.change_history = self.change_history[-self.max_history:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del estado."""
        active = sum(1 for e in self.state.values() if e.status == StateStatus.ACTIVE)
        locked = sum(1 for e in self.state.values() if e.status == StateStatus.LOCKED)
        
        return {
            "name": self.name,
            "total_entries": len(self.state),
            "active_entries": active,
            "locked_entries": locked,
            "change_history_size": len(self.change_history)
        }


# Instancia global de estados
_distributed_states: Dict[str, DistributedState] = {}


def create_distributed_state(name: str) -> DistributedState:
    """
    Crear estado distribuido.
    
    Args:
        name: Nombre del estado
        
    Returns:
        Estado distribuido
    """
    state = DistributedState(name)
    _distributed_states[name] = state
    return state


def get_distributed_state(name: str) -> Optional[DistributedState]:
    """Obtener estado distribuido por nombre."""
    return _distributed_states.get(name)






