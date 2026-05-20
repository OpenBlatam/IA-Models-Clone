"""
Conflict Resolver - Resolvedor de Conflictos Avanzado
======================================================

Sistema avanzado de resolución de conflictos con múltiples estrategias y aprendizaje automático.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Tipo de conflicto."""
    DATA_UPDATE = "data_update"
    RESOURCE_ACCESS = "resource_access"
    VERSION_CONFLICT = "version_conflict"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    CUSTOM = "custom"


class ResolutionStrategy(Enum):
    """Estrategia de resolución."""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MERGE = "merge"
    MANUAL = "manual"
    VOTE = "vote"
    PRIORITY = "priority"
    TIMESTAMP = "timestamp"
    CUSTOM = "custom"


class ConflictStatus(Enum):
    """Estado de conflicto."""
    PENDING = "pending"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    IGNORED = "ignored"


@dataclass
class Conflict:
    """Conflicto."""
    conflict_id: str
    conflict_type: ConflictType
    resource_id: str
    conflicting_values: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    status: ConflictStatus = ConflictStatus.PENDING
    resolution_strategy: Optional[ResolutionStrategy] = None
    resolved_value: Optional[Any] = None
    resolver_id: Optional[str] = None


class ConflictResolver:
    """Resolvedor de conflictos avanzado."""
    
    def __init__(self):
        self.conflicts: Dict[str, Conflict] = {}
        self.resolution_history: deque = deque(maxlen=100000)
        self.resolution_rules: Dict[ConflictType, ResolutionStrategy] = {}
        self.custom_resolvers: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()
    
    def register_conflict(
        self,
        conflict_id: str,
        conflict_type: ConflictType,
        resource_id: str,
        conflicting_values: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar conflicto."""
        conflict = Conflict(
            conflict_id=conflict_id,
            conflict_type=conflict_type,
            resource_id=resource_id,
            conflicting_values=conflicting_values,
            metadata=metadata or {},
        )
        
        async def save_conflict():
            async with self._lock:
                self.conflicts[conflict_id] = conflict
        
        asyncio.create_task(save_conflict())
        
        # Intentar resolución automática
        asyncio.create_task(self._auto_resolve(conflict))
        
        logger.info(f"Registered conflict: {conflict_id}")
        return conflict_id
    
    async def _auto_resolve(self, conflict: Conflict):
        """Intentar resolución automática."""
        # Obtener estrategia para este tipo de conflicto
        strategy = self.resolution_rules.get(
            conflict.conflict_type,
            ResolutionStrategy.LAST_WRITE_WINS
        )
        
        # Resolver
        resolved_value = await self._resolve_with_strategy(conflict, strategy)
        
        if resolved_value is not None:
            await self.apply_resolution(
                conflict.conflict_id,
                resolved_value,
                strategy,
                resolver_id="auto"
            )
    
    async def _resolve_with_strategy(
        self,
        conflict: Conflict,
        strategy: ResolutionStrategy,
    ) -> Optional[Any]:
        """Resolver con estrategia."""
        values = conflict.conflicting_values
        
        if strategy == ResolutionStrategy.LAST_WRITE_WINS:
            # Obtener el más reciente basado en timestamp
            if "timestamps" in conflict.metadata:
                timestamps = conflict.metadata["timestamps"]
                latest_key = max(timestamps.keys(), key=lambda k: timestamps[k])
                return values.get(latest_key)
            return list(values.values())[-1]
        
        elif strategy == ResolutionStrategy.FIRST_WRITE_WINS:
            # Obtener el más antiguo
            if "timestamps" in conflict.metadata:
                timestamps = conflict.metadata["timestamps"]
                earliest_key = min(timestamps.keys(), key=lambda k: timestamps[k])
                return values.get(earliest_key)
            return list(values.values())[0]
        
        elif strategy == ResolutionStrategy.MERGE:
            # Fusionar valores
            if isinstance(list(values.values())[0], dict):
                merged = {}
                for value in values.values():
                    merged.update(value)
                return merged
            elif isinstance(list(values.values())[0], list):
                merged = []
                for value in values.values():
                    merged.extend(value)
                return list(set(merged))  # Remover duplicados
            return list(values.values())
        
        elif strategy == ResolutionStrategy.PRIORITY:
            # Usar valor de mayor prioridad
            if "priorities" in conflict.metadata:
                priorities = conflict.metadata["priorities"]
                highest_key = max(priorities.keys(), key=lambda k: priorities[k])
                return values.get(highest_key)
            return list(values.values())[0]
        
        elif strategy == ResolutionStrategy.VOTE:
            # Usar valor más votado
            if "votes" in conflict.metadata:
                votes = conflict.metadata["votes"]
                most_voted_key = max(votes.keys(), key=lambda k: votes[k])
                return values.get(most_voted_key)
            return list(values.values())[0]
        
        elif strategy == ResolutionStrategy.CUSTOM:
            # Usar resolver personalizado
            resolver_key = f"{conflict.conflict_type.value}_custom"
            resolver = self.custom_resolvers.get(resolver_key)
            
            if resolver:
                if asyncio.iscoroutinefunction(resolver):
                    return await resolver(conflict)
                else:
                    return resolver(conflict)
        
        return None
    
    async def apply_resolution(
        self,
        conflict_id: str,
        resolved_value: Any,
        strategy: ResolutionStrategy,
        resolver_id: str = "system",
    ) -> bool:
        """Aplicar resolución."""
        async with self._lock:
            conflict = self.conflicts.get(conflict_id)
            if not conflict:
                return False
            
            if conflict.status != ConflictStatus.PENDING:
                return False
            
            conflict.status = ConflictStatus.RESOLVED
            conflict.resolved_at = datetime.now()
            conflict.resolution_strategy = strategy
            conflict.resolved_value = resolved_value
            conflict.resolver_id = resolver_id
        
        # Guardar en historial
        async with self._lock:
            self.resolution_history.append({
                "conflict_id": conflict_id,
                "timestamp": datetime.now().isoformat(),
                "strategy": strategy.value,
                "resolver_id": resolver_id,
            })
        
        logger.info(f"Conflict {conflict_id} resolved using {strategy.value}")
        return True
    
    def register_resolution_rule(
        self,
        conflict_type: ConflictType,
        strategy: ResolutionStrategy,
    ):
        """Registrar regla de resolución."""
        async def save_rule():
            async with self._lock:
                self.resolution_rules[conflict_type] = strategy
        
        asyncio.create_task(save_rule())
        
        logger.info(f"Registered resolution rule: {conflict_type.value} -> {strategy.value}")
    
    def register_custom_resolver(
        self,
        conflict_type: ConflictType,
        resolver: Callable,
    ):
        """Registrar resolvedor personalizado."""
        resolver_key = f"{conflict_type.value}_custom"
        self.custom_resolvers[resolver_key] = resolver
        
        logger.info(f"Registered custom resolver for {conflict_type.value}")
    
    def get_conflict(self, conflict_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de conflicto."""
        conflict = self.conflicts.get(conflict_id)
        if not conflict:
            return None
        
        return {
            "conflict_id": conflict.conflict_id,
            "conflict_type": conflict.conflict_type.value,
            "resource_id": conflict.resource_id,
            "conflicting_values": conflict.conflicting_values,
            "status": conflict.status.value,
            "resolution_strategy": conflict.resolution_strategy.value if conflict.resolution_strategy else None,
            "resolved_value": conflict.resolved_value,
            "resolver_id": conflict.resolver_id,
            "created_at": conflict.created_at.isoformat(),
            "resolved_at": conflict.resolved_at.isoformat() if conflict.resolved_at else None,
        }
    
    def get_pending_conflicts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener conflictos pendientes."""
        pending = [
            self.get_conflict(cid)
            for cid, conflict in self.conflicts.items()
            if conflict.status == ConflictStatus.PENDING
        ]
        
        return [c for c in pending if c][:limit]
    
    def get_resolution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de resoluciones."""
        return list(self.resolution_history)[-limit:]
    
    def get_conflict_resolver_summary(self) -> Dict[str, Any]:
        """Obtener resumen del resolvedor."""
        by_status: Dict[str, int] = defaultdict(int)
        by_type: Dict[str, int] = defaultdict(int)
        
        for conflict in self.conflicts.values():
            by_status[conflict.status.value] += 1
            by_type[conflict.conflict_type.value] += 1
        
        return {
            "total_conflicts": len(self.conflicts),
            "conflicts_by_status": dict(by_status),
            "conflicts_by_type": dict(by_type),
            "total_rules": len(self.resolution_rules),
            "total_custom_resolvers": len(self.custom_resolvers),
            "total_history": len(self.resolution_history),
        }


