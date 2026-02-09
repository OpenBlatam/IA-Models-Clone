"""
Performance Profiler - Perfilador de Rendimiento
================================================

Sistema avanzado de profiling de performance con análisis detallado y optimización.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import time
import statistics

logger = logging.getLogger(__name__)


class ProfilerScope(Enum):
    """Alcance del profiler."""
    FUNCTION = "function"
    ENDPOINT = "endpoint"
    OPERATION = "operation"
    QUERY = "query"
    CUSTOM = "custom"


@dataclass
class PerformanceProfile:
    """Perfil de performance."""
    profile_id: str
    scope: ProfilerScope
    name: str
    duration: float
    timestamp: datetime
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSummary:
    """Resumen de performance."""
    name: str
    total_calls: int = 0
    avg_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    p95_duration: float = 0.0
    p99_duration: float = 0.0
    total_memory: float = 0.0
    avg_memory: float = 0.0


class PerformanceProfiler:
    """Perfilador de performance."""
    
    def __init__(self, history_size: int = 100000):
        self.history_size = history_size
        self.profiles: deque = deque(maxlen=history_size)
        self.summaries: Dict[str, PerformanceSummary] = {}
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    def start_profile(
        self,
        profile_id: str,
        scope: ProfilerScope,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Iniciar perfil."""
        start_time = time.time()
        
        profile_data = {
            "profile_id": profile_id,
            "scope": scope,
            "name": name,
            "start_time": start_time,
            "metadata": metadata or {},
        }
        
        self.active_profiles[profile_id] = profile_data
        
        return profile_id
    
    def end_profile(
        self,
        profile_id: str,
        memory_usage: Optional[float] = None,
        cpu_usage: Optional[float] = None,
    ) -> Optional[PerformanceProfile]:
        """Finalizar perfil."""
        profile_data = self.active_profiles.pop(profile_id, None)
        if not profile_data:
            return None
        
        duration = time.time() - profile_data["start_time"]
        
        profile = PerformanceProfile(
            profile_id=profile_id,
            scope=profile_data["scope"],
            name=profile_data["name"],
            duration=duration,
            timestamp=datetime.now(),
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            metadata=profile_data["metadata"],
        )
        
        self.profiles.append(profile)
        
        # Actualizar resumen
        asyncio.create_task(self._update_summary(profile))
        
        return profile
    
    async def _update_summary(self, profile: PerformanceProfile):
        """Actualizar resumen."""
        async with self._lock:
            key = f"{profile.scope.value}_{profile.name}"
            
            if key not in self.summaries:
                self.summaries[key] = PerformanceSummary(name=profile.name)
            
            summary = self.summaries[key]
            summary.total_calls += 1
            
            # Actualizar duraciones
            durations = [p.duration for p in self.profiles if p.scope == profile.scope and p.name == profile.name]
            if durations:
                summary.avg_duration = statistics.mean(durations)
                summary.min_duration = min(durations)
                summary.max_duration = max(durations)
                
                if len(durations) > 20:
                    sorted_durations = sorted(durations)
                    summary.p95_duration = sorted_durations[int(len(sorted_durations) * 0.95)]
                if len(durations) > 100:
                    sorted_durations = sorted(durations)
                    summary.p99_duration = sorted_durations[int(len(sorted_durations) * 0.99)]
            
            # Actualizar memoria
            if profile.memory_usage:
                summary.total_memory += profile.memory_usage
                summary.avg_memory = summary.total_memory / summary.total_calls
    
    def profile_function(
        self,
        func: Callable,
        scope: ProfilerScope = ProfilerScope.FUNCTION,
        name: Optional[str] = None,
    ):
        """Decorador para perfilar función."""
        profile_name = name or func.__name__
        
        def wrapper(*args, **kwargs):
            profile_id = f"func_{profile_name}_{datetime.now().timestamp()}"
            self.start_profile(profile_id, scope, profile_name)
            
            try:
                result = func(*args, **kwargs)
                self.end_profile(profile_id)
                return result
            except Exception as e:
                self.end_profile(profile_id)
                raise e
        
        return wrapper
    
    async def profile_async_function(
        self,
        func: Callable,
        scope: ProfilerScope = ProfilerScope.FUNCTION,
        name: Optional[str] = None,
    ):
        """Decorador para perfilar función async."""
        profile_name = name or func.__name__
        
        async def wrapper(*args, **kwargs):
            profile_id = f"async_func_{profile_name}_{datetime.now().timestamp()}"
            self.start_profile(profile_id, scope, profile_name)
            
            try:
                result = await func(*args, **kwargs)
                self.end_profile(profile_id)
                return result
            except Exception as e:
                self.end_profile(profile_id)
                raise e
        
        return wrapper
    
    def get_summary(
        self,
        scope: Optional[ProfilerScope] = None,
        name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Obtener resumen de performance."""
        if scope and name:
            key = f"{scope.value}_{name}"
            summary = self.summaries.get(key)
        elif name:
            # Buscar por nombre
            summary = next((s for k, s in self.summaries.items() if s.name == name), None)
        else:
            return None
        
        if not summary:
            return None
        
        return {
            "name": summary.name,
            "total_calls": summary.total_calls,
            "avg_duration": summary.avg_duration,
            "min_duration": summary.min_duration if summary.min_duration != float('inf') else 0.0,
            "max_duration": summary.max_duration,
            "p95_duration": summary.p95_duration,
            "p99_duration": summary.p99_duration,
            "avg_memory": summary.avg_memory,
        }
    
    def get_slow_operations(
        self,
        threshold: float = 1.0,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Obtener operaciones lentas."""
        slow = [
            {
                "profile_id": p.profile_id,
                "scope": p.scope.value,
                "name": p.name,
                "duration": p.duration,
                "timestamp": p.timestamp.isoformat(),
            }
            for p in self.profiles
            if p.duration > threshold
        ]
        
        slow.sort(key=lambda x: x["duration"], reverse=True)
        return slow[:limit]
    
    def get_performance_profiler_summary(self) -> Dict[str, Any]:
        """Obtener resumen del profiler."""
        by_scope: Dict[str, int] = defaultdict(int)
        
        for profile in self.profiles:
            by_scope[profile.scope.value] += 1
        
        return {
            "total_profiles": len(self.profiles),
            "profiles_by_scope": dict(by_scope),
            "total_summaries": len(self.summaries),
            "active_profiles": len(self.active_profiles),
        }



