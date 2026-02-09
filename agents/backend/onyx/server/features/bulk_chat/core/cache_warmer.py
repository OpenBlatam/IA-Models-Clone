"""
Cache Warmer - Sistema de Precalentamiento de Cache
====================================================

Sistema inteligente de precalentamiento de cache basado en patrones de acceso y predicción.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class WarmingStrategy(Enum):
    """Estrategia de precalentamiento."""
    PREDICTIVE = "predictive"
    PATTERN_BASED = "pattern_based"
    TIME_BASED = "time_based"
    DEMAND_BASED = "demand_based"
    HYBRID = "hybrid"


@dataclass
class CachePattern:
    """Patrón de acceso a cache."""
    pattern_id: str
    key_pattern: str
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    access_times: List[datetime] = field(default_factory=list)
    frequency: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WarmingRule:
    """Regla de precalentamiento."""
    rule_id: str
    strategy: WarmingStrategy
    key_pattern: str
    fetch_function: Callable
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WarmingTask:
    """Tarea de precalentamiento."""
    task_id: str
    key: str
    rule_id: str
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    success: bool = False
    error: Optional[str] = None


class CacheWarmer:
    """Sistema de precalentamiento de cache."""
    
    def __init__(self, cache_manager: Optional[Any] = None):
        self.cache_manager = cache_manager
        self.patterns: Dict[str, CachePattern] = {}
        self.warming_rules: Dict[str, WarmingRule] = {}
        self.warming_tasks: Dict[str, WarmingTask] = {}
        self.access_history: deque = deque(maxlen=100000)
        self.warming_history: deque = deque(maxlen=10000)
        self._lock = asyncio.Lock()
        self._warming_active = False
    
    def record_access(self, key: str, timestamp: Optional[datetime] = None):
        """Registrar acceso a cache."""
        timestamp = timestamp or datetime.now()
        
        async def save_access():
            async with self._lock:
                self.access_history.append({
                    "key": key,
                    "timestamp": timestamp,
                })
                
                # Actualizar patrón
                pattern_key = self._extract_pattern(key)
                if pattern_key not in self.patterns:
                    self.patterns[pattern_key] = CachePattern(
                        pattern_id=pattern_key,
                        key_pattern=pattern_key,
                    )
                
                pattern = self.patterns[pattern_key]
                pattern.access_count += 1
                pattern.last_accessed = timestamp
                pattern.access_times.append(timestamp)
                
                # Limitar historial de tiempos
                if len(pattern.access_times) > 1000:
                    pattern.access_times.pop(0)
                
                # Calcular frecuencia
                if len(pattern.access_times) > 1:
                    time_diffs = [
                        (pattern.access_times[i] - pattern.access_times[i-1]).total_seconds()
                        for i in range(1, len(pattern.access_times))
                    ]
                    if time_diffs:
                        pattern.frequency = 1.0 / statistics.mean(time_diffs) if statistics.mean(time_diffs) > 0 else 0.0
        
        asyncio.create_task(save_access())
        
        # Trigger precalentamiento si es necesario
        asyncio.create_task(self._check_and_warm(key))
    
    def _extract_pattern(self, key: str) -> str:
        """Extraer patrón de key."""
        # Simplificado: usar prefijo común
        parts = key.split(":")
        if len(parts) > 1:
            return ":".join(parts[:-1]) + ":*"
        return key
    
    async def _check_and_warm(self, accessed_key: str):
        """Verificar y precalentar basado en acceso."""
        if not self._warming_active:
            return
        
        # Buscar reglas aplicables
        applicable_rules = [
            rule for rule in self.warming_rules.values()
            if rule.enabled and self._matches_pattern(accessed_key, rule.key_pattern)
        ]
        
        for rule in applicable_rules:
            # Precalentar keys relacionadas
            asyncio.create_task(self._warm_keys_for_rule(rule, accessed_key))
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Verificar si key coincide con patrón."""
        if "*" in pattern:
            prefix = pattern.replace("*", "")
            return key.startswith(prefix)
        return key == pattern
    
    async def _warm_keys_for_rule(self, rule: WarmingRule, trigger_key: str):
        """Precalentar keys para regla."""
        try:
            # Generar keys a precalentar basado en estrategia
            keys_to_warm = await self._generate_keys_to_warm(rule, trigger_key)
            
            for key in keys_to_warm:
                task_id = f"warm_{key}_{datetime.now().timestamp()}"
                task = WarmingTask(
                    task_id=task_id,
                    key=key,
                    rule_id=rule.rule_id,
                )
                
                async with self._lock:
                    self.warming_tasks[task_id] = task
                
                asyncio.create_task(self._execute_warming_task(task, rule))
        
        except Exception as e:
            logger.error(f"Error warming keys for rule {rule.rule_id}: {e}")
    
    async def _generate_keys_to_warm(self, rule: WarmingRule, trigger_key: str) -> List[str]:
        """Generar keys a precalentar."""
        keys = []
        
        if rule.strategy == WarmingStrategy.PREDICTIVE:
            # Predecir keys basado en patrones históricos
            pattern_key = self._extract_pattern(trigger_key)
            pattern = self.patterns.get(pattern_key)
            
            if pattern and pattern.frequency > 0:
                # Precalentar keys relacionadas
                keys.append(f"{pattern_key}_next")
        
        elif rule.strategy == WarmingStrategy.PATTERN_BASED:
            # Basado en patrones de acceso
            pattern_key = self._extract_pattern(trigger_key)
            if pattern_key in self.patterns:
                # Precalentar variaciones del patrón
                keys.append(trigger_key.replace("_1", "_2"))
        
        elif rule.strategy == WarmingStrategy.TIME_BASED:
            # Basado en hora del día
            now = datetime.now()
            # Precalentar keys comunes para esta hora
            keys.append(f"{trigger_key}_hour_{now.hour}")
        
        elif rule.strategy == WarmingStrategy.DEMAND_BASED:
            # Basado en demanda reciente
            recent_accesses = [
                a for a in self.access_history
                if (datetime.now() - a["timestamp"]).total_seconds() < 3600
            ]
            # Precalentar keys accedidas recientemente
            for access in recent_accesses[:5]:
                keys.append(access["key"])
        
        elif rule.strategy == WarmingStrategy.HYBRID:
            # Combinar múltiples estrategias
            keys.extend(await self._generate_keys_to_warm(
                WarmingRule(rule_id="", strategy=WarmingStrategy.PREDICTIVE, key_pattern=rule.key_pattern, fetch_function=rule.fetch_function),
                trigger_key
            ))
            keys.extend(await self._generate_keys_to_warm(
                WarmingRule(rule_id="", strategy=WarmingStrategy.PATTERN_BASED, key_pattern=rule.key_pattern, fetch_function=rule.fetch_function),
                trigger_key
            ))
        
        return list(set(keys))  # Remover duplicados
    
    async def _execute_warming_task(self, task: WarmingTask, rule: WarmingRule):
        """Ejecutar tarea de precalentamiento."""
        task.started_at = datetime.now()
        task.status = "running"
        
        try:
            # Ejecutar función de fetch
            if asyncio.iscoroutinefunction(rule.fetch_function):
                value = await rule.fetch_function(task.key)
            else:
                value = rule.fetch_function(task.key)
            
            # Guardar en cache si hay cache manager
            if self.cache_manager:
                if hasattr(self.cache_manager, 'set'):
                    await self.cache_manager.set(task.key, value)
                elif hasattr(self.cache_manager, 'set_async'):
                    await self.cache_manager.set_async(task.key, value)
            
            task.status = "completed"
            task.success = True
            task.completed_at = datetime.now()
            
            async with self._lock:
                self.warming_history.append({
                    "task_id": task.task_id,
                    "key": task.key,
                    "success": True,
                    "timestamp": task.completed_at.isoformat(),
                })
        
        except Exception as e:
            task.status = "failed"
            task.success = False
            task.error = str(e)
            task.completed_at = datetime.now()
            
            async with self._lock:
                self.warming_history.append({
                    "task_id": task.task_id,
                    "key": task.key,
                    "success": False,
                    "error": str(e),
                    "timestamp": task.completed_at.isoformat(),
                })
    
    def register_warming_rule(
        self,
        rule_id: str,
        strategy: WarmingStrategy,
        key_pattern: str,
        fetch_function: Callable,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar regla de precalentamiento."""
        rule = WarmingRule(
            rule_id=rule_id,
            strategy=strategy,
            key_pattern=key_pattern,
            fetch_function=fetch_function,
            priority=priority,
            metadata=metadata or {},
        )
        
        async def save_rule():
            async with self._lock:
                self.warming_rules[rule_id] = rule
        
        asyncio.create_task(save_rule())
        
        logger.info(f"Registered warming rule: {rule_id}")
        return rule_id
    
    def start_warming(self):
        """Iniciar precalentamiento."""
        self._warming_active = True
        logger.info("Cache warming started")
    
    def stop_warming(self):
        """Detener precalentamiento."""
        self._warming_active = False
        logger.info("Cache warming stopped")
    
    def get_access_patterns(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtener patrones de acceso."""
        patterns = sorted(
            self.patterns.values(),
            key=lambda p: p.access_count,
            reverse=True
        )
        
        return [
            {
                "pattern_id": p.pattern_id,
                "key_pattern": p.key_pattern,
                "access_count": p.access_count,
                "last_accessed": p.last_accessed.isoformat() if p.last_accessed else None,
                "frequency": p.frequency,
            }
            for p in patterns[:limit]
        ]
    
    def get_warming_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de precalentamiento."""
        total_tasks = len(self.warming_tasks)
        successful_tasks = sum(1 for t in self.warming_tasks.values() if t.success)
        failed_tasks = total_tasks - successful_tasks
        
        return {
            "warming_active": self._warming_active,
            "total_rules": len(self.warming_rules),
            "active_rules": len([r for r in self.warming_rules.values() if r.enabled]),
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            "total_patterns": len(self.patterns),
            "total_accesses": len(self.access_history),
        }
    
    def get_cache_warmer_summary(self) -> Dict[str, Any]:
        """Obtener resumen del warmer."""
        return {
            "warming_active": self._warming_active,
            "total_rules": len(self.warming_rules),
            "total_patterns": len(self.patterns),
            "total_tasks": len(self.warming_tasks),
            "total_history": len(self.warming_history),
        }


