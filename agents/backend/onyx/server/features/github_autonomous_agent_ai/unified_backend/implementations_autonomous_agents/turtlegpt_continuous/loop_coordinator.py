"""
Loop Coordinator Module
=======================

Coordinador del loop principal del agente.
Maneja la ejecución de operaciones periódicas y la coordinación de tareas.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LoopPhase(Enum):
    """Fases del loop principal."""
    TASK_PROCESSING = "task_processing"
    REFLECTION = "reflection"
    PLANNING = "planning"
    LEARNING = "learning"
    METRICS_UPDATE = "metrics_update"
    MAINTENANCE = "maintenance"
    IDLE = "idle"


@dataclass
class LoopOperation:
    """Operación del loop principal."""
    phase: LoopPhase
    handler: Callable
    condition: Optional[Callable] = None
    priority: int = 5
    enabled: bool = True


class LoopCoordinator:
    """
    Coordinador del loop principal del agente.
    
    Maneja la ejecución ordenada de operaciones periódicas,
    el control de flujo y la gestión de errores.
    """
    
    def __init__(
        self,
        operations: Optional[List[LoopOperation]] = None,
        default_sleep: float = 1.0,
        idle_sleep: float = 5.0,
        retry_sleep: float = 2.0
    ):
        """
        Inicializar coordinador.
        
        Args:
            operations: Lista de operaciones del loop
            default_sleep: Tiempo de espera por defecto
            idle_sleep: Tiempo de espera en modo idle
            retry_sleep: Tiempo de espera antes de reintentar
        """
        self.operations = operations or []
        self.default_sleep = default_sleep
        self.idle_sleep = idle_sleep
        self.retry_sleep = retry_sleep
        self.should_stop = False
        self.is_idle = False
    
    def add_operation(self, operation: LoopOperation) -> None:
        """Agregar operación al loop."""
        self.operations.append(operation)
        # Ordenar por prioridad (mayor primero)
        self.operations.sort(key=lambda op: op.priority, reverse=True)
    
    def remove_operation(self, phase: LoopPhase) -> None:
        """Remover operación del loop."""
        self.operations = [op for op in self.operations if op.phase != phase]
    
    def enable_operation(self, phase: LoopPhase) -> None:
        """Habilitar operación."""
        for op in self.operations:
            if op.phase == phase:
                op.enabled = True
    
    def disable_operation(self, phase: LoopPhase) -> None:
        """Deshabilitar operación."""
        for op in self.operations:
            if op.phase == phase:
                op.enabled = False
    
    async def execute_phase(self, operation: LoopOperation) -> bool:
        """
        Ejecutar una fase del loop.
        
        Args:
            operation: Operación a ejecutar
            
        Returns:
            True si se ejecutó exitosamente
        """
        if not operation.enabled:
            return False
        
        # Verificar condición si existe
        if operation.condition:
            try:
                if asyncio.iscoroutinefunction(operation.condition):
                    should_run = await operation.condition()
                else:
                    should_run = operation.condition()
                
                if not should_run:
                    return False
            except Exception as e:
                logger.warning(f"Error checking condition for {operation.phase.value}: {e}")
                return False
        
        # Ejecutar handler
        try:
            if asyncio.iscoroutinefunction(operation.handler):
                await operation.handler()
            else:
                operation.handler()
            
            logger.debug(f"Phase {operation.phase.value} executed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error executing phase {operation.phase.value}: {e}", exc_info=True)
            return False
    
    async def run_cycle(self) -> Dict[str, Any]:
        """
        Ejecutar un ciclo completo del loop.
        
        Returns:
            Estadísticas del ciclo
        """
        cycle_stats = {
            "phases_executed": 0,
            "phases_failed": 0,
            "phases_skipped": 0
        }
        
        for operation in self.operations:
            try:
                success = await self.execute_phase(operation)
                if success:
                    cycle_stats["phases_executed"] += 1
                else:
                    cycle_stats["phases_skipped"] += 1
            except Exception as e:
                logger.error(f"Unexpected error in phase {operation.phase.value}: {e}")
                cycle_stats["phases_failed"] += 1
        
        return cycle_stats
    
    async def run_loop(self, stop_callback: Optional[Callable] = None) -> None:
        """
        Ejecutar el loop principal continuamente.
        
        Se ejecuta hasta que should_stop sea True.
        
        Args:
            stop_callback: Callback para verificar si se debe detener
        """
        logger.info("Starting main loop coordinator")
        
        while not self.should_stop:
            try:
                # Verificar callback de stop si existe
                if stop_callback:
                    if asyncio.iscoroutinefunction(stop_callback):
                        should_stop = await stop_callback()
                    else:
                        should_stop = stop_callback()
                    if should_stop:
                        self.should_stop = True
                        break
                
                # Ejecutar ciclo
                cycle_stats = await self.run_cycle()
                
                # Determinar tiempo de espera
                sleep_time = self.idle_sleep if self.is_idle else self.default_sleep
                
                logger.debug(
                    f"Cycle completed: {cycle_stats['phases_executed']} executed, "
                    f"{cycle_stats['phases_skipped']} skipped, "
                    f"{cycle_stats['phases_failed']} failed"
                )
                
                await asyncio.sleep(sleep_time)
            
            except asyncio.CancelledError:
                logger.info("Loop coordinator cancelled")
                break
            
            except Exception as e:
                logger.error(f"Error in loop coordinator: {e}", exc_info=True)
                await asyncio.sleep(self.retry_sleep)
        
        logger.info("Loop coordinator stopped")
    
    def stop(self) -> None:
        """Detener el loop."""
        logger.info("Stopping loop coordinator")
        self.should_stop = True
    
    def set_idle_mode(self, idle: bool) -> None:
        """Configurar modo idle."""
        self.is_idle = idle
        logger.debug(f"Idle mode: {idle}")


class LoopPhaseBuilder:
    """Builder para crear operaciones del loop de forma fluida."""
    
    def __init__(self):
        self.operations: List[LoopOperation] = []
    
    def add_task_processing(
        self,
        handler: Callable,
        condition: Optional[Callable] = None,
        priority: int = 10
    ) -> 'LoopPhaseBuilder':
        """Agregar fase de procesamiento de tareas."""
        self.operations.append(LoopOperation(
            phase=LoopPhase.TASK_PROCESSING,
            handler=handler,
            condition=condition,
            priority=priority
        ))
        return self
    
    def add_reflection(
        self,
        handler: Callable,
        condition: Optional[Callable] = None,
        priority: int = 7
    ) -> 'LoopPhaseBuilder':
        """Agregar fase de reflexión."""
        self.operations.append(LoopOperation(
            phase=LoopPhase.REFLECTION,
            handler=handler,
            condition=condition,
            priority=priority
        ))
        return self
    
    def add_planning(
        self,
        handler: Callable,
        condition: Optional[Callable] = None,
        priority: int = 6
    ) -> 'LoopPhaseBuilder':
        """Agregar fase de planificación."""
        self.operations.append(LoopOperation(
            phase=LoopPhase.PLANNING,
            handler=handler,
            condition=condition,
            priority=priority
        ))
        return self
    
    def add_learning(
        self,
        handler: Callable,
        condition: Optional[Callable] = None,
        priority: int = 5
    ) -> 'LoopPhaseBuilder':
        """Agregar fase de aprendizaje."""
        self.operations.append(LoopOperation(
            phase=LoopPhase.LEARNING,
            handler=handler,
            condition=condition,
            priority=priority
        ))
        return self
    
    def add_metrics_update(
        self,
        handler: Callable,
        priority: int = 3
    ) -> 'LoopPhaseBuilder':
        """Agregar fase de actualización de métricas."""
        self.operations.append(LoopOperation(
            phase=LoopPhase.METRICS_UPDATE,
            handler=handler,
            priority=priority
        ))
        return self
    
    def add_maintenance(
        self,
        handler: Callable,
        condition: Optional[Callable] = None,
        priority: int = 2
    ) -> 'LoopPhaseBuilder':
        """Agregar fase de mantenimiento."""
        self.operations.append(LoopOperation(
            phase=LoopPhase.MAINTENANCE,
            handler=handler,
            condition=condition,
            priority=priority
        ))
        return self
    
    def build(self) -> List[LoopOperation]:
        """Construir lista de operaciones."""
        return self.operations
