"""
Loop Configurator Module
========================

Configurador del loop principal del agente.
Centraliza la lógica de construcción y configuración del LoopCoordinator.
"""

import logging
from typing import Any, Optional, Callable

from .loop_coordinator import LoopCoordinator, LoopPhaseBuilder
from .models import ContinuousAgentConfig

logger = logging.getLogger(__name__)


class LoopConfigurator:
    """
    Configurador del loop principal del agente.
    
    Centraliza la lógica de construcción del LoopCoordinator
    con todas sus fases y operaciones.
    """
    
    def __init__(
        self,
        agent_config: ContinuousAgentConfig,
        reflection_planner: Any,
        learning_manager: Any,
        metrics_manager: Any,
        maintenance_manager: Any
    ):
        """
        Inicializar configurador de loop.
        
        Args:
            agent_config: Configuración del agente
            reflection_planner: Planner de reflexión
            learning_manager: Manager de aprendizaje
            metrics_manager: Manager de métricas
            maintenance_manager: Manager de mantenimiento
        """
        self.agent_config = agent_config
        self.reflection_planner = reflection_planner
        self.learning_manager = learning_manager
        self.metrics_manager = metrics_manager
        self.maintenance_manager = maintenance_manager
    
    def create_loop_coordinator(
        self,
        process_task_queue_handler: Callable,
        identify_learning_opportunities_handler: Optional[Callable] = None
    ) -> LoopCoordinator:
        """
        Crear y configurar el coordinador del loop principal.
        
        Args:
            process_task_queue_handler: Handler para procesar cola de tareas
            identify_learning_opportunities_handler: Handler para identificar oportunidades de aprendizaje
            
        Returns:
            LoopCoordinator configurado
        """
        builder = LoopPhaseBuilder()
        
        # Fase 1: Procesamiento de tareas (prioridad más alta)
        builder.add_task_processing(
            handler=process_task_queue_handler,
            priority=10
        )
        
        # Fase 2: Reflexión
        builder.add_reflection(
            handler=self.reflection_planner.reflect_on_experiences,
            condition=self.reflection_planner.should_reflect,
            priority=7
        )
        
        # Fase 3: Planificación
        builder.add_planning(
            handler=self.reflection_planner.generate_plan_from_memory,
            priority=6
        )
        
        # Fase 4: Aprendizaje (solo si está habilitado)
        if self.learning_manager.learning_enabled:
            if identify_learning_opportunities_handler:
                builder.add_learning(
                    handler=identify_learning_opportunities_handler,
                    condition=lambda: self.learning_manager.learning_enabled,
                    priority=5
                )
            else:
                logger.warning("Learning enabled but no handler provided")
        
        # Fase 5: Actualización de métricas
        builder.add_metrics_update(
            handler=self.metrics_manager.update_activity,
            priority=3
        )
        
        # Fase 6: Mantenimiento
        builder.add_maintenance(
            handler=self.maintenance_manager.perform_maintenance,
            condition=self.maintenance_manager.should_perform_maintenance,
            priority=2
        )
        
        # Construir operaciones
        operations = builder.build()
        
        # Crear coordinador con configuración del agente
        coordinator = LoopCoordinator(
            operations=operations,
            default_sleep=self.agent_config.loop_sleep_seconds,
            idle_sleep=self.agent_config.idle_sleep_seconds,
            retry_sleep=self.agent_config.retry_sleep_seconds
        )
        
        logger.debug(f"Created loop coordinator with {len(operations)} operations")
        return coordinator
    
    def add_custom_phase(
        self,
        builder: LoopPhaseBuilder,
        phase_type: str,
        handler: Callable,
        condition: Optional[Callable] = None,
        priority: int = 5
    ) -> LoopPhaseBuilder:
        """
        Agregar fase personalizada al builder.
        
        Args:
            builder: Builder del loop
            phase_type: Tipo de fase (task_processing, reflection, planning, learning, metrics_update, maintenance)
            handler: Handler de la fase
            condition: Condición opcional
            priority: Prioridad de la fase
            
        Returns:
            Builder actualizado
        """
        phase_methods = {
            "task_processing": builder.add_task_processing,
            "reflection": builder.add_reflection,
            "planning": builder.add_planning,
            "learning": builder.add_learning,
            "metrics_update": builder.add_metrics_update,
            "maintenance": builder.add_maintenance
        }
        
        if phase_type not in phase_methods:
            logger.warning(f"Unknown phase type: {phase_type}, skipping")
            return builder
        
        method = phase_methods[phase_type]
        if condition:
            method(handler=handler, condition=condition, priority=priority)
        else:
            method(handler=handler, priority=priority)
        
        return builder


def create_loop_configurator(
    agent_config: ContinuousAgentConfig,
    reflection_planner: Any,
    learning_manager: Any,
    metrics_manager: Any,
    maintenance_manager: Any
) -> LoopConfigurator:
    """
    Factory function para crear LoopConfigurator.
    
    Args:
        agent_config: Configuración del agente
        reflection_planner: Planner de reflexión
        learning_manager: Manager de aprendizaje
        metrics_manager: Manager de métricas
        maintenance_manager: Manager de mantenimiento
        
    Returns:
        Instancia de LoopConfigurator
    """
    return LoopConfigurator(
        agent_config=agent_config,
        reflection_planner=reflection_planner,
        learning_manager=learning_manager,
        metrics_manager=metrics_manager,
        maintenance_manager=maintenance_manager
    )


