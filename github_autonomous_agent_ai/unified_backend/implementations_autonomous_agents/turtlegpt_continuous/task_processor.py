"""
Task Processor Module
=====================

Procesador de tareas que maneja la ejecución de estrategias y el flujo completo
de procesamiento de tareas.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .models import AgentTask
from .error_handler import TaskProcessingError, handle_errors
from .strategy_selector import StrategySelector

logger = logging.getLogger(__name__)


@dataclass
class TaskExecutionResult:
    """Resultado de la ejecución de una tarea."""
    task_id: str
    success: bool
    result: Any
    strategy_used: str
    execution_time: float
    error: Optional[Exception] = None


class TaskProcessor:
    """
    Procesador de tareas que maneja la ejecución de estrategias.
    
    Encapsula la lógica de:
    - Selección de estrategia
    - Ejecución de estrategia
    - Manejo de errores
    - Tracking de resultados
    """
    
    def __init__(
        self,
        strategy_selector: StrategySelector,
        strategies: Dict[str, Any],
        agi_manager: Optional[Any] = None,
        episodic_memory: Optional[Any] = None,
        task_manager: Optional[Any] = None,
        metrics_manager: Optional[Any] = None
    ):
        """
        Inicializar procesador de tareas.
        
        Args:
            strategy_selector: Selector de estrategias
            strategies: Diccionario de estrategias disponibles
            agi_manager: Manager de capacidades AGI
            episodic_memory: Memoria episódica
            task_manager: Manager de tareas
            metrics_manager: Manager de métricas
        """
        self.strategy_selector = strategy_selector
        self.strategies = strategies
        self.agi_manager = agi_manager
        self.episodic_memory = episodic_memory
        self.task_manager = task_manager
        self.metrics_manager = metrics_manager
    
    async def execute_strategy(
        self,
        strategy_name: str,
        task_description: str,
        agent: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Ejecutar una estrategia específica.
        
        Args:
            strategy_name: Nombre de la estrategia
            task_description: Descripción de la tarea
            agent: Referencia al agente (para métodos act/observe)
            
        Returns:
            Resultado de la ejecución
        """
        strategy = self.strategies.get(strategy_name)
        
        if not strategy:
            logger.warning(f"Strategy {strategy_name} not available, using standard")
            return {"status": "strategy_not_available", "strategy": strategy_name}
        
        if strategy_name == "lats" and hasattr(strategy, 'execute'):
            return await strategy.execute(task_description)
        elif strategy_name == "tot" and hasattr(strategy, 'execute'):
            return await strategy.execute(task_description)
        elif strategy_name == "react" and hasattr(strategy, 'execute'):
            return await strategy.execute(task_description)
        elif strategy_name == "standard":
            # Ejecutar método estándar del agente
            if agent and hasattr(agent, 'run'):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: agent.run(task_description, max_steps=10)
                )
            return {"status": "standard_not_available"}
        
        return {"status": "unknown_strategy", "strategy": strategy_name}
    
    @handle_errors(
        default_return=None,
        log_error=True,
        reraise=True,
        error_class=TaskProcessingError
    )
    async def process_task(
        self,
        task: AgentTask,
        agent: Optional[Any] = None
    ) -> TaskExecutionResult:
        """
        Procesar una tarea completa.
        
        Args:
            task: Tarea a procesar
            agent: Referencia al agente
            
        Returns:
            Resultado de la ejecución
        """
        import time
        start_time = time.time()
        
        logger.info(f"Processing task: {task.task_id} - {task.description}")
        
        # Seleccionar estrategia
        strategy_info = self.strategy_selector.select_strategy(task)
        strategy_name = strategy_info["strategy"]
        logger.info(f"Selected strategy: {strategy_name} - {strategy_info['reason']}")
        
        # Ejecutar estrategia
        try:
            result = await self.execute_strategy(strategy_name, task.description, agent)
            
            # Evaluar éxito
            success = (
                result.get("status") != "failed" and
                "error" not in str(result).lower() and
                result.get("status") != "strategy_not_available"
            )
            
            # Evaluar capacidades AGI
            if self.agi_manager:
                task_complexity = task.priority / 10.0
                self.agi_manager.evaluate_reasoning(task_complexity, success)
                self.agi_manager.evaluate_problem_solving(success, 0.7 if success else 0.0)
            
            # Guardar en memoria episódica
            if self.episodic_memory:
                self.episodic_memory.add(
                    content=f"Completed task: {task.description}",
                    importance=0.8,
                    metadata={"task_id": task.task_id, "result": result}
                )
            
            # Actualizar estado de tarea
            if self.task_manager:
                self.task_manager.mark_task_completed(task.task_id, result)
            
            # Registrar métricas
            if self.metrics_manager:
                self.metrics_manager.record_task_completed()
            
            execution_time = time.time() - start_time
            logger.info(f"Task completed: {task.task_id} in {execution_time:.2f}s")
            
            return TaskExecutionResult(
                task_id=task.task_id,
                success=success,
                result=result,
                strategy_used=strategy_name,
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error processing task {task.task_id}: {e}", exc_info=True)
            
            # Guardar error en memoria
            if self.episodic_memory:
                self.episodic_memory.add(
                    content=f"Failed task: {task.description} - Error: {str(e)}",
                    importance=0.9,
                    metadata={"task_id": task.task_id, "error": str(e)}
                )
            
            # Actualizar estado
            if self.task_manager:
                self.task_manager.mark_task_failed(task.task_id, str(e))
            
            # Registrar métricas
            if self.metrics_manager:
                self.metrics_manager.record_task_failed()
            
            return TaskExecutionResult(
                task_id=task.task_id,
                success=False,
                result={"status": "failed", "error": str(e)},
                strategy_used=strategy_name,
                execution_time=execution_time,
                error=e
            )
    
    async def process_task_queue(
        self,
        tasks: list,
        agent: Optional[Any] = None,
        max_concurrent: int = 5
    ) -> list[TaskExecutionResult]:
        """
        Procesar múltiples tareas con límite de concurrencia.
        
        Args:
            tasks: Lista de tareas a procesar
            agent: Referencia al agente
            max_concurrent: Número máximo de tareas concurrentes
            
        Returns:
            Lista de resultados
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(task):
            async with semaphore:
                return await self.process_task(task, agent)
        
        tasks_coroutines = [process_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
        
        # Filtrar excepciones
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task processing exception: {result}", exc_info=True)
            else:
                valid_results.append(result)
        
        return valid_results
