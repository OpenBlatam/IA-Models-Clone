"""
Continuous Generative Agent with OpenRouter/TruthGPT
====================================================

Basado en múltiples papers de investigación:

1. "Generative Agents: Interactive Simulacra of Human Behavior" (Stanford, 2023)
   - Memoria episódica y semántica
   - Reflexión periódica sobre experiencias
   - Planificación basada en memoria
   - Funcionamiento continuo 24/7

2. "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
   - Intercalación de razonamiento y acción
   - Ciclo Thought → Action → Observation
   - Integración con herramientas

3. "Language Agent Tree Search Unifies Reasoning, Acting, and Planning" (LATS)
   - Búsqueda en árbol con evaluación LLM
   - Unificación de reasoning, acting y planning
   - Búsqueda de caminos óptimos

4. "From LLM Reasoning to Autonomous AI Agents"
   - Autonomía progresiva
   - Pipeline de razonamiento a acción
   - Auto-monitoreo y adaptación

5. "AI Autonomy: Self-Initiated Open-World Continual Learning"
   - Aprendizaje autónomo iniciado por el agente
   - Identificación de oportunidades de aprendizaje
   - Adaptación continua

6. "Tree of Thoughts: Deliberate Problem Solving" (ToT)
   - Razonamiento basado en árbol de pensamientos
   - Búsqueda BFS/DFS de soluciones
   - Evaluación de estados intermedios

7. "Autonomous Agents Modelling Other Agents: A Comprehensive Survey" (Theory of Mind)
   - Modelado de estados mentales de otros agentes
   - Predicción de acciones
   - Seguimiento de intenciones

8. "Personality-Driven Decision-Making in LLM-Based Autonomous"
   - Decisiones influenciadas por personalidad
   - Estados emocionales y sesgos
   - Expresión de personalidad contextual

9. "Toolformer: Language Models Can Teach Themselves to Use Tools"
   - Aprendizaje autónomo de herramientas
   - Auto-supervisión para uso de herramientas
   - Filtrado basado en pérdida

10. "Sparks of Artificial General Intelligence"
    - Capacidades emergentes de AGI
    - Razonamiento multi-modal
    - Uso de herramientas y generación de código

Este agente combina estos conceptos con OpenRouter/TruthGPT como LLM,
funcionando continuamente 24/7 hasta detención manual.
"""

import asyncio
import logging
import signal
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.tools import ToolRegistry
from ..common.memory import EpisodicMemory, SemanticMemory
from .openrouter_client import OpenRouterTruthGPTClient
from .models import (
    TaskStatus,
    AgentTask,
    AgentMetrics,
    ContinuousAgentConfig
)
from .component_factory import ComponentFactory
from .config_validator import ConfigValidator
from .task_processor import TaskProcessor, TaskExecutionResult
from .llm_service import LLMService, LLMCallTracker
from .status_builder import StatusBuilder, build_agent_status
from .event_publisher import EventPublisher, create_event_publisher
from .signal_handler import SignalHandler, SignalType, create_signal_handler
from .agent_lifecycle import AgentLifecycle, create_agent_lifecycle
from .startup_logger import StartupLogger, create_startup_logger
from .hook_manager import HookManager, HookType, create_hook_manager
from .resilient_operations import CircuitBreaker, CircuitState, create_circuit_breaker, resilient_call, resilient_call_async
from .tool_executor import ToolExecutor, create_tool_executor
from .memory_operations import MemoryOperations, create_memory_operations

logger = logging.getLogger(__name__)


class TurtleGPTContinuousAgent(BaseAgent):
    """
    Agente generativo continuo que combina 10 papers de investigación.
    
    Papers integrados:
    1. Generative Agents: Memoria episódica/semántica, reflexión, planificación
    2. ReAct: Intercalación de razonamiento y acción (Thought → Action → Observation)
    3. LATS: Búsqueda en árbol para reasoning/acting/planning unificados
    4. LLM to Autonomous: Autonomía progresiva y pipeline razonamiento→acción
    5. Self-Initiated Learning: Aprendizaje autónomo y adaptación continua
    6. Tree of Thoughts: Búsqueda deliberada en árbol de pensamientos
    7. Theory of Mind: Modelado de otros agentes y predicción de acciones
    8. Personality-Driven: Decisiones basadas en rasgos de personalidad
    9. Toolformer: Auto-aprendizaje de herramientas
    10. Sparks of AGI: Capacidades tipo AGI y razonamiento avanzado
    
    El agente funciona continuamente 24/7 hasta detención manual.
    Usa OpenRouter/TruthGPT como LLM para razonamiento, planificación y reflexión.
    Selecciona automáticamente la estrategia más apropiada según la prioridad de la tarea:
    - Prioridad ≥ 9: LATS (búsqueda en árbol avanzada)
    - Prioridad ≥ 8: Tree of Thoughts (razonamiento deliberado)
    - Prioridad ≥ 7: ReAct (reasoning-acting interleaved)
    - Otra: Método estándar (Generative Agents)
    """
    
    def __init__(
        self,
        name: str = "TurtleGPTContinuousAgent",
        api_key: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[Dict[str, Any]] = None,
        agent_config: Optional[ContinuousAgentConfig] = None
    ):
        """
        Inicializar agente continuo basado en papers.
        
        Args:
            name: Nombre del agente
            api_key: API key de OpenRouter
            tool_registry: Registro de herramientas
            config: Configuración adicional
            agent_config: Configuración del agente continuo
        """
        super().__init__(name, config=config)
        self.tool_registry = tool_registry or ToolRegistry()
        self.agent_config = agent_config or ContinuousAgentConfig()
        
        # Validar y normalizar configuración
        validated_config = ConfigValidator.merge_with_defaults(config)
        
        # Crear componentes usando factory
        llm_client_raw = ComponentFactory.create_llm_client(api_key=api_key)
        self.episodic_memory, self.semantic_memory = ComponentFactory.create_memory_systems()
        self.task_manager = ComponentFactory.create_task_manager(self.agent_config)
        
        # Crear servicio LLM encapsulado
        self.llm_service = LLMService(
            llm_client=llm_client_raw,
            metrics_manager=None,  # Se asignará después
            enable_retry=True,
            enable_timeout=True
        )
        self.llm_client = llm_client_raw  # Mantener referencia directa para compatibilidad
        self.llm_tracker = LLMCallTracker(self.llm_service)
        
        # Flags de habilitación usando config_helper
        self.react_enabled = get_bool_config(validated_config, "react_enabled", DEFAULT_REACT_ENABLED)
        self.lats_enabled = get_bool_config(validated_config, "lats_enabled", DEFAULT_LATS_ENABLED)
        self.tot_enabled = get_bool_config(validated_config, "tot_enabled", DEFAULT_TOT_ENABLED)
        self.tom_enabled = get_bool_config(validated_config, "tom_enabled", DEFAULT_TOM_ENABLED)
        self.personality_enabled = get_bool_config(validated_config, "personality_enabled", DEFAULT_PERSONALITY_ENABLED)
        self.toolformer_enabled = get_bool_config(validated_config, "toolformer_enabled", DEFAULT_TOOLFORMER_ENABLED)
        self.learning_enabled = get_bool_config(validated_config, "learning_enabled", DEFAULT_LEARNING_ENABLED)
        
        # Crear estrategias usando factory
        strategies = ComponentFactory.create_strategies(
            self.llm_client,
            self.tool_registry,
            self,
            validated_config
        )
        
        # Asignar estrategias individuales
        self.react_strategy = strategies["react_strategy"]
        self.lats_strategy = strategies["lats_strategy"]
        self.tot_strategy = strategies["tot_strategy"]
        self.tom_strategy = strategies["tom_strategy"]
        self.personality_strategy = strategies["personality_strategy"]
        self.toolformer_strategy = strategies["toolformer_strategy"]
        
        # Crear gestor de estrategias
        enabled_flags = {
            "react_enabled": strategies.get("react_enabled", False),
            "lats_enabled": strategies.get("lats_enabled", False),
            "tot_enabled": strategies.get("tot_enabled", False),
            "tom_enabled": strategies.get("tom_enabled", False),
            "personality_enabled": strategies.get("personality_enabled", False),
            "toolformer_enabled": strategies.get("toolformer_enabled", False)
        }
        self.strategy_manager = create_strategy_manager(strategies, enabled_flags)
        
        # Autonomy level usando factory
        self.autonomy_level = ComponentFactory.get_autonomy_level(validated_config)
        
        # Crear managers usando factory
        managers = ComponentFactory.create_managers(
            self.llm_client,
            self.episodic_memory,
            self.semantic_memory,
            self.task_manager,
            self.agent_config,
            validated_config
        )
        self.reflection_planner = managers["reflection_planner"]
        self.metrics_manager = managers["metrics_manager"]
        self.maintenance_manager = managers["maintenance_manager"]
        self.callback_manager = managers["callback_manager"]
        self.strategy_selector = managers["strategy_selector"]
        self.learning_manager = managers["learning_manager"]
        self.agi_manager = managers["agi_manager"]
        
        # Actualizar servicio LLM con metrics_manager
        self.llm_service.metrics_manager = self.metrics_manager
        
        # Manager de tareas asíncronas
        self.async_task_manager = AsyncTaskManager()
        
        # Ejecutor de tareas
        self.task_executor = create_task_executor(
            async_task_manager=self.async_task_manager,
            max_concurrent=5
        )
        
        # Constructor de contexto de memoria
        self.memory_context_builder = create_memory_context_builder(
            self.episodic_memory,
            self.semantic_memory,
            validated_config
        )
        
        # Operaciones de memoria
        self.memory_operations = create_memory_operations(
            self.episodic_memory,
            self.semantic_memory
        )
        
        # Operaciones del agente
        self.agent_operations = AgentOperations(
            llm_client=self.llm_client,
            memory_context_builder=self.memory_context_builder,
            metrics_manager=self.metrics_manager,
            state=self.state
        )
        
        # Estado del agente
        self.is_running = False
        self.should_stop = False
        
        # Crear publisher de eventos
        self.event_publisher = create_event_publisher(
            self.event_bus,
            source=self.name
        )
        
        # Crear gestor de señales
        self.signal_handler = create_signal_handler()
        self.signal_handler.register_stop_handler(self._on_stop_signal)
        
        # Configurar manejo de señales (legacy, ahora usa signal_handler)
        self._setup_signal_handlers()
        
        # Crear validador de tareas
        self.task_validator = create_task_validator()
        
        # Crear configurador de loop
        self.loop_configurator = create_loop_configurator(
            agent_config=self.agent_config,
            reflection_planner=self.reflection_planner,
            learning_manager=self.learning_manager,
            metrics_manager=self.metrics_manager,
            maintenance_manager=self.maintenance_manager
        )
        
        # Crear coordinador de loop (después de inicializar todos los managers)
        self.loop_coordinator = self.loop_configurator.create_loop_coordinator(
            process_task_queue_handler=self._process_task_queue,
            identify_learning_opportunities_handler=self._identify_learning_opportunities
        )
        
        # Crear logger de inicio
        self.startup_logger = create_startup_logger(self.name)
        
        # Crear gestor de ciclo de vida
        self.lifecycle = create_agent_lifecycle(
            agent_name=self.name,
            start_callback=self._on_lifecycle_start,
            stop_callback=self._on_lifecycle_stop,
            cleanup_callback=self._on_lifecycle_cleanup
        )
        
        # Crear gestor de hooks
        self.hook_manager = create_hook_manager()
        
        # Crear gestor de estado
        self.state_manager = create_state_manager(self.state)
        
        # Crear tracker de métricas
        self.metrics_tracker = create_metrics_tracker(self.metrics_manager)
        
        # Crear gestor de envío de tareas
        self.task_submitter = create_task_submitter(
            task_validator=self.task_validator,
            task_manager=self.task_manager,
            event_publisher=self.event_publisher,
            metrics_tracker=self.metrics_tracker
        )
        
        # Crear registro de componentes
        self.component_registry = create_component_registry()
        
        # Crear programador de tareas periódicas
        self.periodic_scheduler = create_periodic_scheduler()
        
        # Registrar componentes principales en el registro
        self._register_components()
        
        logger.info(f"Continuous Autonomous Agent '{name}' initialized (based on research papers)")
    
    def _register_components(self) -> None:
        """
        Registrar todos los componentes principales en el registro.
        
        Facilita el acceso centralizado a componentes.
        """
        # Componentes core
        self.component_registry.register("llm_client", self.llm_client)
        self.component_registry.register("llm_service", self.llm_service)
        self.component_registry.register("episodic_memory", self.episodic_memory)
        self.component_registry.register("semantic_memory", self.semantic_memory)
        self.component_registry.register("task_manager", self.task_manager)
        self.component_registry.register("tool_registry", self.tool_registry)
        
        # Managers
        self.component_registry.register("metrics_manager", self.metrics_manager)
        self.component_registry.register("reflection_planner", self.reflection_planner)
        self.component_registry.register("maintenance_manager", self.maintenance_manager)
        self.component_registry.register("callback_manager", self.callback_manager)
        self.component_registry.register("strategy_selector", self.strategy_selector)
        self.component_registry.register("learning_manager", self.learning_manager)
        self.component_registry.register("agi_manager", self.agi_manager)
        
        # Servicios y operaciones
        self.component_registry.register("memory_operations", self.memory_operations)
        self.component_registry.register("agent_operations", self.agent_operations)
        self.component_registry.register("task_processor", self.task_processor)
        self.component_registry.register("task_executor", self.task_executor)
        self.component_registry.register("tool_executor", self.tool_executor)
        self.component_registry.register("state_manager", self.state_manager)
        self.component_registry.register("metrics_tracker", self.metrics_tracker)
        self.component_registry.register("task_submitter", self.task_submitter)
        
        # Sistema de eventos y otros
        self.component_registry.register("event_publisher", self.event_publisher)
        self.component_registry.register("health_monitor", self.health_monitor)
        self.component_registry.register("hook_manager", self.hook_manager)
        
        logger.debug(f"Registered {self.component_registry.count()} components in registry")
    
    def _setup_signal_handlers(self):
        """
        Configurar manejadores de señales para detención graceful.
        
        Ahora usa SignalHandler para gestión centralizada.
        """
        # El signal_handler ya está configurado en __init__
        # Este método se mantiene por compatibilidad
        pass
    
    def _on_stop_signal(self) -> None:
        """Callback para señales de detención."""
        logger.info("Stop signal received, setting should_stop flag")
        self.should_stop = True
        self.is_running = False
    
    async def _think_async(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Pensar sobre una tarea usando TurtleGPT (versión async).
        
        Args:
            task: Descripción de la tarea
            context: Contexto adicional
        
        Returns:
            Resultado del pensamiento
        """
        # Usar StateManager para transición de estado
        self.state_manager.transition_to(AgentStatus.THINKING)
        self.state_manager.set_current_task(task)
        
        # Construir prompt para pensar usando PromptBuilder
        thinking_prompt = PromptBuilder.build_thinking_prompt(
            task=task,
            context=context,
            memory_context=None  # Podría agregarse contexto de memoria relevante
        )
        
        try:
            # Usar servicio LLM encapsulado
            response = await self.llm_service.generate_text(
                prompt=thinking_prompt,
                max_tokens=2000,
                temperature=0.7
            )
            
            generated_text = response.get("generated_text", "")
            metadata = response.get("_metadata", {})
            tokens_used = metadata.get("tokens_used", 0)
            elapsed = metadata.get("response_time", 0.0)
            
            result = {
                "task": task,
                "reasoning": generated_text,
                "tokens_used": tokens_used,
                "response_time": elapsed,
                "context": context or {}
            }
            
            self.state_manager.add_step("thinking", result)
            return result
            
        except Exception as e:
            logger.error(f"Error in think: {e}", exc_info=True)
            self.metrics_manager.record_error()
            self.state.status = AgentStatus.ERROR
            self.state.add_step("error", {"error": str(e), "task": task})
            raise
    
    def think(self, observation: str, context: Optional[Dict] = None) -> str:
        """
        Pensar sobre una observación (implementación requerida por BaseAgent).
        
        Args:
            observation: Observación actual
            context: Contexto adicional
        
        Returns:
            Pensamiento como string
        """
        # Para compatibilidad con BaseAgent, ejecutamos async en sync
        try:
            result = asyncio.run(self._think_async(observation, context))
            return result.get("reasoning", f"Thinking about: {observation}")
        except Exception as e:
            logger.error(f"Error in think: {e}", exc_info=True)
            return f"Error thinking: {str(e)}"
    
    def act(self, action: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ejecutar una acción.
        
        Args:
            action: Acción a ejecutar
            context: Contexto adicional
        
        Returns:
            Resultado de la acción
        """
        # Usar StateManager para transición de estado
        self.state_manager.transition_to(AgentStatus.ACTING)
        
        action_type = action.get("type", "process")
        action_description = action.get("description", "")
        
        try:
            if action_type == "tool_call":
                # Ejecutar herramienta usando ToolExecutor
                tool_name = action.get("tool")
                tool_args = action.get("args", {})
                
                try:
                    execution_result = self.tool_executor.execute_tool(
                        tool_name=tool_name,
                        args=tool_args,
                        validate_args=True
                    )
                    result = execution_result.get("result", {})
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
                    result = {"error": str(e), "tool": tool_name}
                
            elif action_type == "llm_call":
                # Llamada adicional al LLM usando servicio encapsulado
                prompt = action.get("prompt", action_description)
                response = asyncio.run(self.llm_service.generate_text(
                    prompt=prompt,
                    max_tokens=action.get("max_tokens", 2000),
                    temperature=action.get("temperature", 0.7)
                ))
                
                metadata = response.get("_metadata", {})
                tokens_used = metadata.get("tokens_used", 0)
                
                result = {
                    "response": response.get("generated_text", ""),
                    "tokens_used": tokens_used
                }
                
            else:
                # Acción genérica
                result = {
                    "status": "executed",
                    "action": action,
                    "result": "Action completed"
                }
            
            self.state.add_step("action", result)
            return result
            
        except Exception as e:
            logger.error(f"Error in act: {e}", exc_info=True)
            self.metrics_tracker.track_error(error_type="act_error", context={"action": action})
            self.state_manager.transition_to(AgentStatus.ERROR)
            self.state_manager.add_step("error", {"error": str(e), "action": action})
            raise
    
    def observe(self, observation: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Procesar observación.
        
        Args:
            observation: Observación a procesar
            context: Contexto adicional
        
        Returns:
            Observación procesada
        """
        # Usar StateManager para transición y paso
        self.state_manager.transition_to(AgentStatus.OBSERVING)
        
        processed = {
            "observation": str(observation),
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        self.state_manager.add_step("observation", processed)
        return processed
    
    def _format_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Formatear contexto para el prompt usando utils."""
        from .utils import format_context
        return format_context(context)
    
    async def _react_cycle(self, task: str) -> Dict[str, Any]:
        """Ciclo ReAct usando estrategia refactorizada."""
        if not hasattr(self, 'strategy_manager') or not self.strategy_manager.is_available("react"):
            return {"status": "react_disabled"}
        return await self.react_strategy.execute(task)
    
    async def _lats_search(self, goal: str) -> Dict[str, Any]:
        """Búsqueda LATS usando estrategia refactorizada."""
        if not hasattr(self, 'strategy_manager') or not self.strategy_manager.is_available("lats"):
            return {"status": "lats_disabled"}
        return await self.lats_strategy.execute(goal)
    
    async def _identify_learning_opportunities(self) -> List[Dict[str, Any]]:
        """Identificar oportunidades de aprendizaje usando LearningManager."""
        recent_tasks = self.task_manager.get_recent_tasks(count=5)
        metrics = self.metrics_manager.get_metrics()
        
        return self.learning_manager.identify_opportunities(
            recent_tasks=recent_tasks,
            semantic_memory=self.semantic_memory,
            metrics=metrics
        )
    
    async def start(self) -> None:
        """
        Iniciar el agente en modo continuo (24/7).
        
        El agente seguirá funcionando hasta que se llame a stop() o se reciba una señal.
        """
        if self.is_running:
            logger.warning("Agent is already running")
            return
        
        self.is_running = True
        self.should_stop = False
        self.metrics_manager.start_tracking()
        
        # Loggear inicio usando StartupLogger
        papers = [
            "Generative Agents", "ReAct", "LATS", "Self-Initiated Learning",
            "LLM to Autonomous", "ToT", "Theory of Mind", "Personality-Driven",
            "Toolformer", "Sparks of AGI"
        ]
        self.startup_logger.log_startup_banner(papers=papers)
        
        # Registrar hooks pre-start
        async def pre_start_publish():
            if hasattr(self, 'event_publisher'):
                await self.event_publisher.publish_agent_started()
        
        self.hook_manager.register_hook(HookType.PRE_START, pre_start_publish, priority=10)
        
        # Registrar hooks post-start
        def post_start_log():
            if hasattr(self, 'startup_logger'):
                self.startup_logger.log_ready()
        
        self.hook_manager.register_hook(HookType.POST_START, post_start_log, priority=10)
        
        # Ejecutar hooks pre-start
        await self.hook_manager.execute_hooks(HookType.PRE_START)
        
        # Iniciar programador de tareas periódicas
        await self.periodic_scheduler.start()
        
        # Usar lifecycle manager para gestionar el ciclo de vida
        await self.lifecycle.start(
            main_loop=self._main_loop,
            pre_start_hooks=[],  # Ya ejecutados arriba
            post_start_hooks=[]  # Se ejecutarán después
        )
        
        # Ejecutar hooks post-start
        await self.hook_manager.execute_hooks(HookType.POST_START)
    
    def _on_lifecycle_start(self) -> None:
        """Callback para inicio del ciclo de vida."""
        logger.debug("Lifecycle start callback executed")
    
    async def _on_lifecycle_stop(self) -> None:
        """Callback para detención del ciclo de vida."""
        # Ejecutar hooks pre-stop
        await self.hook_manager.execute_hooks(HookType.PRE_STOP)
        
        self.is_running = False
        self.should_stop = True
        
        # Detener coordinador de loop
        if hasattr(self, 'loop_coordinator'):
            self.loop_coordinator.stop()
        
        # Detener programador de tareas periódicas
        if hasattr(self, 'periodic_scheduler'):
            await self.periodic_scheduler.stop()
        
        # Cancelar tareas asíncronas
        if hasattr(self, 'async_task_manager'):
            self.async_task_manager.cancel_all()
        
        self.startup_logger.log_stopping()
        
        # Ejecutar hooks post-stop
        await self.hook_manager.execute_hooks(HookType.POST_STOP)
    
    async def _on_lifecycle_cleanup(self) -> None:
        """Callback para limpieza del ciclo de vida."""
        # Ejecutar hooks pre-cleanup
        await self.hook_manager.execute_hooks(HookType.PRE_CLEANUP)
        
        # Cerrar cliente LLM
        if hasattr(self, 'llm_client'):
            await self.llm_client.close()
        
        # Imprimir métricas finales
        if hasattr(self, 'metrics_manager'):
            self.metrics_manager.print_final_metrics()
        
        self.startup_logger.log_stopped()
        
        # Ejecutar hooks post-cleanup
        await self.hook_manager.execute_hooks(HookType.POST_CLEANUP)
    
    
    async def _main_loop(self) -> None:
        """
        Loop principal del agente usando LoopCoordinator.
        
        Delega la coordinación del loop al LoopCoordinator.
        """
        # Sincronizar estado inicial
        self.loop_coordinator.should_stop = self.should_stop
        
        # Callback para verificar si se debe detener y actualizar estado
        def should_stop_check():
            # Sincronizar estado
            self.loop_coordinator.should_stop = self.should_stop
            
            # Actualizar modo idle
            is_idle = self.task_manager.should_enter_idle_mode(self.agent_config.enable_idle_mode)
            self.loop_coordinator.set_idle_mode(is_idle)
            
            return self.should_stop
        
        # Ejecutar loop coordinado con callback de stop
        await self.loop_coordinator.run_loop(stop_callback=should_stop_check)
    
    async def _process_task_queue(self) -> None:
        """Procesar cola de tareas usando TaskExecutor."""
        # Get next tasks to process
        tasks_to_process = self.task_manager.get_next_tasks()
        
        if not tasks_to_process:
            return
        
        # Procesar tareas usando TaskExecutor
        async def process_task_wrapper(task: AgentTask):
            return await self._process_task(task)
        
        async def on_task_complete(task: AgentTask, result: Any):
            await self.callback_manager.execute_task_callbacks(task)
        
        async def on_task_error(task: AgentTask, error: Exception):
            await self.callback_manager.execute_error_callbacks(error)
        
        # Ejecutar tareas en background usando TaskExecutor
        for task in tasks_to_process:
            await self.task_executor.execute_task_background(
                task=task,
                processor=process_task_wrapper,
                on_complete=on_task_complete,
                on_error=on_task_error
            )
        
        # Limpiar tareas completadas
        self.task_executor.cleanup_completed()
        
        # Cleanup completed tasks del task_manager
        completed_tasks = self.task_manager.get_completed_tasks_for_cleanup()
        for task in completed_tasks:
            self.task_manager.mark_task_completed(task.task_id, task.result)
    
    async def _process_task(self, task: AgentTask) -> Any:
        """
        Procesar una tarea individual usando TaskProcessor.
        
        Delega toda la lógica de procesamiento al TaskProcessor.
        
        Returns:
            Resultado del procesamiento
        """
        try:
            result = await self.task_processor.process_task(task, agent=self)
            
            # Los callbacks se manejan en TaskExecutor
            return result
        except Exception as e:
            logger.error(f"Unexpected error in task processing: {e}", exc_info=True)
            # El error se manejará en TaskExecutor
            raise
    
    def submit_task(
        self,
        description: str,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enviar una tarea para procesamiento.
        
        Args:
            description: Descripción de la tarea
            priority: Prioridad (1-10, mayor = más prioridad)
            metadata: Metadatos adicionales
        
        Returns:
            ID de la tarea
            
        Raises:
            ValueError: Si la tarea no es válida
        """
        # Usar TaskSubmitter para envío centralizado
        return self.task_submitter.submit_task(
            description=description,
            priority=priority,
            metadata=metadata
        )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual del agente usando StatusBuilder.
        
        Returns:
            Dict con estado completo del agente
        """
        return build_agent_status(self)
    
    def stop(self) -> None:
        """Detener el agente."""
        logger.info("Stop requested for agent")
        self.should_stop = True
        self.is_running = False
        
        # Publicar evento de detención
        if hasattr(self, 'event_publisher'):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.event_publisher.publish_agent_stopped())
                else:
                    asyncio.run(self.event_publisher.publish_agent_stopped())
            except RuntimeError:
                asyncio.run(self.event_publisher.publish_agent_stopped())
        
        # Detener coordinador de loop
        if hasattr(self, 'loop_coordinator'):
            self.loop_coordinator.stop()
        
        # Cancelar tareas asíncronas
        if hasattr(self, 'async_task_manager'):
            self.async_task_manager.cancel_all()
        
        # Limpiar handlers de señales
        if hasattr(self, 'signal_handler'):
            self.signal_handler.cleanup()
    
    def set_task_callback(self, callback: Callable[[AgentTask], None]) -> None:
        """Establecer callback para cuando se complete una tarea."""
        self.callback_manager.add_task_callback(callback)
    
    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Establecer callback para cuando ocurra un error."""
        self.callback_manager.add_error_callback(callback)



