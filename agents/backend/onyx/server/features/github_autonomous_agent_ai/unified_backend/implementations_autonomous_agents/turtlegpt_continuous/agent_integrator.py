"""
Agent Integrator Module
=======================

Integra todos los componentes del agente de forma cohesiva.
Proporciona una capa de abstracción para la inicialización y configuración.
"""

import logging
from typing import Dict, Any, Optional

from .component_factory import ComponentFactory
from .config_validator import ConfigValidator
from .event_system import EventBus, EventType
from .health_monitor import HealthMonitor
from .models import ContinuousAgentConfig
from ..common.tools import ToolRegistry

logger = logging.getLogger(__name__)


class AgentIntegrator:
    """
    Integrador de componentes del agente.
    
    Facilita la integración de todos los componentes y proporciona
    una interfaz unificada para la inicialización.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[Dict[str, Any]] = None,
        agent_config: Optional[ContinuousAgentConfig] = None
    ):
        """
        Inicializar integrador.
        
        Args:
            api_key: API key de OpenRouter
            tool_registry: Registro de herramientas
            config: Configuración del agente
            agent_config: Configuración continua del agente
        """
        self.api_key = api_key
        self.tool_registry = tool_registry or ToolRegistry()
        self.agent_config = agent_config or ContinuousAgentConfig()
        
        # Validar configuración
        self.validated_config = ConfigValidator.merge_with_defaults(config)
        
        # Crear componentes base
        self.llm_client = ComponentFactory.create_llm_client(api_key=api_key)
        self.episodic_memory, self.semantic_memory = ComponentFactory.create_memory_systems()
        self.task_manager = ComponentFactory.create_task_manager(self.agent_config)
        
        # Crear sistema de eventos
        self.event_bus = EventBus()
        
        # Crear monitor de salud
        self.health_monitor = HealthMonitor()
        
        # Crear estrategias (se necesita el agente, se hará después)
        self.strategies = None
        
        # Crear managers
        self.managers = ComponentFactory.create_managers(
            self.llm_client,
            self.episodic_memory,
            self.semantic_memory,
            self.task_manager,
            self.agent_config,
            self.validated_config
        )
        
        logger.info("Agent integrator initialized")
    
    def create_strategies(self, agent: Any) -> Dict[str, Any]:
        """
        Crear estrategias (requiere referencia al agente).
        
        Args:
            agent: Instancia del agente
            
        Returns:
            Dict con estrategias creadas
        """
        self.strategies = ComponentFactory.create_strategies(
            self.llm_client,
            self.tool_registry,
            agent,
            self.validated_config
        )
        return self.strategies
    
    def setup_event_subscriptions(self, agent: Any):
        """
        Configurar suscripciones a eventos.
        
        Args:
            agent: Instancia del agente
        """
        # Suscribir métricas a eventos de tareas
        async def on_task_completed(event):
            agent.metrics_manager.record_task_completed()
        
        async def on_task_failed(event):
            agent.metrics_manager.record_task_failed()
        
        self.event_bus.subscribe(EventType.TASK_COMPLETED, on_task_completed)
        self.event_bus.subscribe(EventType.TASK_FAILED, on_task_failed)
        
        logger.debug("Event subscriptions configured")
    
    def setup_health_checks(self, agent: Any):
        """
        Configurar health checks para componentes.
        
        Args:
            agent: Instancia del agente
        """
        # Health check para task manager
        def task_manager_health():
            stats = agent.task_manager.get_stats()
            return {
                "is_active": True,
                "metrics": {
                    "queue_size": stats.get("queue_size", 0),
                    "active_tasks": stats.get("active_tasks", 0)
                }
            }
        
        # Health check para LLM client
        def llm_client_health():
            return {
                "is_active": agent.llm_client is not None,
                "has_api_key": bool(self.api_key)
            }
        
        # Health check para memory systems
        def memory_health():
            return {
                "is_active": True,
                "metrics": {
                    "episodic_count": len(agent.episodic_memory.memories),
                    "semantic_count": len(agent.semantic_memory.facts)
                }
            }
        
        self.health_monitor.register_component("task_manager", task_manager_health)
        self.health_monitor.register_component("llm_client", llm_client_health)
        self.health_monitor.register_component("memory", memory_health)
        
        logger.debug("Health checks configured")
    
    def get_all_components(self) -> Dict[str, Any]:
        """
        Obtener todos los componentes integrados.
        
        Returns:
            Dict con todos los componentes
        """
        return {
            "llm_client": self.llm_client,
            "episodic_memory": self.episodic_memory,
            "semantic_memory": self.semantic_memory,
            "task_manager": self.task_manager,
            "event_bus": self.event_bus,
            "health_monitor": self.health_monitor,
            "strategies": self.strategies,
            "managers": self.managers,
            "tool_registry": self.tool_registry,
            "agent_config": self.agent_config,
            "validated_config": self.validated_config
        }
