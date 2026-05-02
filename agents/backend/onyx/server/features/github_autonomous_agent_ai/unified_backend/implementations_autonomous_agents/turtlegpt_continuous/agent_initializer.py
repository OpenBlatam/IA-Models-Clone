"""
Agent Initializer Module
========================

Inicialización completa y estructurada del agente.
Centraliza toda la lógica de inicialización de componentes.
"""

import logging
from typing import Dict, Any, Optional

from .component_factory import ComponentFactory
from .config_validator import ConfigValidator
from .models import ContinuousAgentConfig
from ..common.tools import ToolRegistry
from .strategy_manager import StrategyManager, create_strategy_manager

logger = logging.getLogger(__name__)


class AgentInitializer:
    """
    Inicializador completo del agente.
    
    Centraliza toda la lógica de inicialización de componentes,
    estrategias, managers y servicios.
    """
    
    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[Dict[str, Any]] = None,
        agent_config: Optional[ContinuousAgentConfig] = None
    ):
        """
        Inicializar inicializador.
        
        Args:
            name: Nombre del agente
            api_key: API key de OpenRouter
            tool_registry: Registro de herramientas
            config: Configuración adicional
            agent_config: Configuración del agente continuo
        """
        self.name = name
        self.api_key = api_key
        self.tool_registry = tool_registry or ToolRegistry()
        self.agent_config = agent_config or ContinuousAgentConfig()
        
        # Validar y normalizar configuración
        self.validated_config = ConfigValidator.merge_with_defaults(config)
        
        # Componentes inicializados
        self.components: Dict[str, Any] = {}
        self.strategies: Dict[str, Any] = {}
        self.managers: Dict[str, Any] = {}
        self.strategy_manager: Optional[StrategyManager] = None
    
    def initialize_core_components(self) -> None:
        """Inicializar componentes core del agente."""
        logger.debug("Initializing core components")
        
        # Cliente LLM
        llm_client_raw = ComponentFactory.create_llm_client(api_key=self.api_key)
        self.components["llm_client"] = llm_client_raw
        
        # Sistemas de memoria
        episodic_memory, semantic_memory = ComponentFactory.create_memory_systems()
        self.components["episodic_memory"] = episodic_memory
        self.components["semantic_memory"] = semantic_memory
        
        # Task manager
        task_manager = ComponentFactory.create_task_manager(self.agent_config)
        self.components["task_manager"] = task_manager
        
        logger.debug("Core components initialized")
    
    def initialize_strategies(self, agent: Any) -> None:
        """
        Inicializar estrategias de papers.
        
        Args:
            agent: Referencia al agente
        """
        logger.debug("Initializing strategies")
        
        strategies = ComponentFactory.create_strategies(
            self.components["llm_client"],
            self.tool_registry,
            agent,
            self.validated_config
        )
        
        self.strategies = strategies
        
        # Crear strategy manager
        enabled_flags = {
            "react_enabled": strategies.get("react_enabled", False),
            "lats_enabled": strategies.get("lats_enabled", False),
            "tot_enabled": strategies.get("tot_enabled", False),
            "tom_enabled": strategies.get("tom_enabled", False),
            "personality_enabled": strategies.get("personality_enabled", False),
            "toolformer_enabled": strategies.get("toolformer_enabled", False)
        }
        
        self.strategy_manager = create_strategy_manager(strategies, enabled_flags)
        
        logger.debug(f"Strategies initialized: {len([s for s in strategies.values() if s is not None])} active")
    
    def initialize_managers(self) -> None:
        """Inicializar managers del agente."""
        logger.debug("Initializing managers")
        
        managers = ComponentFactory.create_managers(
            self.components["llm_client"],
            self.components["episodic_memory"],
            self.components["semantic_memory"],
            self.components["task_manager"],
            self.agent_config,
            self.validated_config
        )
        
        self.managers = managers
        logger.debug(f"Managers initialized: {len(managers)} managers")
    
    def get_autonomy_level(self) -> Any:
        """
        Obtener nivel de autonomía.
        
        Returns:
            Nivel de autonomía
        """
        return ComponentFactory.get_autonomy_level(self.validated_config)
    
    def get_all_components(self) -> Dict[str, Any]:
        """
        Obtener todos los componentes inicializados.
        
        Returns:
            Dict con todos los componentes
        """
        return {
            **self.components,
            "strategies": self.strategies,
            "managers": self.managers,
            "strategy_manager": self.strategy_manager,
            "validated_config": self.validated_config,
            "autonomy_level": self.get_autonomy_level()
        }


def create_agent_initializer(
    name: str,
    api_key: Optional[str] = None,
    tool_registry: Optional[ToolRegistry] = None,
    config: Optional[Dict[str, Any]] = None,
    agent_config: Optional[ContinuousAgentConfig] = None
) -> AgentInitializer:
    """
    Factory function para crear AgentInitializer.
    
    Args:
        name: Nombre del agente
        api_key: API key de OpenRouter
        tool_registry: Registro de herramientas
        config: Configuración adicional
        agent_config: Configuración del agente continuo
        
    Returns:
        Instancia de AgentInitializer
    """
    return AgentInitializer(
        name=name,
        api_key=api_key,
        tool_registry=tool_registry,
        config=config,
        agent_config=agent_config
    )


