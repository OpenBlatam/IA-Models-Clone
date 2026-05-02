"""
Component Factory Module
========================

Factory para crear e inicializar todos los componentes del agente.
Centraliza la lógica de inicialización.
"""

import logging
from typing import Dict, Any, Optional

from ..common.tools import ToolRegistry
from ..common.memory import EpisodicMemory, SemanticMemory
from .openrouter_client import OpenRouterTruthGPTClient
from .paper_strategies import (
    ReactStrategy,
    LATSStrategy,
    TreeOfThoughtsStrategy,
    TheoryOfMindStrategy,
    PersonalityStrategy,
    ToolformerStrategy
)
from .task_manager import TaskManager
from .reflection_planner import ReflectionPlanner
from .metrics_manager import MetricsManager
from .maintenance_manager import MaintenanceManager
from .callback_manager import CallbackManager
from .strategy_selector import StrategySelector
from .learning_manager import LearningManager
from .agi_capabilities_manager import AGICapabilitiesManager
from .models import ContinuousAgentConfig

logger = logging.getLogger(__name__)


class ComponentFactory:
    """
    Factory para crear componentes del agente.
    
    Centraliza la creación e inicialización de todos los componentes
    para mantener el código del agente principal limpio.
    """
    
    @staticmethod
    def create_llm_client(api_key: Optional[str] = None) -> OpenRouterTruthGPTClient:
        """Crear cliente LLM."""
        return OpenRouterTruthGPTClient(api_key=api_key)
    
    @staticmethod
    def create_memory_systems() -> Tuple[EpisodicMemory, SemanticMemory]:
        """Crear sistemas de memoria."""
        return EpisodicMemory(), SemanticMemory()
    
    @staticmethod
    def create_strategies(
        llm_client: OpenRouterTruthGPTClient,
        tool_registry: ToolRegistry,
        agent: Any,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Crear todas las estrategias de papers.
        
        Returns:
            Dict con todas las estrategias creadas
        """
        config = config or {}
        
        # Flags de habilitación
        react_enabled = config.get("react_enabled", True)
        lats_enabled = config.get("lats_enabled", True)
        tot_enabled = config.get("tot_enabled", True)
        tom_enabled = config.get("tom_enabled", True)
        personality_enabled = config.get("personality_enabled", False)
        toolformer_enabled = config.get("toolformer_enabled", True)
        
        strategies = {
            "react_strategy": (
                ReactStrategy(llm_client, tool_registry, agent)
                if react_enabled else None
            ),
            "lats_strategy": (
                LATSStrategy(llm_client, agent)
                if lats_enabled else None
            ),
            "tot_strategy": (
                TreeOfThoughtsStrategy(
                    llm_client,
                    agent,
                    strategy=config.get("tot_strategy", "bfs")
                )
                if tot_enabled else None
            ),
            "tom_strategy": (
                TheoryOfMindStrategy(llm_client)
                if tom_enabled else None
            ),
            "personality_strategy": (
                PersonalityStrategy(config.get("personality_profile"))
                if personality_enabled else None
            ),
            "toolformer_strategy": (
                ToolformerStrategy(llm_client)
                if toolformer_enabled else None
            ),
            # Flags
            "react_enabled": react_enabled,
            "lats_enabled": lats_enabled,
            "tot_enabled": tot_enabled,
            "tom_enabled": tom_enabled,
            "personality_enabled": personality_enabled,
            "toolformer_enabled": toolformer_enabled,
        }
        
        return strategies
    
    @staticmethod
    def create_managers(
        llm_client: OpenRouterTruthGPTClient,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        task_manager: TaskManager,
        agent_config: ContinuousAgentConfig,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Crear todos los managers especializados.
        
        Returns:
            Dict con todos los managers creados
        """
        config = config or {}
        
        reflection_threshold = config.get("reflection_threshold", 5)
        planning_horizon = config.get("planning_horizon", 3)
        
        reflection_planner = ReflectionPlanner(
            llm_client=llm_client,
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory,
            reflection_threshold=reflection_threshold,
            planning_horizon=planning_horizon
        )
        
        metrics_manager = MetricsManager()
        
        maintenance_manager = MaintenanceManager(
            task_manager=task_manager,
            reflection_planner=reflection_planner,
            maintenance_interval_seconds=agent_config.maintenance_interval_seconds
        )
        
        callback_manager = CallbackManager()
        
        # Strategy Selector
        strategy_selector = StrategySelector(
            react_enabled=config.get("react_enabled", True),
            lats_enabled=config.get("lats_enabled", True),
            tot_enabled=config.get("tot_enabled", True),
            react_threshold=config.get("react_threshold", 7),
            tot_threshold=config.get("tot_threshold", 8),
            lats_threshold=config.get("lats_threshold", 9)
        )
        
        # Learning Manager
        learning_manager = LearningManager(
            learning_enabled=config.get("learning_enabled", True),
            performance_threshold=config.get("performance_threshold", 0.6),
            error_threshold=config.get("error_threshold", 3)
        )
        
        # AGI Capabilities Manager
        agi_manager = AGICapabilitiesManager()
        
        return {
            "reflection_planner": reflection_planner,
            "metrics_manager": metrics_manager,
            "maintenance_manager": maintenance_manager,
            "callback_manager": callback_manager,
            "strategy_selector": strategy_selector,
            "learning_manager": learning_manager,
            "agi_manager": agi_manager
        }
    
    @staticmethod
    def create_task_manager(
        agent_config: ContinuousAgentConfig
    ) -> TaskManager:
        """Crear gestor de tareas."""
        return TaskManager(
            max_concurrent_tasks=agent_config.max_concurrent_tasks
        )
    
    @staticmethod
    def get_autonomy_level(config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Obtener nivel de autonomía.
        
        Returns:
            AutonomyLevel o string como fallback
        """
        config = config or {}
        try:
            from ..llm_to_autonomous.llm_to_autonomous import AutonomyLevel
            autonomy_str = config.get("autonomy_level", "fully_autonomous")
            if isinstance(autonomy_str, str):
                return getattr(
                    AutonomyLevel,
                    autonomy_str.upper(),
                    AutonomyLevel.FULLY_AUTONOMOUS
                )
            return autonomy_str
        except (ImportError, AttributeError):
            return "fully_autonomous"
