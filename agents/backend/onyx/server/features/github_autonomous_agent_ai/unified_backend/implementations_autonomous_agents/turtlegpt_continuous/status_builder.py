"""
Status Builder Module
====================

Constructor centralizado del estado del agente.
Agrega información de todos los componentes de forma organizada.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class StatusBuilder:
    """
    Constructor del estado del agente.
    
    Agrega información de todos los componentes de forma organizada
    y estructurada.
    """
    
    def __init__(self, agent):
        """
        Inicializar constructor de estado.
        
        Args:
            agent: Referencia al agente
        """
        self.agent = agent
    
    def build_status(self) -> Dict[str, Any]:
        """
        Construir estado completo del agente.
        
        Returns:
            Dict con todo el estado del agente
        """
        status = {
            "name": self.agent.name,
            "is_running": getattr(self.agent, 'is_running', False),
            "should_stop": getattr(self.agent, 'should_stop', False),
            "timestamp": datetime.now().isoformat()
        }
        
        # Estado básico
        if hasattr(self.agent, 'state'):
            try:
                status["state"] = self.agent.state.to_dict()
            except Exception as e:
                logger.warning(f"Error getting state: {e}")
                status["state"] = {}
        
        # Métricas
        if hasattr(self.agent, 'metrics_manager'):
            try:
                status["metrics"] = self.agent.metrics_manager.get_metrics()
            except Exception as e:
                logger.warning(f"Error getting metrics: {e}")
                status["metrics"] = {}
        
        # Tareas
        status["tasks"] = self._build_task_status()
        
        # Memoria
        status["memory"] = self._build_memory_status()
        
        # Planificación
        status["planning"] = self._build_planning_status()
        
        # Papers integrados
        status["papers_integrated"] = self._build_papers_status()
        
        # Servicios
        status["services"] = self._build_services_status()
        
        return status
    
    def _build_task_status(self) -> Dict[str, Any]:
        """Construir estado de tareas."""
        if not hasattr(self.agent, 'task_manager'):
            return {}
        
        try:
            task_stats = self.agent.task_manager.get_stats()
            return {
                "queue_size": task_stats.get("queue_size", 0),
                "active_tasks": task_stats.get("active_tasks", 0),
                "completed_tasks": task_stats.get("completed_tasks", 0),
                "failed_tasks": task_stats.get("failed_tasks", 0)
            }
        except Exception as e:
            logger.warning(f"Error building task status: {e}")
            return {}
    
    def _build_memory_status(self) -> Dict[str, Any]:
        """Construir estado de memoria."""
        memory_status = {}
        
        # Memoria episódica
        if hasattr(self.agent, 'episodic_memory'):
            try:
                memories = getattr(self.agent.episodic_memory, 'memories', [])
                memory_status["episodic_count"] = len(memories)
            except Exception as e:
                logger.warning(f"Error getting episodic memory: {e}")
                memory_status["episodic_count"] = 0
        
        # Memoria semántica
        if hasattr(self.agent, 'semantic_memory'):
            try:
                facts = getattr(self.agent.semantic_memory, 'facts', [])
                memory_status["semantic_count"] = len(facts)
            except Exception as e:
                logger.warning(f"Error getting semantic memory: {e}")
                memory_status["semantic_count"] = 0
        
        # Insights de reflexión
        if hasattr(self.agent, 'reflection_planner'):
            try:
                reflection_status = self.agent.reflection_planner.get_status()
                memory_status["insights_count"] = reflection_status.get("insights_count", 0)
                memory_status["last_reflection"] = reflection_status.get("last_reflection")
            except Exception as e:
                logger.warning(f"Error getting reflection status: {e}")
                memory_status["insights_count"] = 0
                memory_status["last_reflection"] = None
        
        return memory_status
    
    def _build_planning_status(self) -> Dict[str, Any]:
        """Construir estado de planificación."""
        if not hasattr(self.agent, 'reflection_planner'):
            return {}
        
        try:
            reflection_status = self.agent.reflection_planner.get_status()
            return {
                "current_plan_size": reflection_status.get("current_plan_size", 0),
                "planning_horizon": reflection_status.get("planning_horizon", 0),
                "last_plan_update": reflection_status.get("last_plan_update")
            }
        except Exception as e:
            logger.warning(f"Error building planning status: {e}")
            return {}
    
    def _build_papers_status(self) -> Dict[str, Any]:
        """Construir estado de papers integrados."""
        papers = {}
        
        # ReAct
        if hasattr(self.agent, 'react_strategy') and self.agent.react_strategy:
            papers["react"] = {
                "enabled": getattr(self.agent, 'react_enabled', False),
                "reasoning_steps": len(getattr(self.agent.react_strategy, 'reasoning_history', []))
            }
        
        # LATS
        if hasattr(self.agent, 'lats_strategy') and self.agent.lats_strategy:
            papers["lats"] = {
                "enabled": getattr(self.agent, 'lats_enabled', False),
                "has_search_tree": getattr(self.agent.lats_strategy, 'search_tree', None) is not None
            }
        
        # Tree of Thoughts
        if hasattr(self.agent, 'tot_strategy') and self.agent.tot_strategy:
            papers["tree_of_thoughts"] = {
                "enabled": getattr(self.agent, 'tot_enabled', False),
                "strategy": getattr(self.agent.tot_strategy, 'strategy', None)
            }
        
        # Theory of Mind
        if hasattr(self.agent, 'tom_strategy') and self.agent.tom_strategy:
            papers["theory_of_mind"] = {
                "enabled": getattr(self.agent, 'tom_enabled', False),
                "agents_modeled": len(getattr(self.agent.tom_strategy, 'agent_models', []))
            }
        
        # Personality
        if hasattr(self.agent, 'personality_strategy') and self.agent.personality_strategy:
            papers["personality_driven"] = {
                "enabled": getattr(self.agent, 'personality_enabled', False),
                "has_profile": getattr(self.agent.personality_strategy, 'personality_profile', None) is not None
            }
        
        # Toolformer
        if hasattr(self.agent, 'toolformer_strategy') and self.agent.toolformer_strategy:
            papers["toolformer"] = {
                "enabled": getattr(self.agent, 'toolformer_enabled', False),
                "tools_learned": len(getattr(self.agent.toolformer_strategy, 'learned_tools', []))
            }
        
        # Self-Initiated Learning
        if hasattr(self.agent, 'learning_manager'):
            try:
                learning_stats = self.agent.learning_manager.get_learning_stats()
                papers["self_initiated_learning"] = {
                    "enabled": getattr(self.agent.learning_manager, 'learning_enabled', False),
                    "opportunities_count": len(getattr(self.agent.learning_manager, 'learning_opportunities', [])),
                    "concepts_learned": learning_stats.get("concepts_learned", 0)
                }
            except Exception as e:
                logger.warning(f"Error getting learning stats: {e}")
                papers["self_initiated_learning"] = {
                    "enabled": False,
                    "opportunities_count": 0
                }
        
        # Autonomy Level
        if hasattr(self.agent, 'autonomy_level'):
            autonomy = self.agent.autonomy_level
            if hasattr(autonomy, 'value'):
                papers["autonomy_level"] = autonomy.value
            else:
                papers["autonomy_level"] = str(autonomy) if isinstance(autonomy, str) else "fully_autonomous"
        
        # AGI Capabilities
        if hasattr(self.agent, 'agi_manager'):
            try:
                papers["agi_capabilities"] = self.agent.agi_manager.get_capabilities_report()
            except Exception as e:
                logger.warning(f"Error getting AGI capabilities: {e}")
                papers["agi_capabilities"] = {}
        
        # Strategy Selector
        if hasattr(self.agent, 'strategy_selector'):
            try:
                papers["strategy_selector"] = self.agent.strategy_selector.get_strategy_info()
            except Exception as e:
                logger.warning(f"Error getting strategy selector info: {e}")
                papers["strategy_selector"] = {}
        
        return papers
    
    def _build_services_status(self) -> Dict[str, Any]:
        """Construir estado de servicios."""
        services = {}
        
        # LLM Service
        if hasattr(self.agent, 'llm_service'):
            try:
                services["llm_service"] = self.agent.llm_service.get_stats()
            except Exception as e:
                logger.warning(f"Error getting LLM service stats: {e}")
                services["llm_service"] = {}
        
        # LLM Tracker
        if hasattr(self.agent, 'llm_tracker'):
            try:
                services["llm_tracker"] = self.agent.llm_tracker.get_call_statistics()
            except Exception as e:
                logger.warning(f"Error getting LLM tracker stats: {e}")
                services["llm_tracker"] = {}
        
        # Async Task Manager
        if hasattr(self.agent, 'async_task_manager'):
            try:
                services["async_tasks"] = {
                    "active_count": self.agent.async_task_manager.get_active_count()
                }
            except Exception as e:
                logger.warning(f"Error getting async task manager stats: {e}")
                services["async_tasks"] = {}
        
        # Loop Coordinator
        if hasattr(self.agent, 'loop_coordinator'):
            try:
                services["loop_coordinator"] = {
                    "should_stop": getattr(self.agent.loop_coordinator, 'should_stop', False),
                    "is_idle": getattr(self.agent.loop_coordinator, 'is_idle', False),
                    "operations_count": len(getattr(self.agent.loop_coordinator, 'operations', []))
                }
            except Exception as e:
                logger.warning(f"Error getting loop coordinator stats: {e}")
                services["loop_coordinator"] = {}
        
        return services


def build_agent_status(agent) -> Dict[str, Any]:
    """
    Factory function para construir estado del agente.
    
    Args:
        agent: Instancia del agente
        
    Returns:
        Dict con estado completo
    """
    builder = StatusBuilder(agent)
    return builder.build_status()


